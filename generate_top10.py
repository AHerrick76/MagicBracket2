"""
generate_top10.py — Build the top-10% bracket from completed queue Elo ratings.

For each queue, selects the top 10% of cards by Elo (rounded up).  Normalises
Elo ratings across queues by linearly shifting each queue's block so that all
queues share the same average starting Elo (the grand mean across all selected
cards).  Outputs:

  top_10_queue.json        — card list with queue IDs and normalised starting Elos
  templates/universe.html  — static browsable gallery (no Elo / vote counts shown)

Also creates the new DB tables (votes_top10, elo_ratings_top10) and seeds them
with the normalised starting Elos.  Existing rows are left untouched, so this
script is safe to re-run for diagnostics without corrupting live data.

Usage:
    DATABASE_URL=postgresql://... python generate_top10.py
    DATABASE_URL=postgresql://... python generate_top10.py --dry-run
"""

import argparse
import json
import math
import os
import sys
import webbrowser
from collections import defaultdict
from datetime import datetime

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_data import load_processed_cards

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true',
                    help='Print plan without writing to disk or DB')
parser.add_argument('--html-only', action='store_true',
                    help='Regenerate universe.html without touching top_10_queue.json or the DB')
parser.add_argument('--out-json', default='top_10_queue.json',
                    help='Output path for the queue JSON (default: top_10_queue.json)')
parser.add_argument('--out-html', default=None,
                    help='Output path for universe HTML (default: templates/universe.html)')
args = parser.parse_args()

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
QUEUES_PATH  = os.path.join(BASE_DIR, 'queues.json')
JSON_OUT     = os.path.join(BASE_DIR, args.out_json)
HTML_OUT     = args.out_html or os.path.join(BASE_DIR, 'templates', 'universe.html')
INITIAL_ELO  = 1500.0

DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)

# ── Load queues (skipped in --html-only mode) ─────────────────────────────────

if args.html_only:
    print(f'Loading card list from {JSON_OUT}...')
    with open(JSON_OUT, encoding='utf-8') as f:
        _existing = json.load(f)
    card_records = _existing['cards']
    grand_mean   = _existing.get('grand_mean_elo', INITIAL_ELO)
    print(f'{len(card_records)} cards loaded.')
else:
    print('Loading queues...')
    with open(QUEUES_PATH, encoding='utf-8') as f:
        queues_data = json.load(f)

    queue_map = {q['id']: q['cards'] for q in queues_data['queues']}
    card_to_queue = {}
    for qid, cards in queue_map.items():
        for name in cards:
            card_to_queue[name] = qid

    all_queued = set(card_to_queue)
    print(f'{len(all_queued)} cards across {len(queue_map)} queues.')

    # ── Load Elo ratings from DB ───────────────────────────────────────────────

    print('Loading Elo ratings...')
    elo_map = {}  # card_name → float

    if DATABASE_URL:
        conn = psycopg2.connect(DATABASE_URL)
        elo_df = pd.read_sql(
            'SELECT card_name, rating FROM elo_ratings WHERE card_name = ANY(%s)',
            conn, params=(list(all_queued),)
        )
        conn.close()
        elo_map = dict(zip(elo_df['card_name'], elo_df['rating'].astype(float)))
        print(f'Loaded Elos for {len(elo_map):,} / {len(all_queued):,} cards.')
    else:
        print('Warning: DATABASE_URL not set — using INITIAL_ELO for all cards.')

def get_elo(name):
    return elo_map.get(name, INITIAL_ELO)

if not args.html_only:
    # ── Select top 10% per queue ───────────────────────────────────────────────

    print('\nSelecting top 10% per queue...')
    selected_by_queue = {}  # qid → [(name, elo), ...]

    for qid, cards in queue_map.items():
        if qid == 1:
            continue
        cutoff = math.ceil(len(cards) * 0.10)
        ranked = sorted(cards, key=lambda c: get_elo(c), reverse=True)
        top_cards = [(name, get_elo(name)) for name in ranked[:cutoff]]
        selected_by_queue[qid] = top_cards
        print(f'  Q{qid}: {len(cards)} cards → top {cutoff} '
              f'(Elo range {top_cards[-1][1]:.0f}–{top_cards[0][1]:.0f})')

    total_selected = sum(len(v) for v in selected_by_queue.values())
    print(f'\nTotal selected: {total_selected:,} cards')

    # ── Normalise Elos across queues ───────────────────────────────────────────

    all_raw_elos = [elo for cards in selected_by_queue.values() for _, elo in cards]
    grand_mean   = sum(all_raw_elos) / len(all_raw_elos)
    print(f'\nGrand mean Elo (pre-normalisation): {grand_mean:.1f}')

    print('\nPer-queue normalisation shifts:')
    normalised_by_queue = {}
    for qid, cards in selected_by_queue.items():
        queue_mean = sum(elo for _, elo in cards) / len(cards)
        shift      = grand_mean - queue_mean
        normalised = [(name, round(elo + shift, 2)) for name, elo in cards]
        normalised_by_queue[qid] = normalised
        print(f'  Q{qid}: queue mean {queue_mean:.1f} → shift {shift:+.1f}  '
              f'(normalised range {normalised[-1][1]:.0f}–{normalised[0][1]:.0f})')

    # ── Build flat card list ───────────────────────────────────────────────────

    card_records = []
    for qid, cards in normalised_by_queue.items():
        for name, norm_elo in cards:
            card_records.append({'name': name, 'queue_id': qid, 'normalized_elo': norm_elo})

    card_records.sort(key=lambda c: c['name'])

    check_mean = sum(r['normalized_elo'] for r in card_records) / len(card_records)
    print(f'\nPost-normalisation grand mean: {check_mean:.1f}  (target {grand_mean:.1f})')

# ── Write top_10_queue.json ────────────────────────────────────────────────────

output = {
    'generated_at':   datetime.now().isoformat(),
    'total_cards':    len(card_records),
    'grand_mean_elo': round(grand_mean, 2),
    'cards':          card_records,
}

if args.dry_run or args.html_only:
    print(f'\n[{"dry-run" if args.dry_run else "html-only"}] Skipping {JSON_OUT}')
else:
    with open(JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\nWrote {JSON_OUT}')

# ── Init DB tables and seed starting Elos ─────────────────────────────────────

if args.dry_run or args.html_only:
    print(f'[{"dry-run" if args.dry_run else "html-only"}] Skipping DB init.')
elif DATABASE_URL:
    print('\nInitialising DB tables...')
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS votes_top10 (
            id          SERIAL  PRIMARY KEY,
            timestamp   TEXT    NOT NULL,
            ip_address  TEXT,
            session_id  TEXT,
            card_a      TEXT    NOT NULL,
            card_b      TEXT    NOT NULL,
            chosen      TEXT    NOT NULL,
            config_name TEXT,
            device      TEXT
        )
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS elo_ratings_top10 (
            card_name    TEXT    PRIMARY KEY,
            rating       REAL    NOT NULL DEFAULT 1500.0,
            wins         INTEGER NOT NULL DEFAULT 0,
            losses       INTEGER NOT NULL DEFAULT 0,
            last_updated TEXT
        )
    ''')

    # Seed starting Elos — ON CONFLICT DO NOTHING so re-runs are safe
    rows = [(r['name'], r['normalized_elo'], 0, 0) for r in card_records]
    execute_values(
        cur,
        'INSERT INTO elo_ratings_top10 (card_name, rating, wins, losses) '
        'VALUES %s ON CONFLICT DO NOTHING',
        rows,
    )
    conn.commit()
    conn.close()
    print(f'DB ready. {len(rows)} cards seeded into elo_ratings_top10.')
else:
    print('Warning: DATABASE_URL not set — skipping DB initialisation.')

# ── Generate universe.html ─────────────────────────────────────────────────────

print('\nGenerating universe.html...')

selected_names = set(r['name'] for r in card_records)
name_to_queue  = {r['name']: r['queue_id'] for r in card_records}

df = load_processed_cards()
df = df[df['name'].isin(selected_names)].copy()
df['queue_id'] = df['name'].map(name_to_queue)

# Override promo/deficient images with preferred printings
_IMAGE_OVERRIDES = {
    'Sothera, the Supervoid':  'https://cards.scryfall.io/large/front/e/9/e99d6fc0-dcf2-4b25-81c2-02c230a36246.jpg',
    'The Wandering Emperor':   'https://cards.scryfall.io/large/front/f/a/fab2d8a9-ab4c-4225-a570-22636293c17d.jpg',
}
for _n, _url in _IMAGE_OVERRIDES.items():
    df.loc[df['name'] == _n, 'img_front'] = _url

ALL_TYPES = ['Creature', 'Instant', 'Sorcery', 'Enchantment',
             'Artifact', 'Planeswalker', 'Land', 'Battle']

def get_primary_types(type_line):
    tl = str(type_line)
    found = [t for t in ALL_TYPES if t in tl]
    return found if found else ['Other']

def color_sort_key(row):
    if 'Land' in str(row.get('type_line', '')):
        return 7
    colors = row.get('colors')
    if colors is None or (hasattr(colors, '__len__') and len(colors) == 0):
        return 6
    try:
        c = list(colors)
    except TypeError:
        return 6
    if len(c) > 1:
        return 5
    if len(c) == 0:
        return 6
    return {'W': 0, 'U': 1, 'B': 2, 'R': 3, 'G': 4}.get(c[0], 6)

df['color_sort']    = df.apply(color_sort_key, axis=1)
df['primary_types'] = df['type_line'].apply(get_primary_types)

cards_list = []
for _, row in df.iterrows():
    name = row['name']
    colors_raw = row.get('colors')
    try:
        colors = list(colors_raw) if colors_raw is not None else []
    except TypeError:
        colors = []

    cards_list.append({
        'name':        name,
        'set':         str(row.get('set', '')),
        'set_name':    str(row.get('set_name', '')),
        'released_at': str(row['released_at'])[:10],
        'rarity':      str(row.get('rarity', '')),
        'type_line':   str(row.get('type_line', '')),
        'types':       get_primary_types(str(row.get('type_line', ''))),
        'cmc':         float(row['cmc']) if pd.notna(row.get('cmc')) else 0.0,
        'colors':      colors,
        'color_sort':  int(row['color_sort']),
        'front':       str(row.get('img_front', '') or ''),
        'back':        str(row.get('img_back', '') or '') or None,
        'queue':       int(row['queue_id']),
    })

# Sort: released_at → set_name → color_sort → cmc → name
cards_list.sort(key=lambda c: (c['released_at'], c['set_name'], c['color_sort'], c['cmc'], c['name']))

queue_ids    = sorted(set(r['queue_id'] for r in card_records))
queue_meta   = [{'id': qid, 'size': len([r for r in card_records if r['queue_id'] == qid])}
                for qid in queue_ids]

cards_json_str  = json.dumps(cards_list, ensure_ascii=False)
queue_meta_json = json.dumps(queue_meta)
total_cards     = len(cards_list)
n_queues        = len(queue_ids)

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Top 10% Universe</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #1a1a1a; color: #ddd; font-family: sans-serif; }

/* ── Page header ── */
#page-header {
  background: #111; padding: 10px 14px 8px;
  border-bottom: 1px solid #2a2a2a;
  display: flex; align-items: baseline; gap: 1rem;
}
#page-header h1 { font-size: 1rem; color: #c9a84c; letter-spacing: 0.04em; }
#page-header p  { font-size: 0.75rem; color: #666; }

/* ── Filter bar — sticky, collapses on scroll ── */
#filter-bar {
  position: sticky; top: 0; z-index: 100;
  background: #222; border-bottom: 1px solid #3a3a3a;
  overflow: hidden;
  transition: max-height 0.28s ease, padding 0.28s ease;
  max-height: 600px; /* enough for any expanded state */
}
#filter-bar.collapsed {
  max-height: 36px;
}
#filter-bar-handle {
  display: flex; justify-content: space-between; align-items: center;
  padding: 6px 14px; min-height: 36px; cursor: pointer;
  user-select: none;
}
#filter-bar-handle .handle-left {
  font-size: 0.7rem; color: #888; display: flex; align-items: center; gap: 8px;
}
#filter-bar-handle .handle-left em {
  color: #c9a84c; font-style: normal; font-weight: 600;
}
#filter-toggle-btn {
  font-size: 0.68rem; padding: 2px 8px;
  background: #333; border: 1px solid #555; border-radius: 4px;
  color: #999; cursor: pointer; white-space: nowrap;
}
#filter-bar-inner { padding: 0 14px 8px; }

.filter-row {
  display: flex; flex-wrap: wrap; gap: 6px 12px;
  align-items: center; margin-bottom: 5px;
}
.filter-row:last-child { margin-bottom: 0; }
.filter-label { font-size: 0.7rem; color: #777; white-space: nowrap; }

.filter-input {
  background: #2e2e2e; border: 1px solid #444; border-radius: 5px;
  color: #ddd; font-size: 0.83rem; padding: 3px 9px;
}
.filter-input:focus { outline: none; border-color: #c9a84c; }
#name-filter { width: 150px; }
#set-filter  { width: 120px; }

.pill {
  cursor: pointer; border: 1px solid #555; border-radius: 11px;
  padding: 2px 8px; font-size: 0.72rem; color: #999;
  background: #2e2e2e; user-select: none;
  transition: background 0.1s, color 0.1s, border-color 0.1s;
  white-space: nowrap;
}
.pill.active { background: #c9a84c; border-color: #c9a84c; color: #111; font-weight: 600; }
.pill.q-pill { font-size: 0.68rem; padding: 1px 6px; }
.pill.q-pill.active { background: #3a6ea0; border-color: #3a6ea0; color: #ddd; }

.pill.c-W.active { background: #e8e4d0; border-color: #e8e4d0; color: #111; }
.pill.c-U.active { background: #3a78c9; border-color: #3a78c9; color: #fff; }
.pill.c-B.active { background: #3a3a3a; border-color: #888;    color: #ddd; }
.pill.c-R.active { background: #c94444; border-color: #c94444; color: #fff; }
.pill.c-G.active { background: #2e8c50; border-color: #2e8c50; color: #fff; }
.pill.c-M.active { background: #a07830; border-color: #c9a84c; color: #eee; }
.pill.c-X.active { background: #666;    border-color: #999;    color: #eee; }
.pill.c-L.active { background: #5a7a50; border-color: #8aaa80; color: #eee; }

.pill[data-rarity="common"].active   { background: #888; border-color: #aaa; color: #111; }
.pill[data-rarity="uncommon"].active { background: #6090b8; border-color: #6090b8; color: #fff; }
.pill[data-rarity="rare"].active     { background: #c9b040; border-color: #c9b040; color: #111; }
.pill[data-rarity="mythic"].active   { background: #d06020; border-color: #d06020; color: #fff; }

.filter-select {
  background: #2e2e2e; border: 1px solid #444; border-radius: 5px;
  color: #ddd; font-size: 0.8rem; padding: 3px 7px; cursor: pointer;
}
.filter-select:focus { outline: none; border-color: #c9a84c; }

.mini-btn {
  font-size: 0.68rem; padding: 2px 7px;
  background: #333; border: 1px solid #555; border-radius: 4px;
  color: #999; cursor: pointer;
}
.mini-btn:hover { border-color: #888; color: #ddd; }

/* ── Cols segmented control (mobile only) ── */
.cols-control {
  display: flex; border: 1px solid #555; border-radius: 4px; overflow: hidden;
}
.cols-btn {
  font-size: 0.7rem; padding: 2px 8px;
  background: #2e2e2e; border: none; border-right: 1px solid #555;
  color: #888; cursor: pointer;
}
.cols-btn:last-child { border-right: none; }
.cols-btn.active { background: #444; color: #ddd; font-weight: 600; }

/* ── Stats bar ── */
#stats-bar {
  padding: 5px 14px; font-size: 0.78rem; color: #777;
  background: #1d1d1d; border-bottom: 1px solid #2a2a2a;
}
#stats-bar em { color: #c9a84c; font-style: normal; }

/* ── Grid — CSS grid, columns driven by --cols var ── */
#grid-container { padding: 12px 14px; }

.set-header {
  grid-column: 1 / -1;
  padding: 10px 2px 5px;
  font-size: 0.78rem; color: #c9a84c;
  border-bottom: 1px solid #333; margin-bottom: 4px;
  letter-spacing: 0.02em;
}
.set-header .year { color: #555; }

#grid {
  display: grid;
  grid-template-columns: repeat(var(--cols, 2), 1fr);
  gap: 8px;
}

.card-wrap { display: flex; flex-direction: column; align-items: center; width: 100%; }
.card-wrap img { width: 100%; border-radius: 6px; display: block; }
.card-wrap img.dfc { cursor: crosshair; outline: 2px solid #c9a84c; outline-offset: 2px; }

.card-label {
  font-size: 0.6rem; color: #888; text-align: center;
  margin-top: 3px; line-height: 1.4; word-break: break-word; width: 100%;
}
.card-label strong    { color: #ccc; font-size: 0.63rem; display: block; }
.card-label .set-line { color: #666; }
.card-label .q-tag {
  display: inline-block; background: #1e3a58; color: #6aacdf;
  font-size: 0.58rem; padding: 0 4px; border-radius: 2px;
}
.rc { color: #999; } .ru { color: #6aacdf; } .rr { color: #d4b040; } .rm { color: #d06020; }

/* ── DFC hover preview ── */
#preview {
  display: none; position: fixed; top: 50%; left: 50%;
  transform: translate(-50%, -50%); gap: 12px;
  background: rgba(0,0,0,0.92); padding: 16px; border-radius: 14px;
  z-index: 300; pointer-events: none;
}
#preview.visible { display: flex; }
#preview img { width: 280px; border-radius: 10px; }

#no-results {
  display: none; grid-column: 1 / -1; text-align: center;
  padding: 60px; color: #555; font-size: 1rem;
}
#load-more-wrap { text-align: center; padding: 20px; }
#load-more {
  background: #2e2e2e; border: 1px solid #555; border-radius: 6px;
  color: #aaa; font-size: 0.85rem; padding: 8px 24px; cursor: pointer;
}
#load-more:hover { border-color: #c9a84c; color: #c9a84c; }

/* ── Desktop: fixed-width cards, always-visible filter bar ── */
@media (min-width: 769px) {
  #filter-bar { max-height: none !important; overflow: visible; }
  #filter-bar-handle { display: none; }
  #filter-bar-inner { padding-top: 8px; }
  .cols-control, .cols-label { display: none; }
  #grid { grid-template-columns: repeat(auto-fill, minmax(270px, 1fr)); }
}

/* ── Mobile: collapsible bar, variable columns ── */
@media (max-width: 768px) {
  #filter-bar-inner { padding-top: 0; }
  #name-filter { width: 120px; }
  #set-filter  { width: 100px; }
}
</style>
</head>
<body>

<div id="page-header">
  <h1>Top 10% Universe</h1>
  <p>__TOTAL__ cards &mdash; static snapshot</p>
</div>

<div id="filter-bar">
  <!-- Always-visible handle row -->
  <div id="filter-bar-handle">
    <span class="handle-left">
      Showing <em id="stats-shown">&hellip;</em> of __TOTAL__
    </span>
    <button id="filter-toggle-btn">Filters ▲</button>
  </div>

  <!-- Collapsible filter content -->
  <div id="filter-bar-inner">
    <div class="filter-row">
      <span class="filter-label">Queue</span>
      <div id="queue-pills" style="display:flex;flex-wrap:wrap;gap:4px"></div>
      <button class="mini-btn" id="all-queues">All</button>
      <button class="mini-btn" id="no-queues">None</button>
    </div>
    <div class="filter-row">
      <span class="filter-label">Rarity</span>
      <span class="pill active" data-rarity="common">C</span>
      <span class="pill active" data-rarity="uncommon">U</span>
      <span class="pill active" data-rarity="rare">R</span>
      <span class="pill active" data-rarity="mythic">M</span>
      <span class="filter-label" style="margin-left:6px">Color</span>
      <span class="pill c-W active" data-color="W">W</span>
      <span class="pill c-U active" data-color="U">U</span>
      <span class="pill c-B active" data-color="B">B</span>
      <span class="pill c-R active" data-color="R">R</span>
      <span class="pill c-G active" data-color="G">G</span>
      <span class="pill c-M active" data-color="M">Multi</span>
      <span class="pill c-X active" data-color="X">Colorless</span>
      <span class="pill c-L active" data-color="L">Land</span>
    </div>
    <div class="filter-row">
      <span class="filter-label">Type</span>
      <div id="type-pills" style="display:flex;flex-wrap:wrap;gap:4px"></div>
    </div>
    <div class="filter-row">
      <input class="filter-input" id="name-filter" type="text" placeholder="Card name&hellip;">
      <input class="filter-input" id="set-filter"  type="text" placeholder="Set name&hellip;">
      <label style="font-size:0.72rem;color:#888;cursor:pointer;display:flex;align-items:center;gap:4px">
        <input type="checkbox" id="group-by-set" checked> Group by set
      </label>
      <label style="font-size:0.72rem;color:#888;cursor:pointer;display:flex;align-items:center;gap:4px">
        <input type="checkbox" id="dfc-only"> DFC only
      </label>
      <span class="filter-label" style="margin-left:4px">Cols</span>
      <div class="cols-control">
        <button class="cols-btn" data-cols="1">1</button>
        <button class="cols-btn active" data-cols="2">2</button>
        <button class="cols-btn" data-cols="3">3</button>
      </div>
    </div>
  </div>
</div>

<div id="grid-container">
  <div id="grid">
    <div id="no-results">No cards match the current filters.</div>
  </div>
  <div id="load-more-wrap" style="display:none">
    <button id="load-more">Load more cards</button>
    <span id="load-more-label" style="font-size:0.75rem;color:#666;margin-left:10px"></span>
  </div>
</div>

<div id="preview">
  <img id="prev-front" src="" alt="Front">
  <img id="prev-back"  src="" alt="Back">
</div>

<script>
const CARDS      = __CARDS_JSON__;
const QUEUE_META = __QUEUE_META_JSON__;
const PAGE_SIZE  = 400;

const activeQueues = new Set(QUEUE_META.map(q => q.id));
const activeRarity = new Set(['common','uncommon','rare','mythic','special']);
const activeColors = new Set(['W','U','B','R','G','M','X','L']);
const activeTypes  = new Set(['Creature','Instant','Sorcery','Enchantment',
                               'Artifact','Planeswalker','Land','Battle','Other']);

let visibleCards = [];
let renderOffset = 0;

// ── Columns control (mobile only — desktop uses CSS auto-fill) ────────────────
const gridEl = document.getElementById('grid');
gridEl.style.setProperty('--cols', '2');
document.querySelectorAll('.cols-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.cols-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    gridEl.style.setProperty('--cols', btn.dataset.cols);
  });
});

// ── Filter bar collapse / scroll ──────────────────────────────────────────────
const filterBar    = document.getElementById('filter-bar');
const toggleBtn    = document.getElementById('filter-toggle-btn');
let filterCollapsed = false;
let _manualClosed   = false;  // true when user explicitly closed via button
let lastScrollY     = window.scrollY;
let scrollTimer     = null;

function setCollapsed(collapsed, fromScroll = false) {
  if (fromScroll && collapsed === false && _manualClosed) return; // button close overrides scroll-up
  filterCollapsed = collapsed;
  filterBar.classList.toggle('collapsed', collapsed);
  toggleBtn.textContent = collapsed ? 'Filters ▼' : 'Filters ▲';
}

toggleBtn.addEventListener('click', e => {
  e.stopPropagation();
  const closing = !filterCollapsed;
  _manualClosed = closing;
  setCollapsed(closing);
});

document.getElementById('filter-bar-handle').addEventListener('click', () => {
  if (filterCollapsed) { _manualClosed = false; setCollapsed(false); }
});

const isMobile = () => window.innerWidth <= 768;

window.addEventListener('scroll', () => {
  if (!isMobile()) return;
  clearTimeout(scrollTimer);
  scrollTimer = setTimeout(() => {
    const y = window.scrollY;
    if (y > lastScrollY + 40 && y > 80)  setCollapsed(true,  true);
    if (y < lastScrollY - 20)             setCollapsed(false, true);
    lastScrollY = y;
  }, 50);
}, { passive: true });

// Re-expand if user rotates to desktop width
window.addEventListener('resize', () => {
  if (!isMobile()) setCollapsed(false);
}, { passive: true });

// ── Queue pills ───────────────────────────────────────────────────────────────
const qPillsEl = document.getElementById('queue-pills');
QUEUE_META.forEach(q => {
  const p = document.createElement('span');
  p.className = 'pill q-pill active';
  p.dataset.queue = q.id;
  p.textContent = 'Q' + q.id;
  p.title = q.size + ' cards';
  p.addEventListener('click', () => {
    p.classList.toggle('active');
    p.classList.contains('active') ? activeQueues.add(q.id) : activeQueues.delete(q.id);
    applyFilters();
  });
  qPillsEl.appendChild(p);
});
document.getElementById('all-queues').addEventListener('click', () => {
  qPillsEl.querySelectorAll('.pill').forEach(p => { p.classList.add('active'); activeQueues.add(+p.dataset.queue); });
  applyFilters();
});
document.getElementById('no-queues').addEventListener('click', () => {
  qPillsEl.querySelectorAll('.pill').forEach(p => { p.classList.remove('active'); activeQueues.delete(+p.dataset.queue); });
  applyFilters();
});

// ── Type pills ────────────────────────────────────────────────────────────────
const TYPE_LABELS = {Creature:'Creature',Instant:'Instant',Sorcery:'Sorcery',
  Enchantment:'Enchant',Artifact:'Artifact',Planeswalker:'PW',Land:'Land',Battle:'Battle',Other:'Other'};
const typePillsEl = document.getElementById('type-pills');
Object.entries(TYPE_LABELS).forEach(([t, label]) => {
  const p = document.createElement('span');
  p.className = 'pill active';
  p.dataset.type = t;
  p.textContent = label;
  p.addEventListener('click', () => {
    p.classList.toggle('active');
    p.classList.contains('active') ? activeTypes.add(t) : activeTypes.delete(t);
    applyFilters();
  });
  typePillsEl.appendChild(p);
});

// ── Pill + input listeners ────────────────────────────────────────────────────
document.querySelectorAll('[data-rarity]').forEach(p =>
  p.addEventListener('click', () => {
    p.classList.toggle('active');
    const r = p.dataset.rarity;
    p.classList.contains('active') ? activeRarity.add(r) : activeRarity.delete(r);
    applyFilters();
  })
);
document.querySelectorAll('[data-color]').forEach(p =>
  p.addEventListener('click', () => {
    p.classList.toggle('active');
    const c = p.dataset.color;
    p.classList.contains('active') ? activeColors.add(c) : activeColors.delete(c);
    applyFilters();
  })
);

let nameTimer = null;
document.getElementById('name-filter').addEventListener('input', () => {
  clearTimeout(nameTimer); nameTimer = setTimeout(applyFilters, 200);
});
let setTimer = null;
document.getElementById('set-filter').addEventListener('input', () => {
  clearTimeout(setTimer); setTimer = setTimeout(applyFilters, 200);
});
['group-by-set','dfc-only'].forEach(id =>
  document.getElementById(id).addEventListener('change', applyFilters)
);

// ── Filter + sort ─────────────────────────────────────────────────────────────
function colorKey(card) {
  if (card.color_sort === 7) return 'L';
  if (card.color_sort === 6) return 'X';
  if (card.color_sort === 5) return 'M';
  return (card.colors && card.colors[0]) || 'X';
}

function applyFilters() {
  const nameQ   = document.getElementById('name-filter').value.toLowerCase().trim();
  const setQ    = document.getElementById('set-filter').value.toLowerCase().trim();
  const dfcOnly = document.getElementById('dfc-only').checked;

  visibleCards = CARDS.filter(c => {
    if (!activeQueues.has(c.queue))                           return false;
    if (!activeRarity.has(c.rarity))                         return false;
    if (!activeColors.has(colorKey(c)))                      return false;
    if (!c.types.some(t => activeTypes.has(t)))              return false;
    if (nameQ && !c.name.toLowerCase().includes(nameQ))      return false;
    if (setQ  && !c.set_name.toLowerCase().includes(setQ)
              && !c.set.toLowerCase().includes(setQ))        return false;
    if (dfcOnly && !c.back)                                  return false;
    return true;
  });

  document.getElementById('stats-shown').textContent = visibleCards.length.toLocaleString();
  renderOffset = 0;
  renderPage(true);
}

// ── Render ────────────────────────────────────────────────────────────────────
const RARITY_CLS = {common:'rc', uncommon:'ru', rare:'rr', mythic:'rm'};
const RARITY_LBL = {common:'C', uncommon:'U', rare:'R', mythic:'M', special:'S'};
const noResults    = document.getElementById('no-results');
const loadMoreWrap = document.getElementById('load-more-wrap');
const loadMoreBtn  = document.getElementById('load-more');
const loadMoreLbl  = document.getElementById('load-more-label');
const preview      = document.getElementById('preview');
const prevF        = document.getElementById('prev-front');
const prevB        = document.getElementById('prev-back');

function renderPage(reset) {
  if (reset) {
    // Remove all children except #no-results
    Array.from(gridEl.children).forEach(el => { if (el.id !== 'no-results') el.remove(); });
    noResults.style.display = 'none';
  }
  if (visibleCards.length === 0) {
    noResults.style.display = 'block';
    loadMoreWrap.style.display = 'none';
    return;
  }

  const groupBySet = document.getElementById('group-by-set').checked;
  const slice      = visibleCards.slice(renderOffset, renderOffset + PAGE_SIZE);
  const frag       = document.createDocumentFragment();
  let lastKey      = reset ? null : gridEl.dataset.lastKey;

  slice.forEach(c => {
    if (groupBySet) {
      const key = c.set;
      if (key !== lastKey) {
        const hdr = document.createElement('div');
        hdr.className = 'set-header';
        hdr.innerHTML = c.set_name + ' <span class="year">(' + c.released_at.slice(0,4) + ')</span>';
        frag.appendChild(hdr);
        lastKey = key;
      }
    }

    const wrap = document.createElement('div');
    wrap.className = 'card-wrap';

    const img = document.createElement('img');
    img.src     = c.front;
    img.alt     = c.name;
    img.loading = 'lazy';

    if (c.back) {
      img.classList.add('dfc');
      img.addEventListener('mouseenter', () => {
        prevF.src = c.front; prevB.src = c.back;
        preview.classList.add('visible');
      });
      img.addEventListener('mouseleave', () => preview.classList.remove('visible'));
    }

    const rcls = RARITY_CLS[c.rarity] || '';
    const rlbl = RARITY_LBL[c.rarity] || c.rarity;

    const label = document.createElement('div');
    label.className = 'card-label';
    label.innerHTML =
      '<span class="q-tag">Q' + c.queue + '</span> <span class="' + rcls + '">' + rlbl + '</span><br>' +
      '<strong>' + c.name + '</strong>' +
      '<span class="set-line">' + c.set_name + '</span>';

    wrap.appendChild(img);
    wrap.appendChild(label);
    frag.appendChild(wrap);
  });

  gridEl.dataset.lastKey = lastKey;
  gridEl.appendChild(frag);
  renderOffset += slice.length;

  const remaining = visibleCards.length - renderOffset;
  if (remaining > 0) {
    loadMoreWrap.style.display = 'block';
    loadMoreLbl.textContent = remaining.toLocaleString() + ' more';
  } else {
    loadMoreWrap.style.display = 'none';
  }
}

loadMoreBtn.addEventListener('click', () => renderPage(false));
applyFilters();
</script>
</body>
</html>'''

html = (HTML
    .replace('__CARDS_JSON__',      cards_json_str)
    .replace('__QUEUE_META_JSON__', queue_meta_json)
    .replace('__TOTAL__',           str(total_cards))
    .replace('__NQUEUES__',         str(n_queues))
)

if args.dry_run:
    print(f'[dry-run] Would write universe HTML to {HTML_OUT}')
else:
    os.makedirs(os.path.dirname(HTML_OUT), exist_ok=True)
    with open(HTML_OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Wrote {HTML_OUT}')
    webbrowser.open(f'file:///{HTML_OUT.replace(os.sep, "/")}')

print('\nDone.')
