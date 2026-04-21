"""
generate_card_browser.py — Generate the two-tab public card browser.

Tab 1 — Top 10%: ~1,500 cards from the top-10% tournament, sorted by
         tournament Elo, with global tournament rank shown.
Tab 2 — All Cards: all ~15,000 queued cards, with per-queue Elo and
         intra-queue rank shown.

Vote counts are intentionally omitted from all public displays.

Usage:
    DATABASE_URL=postgresql://... python generate_card_browser.py
    DATABASE_URL=postgresql://... python generate_card_browser.py --out templates/universe.html
"""

import argparse
import json
import os
import sys
import webbrowser
from collections import defaultdict

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_data import load_processed_cards

DATABASE_URL    = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)
QUEUES_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queues.json')
TOP10_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'top_10_queue.json')

parser = argparse.ArgumentParser(description='Generate the two-tab public card browser.')
parser.add_argument('--out', type=str, default=None, help='Output HTML path (default: templates/universe.html)')
args = parser.parse_args()

# ── Load queues ───────────────────────────────────────────────────────────────

print('Loading queues...')
with open(QUEUES_PATH) as f:
    queues_data = json.load(f)

card_to_queue = {}
queue_sizes   = {}
for q in queues_data['queues']:
    qid = q['id']
    queue_sizes[qid] = len(q['cards'])
    for name in q['cards']:
        card_to_queue[name] = qid

queued_cards = set(card_to_queue)
print(f'{len(queued_cards)} cards across {len(queues_data["queues"])} queues.')

# ── Load top-10% card set ──────────────────────────────────────────────────────

top10_cards = set()
if os.path.exists(TOP10_JSON_PATH):
    with open(TOP10_JSON_PATH, encoding='utf-8') as f:
        top10_data = json.load(f)
    top10_cards = {e['name'] for e in top10_data['cards']}
    print(f'{len(top10_cards)} cards in top-10% pool.')
else:
    print('top_10_queue.json not found; top-10% tab will be empty.')

# ── Load card metadata ────────────────────────────────────────────────────────

print('Loading card data...')
df = load_processed_cards()
df = df[df['name'].isin(queued_cards)].copy()
df['queue_id'] = df['name'].map(card_to_queue)

_IMAGE_OVERRIDES = {
    'Sothera, the Supervoid': 'https://cards.scryfall.io/large/front/e/9/e99d6fc0-dcf2-4b25-81c2-02c230a36246.jpg',
    'The Wandering Emperor':  'https://cards.scryfall.io/large/front/f/a/fab2d8a9-ab4c-4225-a570-22636293c17d.jpg',
}
for _n, _url in _IMAGE_OVERRIDES.items():
    df.loc[df['name'] == _n, 'img_front'] = _url

ALL_TYPES = ['Creature', 'Instant', 'Sorcery', 'Enchantment',
             'Artifact', 'Planeswalker', 'Land', 'Battle']


def get_primary_types(type_line):
    tl    = str(type_line)
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

# ── Load Elo ratings ──────────────────────────────────────────────────────────

print('Loading Elo ratings...')
INITIAL_ELO   = 1500.0
elo_map       = {}   # card_name → float (full-queue phase Elo)
elo_top10_map = {}   # card_name → float (top-10% tournament Elo)

if DATABASE_URL:
    try:
        conn   = psycopg2.connect(DATABASE_URL)
        elo_df = pd.read_sql(
            'SELECT card_name, rating FROM elo_ratings WHERE card_name = ANY(%s)',
            conn, params=(list(queued_cards),),
        )
        elo_map = {r['card_name']: float(r['rating']) for _, r in elo_df.iterrows()}
        print(f'Full-queue Elo loaded for {len(elo_map):,} / {len(queued_cards):,} cards.')

        if top10_cards:
            t10_df = pd.read_sql(
                'SELECT card_name, rating FROM elo_ratings_top10 WHERE card_name = ANY(%s)',
                conn, params=(list(top10_cards),),
            )
            elo_top10_map = {r['card_name']: float(r['rating']) for _, r in t10_df.iterrows()}
            print(f'Top-10% Elo loaded for {len(elo_top10_map):,} cards.')

        conn.close()
    except Exception as e:
        print(f'Warning: could not load Elo ({e}); ratings omitted.')
else:
    print('Warning: DATABASE_URL not set; ratings will be omitted.')

# ── Compute intra-queue ranks ─────────────────────────────────────────────────

queue_card_elos = defaultdict(dict)
for _, row in df.iterrows():
    name = row['name']
    elo  = elo_map.get(name, INITIAL_ELO)
    queue_card_elos[row['queue_id']][name] = elo

iq_rank_map = {}
iq_pct_map  = {}
for qid, card_elos in queue_card_elos.items():
    sorted_cards = sorted(card_elos.items(), key=lambda x: (-x[1], x[0]))
    n = len(sorted_cards)
    for rank_0, (name, _) in enumerate(sorted_cards):
        rank = rank_0 + 1
        iq_rank_map[name] = rank
        iq_pct_map[name]  = round((n - rank) / max(n - 1, 1) * 100, 1)

# ── Compute top-10% tournament ranks ─────────────────────────────────────────

top10_rank_map = {}
if elo_top10_map:
    sorted_top10 = sorted(top10_cards, key=lambda n: (-(elo_top10_map.get(n, 0)), n))
    for rank_0, name in enumerate(sorted_top10):
        top10_rank_map[name] = rank_0 + 1

n_top10 = len(top10_cards)

# ── Build cards list ──────────────────────────────────────────────────────────

print('Building card data...')
cards_list = []
for _, row in df.iterrows():
    name      = row['name']
    elo       = elo_map.get(name)
    elo_top10 = elo_top10_map.get(name)
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
        'back':        str(row.get('img_back',  '') or '') or None,
        'queue':       int(row['queue_id']),
        'q_size':      queue_sizes.get(int(row['queue_id']), 0),
        'elo':         round(elo, 1)       if elo       is not None else None,
        'elo_top10':   round(elo_top10, 1) if elo_top10 is not None else None,
        'iq_rank':     iq_rank_map.get(name, 0),
        'iq_pct':      iq_pct_map.get(name, 50.0),
        'top10':       name in top10_cards,
        'top10_rank':  top10_rank_map.get(name),
    })

# Default sort: released_at → set_name → color_sort → cmc → name
cards_list.sort(key=lambda c: (c['released_at'], c['set_name'], c['color_sort'], c['cmc'], c['name']))

cards_json_str  = json.dumps(cards_list, ensure_ascii=False)
queue_meta_json = json.dumps([
    {'id': q['id'], 'size': len(q['cards'])}
    for q in queues_data['queues']
])
total_cards = len(cards_list)
n_queues    = len(queues_data['queues'])

print(f'Built {total_cards:,} card records ({len(cards_json_str) / 1e6:.1f} MB).')

# ── HTML template ─────────────────────────────────────────────────────────────

HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Card Browser &mdash; Magic Bracket</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #1a1a1a; color: #ddd; font-family: sans-serif; }

/* ── Nav ── */
nav {
  display: flex; justify-content: center; gap: 2rem;
  padding: 0.75rem 1rem;
  background: #13132a; border-bottom: 1px solid #2a2a4a;
  flex-wrap: wrap;
}
nav a { color: #c8a96e; text-decoration: none; font-size: 0.85rem; letter-spacing: 0.03em; }
nav a:hover { text-decoration: underline; }

/* ── Tab bar ── */
#tab-bar {
  display: flex; gap: 0; border-bottom: 1px solid #3a3a3a;
  background: #1d1d1d;
}
.tab-btn {
  padding: 0.6rem 1.8rem; font-size: 0.85rem; font-family: sans-serif;
  background: none; border: none; border-bottom: 3px solid transparent;
  color: #777; cursor: pointer; letter-spacing: 0.03em;
  transition: color 0.15s, border-color 0.15s;
}
.tab-btn:hover { color: #bbb; }
.tab-btn.active { color: #c9a84c; border-bottom-color: #c9a84c; font-weight: 600; }
.tab-count { font-size: 0.72rem; color: #555; margin-left: 5px; }
.tab-btn.active .tab-count { color: #9a7838; }

/* ── Filter bar ── */
#filter-bar {
  position: sticky; top: 0; z-index: 100;
  background: #222; border-bottom: 1px solid #3a3a3a;
  overflow: hidden;
  transition: max-height 0.28s ease;
  max-height: 600px;
}
#filter-bar.collapsed { max-height: 38px; }

#filter-bar-handle {
  display: flex; justify-content: space-between; align-items: center;
  padding: 6px 14px; min-height: 38px; cursor: pointer; user-select: none;
}
#filter-bar-handle .handle-left {
  font-size: 0.72rem; color: #888; display: flex; align-items: center; gap: 8px;
}
#filter-bar-handle .handle-left em { color: #c9a84c; font-style: normal; font-weight: 600; }
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
.num-input   { width: 72px; }

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

/* ── Stats bar ── */
#stats-bar {
  padding: 5px 14px; font-size: 0.78rem; color: #777;
  background: #1d1d1d; border-bottom: 1px solid #2a2a2a;
}
#stats-bar em { color: #c9a84c; font-style: normal; }

/* ── Grid ── */
#grid-container { padding: 12px 14px; }

.set-header {
  width: 100%; padding: 10px 2px 5px;
  font-size: 0.78rem; color: #c9a84c;
  border-bottom: 1px solid #333; margin-bottom: 8px;
  letter-spacing: 0.02em;
}
.set-header .year { color: #555; }

#grid { display: flex; flex-wrap: wrap; gap: 8px; }

.card-wrap { display: flex; flex-direction: column; align-items: center; width: 230px; }
.card-wrap img { width: 230px; border-radius: 6px; display: block; }
.card-wrap img.dfc { cursor: crosshair; outline: 2px solid #c9a84c; outline-offset: 2px; }

.card-label {
  font-size: 0.6rem; color: #888; text-align: center;
  margin-top: 4px; line-height: 1.5; word-break: break-word; width: 100%;
}
.card-label strong  { color: #ccc; font-size: 0.63rem; display: block; }
.card-label .set-ln { color: #666; }
.card-label .elo-v  { color: #c9a84c; }
.card-label .rank-v { color: #888; }
.q-tag {
  display: inline-block; background: #1e3a58; color: #6aacdf;
  font-size: 0.58rem; padding: 0 4px; border-radius: 2px; margin-right: 2px;
}
.rc { color: #999; } .ru { color: #6aacdf; } .rr { color: #d4b040; } .rm { color: #d06020; }

/* ── DFC hover ── */
#preview {
  display: none; position: fixed; top: 50%; left: 50%;
  transform: translate(-50%, -50%); gap: 12px;
  background: rgba(0,0,0,0.92); padding: 16px; border-radius: 14px;
  z-index: 300; pointer-events: none;
}
#preview.visible { display: flex; }
#preview img { width: 280px; border-radius: 10px; }

#no-results {
  display: none; width: 100%; text-align: center;
  padding: 60px; color: #555; font-size: 1rem;
}
#load-more-wrap { text-align: center; padding: 20px; display: none; }
#load-more {
  background: #2e2e2e; border: 1px solid #555; border-radius: 6px;
  color: #aaa; font-size: 0.85rem; padding: 8px 24px; cursor: pointer;
}
#load-more:hover { border-color: #c9a84c; color: #c9a84c; }
</style>
</head>
<body>

<nav>
  <a href="/">Home</a>
  <a href="/bracket">Full Bracket</a>
  <a href="/honorable-mentions">Honorable Mentions</a>
  <a href="/community-favorites">Community Favorites</a>
  <a href="/universe">Card Browser</a>
  <a href="/vote-counts">Vote Counts</a>
  <a href="/faq">FAQ</a>
  <a href="/share">Share</a>
</nav>

<div id="tab-bar">
  <button class="tab-btn active" id="tab-top10" onclick="switchTab('top10')">
    Top 10% <span class="tab-count" id="count-top10"></span>
  </button>
  <button class="tab-btn" id="tab-all" onclick="switchTab('all')">
    All Cards <span class="tab-count" id="count-all"></span>
  </button>
</div>

<div id="filter-bar">
  <div id="filter-bar-handle" onclick="toggleFilters()">
    <div class="handle-left">
      Showing <em id="stats-shown">&#8230;</em>
      <span id="stats-context" style="color:#555"></span>
    </div>
    <button id="filter-toggle-btn" onclick="event.stopPropagation(); toggleFilters()">Filters &#9662;</button>
  </div>
  <div id="filter-bar-inner">

    <!-- Row 1: Queue pills -->
    <div class="filter-row">
      <span class="filter-label">Queue</span>
      <div id="queue-pills" style="display:flex;flex-wrap:wrap;gap:4px"></div>
      <button class="mini-btn" id="all-queues">All</button>
      <button class="mini-btn" id="no-queues">None</button>
    </div>

    <!-- Row 2: Rarity · Color · Type · Sort · Options -->
    <div class="filter-row">
      <span class="filter-label">Rarity</span>
      <span class="pill active" data-rarity="common">C</span>
      <span class="pill active" data-rarity="uncommon">U</span>
      <span class="pill active" data-rarity="rare">R</span>
      <span class="pill active" data-rarity="mythic">M</span>

      <span class="filter-label" style="margin-left:4px">Color</span>
      <span class="pill c-W active" data-color="W">W</span>
      <span class="pill c-U active" data-color="U">U</span>
      <span class="pill c-B active" data-color="B">B</span>
      <span class="pill c-R active" data-color="R">R</span>
      <span class="pill c-G active" data-color="G">G</span>
      <span class="pill c-M active" data-color="M">Multi</span>
      <span class="pill c-X active" data-color="X">Colorless</span>
      <span class="pill c-L active" data-color="L">Land</span>

      <span class="filter-label" style="margin-left:4px">Type</span>
      <div id="type-pills" style="display:flex;flex-wrap:wrap;gap:4px"></div>

      <span class="filter-label" style="margin-left:4px">Sort</span>
      <select class="filter-select" id="sort-select">
        <option value="set">Set order</option>
        <option value="set_elo">Set order (Elo &darr;)</option>
        <option value="elo">Elo &darr;</option>
        <option value="elo_asc">Elo &uarr;</option>
        <option value="iq_pct">Queue rank &darr;</option>
        <option value="iq_pct_asc">Queue rank &uarr;</option>
        <option value="name">Name A&ndash;Z</option>
      </select>

      <label style="font-size:0.72rem;color:#888;cursor:pointer;display:flex;align-items:center;gap:4px">
        <input type="checkbox" id="group-by-set" checked> Group by set
      </label>
      <label style="font-size:0.72rem;color:#888;cursor:pointer;display:flex;align-items:center;gap:4px">
        <input type="checkbox" id="dfc-only"> DFC only
      </label>
      <label id="top10-only-wrap" style="font-size:0.72rem;color:#888;cursor:pointer;display:flex;align-items:center;gap:4px">
        <input type="checkbox" id="top10-only"> In Top 10%
      </label>
    </div>

    <!-- Row 3: Text search · Elo range · Queue rank -->
    <div class="filter-row">
      <input class="filter-input" id="name-filter" type="text" placeholder="Card name&hellip;">
      <input class="filter-input" id="set-filter"  type="text" placeholder="Set name&hellip;">
      <span class="filter-label">Min Elo</span>
      <input class="filter-input num-input" id="min-elo" type="number" placeholder="&mdash;">
      <span class="filter-label">Max Elo</span>
      <input class="filter-input num-input" id="max-elo" type="number" placeholder="&mdash;">
      <span class="filter-label">Queue rank</span>
      <select class="filter-select" id="iq-filter">
        <option value="all">All</option>
        <option value="top10">Top 10%</option>
        <option value="top25">Top 25%</option>
        <option value="top50">Top 50%</option>
        <option value="bot50">Bottom 50%</option>
        <option value="bot25">Bottom 25%</option>
      </select>
    </div>

  </div><!-- /filter-bar-inner -->
</div><!-- /filter-bar -->

<div id="stats-bar">
  Showing <em id="stats-shown-2">&#8230;</em> of <em id="stats-total">&#8230;</em>
  &nbsp;&middot;&nbsp; <span id="stats-label"></span>
</div>

<div id="grid-container">
  <div id="grid"></div>
  <div id="no-results">No cards match the current filters.</div>
  <div id="load-more-wrap">
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
const N_TOP10    = __N_TOP10__;
const TOTAL_ALL  = __TOTAL__;
const PAGE_SIZE  = 400;

// ── State ────────────────────────────────────────────────────────────────────

let activeTab    = 'top10';
let visibleCards = [];
let renderOffset = 0;

const activeQueues = new Set(QUEUE_META.map(q => q.id));
const activeRarity = new Set(['common','uncommon','rare','mythic','special']);
const activeColors = new Set(['W','U','B','R','G','M','X','L']);
const activeTypes  = new Set(['Creature','Instant','Sorcery','Enchantment',
                               'Artifact','Planeswalker','Land','Battle','Other']);

// Populate tab counts on load
document.getElementById('count-top10').textContent = '(' + N_TOP10.toLocaleString() + ')';
document.getElementById('count-all').textContent   = '(' + TOTAL_ALL.toLocaleString() + ')';

// ── Tab switching ─────────────────────────────────────────────────────────────

function switchTab(tab) {
  activeTab = tab;
  document.getElementById('tab-top10').classList.toggle('active', tab === 'top10');
  document.getElementById('tab-all').classList.toggle('active', tab === 'all');
  // "In Top 10%" filter only meaningful on the All tab
  document.getElementById('top10-only-wrap').style.display = (tab === 'all') ? '' : 'none';
  applyFilters();
}

// ── Filter bar collapse ───────────────────────────────────────────────────────

let filtersOpen = true;
function toggleFilters() {
  filtersOpen = !filtersOpen;
  document.getElementById('filter-bar').classList.toggle('collapsed', !filtersOpen);
  document.getElementById('filter-toggle-btn').textContent = filtersOpen ? 'Filters \u25be' : 'Filters \u25b8';
}

// ── Queue pills ───────────────────────────────────────────────────────────────

const qPillsEl = document.getElementById('queue-pills');
QUEUE_META.forEach(q => {
  const p = document.createElement('span');
  p.className     = 'pill q-pill active';
  p.dataset.queue = q.id;
  p.textContent   = 'Q' + q.id;
  p.title         = q.size + ' cards';
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

const TYPE_LABELS = {
  Creature:'Creature', Instant:'Instant', Sorcery:'Sorcery',
  Enchantment:'Enchant', Artifact:'Artifact', Planeswalker:'PW',
  Land:'Land', Battle:'Battle', Other:'Other'
};
const typePillsEl = document.getElementById('type-pills');
Object.entries(TYPE_LABELS).forEach(([t, label]) => {
  const p = document.createElement('span');
  p.className   = 'pill active';
  p.dataset.type = t;
  p.textContent  = label;
  p.addEventListener('click', () => {
    p.classList.toggle('active');
    p.classList.contains('active') ? activeTypes.add(t) : activeTypes.delete(t);
    applyFilters();
  });
  typePillsEl.appendChild(p);
});

// ── Pill listeners ────────────────────────────────────────────────────────────

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

// ── Input listeners ───────────────────────────────────────────────────────────

let nameTimer = null;
document.getElementById('name-filter').addEventListener('input', () => {
  clearTimeout(nameTimer); nameTimer = setTimeout(applyFilters, 200);
});
let setTimer = null;
document.getElementById('set-filter').addEventListener('input', () => {
  clearTimeout(setTimer); setTimer = setTimeout(applyFilters, 200);
});
['min-elo','max-elo','iq-filter','sort-select','group-by-set','dfc-only','top10-only']
  .forEach(id => document.getElementById(id).addEventListener('change', applyFilters));
['min-elo','max-elo']
  .forEach(id => document.getElementById(id).addEventListener('input', applyFilters));

// ── Helpers ───────────────────────────────────────────────────────────────────

function displayElo(card) {
  return activeTab === 'top10'
    ? (card.elo_top10 != null ? card.elo_top10 : card.elo)
    : card.elo;
}

function colorKey(card) {
  if (card.color_sort === 7) return 'L';
  if (card.color_sort === 6) return 'X';
  if (card.color_sort === 5) return 'M';
  return (card.colors && card.colors[0]) || 'X';
}

// ── Filter + sort ─────────────────────────────────────────────────────────────

function applyFilters() {
  const isTop10Tab = activeTab === 'top10';
  const nameQ      = document.getElementById('name-filter').value.toLowerCase().trim();
  const setQ       = document.getElementById('set-filter').value.toLowerCase().trim();
  const minElo     = parseFloat(document.getElementById('min-elo').value) || null;
  const maxElo     = parseFloat(document.getElementById('max-elo').value) || null;
  const iqVal      = document.getElementById('iq-filter').value;
  const sortBy     = document.getElementById('sort-select').value;
  const dfcOnly    = document.getElementById('dfc-only').checked;
  const top10Only  = document.getElementById('top10-only').checked;

  visibleCards = CARDS.filter(c => {
    if (isTop10Tab && !c.top10)                               return false;
    if (!activeQueues.has(c.queue))                           return false;
    if (!activeRarity.has(c.rarity))                          return false;
    if (!activeColors.has(colorKey(c)))                       return false;
    if (!c.types.some(t => activeTypes.has(t)))               return false;
    if (nameQ && !c.name.toLowerCase().includes(nameQ))       return false;
    if (setQ  && !c.set_name.toLowerCase().includes(setQ)
              && !c.set.toLowerCase().includes(setQ))         return false;
    const elo = displayElo(c);
    if (minElo != null && (elo == null || elo < minElo))      return false;
    if (maxElo != null && (elo == null || elo > maxElo))      return false;
    if (dfcOnly   && !c.back)                                 return false;
    if (!isTop10Tab && top10Only && !c.top10)                 return false;
    if (iqVal === 'top10' && c.iq_pct < 90)                   return false;
    if (iqVal === 'top25' && c.iq_pct < 75)                   return false;
    if (iqVal === 'top50' && c.iq_pct < 50)                   return false;
    if (iqVal === 'bot50' && c.iq_pct >= 50)                  return false;
    if (iqVal === 'bot25' && c.iq_pct >= 25)                  return false;
    return true;
  });

  if (sortBy === 'set_elo') {
    visibleCards.sort((a, b) =>
      a.released_at.localeCompare(b.released_at) ||
      a.set_name.localeCompare(b.set_name) ||
      (displayElo(b) || 0) - (displayElo(a) || 0) ||
      a.name.localeCompare(b.name)
    );
  } else if (sortBy !== 'set') {
    visibleCards.sort((a, b) => {
      if (sortBy === 'elo')        return (displayElo(b) || 0) - (displayElo(a) || 0);
      if (sortBy === 'elo_asc')    return (displayElo(a) || 0) - (displayElo(b) || 0);
      if (sortBy === 'iq_pct')     return b.iq_pct - a.iq_pct;
      if (sortBy === 'iq_pct_asc') return a.iq_pct - b.iq_pct;
      if (sortBy === 'name')       return a.name.localeCompare(b.name);
      return 0;
    });
  }

  const total = isTop10Tab ? N_TOP10 : TOTAL_ALL;
  document.getElementById('stats-shown').textContent   = visibleCards.length.toLocaleString();
  document.getElementById('stats-shown-2').textContent = visibleCards.length.toLocaleString();
  document.getElementById('stats-total').textContent   = total.toLocaleString();
  document.getElementById('stats-label').textContent   =
    isTop10Tab ? 'top-10% tournament cards' : (QUEUE_META.length + ' queues');

  renderOffset = 0;
  renderPage(true);
}

// ── Render ────────────────────────────────────────────────────────────────────

const RARITY_CLS = {common:'rc', uncommon:'ru', rare:'rr', mythic:'rm'};
const RARITY_LBL = {common:'C', uncommon:'U', rare:'R', mythic:'M', special:'S'};

const grid         = document.getElementById('grid');
const noResults    = document.getElementById('no-results');
const loadMoreWrap = document.getElementById('load-more-wrap');
const loadMoreBtn  = document.getElementById('load-more');
const loadMoreLbl  = document.getElementById('load-more-label');
const preview      = document.getElementById('preview');
const prevF        = document.getElementById('prev-front');
const prevB        = document.getElementById('prev-back');

function renderPage(reset) {
  if (reset) {
    grid.innerHTML         = '';
    noResults.style.display = 'none';
  }

  if (visibleCards.length === 0) {
    noResults.style.display   = 'block';
    loadMoreWrap.style.display = 'none';
    return;
  }

  const groupBySet = document.getElementById('group-by-set').checked;
  const isTop10Tab = activeTab === 'top10';
  const slice      = visibleCards.slice(renderOffset, renderOffset + PAGE_SIZE);
  const frag       = document.createDocumentFragment();

  let lastKey = reset ? null : grid.dataset.lastKey;

  slice.forEach(c => {
    if (groupBySet) {
      const key = c.set;
      if (key !== lastKey) {
        const hdr = document.createElement('div');
        hdr.className       = 'set-header';
        hdr.style.flexBasis = '100%';
        hdr.innerHTML       = c.set_name + ' <span class="year">(' + c.released_at.slice(0,4) + ')</span>';
        frag.appendChild(hdr);
        lastKey = key;
      }
    }

    const wrap = document.createElement('div');
    wrap.className = 'card-wrap';

    const img    = document.createElement('img');
    img.src      = c.front;
    img.alt      = c.name;
    img.loading  = 'lazy';

    if (c.back) {
      img.classList.add('dfc');
      img.addEventListener('mouseenter', () => {
        prevF.src = c.front; prevB.src = c.back;
        preview.classList.add('visible');
      });
      img.addEventListener('mouseleave', () => preview.classList.remove('visible'));
    }

    const elo    = displayElo(c);
    const eloStr = elo != null
      ? '<span class="elo-v">' + Math.round(elo) + '</span>'
      : '<span style="color:#444">&mdash;</span>';

    const rcls   = RARITY_CLS[c.rarity] || '';
    const rlbl   = RARITY_LBL[c.rarity] || c.rarity;

    let rankStr;
    if (isTop10Tab && c.top10_rank != null) {
      rankStr = '<span class="rank-v">#' + c.top10_rank + ' of ' + N_TOP10.toLocaleString() + '</span>';
    } else {
      rankStr = '<span class="rank-v">#' + c.iq_rank + ' of ' + c.q_size + '</span>';
    }

    const label = document.createElement('div');
    label.className = 'card-label';
    label.innerHTML =
      '<span class="q-tag">Q' + c.queue + '</span>' +
      '<span class="' + rcls + '">' + rlbl + '</span><br>' +
      '<strong>' + c.name + '</strong>' +
      '<span class="set-ln">' + c.set_name + '</span><br>' +
      eloStr + '&nbsp;&nbsp;' + rankStr;

    wrap.appendChild(img);
    wrap.appendChild(label);
    frag.appendChild(wrap);
  });

  grid.dataset.lastKey = lastKey;
  grid.appendChild(frag);
  renderOffset += slice.length;

  const remaining = visibleCards.length - renderOffset;
  if (remaining > 0) {
    loadMoreWrap.style.display = 'block';
    loadMoreLbl.textContent    = remaining.toLocaleString() + ' more';
  } else {
    loadMoreWrap.style.display = 'none';
  }
}

loadMoreBtn.addEventListener('click', () => renderPage(false));

// ── Initial render ────────────────────────────────────────────────────────────
applyFilters();
</script>
</body>
</html>'''

# Inject data
html = (HTML
    .replace('__CARDS_JSON__',      cards_json_str)
    .replace('__QUEUE_META_JSON__', queue_meta_json)
    .replace('__TOTAL__',           str(total_cards))
    .replace('__N_TOP10__',         str(n_top10))
    .replace('__NQUEUES__',         str(n_queues))
)

# ── Write output ──────────────────────────────────────────────────────────────

default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'universe.html')
out_path    = args.out or default_out

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f'Saved to {out_path}')
webbrowser.open(f'file:///{out_path.replace(os.sep, "/")}')
