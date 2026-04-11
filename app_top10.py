'''
app_top10.py — Flask app for the top-10% bracket phase.

Run locally with:
    python app_top10.py

Then open http://127.0.0.1:5000

Differences from app.py (full-queue phase):
  - Pool is loaded from top_10_queue.json (single fixed pool, no daily rotation)
  - No similarity models loaded — matchups are Elo-proximity weighted random
  - card_a: Efraimidis-Spirakis weighted shuffle, 1.5:1 Elo curve (0.80–1.20)
  - card_b: inverse-linear soft draw by Elo distance (scale=250)
  - Seen-pair reroll retained at 90%
  - Elo formula clamps implied difference at ±241 (≈ 80/20 max odds)
  - K=32, K_DECAY=250 (halves at 250 games; all cards start with 0 games played)
  - Votes written to votes_top10 table; Elos in elo_ratings_top10 table
  - No mode slider (no similarity modes)
  - Elo not shown to user before voting (visible in session log post-vote)
'''

import json
import os
import random
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session

load_dotenv()

sys_path_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_path_dir)
from parse_data import load_processed_cards


# ── App setup ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.urandom(24)

DATABASE_URL    = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)
TOP10_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'top_10_queue.json')
STATS_TOKEN     = os.environ.get('STATS_TOKEN', 'bracketstats')

_pool = None


@contextmanager
def _get_db():
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


# ── Elo constants ──────────────────────────────────────────────────────────────

INITIAL_ELO         = 1500.0
ELO_K               = 32
ELO_K_DECAY         = 250    # K halves after 250 games (vs 30 in full-queue phase)
ELO_DIFF_CAP        = 241.0  # clamp implied Elo diff at ±241 (≈ 80/20 max odds)

# card_a weighted shuffle — Elo-percentile multipliers (1.5:1 ratio, symmetric)
_ELO_PCT_XP = [0.00, 0.10, 0.50, 0.90, 1.00]
_ELO_PCT_FP = [0.80, 0.80, 1.00, 1.20, 1.20]

# card_b soft draw — inverse-linear decay by Elo distance
CARD_B_ELO_SCALE = 250.0  # probability halves every 250 Elo points

# Seen-pair reroll probability
SEEN_REROLL_PROB = 0.90

# Card types
DOUBLE_FACED_LAYOUTS = {'transform', 'modal_dfc'}
SIDEWAYS_TYPES       = {'Battle', 'Room'}
SIDEWAYS_LAYOUTS     = {'split'}
SIDEWAYS_LAYOUT_KEYWORD_EXCLUSIONS = {'Aftermath'}


# ── Per-session state ──────────────────────────────────────────────────────────

_session_card_a_queue: dict = {}   # session_id → shuffled list (popped per matchup)
_session_seen_pairs:   dict = {}   # session_id → set of frozensets({card_a, card_b})


# ── In-memory Elo cache ────────────────────────────────────────────────────────

_elo_cache:       dict = {}   # card_name → float
_elo_sorted_names: list = []  # all pool cards sorted by Elo ascending


def _rebuild_elo_sorted():
    global _elo_sorted_names
    _elo_sorted_names = sorted(_card_names, key=lambda nm: _elo_cache.get(nm, INITIAL_ELO))


# ── Database ───────────────────────────────────────────────────────────────────

def _init_pool():
    global _pool
    if not DATABASE_URL:
        raise RuntimeError(
            'DATABASE_URL environment variable is not set. '
            'Set it to your PostgreSQL connection URL.'
        )
    _pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=DATABASE_URL)


def init_db():
    _init_pool()
    with _get_db() as conn:
        cur = conn.cursor()
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
        cur.execute('''
            CREATE TABLE IF NOT EXISTS page_views (
                id          SERIAL  PRIMARY KEY,
                timestamp   TEXT    NOT NULL,
                page        TEXT    NOT NULL,
                ip_address  TEXT,
                session_id  TEXT,
                device      TEXT
            )
        ''')


def init_elo_ratings(card_entries):
    '''Seed elo_ratings_top10 from top_10_queue.json. ON CONFLICT DO NOTHING — safe to re-run.'''
    with _get_db() as conn:
        cur = conn.cursor()
        execute_values(
            cur,
            'INSERT INTO elo_ratings_top10 (card_name, rating, wins, losses) '
            'VALUES %s ON CONFLICT DO NOTHING',
            [(e['name'], e['normalized_elo'], 0, 0) for e in card_entries],
        )


def _init_elo_cache():
    with _get_db() as conn:
        cur = conn.cursor()
        cur.execute('SELECT card_name, rating FROM elo_ratings_top10')
        rows = cur.fetchall()
    _elo_cache.update({name: rating for name, rating in rows})
    for name in _card_names:
        if name not in _elo_cache:
            _elo_cache[name] = INITIAL_ELO


def get_current_elos(*card_names):
    with _get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            'SELECT card_name, rating FROM elo_ratings_top10 WHERE card_name = ANY(%s)',
            (list(card_names),),
        )
        rows = cur.fetchall()
    found = {name: rating for name, rating in rows}
    return {name: round(found.get(name, INITIAL_ELO), 1) for name in card_names}


def _get_elo_and_games(cur, card_name):
    cur.execute(
        'SELECT rating, wins + losses FROM elo_ratings_top10 WHERE card_name = %s',
        (card_name,),
    )
    row = cur.fetchone()
    return (row[0], row[1]) if row else (INITIAL_ELO, 0)


def _effective_k(total_games):
    return ELO_K * ELO_K_DECAY / (ELO_K_DECAY + total_games)


def _expected_score(rating_a, rating_b):
    '''Standard Elo expected score with ±ELO_DIFF_CAP clamp on the implied difference.'''
    diff = max(-ELO_DIFF_CAP, min(ELO_DIFF_CAP, rating_b - rating_a))
    return 1.0 / (1.0 + 10.0 ** (diff / 400.0))


def update_elo(winner_name, loser_name):
    with _get_db() as conn:
        cur = conn.cursor()
        r_w, games_w = _get_elo_and_games(cur, winner_name)
        r_l, games_l = _get_elo_and_games(cur, loser_name)

        k_w = _effective_k(games_w)
        k_l = _effective_k(games_l)
        e_w = _expected_score(r_w, r_l)

        new_r_w = r_w + k_w * (1.0 - e_w)
        new_r_l = r_l + k_l * (0.0 - (1.0 - e_w))
        delta_w = new_r_w - r_w
        delta_l = new_r_l - r_l

        ts = datetime.now(timezone.utc).isoformat()
        cur.execute('''
            INSERT INTO elo_ratings_top10 (card_name, rating, wins, losses, last_updated)
            VALUES (%s, %s, 1, 0, %s)
            ON CONFLICT(card_name) DO UPDATE SET
                rating = EXCLUDED.rating,
                wins = elo_ratings_top10.wins + 1,
                last_updated = EXCLUDED.last_updated
        ''', (winner_name, new_r_w, ts))
        cur.execute('''
            INSERT INTO elo_ratings_top10 (card_name, rating, wins, losses, last_updated)
            VALUES (%s, %s, 0, 1, %s)
            ON CONFLICT(card_name) DO UPDATE SET
                rating = EXCLUDED.rating,
                losses = elo_ratings_top10.losses + 1,
                last_updated = EXCLUDED.last_updated
        ''', (loser_name, new_r_l, ts))

    _elo_cache[winner_name] = new_r_w
    _elo_cache[loser_name]  = new_r_l
    _rebuild_elo_sorted()
    return new_r_w, new_r_l, delta_w, delta_l


def log_vote(ip, session_id, card_a, card_b, chosen, config_name, device=None):
    ts = datetime.now(timezone.utc).isoformat()
    with _get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            'INSERT INTO votes_top10 '
            '(timestamp, ip_address, session_id, card_a, card_b, chosen, config_name, device) '
            'VALUES (%s, %s, %s, %s, %s, %s, %s, %s)',
            (ts, ip, session_id, card_a, card_b, chosen, config_name, device),
        )


def _detect_device(user_agent: str) -> str:
    ua = (user_agent or '').lower()
    if any(t in ua for t in ('mobile', 'android', 'iphone', 'ipad', 'ipod')):
        return 'mobile'
    return 'desktop'


def log_page_view(page: str):
    '''Record a page view. Called from route handlers; silently swallows errors.'''
    try:
        ts  = datetime.now(timezone.utc).isoformat()
        ip  = request.headers.get('X-Forwarded-For', request.remote_addr)
        sid = session.get('session_id')
        dev = _detect_device(request.headers.get('User-Agent', ''))
        with _get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                'INSERT INTO page_views (timestamp, page, ip_address, session_id, device) '
                'VALUES (%s, %s, %s, %s, %s)',
                (ts, page, ip, sid, dev),
            )
    except Exception:
        pass


# ── Card data (loaded once at startup) ────────────────────────────────────────

print('Loading top_10_queue.json...')
with open(TOP10_JSON_PATH, encoding='utf-8') as _f:
    _top10_data = json.load(_f)

_card_entries = _top10_data['cards']               # list of {name, queue_id, normalized_elo}
_card_names   = [e['name'] for e in _card_entries]
_card_name_set = set(_card_names)
print(f'{len(_card_names)} cards in top-10% pool.')

print('Loading card metadata...')
_df = load_processed_cards()
# Build lookup dicts — include all cards in the parquet (some top-10 cards may be
# in both post- and pre-C16 datasets; match by name regardless of release date)
_meta = _df[_df['name'].isin(_card_name_set)].drop_duplicates('name').set_index('name')

def _meta_get(name, col, default=''):
    try:
        return _meta.at[name, col]
    except KeyError:
        return default

_name_to_layout       = {n: str(_meta_get(n, 'layout'))    for n in _card_names}
_name_to_type         = {n: str(_meta_get(n, 'type_line')) for n in _card_names}
_name_to_keywords     = {n: (list(kw) if (kw := _meta_get(n, 'keywords', None)) is not None and not isinstance(kw, float) else []) for n in _card_names}
_name_to_set          = {n: str(_meta_get(n, 'set_name'))  for n in _card_names}
_name_to_year         = {n: pd.to_datetime(_meta_get(n, 'released_at', pd.NaT)).year
                          if pd.notna(_meta_get(n, 'released_at', pd.NaT)) else ''
                          for n in _card_names}
_name_to_img_front    = {n: _meta_get(n, 'img_front') or None for n in _card_names}
_name_to_img_back     = {n: _meta_get(n, 'img_back')  or None for n in _card_names}

# Override promo/deficient images with preferred printings
_IMAGE_OVERRIDES = {
    'Sothera, the Supervoid':  'https://cards.scryfall.io/large/front/e/9/e99d6fc0-dcf2-4b25-81c2-02c230a36246.jpg',
    'The Wandering Emperor':   'https://cards.scryfall.io/large/front/f/a/fab2d8a9-ab4c-4225-a570-22636293c17d.jpg',
}
for _n, _url in _IMAGE_OVERRIDES.items():
    if _n in _name_to_img_front:
        _name_to_img_front[_n] = _url
_name_to_vintage_legal = {n: _meta_get(n, 'legal_vintage', 'not_legal') for n in _card_names}

print('Initialising database and Elo ratings...')
init_db()
init_elo_ratings(_card_entries)
_init_elo_cache()
_rebuild_elo_sorted()
print(f'Ready. Visit http://127.0.0.1:5000')


# ── Card info cache ────────────────────────────────────────────────────────────

_card_info_cache = {}


def get_card_info(card_name):
    if card_name in _card_info_cache:
        return _card_info_cache[card_name]

    layout    = _name_to_layout.get(card_name, '')
    type_line = _name_to_type.get(card_name, '')
    keywords  = _name_to_keywords.get(card_name, [])
    vintage_legal = _name_to_vintage_legal.get(card_name, 'not_legal') in {'legal', 'restricted'}
    is_sideways = (
        (layout in SIDEWAYS_LAYOUTS
         and vintage_legal
         and not any(k in keywords for k in SIDEWAYS_LAYOUT_KEYWORD_EXCLUSIONS))
        or any(t in type_line for t in SIDEWAYS_TYPES)
    )

    img_front = _name_to_img_front.get(card_name)
    img_back  = _name_to_img_back.get(card_name)
    if pd.isna(img_front) if isinstance(img_front, float) else False: img_front = None
    if pd.isna(img_back)  if isinstance(img_back,  float) else False: img_back  = None

    info = {'img_front': img_front, 'img_back': img_back, 'is_sideways': is_sideways}
    _card_info_cache[card_name] = info
    return info


# ── Matchup logic ──────────────────────────────────────────────────────────────

def _weighted_shuffle(cards):
    '''
    Efraimidis-Spirakis weighted shuffle for card_a ordering.
    Weight is interpolated from each card's Elo percentile within the pool:
      ≤10th pct → ×0.80,  50th → ×1.00,  ≥90th → ×1.20  (1.5:1 ratio)
    '''
    cards = list(cards)
    n = len(cards)
    if n == 0:
        return cards

    elos  = np.array([_elo_cache.get(c, INITIAL_ELO) for c in cards], dtype=float)
    order = np.argsort(elos)
    pct   = np.empty(n, dtype=float)
    pct[order] = np.arange(n) / max(n - 1, 1)

    elo_weights = np.interp(pct, _ELO_PCT_XP, _ELO_PCT_FP)

    keyed = [(random.random() ** (1.0 / w), c) for c, w in zip(cards, elo_weights)]
    keyed.sort()  # ascending — highest weight pops first
    return [c for _, c in keyed]


def _next_card_a(session_id):
    queue = _session_card_a_queue.get(session_id)
    if not queue:
        _session_card_a_queue[session_id] = _weighted_shuffle(_card_names)
    return _session_card_a_queue[session_id].pop()


def _card_b_weights(card_a_name):
    '''
    Inverse-linear soft weights for card_b selection.
    w(card) = 1 / (1 + |elo_card - elo_a| / CARD_B_ELO_SCALE)
    card_a itself gets weight 0.
    '''
    elo_a = _elo_cache.get(card_a_name, INITIAL_ELO)
    weights = []
    for name in _card_names:
        if name == card_a_name:
            weights.append(0.0)
        else:
            diff = abs(_elo_cache.get(name, INITIAL_ELO) - elo_a)
            weights.append(1.0 / (1.0 + diff / CARD_B_ELO_SCALE))
    return weights


def get_matchup(session_id=None, unusual=False):
    '''
    Generate a card pair for the top-10% bracket.

    card_a: next from per-session weighted shuffle (Elo-biased, no repeats per cycle)
    card_b: soft inverse-linear draw by Elo proximity; 90% reroll if pair already seen
    unusual=True: force card_a to be a Battle or double-faced card (test helper)
    '''
    pool = _card_names

    if unusual:
        unusual_pool = [c for c in pool
                        if any(t in _name_to_type.get(c, '') for t in SIDEWAYS_TYPES)
                        or _name_to_layout.get(c, '') in DOUBLE_FACED_LAYOUTS]
        card_a = random.choice(unusual_pool) if unusual_pool else random.choice(pool)
    elif session_id:
        card_a = _next_card_a(session_id)
    else:
        card_a = random.choice(pool)

    seen_pairs = _session_seen_pairs.get(session_id, set()) if session_id else set()

    # Build soft weights and draw card_b
    weights = _card_b_weights(card_a)

    # Up to 10 attempts to avoid seen pairs (90% reroll probability)
    card_b = None
    for _ in range(10):
        candidate = random.choices(_card_names, weights=weights, k=1)[0]
        if candidate == card_a:
            continue
        pair = frozenset({card_a, candidate})
        if pair in seen_pairs and random.random() < SEEN_REROLL_PROB:
            continue
        card_b = candidate
        break

    # Fallback: any card not card_a
    if card_b is None:
        others = [c for c in pool if c != card_a]
        card_b = random.choice(others)

    if session_id:
        _session_seen_pairs.setdefault(session_id, set()).add(frozenset({card_a, card_b}))

    info_a = get_card_info(card_a)
    info_b = get_card_info(card_b)
    elos   = get_current_elos(card_a, card_b)

    return {
        'card_a':        card_a,
        'card_b':        card_b,
        'config':        'elo_soft',
        'img_a_front':   info_a['img_front'],
        'img_a_back':    info_a['img_back'],
        'img_b_front':   info_b['img_front'],
        'img_b_back':    info_b['img_back'],
        'sideways_a':    info_a['is_sideways'],
        'sideways_b':    info_b['is_sideways'],
        'set_a':         _name_to_set.get(card_a, ''),
        'year_a':        int(_name_to_year.get(card_a, 0) or 0) or '',
        'set_b':         _name_to_set.get(card_b, ''),
        'year_b':        int(_name_to_year.get(card_b, 0) or 0) or '',
        'elo_a_current': elos[card_a],
        'elo_b_current': elos[card_b],
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

def _ensure_session():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())


@app.route('/')
def index():
    _ensure_session()
    log_page_view('main')
    return render_template('index.html')


@app.route('/faq')
def faq():
    _ensure_session()
    log_page_view('faq')
    return render_template('faq.html')


@app.route('/share')
def share():
    return render_template('share.html')


@app.route('/universe')
def universe():
    _ensure_session()
    log_page_view('universe')
    return render_template('universe.html')


@app.route('/stats/<token>')
def stats(token):
    if token != STATS_TOKEN:
        return 'Not found', 404

    # Load queue sizes from queues.json
    queues_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queues.json')
    with open(queues_path, encoding='utf-8') as _f:
        _queues_data = json.load(_f)
    _queue_sizes = {q['id']: len(q['cards']) for q in _queues_data['queues']}

    # Fresh connection — bypasses the pool to avoid stale-connection errors on a low-traffic route
    conn = psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor()

        # Top-10% phase total
        cur.execute('SELECT COUNT(*) FROM votes_top10')
        top10_total = cur.fetchone()[0]

        # Full-phase votes per queue
        cur.execute('''
            SELECT queue_id, COUNT(*) AS total_votes
            FROM votes
            WHERE queue_id IS NOT NULL
            GROUP BY queue_id
            ORDER BY queue_id
        ''')
        queue_rows = cur.fetchall()

        # Full-phase grand total
        cur.execute('SELECT COUNT(*) FROM votes')
        full_total = cur.fetchone()[0]

        # Last 24h hourly from top-10% table
        cur.execute('''
            SELECT
                date_trunc('hour', timestamp::timestamptz AT TIME ZONE 'America/New_York') AS hour,
                COUNT(*) AS cnt
            FROM votes_top10
            WHERE timestamp::timestamptz >= NOW() - INTERVAL '24 hours'
            GROUP BY 1
            ORDER BY 1
        ''')
        hourly_rows = cur.fetchall()

    finally:
        conn.close()

    # Top-10% summary row
    top10_pool = len(_card_names)
    top10_avg  = round(top10_total * 2 / top10_pool, 2) if top10_pool else 0
    top10_row  = (
        f'<tr style="color:#c9a84c;font-weight:600">'
        f'<td>Top 10%</td>'
        f'<td style="text-align:right">{top10_pool:,}</td>'
        f'<td style="text-align:right">{top10_total:,}</td>'
        f'<td style="text-align:right">{top10_avg:,}</td></tr>\n'
        f'<tr><td colspan="4" style="color:#555;font-size:0.78rem;padding:6px 8px 2px">Full phase</td></tr>\n'
    )

    # Per-queue table rows
    vote_count_map = {qid: total for qid, total in queue_rows}
    all_qids = sorted(set(list(_queue_sizes.keys()) + [qid for qid, _ in queue_rows]))
    queue_table = ''
    for qid in all_qids:
        total_votes = vote_count_map.get(qid, 0)
        qsize = _queue_sizes.get(qid)
        avg = round(total_votes * 2 / qsize, 2) if qsize else '?'
        qsize_str = f'{qsize:,}' if qsize is not None else '?'
        avg_str   = f'{avg:,}' if isinstance(avg, float) else avg
        queue_table += (
            f'<tr><td>Q{qid}</td>'
            f'<td style="text-align:right">{qsize_str}</td>'
            f'<td style="text-align:right">{total_votes:,}</td>'
            f'<td style="text-align:right">{avg_str}</td></tr>\n'
        )

    # Hourly table
    hourly_table = ''
    for hour, cnt in hourly_rows:
        hourly_table += (
            f'<tr><td>{str(hour)[:16]}</td>'
            f'<td style="text-align:right">{cnt:,}</td></tr>\n'
        )

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stats</title>
<style>
body {{ background:#1a1a1a; color:#ddd; font-family:sans-serif; padding:20px; max-width:520px; margin:0 auto; }}
h2 {{ color:#c9a84c; margin:24px 0 8px; }}
table {{ width:100%; border-collapse:collapse; font-size:0.9rem; }}
th {{ text-align:left; color:#888; border-bottom:1px solid #444; padding:4px 8px; }}
td {{ padding:4px 8px; border-bottom:1px solid #2a2a2a; }}
.total {{ color:#c9a84c; font-weight:bold; margin:6px 0 20px; }}
</style>
</head>
<body>
<h2>Vote totals by queue</h2>
<table>
<tr><th>Queue</th><th style="text-align:right">Size</th><th style="text-align:right">Votes</th><th style="text-align:right">Avg/card</th></tr>
{top10_row}{queue_table}
</table>
<p class="total">Full-phase total: {full_total:,} votes</p>

<h2>Last 24 hours (hourly, ET)</h2>
<table>
<tr><th>Hour</th><th style="text-align:right">Votes</th></tr>
{hourly_table}
</table>
</body>
</html>'''
    return html


@app.route('/api/matchup')
def api_matchup():
    _ensure_session()
    sid     = session.get('session_id')
    unusual = request.args.get('unusual', '0') == '1'
    try:
        return jsonify(get_matchup(session_id=sid, unusual=unusual))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/matchup/unusual')
def api_matchup_unusual():
    _ensure_session()
    try:
        return jsonify(get_matchup(session_id=None, unusual=True))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/vote', methods=['POST'])
def api_vote():
    data   = request.get_json()
    card_a = data['card_a']
    card_b = data['card_b']
    chosen = data['chosen']
    cfg    = data.get('config', '')
    device = data.get('device')

    winner = chosen
    loser  = card_b if chosen == card_a else card_a

    _ensure_session()
    sid = session.get('session_id')
    ip  = request.headers.get('X-Forwarded-For', request.remote_addr)

    log_vote(ip, sid, card_a, card_b, chosen, cfg, device=device)

    new_w, new_l, delta_w, delta_l = update_elo(winner, loser)

    if chosen == card_a:
        elo_a, delta_a = new_w, delta_w
        elo_b, delta_b = new_l, delta_l
    else:
        elo_a, delta_a = new_l, delta_l
        elo_b, delta_b = new_w, delta_w

    return jsonify({
        'elo_a':   round(elo_a, 1),
        'delta_a': round(delta_a, 1),
        'elo_b':   round(elo_b, 1),
        'delta_b': round(delta_b, 1),
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
