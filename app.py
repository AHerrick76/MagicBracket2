'''
Flask web applet for the Magic Bracket voting system.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
Votes are logged to a PostgreSQL database (set DATABASE_URL env var).
'''

import os
import random
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool

import pandas as pd
from flask import Flask, jsonify, render_template, request, session

sys_path_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_path_dir)
from parse_data import load_processed_cards
from similarity import build_candidate_models, get_candidates


# ── App setup ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
# Ephemeral secret key: sessions (and thus session IDs) reset when the server restarts.
# This is fine for local use; swap for a fixed key if persistence matters.
app.secret_key = os.urandom(24)

# PostgreSQL connection URL — set this environment variable before running.
# Format: postgresql://user:password@host:5432/dbname
# Railway and Render export this automatically as DATABASE_URL.
# Replace 'postgres://' prefix (legacy Heroku/Railway format) with 'postgresql://'.
DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)

_pool = None   # ThreadedConnectionPool; initialised in init_db()


@contextmanager
def _get_db():
    '''Yield a psycopg2 connection from the pool; commits on success, rolls back on error.'''
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)

INITIAL_ELO   = 1500.0
ELO_K         = 32    # maximum K (applies to a card with 0 prior votes)
ELO_K_DECAY   = 30    # games played at which effective K halves (K = ELO_K/2 at this point)

# ── DEV ONLY ── remove (or gate behind auth) before deploying to production ──
CLOSED_LOOP_SIZE = 25   # number of cards in a closed-loop test session

CANDIDATE_WEIGHTS = [0.45, 0.30, 0.15, 0.07, 0.03]

# Mode 2: wider candidate pool, uniform selection
MODE2_N_NEIGHBORS = 15

# Mode 3: uniform pick from the closest 25% of all cards (balanced model).
# Computed at startup once _card_names is available.
MODE3_FRACTION = 0.25

# Per-session seen-card tracking (server-side, resets with server restart).
# Maps session_id -> set of card names the session has already been shown.
_session_seen: dict = {}
# Probability of re-rolling card_a when it has already been seen this session.
SEEN_REROLL_PROB = 0.90

# Elo-bracket pairing: card_b is chosen from within a percentile window around
# card_a's Elo rank before applying the similarity index.  One entry is chosen
# uniformly at random per matchup; None means no Elo filter is applied.
# Values are half-widths in percentile units (0.05 → ±5 pp window ≈ 10% of cards).
ELO_BRACKET_HALF_WIDTHS = [0.05, 0.20, 0.40, None]

# In-memory Elo cache: populated at startup, kept in sync on every vote.
# Used for fast weighted card_a selection and percentile-bracket calculations.
_elo_cache: dict = {}

# ── DEV ONLY ── remove before deploying to production ────────────────────────
# Maps session_id -> list of CLOSED_LOOP_SIZE card names for closed-loop testing.
_session_closed_loop: dict = {}

# Layouts that have two distinct faces, each with their own Scryfall image URL.
# These get a hover popup showing both faces.
DOUBLE_FACED_LAYOUTS = {'transform', 'modal_dfc'}

# Card types whose text is oriented sideways and should be displayed rotated 90°.
# Battle cards are physically played sideways; Room cards are landscape split cards.
SIDEWAYS_TYPES = {'Battle', 'Room'}

# Layouts whose card image is printed in landscape orientation (rotated 90°).
# 'split' covers regular split cards and fuse cards (e.g. Fire // Ice, Beck // Call).
# Aftermath cards also have layout='split' in Scryfall data but are portrait-oriented;
# they are excluded at runtime via SIDEWAYS_LAYOUT_KEYWORD_EXCLUSIONS.
SIDEWAYS_LAYOUTS = {'split'}
SIDEWAYS_LAYOUT_KEYWORD_EXCLUSIONS = {'Aftermath'}


# ── Database ───────────────────────────────────────────────────────────────────

def _init_pool():
    '''Create (or replace) the connection pool. Called at startup and after fork.'''
    global _pool
    if not DATABASE_URL:
        raise RuntimeError(
            'DATABASE_URL environment variable is not set. '
            'In Railway: go to your web service → Variables and add '
            'DATABASE_URL = ${{ Postgres.DATABASE_URL }}'
        )
    _pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=DATABASE_URL)


def init_db():
    _init_pool()
    with _get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS votes (
                id          SERIAL  PRIMARY KEY,
                timestamp   TEXT    NOT NULL,
                ip_address  TEXT,
                session_id  TEXT,
                card_a      TEXT    NOT NULL,
                card_b      TEXT    NOT NULL,
                chosen      TEXT    NOT NULL,
                config_name TEXT
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS elo_ratings (
                card_name    TEXT    PRIMARY KEY,
                rating       REAL    NOT NULL DEFAULT 1500.0,
                wins         INTEGER NOT NULL DEFAULT 0,
                losses       INTEGER NOT NULL DEFAULT 0,
                last_updated TEXT
            )
        ''')


def get_current_elos(*card_names):
    '''Fetch current Elo ratings for one or more cards in a single query.'''
    with _get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            'SELECT card_name, rating FROM elo_ratings WHERE card_name = ANY(%s)',
            (list(card_names),),
        )
        rows = cur.fetchall()
    found = {name: rating for name, rating in rows}
    return {name: round(found.get(name, INITIAL_ELO), 1) for name in card_names}


def init_elo_ratings(card_names):
    '''Ensure every card has an elo_ratings row. Existing rows are unchanged.'''
    with _get_db() as conn:
        cur = conn.cursor()
        execute_values(
            cur,
            'INSERT INTO elo_ratings (card_name, rating, wins, losses) VALUES %s ON CONFLICT DO NOTHING',
            [(name, INITIAL_ELO, 0, 0) for name in card_names],
        )


def _get_elo(cur, card_name):
    '''Return (rating, total_games) for a card; defaults to (INITIAL_ELO, 0).'''
    cur.execute(
        'SELECT rating, wins + losses FROM elo_ratings WHERE card_name = %s', (card_name,)
    )
    row = cur.fetchone()
    return (row[0], row[1]) if row else (INITIAL_ELO, 0)


def _effective_k(total_games):
    '''K decays as a card accumulates votes; halves at ELO_K_DECAY games.'''
    return ELO_K * ELO_K_DECAY / (ELO_K_DECAY + total_games)


def update_elo(winner_name, loser_name):
    '''
    Compute and store updated Elo ratings after a vote.
    Each card uses its own effective K based on how many games it has played,
    so ratings stabilise as a card accumulates more votes.
    Returns (winner_new_elo, loser_new_elo, winner_delta, loser_delta).
    '''
    with _get_db() as conn:
        cur = conn.cursor()
        r_w, games_w = _get_elo(cur, winner_name)
        r_l, games_l = _get_elo(cur, loser_name)

        k_w = _effective_k(games_w)
        k_l = _effective_k(games_l)

        e_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))

        new_r_w = r_w + k_w * (1.0 - e_w)
        new_r_l = r_l + k_l * (0.0 - (1.0 - e_w))
        delta_w = new_r_w - r_w
        delta_l = new_r_l - r_l

        ts = datetime.now(timezone.utc).isoformat()
        cur.execute('''
            INSERT INTO elo_ratings (card_name, rating, wins, losses, last_updated)
            VALUES (%s, %s, 1, 0, %s)
            ON CONFLICT(card_name) DO UPDATE SET
                rating = EXCLUDED.rating,
                wins = elo_ratings.wins + 1,
                last_updated = EXCLUDED.last_updated
        ''', (winner_name, new_r_w, ts))
        cur.execute('''
            INSERT INTO elo_ratings (card_name, rating, wins, losses, last_updated)
            VALUES (%s, %s, 0, 1, %s)
            ON CONFLICT(card_name) DO UPDATE SET
                rating = EXCLUDED.rating,
                losses = elo_ratings.losses + 1,
                last_updated = EXCLUDED.last_updated
        ''', (loser_name, new_r_l, ts))

    _elo_cache[winner_name] = new_r_w
    _elo_cache[loser_name]  = new_r_l
    _rebuild_elo_sorted()
    return new_r_w, new_r_l, delta_w, delta_l


# ── DEV ONLY ── remove (or gate behind auth) before deploying to production ──
def reset_elo():
    '''Reset every card's rating to INITIAL_ELO and clear win/loss counts.'''
    with _get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            'UPDATE elo_ratings SET rating = %s, wins = 0, losses = 0, last_updated = NULL',
            (INITIAL_ELO,),
        )
    for name in _card_names:
        _elo_cache[name] = INITIAL_ELO
    _rebuild_elo_sorted()


def _init_elo_cache():
    '''Load all Elo ratings from the DB into _elo_cache. Call after init_elo_ratings.'''
    with _get_db() as conn:
        cur = conn.cursor()
        cur.execute('SELECT card_name, rating FROM elo_ratings')
        rows = cur.fetchall()
    _elo_cache.update({name: rating for name, rating in rows})
    for name in _card_names:
        if name not in _elo_cache:
            _elo_cache[name] = INITIAL_ELO


# Sorted card names by Elo — kept up-to-date by update_elo() and reset_elo() so
# _get_elo_bracket_pool doesn't need to re-sort on every matchup request.
_elo_sorted_names: list = []


def _rebuild_elo_sorted():
    global _elo_sorted_names
    _elo_sorted_names = sorted(_card_names, key=lambda nm: _elo_cache.get(nm, INITIAL_ELO))


def _get_closed_loop_pool(session_id, enabled):
    '''
    Return the 25-card closed-loop pool for this session, creating it if needed.
    Clears the pool and returns None when enabled=False.
    DEV ONLY — remove before deploying to production.
    '''
    if not enabled:
        _session_closed_loop.pop(session_id, None)
        return None
    if session_id not in _session_closed_loop:
        _session_closed_loop[session_id] = random.sample(_card_names, CLOSED_LOOP_SIZE)
    return _session_closed_loop[session_id]
# ─────────────────────────────────────────────────────────────────────────────


def log_vote(ip, session_id, card_a, card_b, chosen, config_name):
    ts = datetime.now(timezone.utc).isoformat()
    with _get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            'INSERT INTO votes '
            '(timestamp, ip_address, session_id, card_a, card_b, chosen, config_name) '
            'VALUES (%s, %s, %s, %s, %s, %s, %s)',
            (ts, ip, session_id, card_a, card_b, chosen, config_name),
        )


# ── Card data & models (loaded once at startup) ────────────────────────────────

print('Loading card data...')
_df = load_processed_cards()
_post_c16 = _df[_df['released_at'] > pd.Timestamp('2016-11-11')].reset_index(drop=True)
_card_names      = _post_c16['name'].tolist()
_name_to_id      = dict(zip(_post_c16['name'], _post_c16['id']))
_name_to_layout   = dict(zip(_post_c16['name'], _post_c16['layout'].fillna('')))
_name_to_type     = dict(zip(_post_c16['name'], _post_c16['type_line'].fillna('')))
_name_to_keywords = dict(zip(_post_c16['name'], _post_c16['keywords']))
_name_to_set      = dict(zip(_post_c16['name'], _post_c16['set_name'].fillna('')))
_name_to_year     = dict(zip(_post_c16['name'], pd.to_datetime(_post_c16['released_at']).dt.year))
_name_to_img_front = dict(zip(_post_c16['name'], _post_c16['img_front']))
_name_to_img_back  = dict(zip(_post_c16['name'], _post_c16['img_back']))
print(f'Post-C16 cards: {len(_card_names)}')
_mode3_n_neighbors = max(1, int(len(_card_names) * MODE3_FRACTION))

print('Building candidate models (this takes a moment)...')
_models = build_candidate_models(_post_c16)

print('Initialising Elo ratings...')
init_db()
init_elo_ratings(_card_names)
_init_elo_cache()
_rebuild_elo_sorted()
print('Ready. Visit http://127.0.0.1:5000')

# Pre-compute the pool of "unusual" cards for the test button:
# any double-faced card or any Battle card.
_unusual_names = [
    n for n in _card_names
    if _name_to_layout.get(n) in DOUBLE_FACED_LAYOUTS
    or any(t in _name_to_type.get(n, '') for t in SIDEWAYS_TYPES)
]


# ── Scryfall card info cache ───────────────────────────────────────────────────

_card_info_cache = {}


def get_card_info(card_name):
    '''
    Return display info for a card using image URLs extracted from the bulk data.
    No network calls — all data is preloaded at startup.
    Returns a dict with:
        img_front  : URL of the front-face image (large)
        img_back   : URL of the back-face image, or None for single-faced cards
        is_sideways: True if the card should be displayed rotated 90°
    Result is cached in memory by card name.
    '''
    if card_name in _card_info_cache:
        return _card_info_cache[card_name]

    layout    = _name_to_layout.get(card_name, '')
    type_line = _name_to_type.get(card_name, '')
    keywords  = _name_to_keywords.get(card_name, [])
    is_sideways = (
        (layout in SIDEWAYS_LAYOUTS
         and not any(k in keywords for k in SIDEWAYS_LAYOUT_KEYWORD_EXCLUSIONS))
        or any(t in type_line for t in SIDEWAYS_TYPES)
    )

    img_front = _name_to_img_front.get(card_name) or None
    img_back  = _name_to_img_back.get(card_name)  or None

    info = {'img_front': img_front, 'img_back': img_back, 'is_sideways': is_sideways}
    _card_info_cache[card_name] = info
    return info


# ── Matchup logic ──────────────────────────────────────────────────────────────

def _pick_card_weighted(pool):
    '''Pick a card from pool with probability proportional to Elo rating.'''
    weights = [_elo_cache.get(name, INITIAL_ELO) for name in pool]
    return random.choices(pool, weights=weights, k=1)[0]


def _get_elo_bracket_pool(card_a_name):
    '''
    Choose an Elo bracket half-width uniformly at random from ELO_BRACKET_HALF_WIDTHS,
    then return the subset of _card_names whose Elo rank falls within that window
    around card_a.

    Returns (pool_or_None, label):
      pool_or_None — list of eligible card names, or None if no filter applies
      label        — string tag logged to votes.db (e.g. 'elo_5pct', 'elo_any')
    '''
    half_width = random.choice(ELO_BRACKET_HALF_WIDTHS)
    if half_width is None:
        return None, 'elo_any'

    sorted_names = _elo_sorted_names
    n = len(sorted_names)
    rank_map = {nm: i for i, nm in enumerate(sorted_names)}

    a_rank = rank_map.get(card_a_name, n // 2)
    a_pct  = a_rank / n

    lo_idx = int(max(0.0, a_pct - half_width) * n)
    hi_idx = int(min(1.0, a_pct + half_width) * n)
    hi_idx = max(lo_idx + 1, hi_idx)  # ensure at least one card

    pool = sorted_names[lo_idx:hi_idx]
    return (pool or None), f'elo_{int(half_width * 100)}pct'


def pick_matchup(unusual=False, mode=1, session_id=None, closed_loop_pool=None):
    '''
    Pick a card pair and return display info for both.

    closed_loop_pool: if provided, both cards are drawn from this list and all
                      mode/unusual/seen-reroll logic is bypassed.
    mode=1 (similar):  5 candidates per config, weighted selection.
    mode=2 (diverse):  15 candidates per config, uniform selection.
    mode=3 (broad):    uniform pick from closest 25% of all cards (balanced model).
    mode=4 (wild):     card_b drawn uniformly from all cards.
    unusual=True: forces card_a to be a Battle or double-faced card (for testing).
    session_id: if provided, avoids repeating recently seen cards (90% re-roll).
    '''
    if closed_loop_pool is not None:
        card_a      = random.choice(closed_loop_pool)
        card_b      = random.choice([c for c in closed_loop_pool if c != card_a])
        config_name = 'closed_loop'
    else:
        pool = _unusual_names if (unusual and _unusual_names) else _card_names
        seen = _session_seen.get(session_id, set()) if session_id else set()

        # card_a: weighted by Elo so higher-rated cards appear more often
        card_a = _pick_card_weighted(pool)
        if card_a in seen and random.random() < SEEN_REROLL_PROB:
            card_a = _pick_card_weighted(pool)

        # Elo bracket: restrict card_b candidates to cards near card_a's Elo rank
        elo_pool, elo_label = _get_elo_bracket_pool(card_a)
        # allowed_names=None means no filter (elo_any); similarity.py handles this correctly
        fallback_pool = elo_pool if elo_pool is not None else _card_names

        if mode == 4:
            card_b      = random.choice(fallback_pool)
            config_name = f'random+{elo_label}'
        elif mode == 3:
            n3          = min(_mode3_n_neighbors, max(1, len(fallback_pool)))
            candidates  = get_candidates(card_a, _models, n_neighbors=n3,
                                         allowed_names=elo_pool)
            card_list   = candidates['balanced'] or fallback_pool
            card_b      = random.choice(card_list)
            config_name = f'broad_balanced+{elo_label}'
        elif mode == 2:
            candidates    = get_candidates(card_a, _models, n_neighbors=MODE2_N_NEIGHBORS,
                                           allowed_names=elo_pool)
            chosen_config = random.choice(list(candidates.keys()))
            card_list     = candidates[chosen_config] or fallback_pool
            card_b        = random.choice(card_list)
            config_name   = f'{chosen_config}+{elo_label}'
        else:
            candidates  = get_candidates(card_a, _models, allowed_names=elo_pool)
            chosen_config = random.choice(list(candidates.keys()))
            card_list   = candidates[chosen_config] or fallback_pool
            card_b      = random.choices(card_list, weights=CANDIDATE_WEIGHTS[:len(card_list)], k=1)[0]
            config_name = f'{chosen_config}+{elo_label}'

        # Guard: ensure card_b differs from card_a
        if card_b == card_a:
            others = [c for c in fallback_pool if c != card_a]
            if others:
                card_b = random.choice(others)

        if session_id:
            seen.add(card_a)
            seen.add(card_b)
            _session_seen[session_id] = seen

    info_a = get_card_info(card_a)
    info_b = get_card_info(card_b)
    elos = {card_a: round(_elo_cache.get(card_a, INITIAL_ELO), 1),
            card_b: round(_elo_cache.get(card_b, INITIAL_ELO), 1)}

    return {
        'card_a':        card_a,
        'card_b':        card_b,
        'config':        config_name,
        'img_a_front':   info_a['img_front'],
        'img_a_back':    info_a['img_back'],
        'img_b_front':   info_b['img_front'],
        'img_b_back':    info_b['img_back'],
        'sideways_a':    info_a['is_sideways'],
        'sideways_b':    info_b['is_sideways'],
        'set_a':         _name_to_set.get(card_a, ''),
        'year_a':        int(_name_to_year.get(card_a, 0)) or '',
        'set_b':         _name_to_set.get(card_b, ''),
        'year_b':        int(_name_to_year.get(card_b, 0)) or '',
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
    return render_template('index.html')


@app.route('/faq')
def faq():
    return render_template('faq.html')



@app.route('/api/matchup')
def matchup():
    _ensure_session()
    sid  = session['session_id']
    mode = int(request.args.get('mode', 1))
    pool = _get_closed_loop_pool(sid, request.args.get('closed_loop', '0') == '1')
    return jsonify(pick_matchup(mode=mode, session_id=sid, closed_loop_pool=pool))


@app.route('/api/matchup/unusual')
def matchup_unusual():
    _ensure_session()
    sid  = session['session_id']
    mode = int(request.args.get('mode', 1))
    pool = _get_closed_loop_pool(sid, request.args.get('closed_loop', '0') == '1')
    return jsonify(pick_matchup(unusual=True, mode=mode, session_id=sid, closed_loop_pool=pool))


@app.route('/api/vote', methods=['POST'])
def vote():
    _ensure_session()
    data   = request.get_json()
    card_a = data['card_a']
    card_b = data['card_b']
    chosen = data['chosen']
    loser  = card_b if chosen == card_a else card_a

    log_vote(
        ip          = request.remote_addr,
        session_id  = session['session_id'],
        card_a      = card_a,
        card_b      = card_b,
        chosen      = chosen,
        config_name = data.get('config', ''),
    )

    winner_elo, loser_elo, delta_w, delta_l = update_elo(chosen, loser)
    if chosen == card_a:
        elo_a, delta_a = winner_elo, delta_w
        elo_b, delta_b = loser_elo,  delta_l
    else:
        elo_a, delta_a = loser_elo,  delta_l
        elo_b, delta_b = winner_elo, delta_w

    return jsonify({
        'elo_a':   round(elo_a,   1),
        'delta_a': round(delta_a, 1),
        'elo_b':   round(elo_b,   1),
        'delta_b': round(delta_b, 1),
    })


# ── DEV ONLY ── remove (or gate behind auth) before deploying to production ──
@app.route('/api/reset_elo', methods=['POST'])
def reset_elo_route():
    reset_elo()
    return jsonify({'status': 'ok', 'reset_to': INITIAL_ELO})
# ─────────────────────────────────────────────────────────────────────────────


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_db()
    app.run(debug=False, host='127.0.0.1', port=5000)
