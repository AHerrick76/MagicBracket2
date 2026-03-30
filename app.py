'''
Flask web applet for the Magic Bracket voting system.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
Votes are logged to a PostgreSQL database (set DATABASE_URL env var).
'''

import json
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
QUEUES_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queues.json')

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
ELO_K_DECAY   = 30    # games played at which effective K halves

# ── DEV ONLY ── remove (or gate behind auth) before deploying to production ──
CLOSED_LOOP_SIZE = 25   # number of cards in a closed-loop test session

CANDIDATE_WEIGHTS = [0.45, 0.30, 0.15, 0.07, 0.03]

# Mode 2: wider candidate pool, uniform selection
MODE2_N_NEIGHBORS = 15

# Mode 3: uniform pick from the closest 25% of all cards (balanced model).
# Computed at startup once _card_names is available.
MODE3_FRACTION = 0.25

# Per-session state (server-side, resets with server restart).
# card_a queue: session_id → shuffled list of card names; popped one per matchup.
# seen pairs:  session_id → set of frozensets({card_a, card_b}) shown this session.
_session_card_a_queue: dict = {}
_session_seen_pairs:   dict = {}

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
                config_name TEXT,
                queue_id    INTEGER
            )
        ''')
        # queue_id column may not exist on databases created before this feature
        cur.execute('''
            ALTER TABLE votes ADD COLUMN IF NOT EXISTS queue_id INTEGER
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
        cur.execute('''
            CREATE TABLE IF NOT EXISTS queue_transitions (
                id           SERIAL  PRIMARY KEY,
                queue_id     INTEGER NOT NULL,
                activated_at TEXT    NOT NULL
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


# Sorted card names by Elo (all cards) — kept up-to-date after every vote.
_elo_sorted_names: list = []

# Active queue pool — the subset of cards in today's bracket.
# When no queue is active (queue_id=0) this equals _card_names.
_active_queue_id:   int  = 0
_active_pool:       list = []
_active_pool_set:   set  = set()
_active_elo_sorted: list = []   # _active_pool cards sorted by Elo


def _rebuild_elo_sorted():
    global _elo_sorted_names, _active_elo_sorted
    _elo_sorted_names  = sorted(_card_names, key=lambda nm: _elo_cache.get(nm, INITIAL_ELO))
    _active_elo_sorted = [nm for nm in _elo_sorted_names if nm in _active_pool_set]


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


# ── Queue management ───────────────────────────────────────────────────────────

def _load_active_queue(queue_id):
    '''
    Activate a queue by ID.  Updates all pool-related globals and clears
    per-session state so cards from the previous queue don't linger.
    queue_id=0 means no active queue — falls back to all post-C16 cards.
    '''
    global _active_queue_id, _active_pool, _active_pool_set, _active_elo_sorted
    global _mode3_n_neighbors, _unusual_names

    if queue_id == 0 or not _queues:
        _active_pool = _card_names
    else:
        raw = _queues.get(queue_id, {}).get('cards', [])
        # Guard against queues.json being out of sync with the card dataset
        _active_pool = [c for c in raw if c in _name_to_id]
        if not _active_pool:
            print(f'Warning: queue {queue_id} yielded 0 valid cards; using all cards.')
            _active_pool = _card_names

    _active_queue_id   = queue_id
    _active_pool_set   = set(_active_pool)
    _mode3_n_neighbors = max(1, int(len(_active_pool) * MODE3_FRACTION))
    _unusual_names     = [
        n for n in _active_pool
        if _name_to_layout.get(n) in DOUBLE_FACED_LAYOUTS
        or any(t in _name_to_type.get(n, '') for t in SIDEWAYS_TYPES)
    ]
    _active_elo_sorted = [nm for nm in _elo_sorted_names if nm in _active_pool_set]

    # Clear per-session state so queues rebuild against the new pool
    _session_card_a_queue.clear()
    _session_seen_pairs.clear()
    print(f'Active queue: {queue_id} ({len(_active_pool)} cards)')


def _load_queue_from_db():
    '''Read the latest queue_id from queue_transitions and activate it.'''
    with _get_db() as conn:
        cur = conn.cursor()
        cur.execute('SELECT queue_id FROM queue_transitions ORDER BY id DESC LIMIT 1')
        row = cur.fetchone()
    _load_active_queue(row[0] if row else 0)


def log_vote(ip, session_id, card_a, card_b, chosen, config_name, queue_id=None):
    ts = datetime.now(timezone.utc).isoformat()
    with _get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            'INSERT INTO votes '
            '(timestamp, ip_address, session_id, card_a, card_b, chosen, config_name, queue_id) '
            'VALUES (%s, %s, %s, %s, %s, %s, %s, %s)',
            (ts, ip, session_id, card_a, card_b, chosen, config_name, queue_id),
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
_name_to_img_front = dict(zip(_post_c16['name'], _post_c16['img_front'].where(_post_c16['img_front'].notna(), None)))
_name_to_img_back  = dict(zip(_post_c16['name'], _post_c16['img_back'].where(_post_c16['img_back'].notna(),  None)))
# Vintage legality: used to exclude non-tournament split cards (e.g. Mystery Booster
# playtest cards) from sideways rotation — they use layout='split' but are portrait.
_name_to_vintage_legal = dict(zip(_post_c16['name'], _post_c16.get('legal_vintage', pd.Series('not_legal', index=_post_c16.index))))
print(f'Post-C16 cards: {len(_card_names)}')

# Load queue schedule (may not exist yet if generate_queues.py hasn't been run)
if os.path.exists(QUEUES_PATH):
    with open(QUEUES_PATH, encoding='utf-8') as _f:
        _queue_data = json.load(_f)
    _queues = {q['id']: q for q in _queue_data['queues']}
    print(f'Loaded {_queue_data["total_queues"]} queues from queues.json.')
else:
    _queues = {}
    print('queues.json not found — run generate_queues.py to enable daily queue mode.')

print('Building candidate models (this takes a moment)...')
_models = build_candidate_models(_post_c16)

print('Initialising Elo ratings...')
init_db()
init_elo_ratings(_card_names)
_init_elo_cache()
_rebuild_elo_sorted()
_load_queue_from_db()   # sets _active_pool, _active_queue_id, _unusual_names, etc.
print('Ready. Visit http://127.0.0.1:5000')


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
    vintage_legal = _name_to_vintage_legal.get(card_name, 'not_legal') in {'legal', 'restricted'}
    is_sideways = (
        (layout in SIDEWAYS_LAYOUTS
         and vintage_legal
         and not any(k in keywords for k in SIDEWAYS_LAYOUT_KEYWORD_EXCLUSIONS))
        or any(t in type_line for t in SIDEWAYS_TYPES)
    )

    img_front = _name_to_img_front.get(card_name)
    img_back  = _name_to_img_back.get(card_name)
    if pd.isna(img_front): img_front = None
    if pd.isna(img_back):  img_back  = None

    info = {'img_front': img_front, 'img_back': img_back, 'is_sideways': is_sideways}
    _card_info_cache[card_name] = info
    return info


# ── Matchup logic ──────────────────────────────────────────────────────────────

def _next_card_a(session_id, pool):
    '''
    Return the next card_a from the per-session shuffled queue.
    When the queue empties, refill it with a fresh shuffle of pool,
    so every card gets a turn before any card repeats.
    '''
    queue = _session_card_a_queue.get(session_id)
    if not queue:
        new_queue = list(pool)
        random.shuffle(new_queue)
        _session_card_a_queue[session_id] = new_queue
    return _session_card_a_queue[session_id].pop()


def _compute_elo_bracket_pool(card_a_name, half_width):
    '''
    Return the subset of the active pool whose Elo rank falls within half_width
    of card_a's current rank.  Returns None when half_width is None (no filter).
    '''
    if half_width is None:
        return None

    sorted_names = _active_elo_sorted
    n            = len(sorted_names)
    try:
        a_rank = sorted_names.index(card_a_name)
    except ValueError:
        a_rank = n // 2
    a_pct  = a_rank / n
    lo_idx = int(max(0.0, a_pct - half_width) * n)
    hi_idx = int(min(1.0, a_pct + half_width) * n)
    hi_idx = max(lo_idx + 1, hi_idx)
    pool   = sorted_names[lo_idx:hi_idx]
    return pool if pool else None


def _bracket_fallback_sequence(initial_half_width):
    '''
    Return ELO_BRACKET_HALF_WIDTHS starting from initial_half_width, ordered
    narrow → wide.  Used to expand the bracket when all candidates at the
    current width have already been seen this session.
    e.g. initial=0.05 → [0.05, 0.20, 0.40, None]
         initial=0.40 → [0.40, None]
         initial=None → [None]
    '''
    try:
        idx = ELO_BRACKET_HALF_WIDTHS.index(initial_half_width)
    except ValueError:
        idx = len(ELO_BRACKET_HALF_WIDTHS) - 1
    return ELO_BRACKET_HALF_WIDTHS[idx:]


def pick_matchup(unusual=False, mode=1, session_id=None, closed_loop_pool=None):
    '''
    Pick a card pair and return display info for both.

    closed_loop_pool: if provided, both cards are drawn from this list and all
                      other logic is bypassed.
    mode=1 (similar):  5 candidates per config, weighted selection.
    mode=2 (diverse):  15 candidates per config, uniform selection.
    mode=3 (broad):    uniform pick from closest 25% of all cards (balanced model).
    mode=4 (wild):     card_b drawn uniformly from all cards.
    unusual=True: forces card_a to be a Battle or double-faced card (for testing).
    session_id: if provided, iterates card_a through a per-session shuffled queue
                (every card appears once per cycle) and avoids repeating seen pairs
                for card_b, expanding the Elo bracket as a fallback if needed.
    '''
    if closed_loop_pool is not None:
        card_a      = random.choice(closed_loop_pool)
        card_b      = random.choice([c for c in closed_loop_pool if c != card_a])
        config_name = 'closed_loop'
    else:
        pool = _unusual_names if (unusual and _unusual_names) else _active_pool

        # card_a: next from the shuffled per-session queue so every card gets a
        # turn before repeating.  Unusual mode bypasses the queue (separate pool).
        if unusual or session_id is None:
            card_a = random.choice(pool)
        else:
            card_a = _next_card_a(session_id, pool)

        seen_pairs = _session_seen_pairs.get(session_id, set()) if session_id else set()

        # Try Elo brackets from the randomly chosen width up to 'no filter',
        # stopping at the first width that yields at least one unseen candidate.
        initial_half_width = random.choice(ELO_BRACKET_HALF_WIDTHS)
        card_b, config_name = None, None

        for half_width in _bracket_fallback_sequence(initial_half_width):
            elo_pool      = _compute_elo_bracket_pool(card_a, half_width)
            elo_label     = f'elo_{int(half_width * 100)}pct' if half_width is not None else 'elo_any'
            # When the Elo bracket is unrestricted, still confine candidates to
            # the active pool so cards from other queues can't appear.
            effective_allowed = elo_pool if elo_pool is not None else (
                _active_pool if _active_queue_id else None
            )
            fallback_pool = effective_allowed if effective_allowed is not None else _active_pool

            if mode == 4:
                card_list    = [c for c in fallback_pool if c != card_a]
                chosen_label = 'random'
            elif mode == 3:
                n3           = min(_mode3_n_neighbors, max(1, len(fallback_pool)))
                raw          = get_candidates(card_a, _models, n_neighbors=n3,
                                              allowed_names=effective_allowed)
                card_list    = raw['balanced'] or fallback_pool
                chosen_label = 'broad_balanced'
            elif mode == 2:
                raw           = get_candidates(card_a, _models, n_neighbors=MODE2_N_NEIGHBORS,
                                               allowed_names=effective_allowed)
                chosen_config = random.choice(list(raw.keys()))
                card_list     = raw[chosen_config] or fallback_pool
                chosen_label  = chosen_config
            else:  # mode 1
                raw           = get_candidates(card_a, _models, allowed_names=effective_allowed)
                chosen_config = random.choice(list(raw.keys()))
                card_list     = raw[chosen_config] or fallback_pool
                chosen_label  = chosen_config

            # Filter to unseen pairs.  Mode 1 retains positional weights for
            # the remaining candidates; other modes use uniform selection.
            if mode == 1:
                weighted = list(zip(card_list, CANDIDATE_WEIGHTS[:len(card_list)]))
                unseen   = [(c, w) for c, w in weighted
                            if frozenset({card_a, c}) not in seen_pairs and c != card_a]
                if unseen:
                    cards, weights = zip(*unseen)
                    card_b      = random.choices(cards, weights=list(weights), k=1)[0]
                    config_name = f'{chosen_label}+{elo_label}'
                    break
            else:
                unseen = [c for c in card_list
                          if frozenset({card_a, c}) not in seen_pairs and c != card_a]
                if unseen:
                    card_b      = random.choice(unseen)
                    config_name = f'{chosen_label}+{elo_label}'
                    break

        # Absolute fallback: any card in the pool not yet paired with card_a.
        if card_b is None:
            others = [c for c in pool if frozenset({card_a, c}) not in seen_pairs and c != card_a]
            card_b      = random.choice(others) if others else random.choice(
                              [c for c in pool if c != card_a])
            config_name = 'fallback+elo_any'

        if session_id:
            _session_seen_pairs.setdefault(session_id, set()).add(frozenset({card_a, card_b}))

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
        queue_id    = _active_queue_id or None,
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


@app.route('/api/reload_queue', methods=['POST'])
def reload_queue():
    '''Re-read the active queue from the database and reload the pool in place.'''
    _load_queue_from_db()
    return jsonify({
        'status':    'ok',
        'queue_id':  _active_queue_id,
        'pool_size': len(_active_pool),
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
