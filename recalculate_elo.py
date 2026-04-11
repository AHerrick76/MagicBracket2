"""
recalculate_elo.py — Replay all votes and rewrite elo_ratings from scratch.

Reads every vote from the database in insertion order (by id), replays the
Elo formula with the current parameters, then updates the elo table in-place.

Usage:
    DATABASE_URL=postgresql://... python recalculate_elo.py
    DATABASE_URL=postgresql://... python recalculate_elo.py --phase top10

Safe to re-run: existing rows are overwritten, not duplicated.
"""

import argparse
import json
import os
import sys
from dotenv import load_dotenv
load_dotenv()

import psycopg2
from psycopg2.extras import execute_values

_parser = argparse.ArgumentParser()
_parser.add_argument('--phase', choices=['full', 'top10'], default='full',
                     help='Which phase to recalculate: full (default) or top10')
_args = _parser.parse_args()

# ── Parameters — must match the app for the chosen phase ─────────────────────

if _args.phase == 'top10':
    INITIAL_ELO = 1500.0   # note: top10 uses normalised starting Elos seeded in the DB
    ELO_K       = 32
    ELO_K_DECAY = 250
    VOTES_TABLE = 'votes_top10'
    ELO_TABLE   = 'elo_ratings_top10'
else:
    INITIAL_ELO = 1500.0
    ELO_K       = 32
    ELO_K_DECAY = 30
    VOTES_TABLE = 'votes'
    ELO_TABLE   = 'elo_ratings'

print(f'Phase: {_args.phase}  (votes={VOTES_TABLE}, elo={ELO_TABLE})')

DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)

if not DATABASE_URL:
    print('ERROR: DATABASE_URL environment variable is not set.', file=sys.stderr)
    sys.exit(1)


# ── Load votes ─────────────────────────────────────────────────────────────────

print('Connecting to database...')
conn = psycopg2.connect(DATABASE_URL)
cur  = conn.cursor()

cur.execute(f'SELECT id, card_a, card_b, chosen FROM {VOTES_TABLE} ORDER BY id')
votes = cur.fetchall()
print(f'Loaded {len(votes)} votes.')

if not votes:
    print('No votes to replay — nothing to do.')
    conn.close()
    sys.exit(0)


# ── Replay ─────────────────────────────────────────────────────────────────────

wins    = {}   # card_name → int
losses  = {}   # card_name → int

# Seed starting ratings: top10 uses normalized Elos from top_10_queue.json,
# not a flat 1500 starting value.
if _args.phase == 'top10':
    _top10_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'top_10_queue.json')
    with open(_top10_path, encoding='utf-8') as _f:
        _top10_data = json.load(_f)
    ratings = {entry['name']: entry['normalized_elo'] for entry in _top10_data['cards']}
    print(f'Seeded {len(ratings)} starting Elos from top_10_queue.json.')
else:
    ratings = {}   # card_name → float rating

def get_rating(name):
    return ratings.get(name, INITIAL_ELO)

def get_games(name):
    return wins.get(name, 0) + losses.get(name, 0)

def effective_k(name):
    return ELO_K * ELO_K_DECAY / (ELO_K_DECAY + get_games(name))

for vote_id, card_a, card_b, chosen in votes:
    winner = chosen
    loser  = card_b if chosen == card_a else card_a

    r_w = get_rating(winner)
    r_l = get_rating(loser)
    e_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))

    ratings[winner] = r_w + effective_k(winner) * (1.0 - e_w)
    ratings[loser]  = r_l + effective_k(loser)  * (0.0 - (1.0 - e_w))
    wins[winner]    = wins.get(winner, 0) + 1
    losses[loser]   = losses.get(loser, 0) + 1

print(f'Replayed {len(votes)} votes across {len(ratings)} cards.')
print(f'Rating range: [{min(ratings.values()):.1f}, {max(ratings.values()):.1f}]')


# ── Write results ──────────────────────────────────────────────────────────────

rows = [
    (name, round(ratings[name], 4), wins.get(name, 0), losses.get(name, 0))
    for name in ratings
]

print(f'Writing {len(rows)} updated ratings...')
execute_values(
    cur,
    f'''INSERT INTO {ELO_TABLE} (card_name, rating, wins, losses)
       VALUES %s
       ON CONFLICT (card_name) DO UPDATE SET
           rating       = EXCLUDED.rating,
           wins         = EXCLUDED.wins,
           losses       = EXCLUDED.losses,
           last_updated = NULL''',
    rows,
)

conn.commit()
cur.close()
conn.close()
print('Done.')
