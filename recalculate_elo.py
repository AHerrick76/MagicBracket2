"""
recalculate_elo.py — Replay all votes and rewrite elo_ratings from scratch.

Reads every vote from the database in insertion order (by id), replays the
Elo formula with the current app.py parameters (flat K, no decay), then
updates the elo_ratings table in-place.

Usage:
    DATABASE_URL=postgresql://... python recalculate_elo.py

Safe to re-run: existing elo_ratings rows are overwritten, not duplicated.
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

import psycopg2
from psycopg2.extras import execute_values

# ── Parameters — must match app.py ────────────────────────────────────────────

INITIAL_ELO = 1500.0
ELO_K       = 32

DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)

if not DATABASE_URL:
    print('ERROR: DATABASE_URL environment variable is not set.', file=sys.stderr)
    sys.exit(1)


# ── Load votes ─────────────────────────────────────────────────────────────────

print('Connecting to database...')
conn = psycopg2.connect(DATABASE_URL)
cur  = conn.cursor()

cur.execute('SELECT id, card_a, card_b, chosen FROM votes ORDER BY id')
votes = cur.fetchall()
print(f'Loaded {len(votes)} votes.')

if not votes:
    print('No votes to replay — nothing to do.')
    conn.close()
    sys.exit(0)


# ── Replay ─────────────────────────────────────────────────────────────────────

ratings = {}   # card_name → float rating
wins    = {}   # card_name → int
losses  = {}   # card_name → int

def get_rating(name):
    return ratings.get(name, INITIAL_ELO)

for vote_id, card_a, card_b, chosen in votes:
    winner = chosen
    loser  = card_b if chosen == card_a else card_a

    r_w = get_rating(winner)
    r_l = get_rating(loser)
    e_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))

    ratings[winner] = r_w + ELO_K * (1.0 - e_w)
    ratings[loser]  = r_l + ELO_K * (0.0 - (1.0 - e_w))
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
    '''INSERT INTO elo_ratings (card_name, rating, wins, losses)
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
