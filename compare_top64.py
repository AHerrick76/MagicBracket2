"""
compare_top64.py — Compare the current top-64 cards in elo_ratings_top10
against the snapshot saved in top_64_candidates.json.

Default mode: prints cards that appear in only one of the two lists, with
their rank in each (or "not ranked" if absent).

--full mode: prints all 64 current cards with current rank, snapshot rank,
and movement (positive = climbed, negative = fell, NEW = not in snapshot).

Usage:
    DATABASE_URL=postgresql://... python compare_top64.py
    DATABASE_URL=postgresql://... python compare_top64.py --full
"""

import argparse
import json
import os
import sys

import psycopg2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--full', action='store_true',
                    help='Show all 64 current cards with rank movement vs snapshot')
args = parser.parse_args()

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_PATH  = os.path.join(BASE_DIR, 'top_64_candidates.json')
N              = 64

DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)
if not DATABASE_URL:
    sys.exit('Error: DATABASE_URL not set.')

# ── Load current top-N from DB ────────────────────────────────────────────────

print('Fetching current top-64 from elo_ratings_top10...')
conn = psycopg2.connect(DATABASE_URL)
df = pd.read_sql(
    'SELECT card_name, rating, wins, losses FROM elo_ratings_top10 ORDER BY rating DESC',
    conn
)
conn.close()

df['rank'] = range(1, len(df) + 1)
current_top = df.head(N).copy()
current_rank = dict(zip(current_top['card_name'], current_top['rank']))

print(f'  {len(df):,} total cards in table; top {N} fetched.')

# ── Load snapshot ─────────────────────────────────────────────────────────────

print(f'Loading snapshot from {SNAPSHOT_PATH}...')
with open(SNAPSHOT_PATH, encoding='utf-8') as f:
    snap = json.load(f)

snapshot_cards = snap['cards']
snapshot_rank  = {c['name']: c['rank'] for c in snapshot_cards}
snapshot_ts    = snap.get('generated_at', 'unknown')
print(f'  {len(snapshot_cards)} cards in snapshot (generated {snapshot_ts})')

# ── Find differences ──────────────────────────────────────────────────────────

current_set  = set(current_rank)
snapshot_set = set(snapshot_rank)

only_current  = current_set  - snapshot_set   # new entrants
only_snapshot = snapshot_set - current_set    # cards that fell out

col_w = max((len(n) for n in current_set | snapshot_set), default=20)
col_w = max(col_w, 20)

# ── --full mode: all 64 current cards with movement ───────────────────────────

if args.full:
    print()
    header  = f'  {"Card":<{col_w}}  {"Cur":>5}  {"Prev":>5}  {"Move":>6}'
    divider = '  ' + '-' * (col_w + 22)
    print(f'All {N} current top-{N} cards (snapshot: {snapshot_ts}):')
    print(header)
    print(divider)
    for name in sorted(current_rank, key=lambda n: current_rank[n]):
        cur_r  = current_rank[name]
        snap_r = snapshot_rank.get(name)
        if snap_r is None:
            prev_s = '  NEW'
            move_s = '   NEW'
        else:
            delta  = snap_r - cur_r   # positive = climbed
            prev_s = f'{snap_r:>5}'
            move_s = f'{delta:>+6}' if delta != 0 else f'{"—":>6}'
        print(f'  {name:<{col_w}}  {cur_r:>5}  {prev_s}  {move_s}')

    if only_snapshot:
        print()
        print(f'Dropped out of top-{N} ({len(only_snapshot)}):')
        print(header)
        print(divider)
        for name in sorted(only_snapshot, key=lambda n: snapshot_rank[n]):
            snap_r  = snapshot_rank[name]
            cur_row = df[df['card_name'] == name]
            cur_s   = str(int(cur_row.iloc[0]['rank'])) if not cur_row.empty else 'not in DB'
            print(f'  {name:<{col_w}}  {cur_s:>5}  {snap_r:>5}  {"OUT":>6}')

    sys.exit(0)

# ── Default mode: only differences ───────────────────────────────────────────

print()
if not only_current and not only_snapshot:
    print('No differences — both lists are identical.')
    sys.exit(0)

header = f'  {"Card":<{col_w}}  {"Current rank":>13}  {"Snapshot rank":>13}'
divider = '  ' + '-' * (col_w + 30)

if only_current:
    print(f'Cards in current top-{N} but NOT in snapshot ({len(only_current)}):')
    print(header)
    print(divider)
    for name in sorted(only_current, key=lambda n: current_rank[n]):
        cur_r  = current_rank[name]
        snap_r = snapshot_rank.get(name, None)
        snap_s = str(snap_r) if snap_r else 'not in snapshot'
        print(f'  {name:<{col_w}}  {cur_r:>13}  {snap_s:>13}')

print()

if only_snapshot:
    print(f'Cards in snapshot but NOT in current top-{N} ({len(only_snapshot)}):')
    print(header)
    print(divider)
    for name in sorted(only_snapshot, key=lambda n: snapshot_rank[n]):
        snap_r = snapshot_rank[name]
        # show their current full-table rank if they exist in the DB at all
        cur_row = df[df['card_name'] == name]
        if cur_row.empty:
            cur_s = 'not in DB'
        else:
            cur_s = str(int(cur_row.iloc[0]['rank']))
        print(f'  {name:<{col_w}}  {cur_s:>13}  {snap_r:>13}')
