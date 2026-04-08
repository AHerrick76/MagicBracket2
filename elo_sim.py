"""
elo_sim.py — Compare Elo K-factor schemes on real vote history.

Two analyses:

  1. Chronological replay
     Replay votes in insertion order under each K-factor scheme.
     Reports final rating distributions and Spearman rank correlations
     between schemes (do cards end up in roughly the same order?).

  2. Shuffle test
     Replay the vote sequence N times in random order under each scheme.
     Reports per-card std dev of final rating across shuffles — the direct
     measure of how much vote-ordering affects a card's final rating.

Note: even flat-K is slightly order-sensitive because expected scores depend
on current ratings, which evolve as votes are processed.  The shuffle test
quantifies the *extra* sensitivity introduced by K-decay.

Usage:
    DATABASE_URL=postgresql://... python elo_sim.py
    DATABASE_URL=postgresql://... python elo_sim.py --shuffles 200
    DATABASE_URL=postgresql://... python elo_sim.py --shuffles 200 --queue 1
    DATABASE_URL=postgresql://... python elo_sim.py --no-shuffle
"""

import argparse
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from scipy.stats import spearmanr

load_dotenv()

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Compare Elo K-factor schemes on real votes.')
parser.add_argument('--shuffles', type=int, default=100,
                    help='Number of random shuffles for the shuffle test (default: 100).')
parser.add_argument('--no-shuffle', action='store_true',
                    help='Skip the shuffle test (faster).')
parser.add_argument('--queue', type=int, default=None,
                    help='Restrict to votes from a single queue_id.')
parser.add_argument('--top-n', type=int, default=15,
                    help='Number of most/least order-sensitive cards to show (default: 15).')
args = parser.parse_args()

# ── K-factor schemes ───────────────────────────────────────────────────────────
#
# Each scheme is a dict with:
#   label  : display name
#   k_fn   : callable(games_played) -> effective K for this vote

INITIAL_ELO = 1500.0

SCHEMES = [
    {'label': 'flat_16',       'k_fn': lambda g: 16.0},
    {'label': 'flat_32',       'k_fn': lambda g: 32.0},
    {'label': 'decay_fast',    'k_fn': lambda g: 32 * 15  / (15  + g)},  # halves at 15 games
    {'label': 'decay_current', 'k_fn': lambda g: 32 * 30  / (30  + g)},  # halves at 30 games (app.py)
    {'label': 'decay_slow',    'k_fn': lambda g: 32 * 60  / (60  + g)},  # halves at 60 games
]

# ── Database ───────────────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)
if not DATABASE_URL:
    print('ERROR: DATABASE_URL environment variable is not set.', file=sys.stderr)
    sys.exit(1)

print('Loading votes from database...')
conn = psycopg2.connect(DATABASE_URL)
cur  = conn.cursor()

if args.queue is not None:
    cur.execute(
        'SELECT card_a, card_b, chosen FROM votes WHERE queue_id = %s ORDER BY id',
        (args.queue,),
    )
    queue_label = f' (queue {args.queue})'
else:
    cur.execute('SELECT card_a, card_b, chosen FROM votes ORDER BY id')
    queue_label = ''

votes = cur.fetchall()
conn.close()

if not votes:
    print(f'No votes found{queue_label}. Nothing to do.')
    sys.exit(0)

print(f'Loaded {len(votes):,} votes{queue_label}.')

# ── Core replay function ───────────────────────────────────────────────────────

def replay(vote_sequence, k_fn):
    """
    Replay a sequence of (card_a, card_b, chosen) tuples under k_fn.
    Returns a dict {card_name: final_rating}.
    """
    ratings = defaultdict(lambda: INITIAL_ELO)
    games   = defaultdict(int)

    for card_a, card_b, chosen in vote_sequence:
        winner = chosen
        loser  = card_b if chosen == card_a else card_a

        r_w = ratings[winner]
        r_l = ratings[loser]
        e_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))

        ratings[winner] += k_fn(games[winner]) * (1.0 - e_w)
        ratings[loser]  += k_fn(games[loser])  * (0.0 - (1.0 - e_w))
        games[winner]   += 1
        games[loser]    += 1

    return dict(ratings)

# ── 1. Chronological replay ────────────────────────────────────────────────────

print('\n' + '=' * 72)
print('1. Chronological replay')
print('=' * 72)

chron_results = {}   # label → {card: rating}
for scheme in SCHEMES:
    chron_results[scheme['label']] = replay(votes, scheme['k_fn'])

# Build a combined DataFrame (only cards that appear in at least one result)
all_cards = sorted({c for r in chron_results.values() for c in r})

chron_df = pd.DataFrame(
    {label: [res.get(card, INITIAL_ELO) for card in all_cards] for label, res in chron_results.items()},
    index=all_cards,
)
chron_df.index.name = 'card_name'

# Summary stats
summary_rows = []
for scheme in SCHEMES:
    label = scheme['label']
    vals  = chron_df[label]
    summary_rows.append({
        'scheme':     label,
        'min':        round(vals.min(), 1),
        'max':        round(vals.max(), 1),
        'mean':       round(vals.mean(), 1),
        'std':        round(vals.std(), 1),
        'cards_seen': sum(1 for c in all_cards if c in chron_results[label]),
    })

print(pd.DataFrame(summary_rows).to_string(index=False))

# Spearman rank correlations
print('\nSpearman rank correlation between schemes:')
labels = [s['label'] for s in SCHEMES]
corr_matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)

for a in labels:
    for b in labels:
        rho, _ = spearmanr(chron_df[a], chron_df[b])
        corr_matrix.loc[a, b] = round(rho, 4)

print(corr_matrix.to_string())

# Top/bottom cards under current decay vs flat_32
print(f'\nTop {args.top_n} cards by rating — decay_current vs flat_32:')
compare = chron_df[['decay_current', 'flat_32']].copy()
compare['rank_decay']   = compare['decay_current'].rank(ascending=False).astype(int)
compare['rank_flat32']  = compare['flat_32'].rank(ascending=False).astype(int)
compare['rank_delta']   = compare['rank_decay'] - compare['rank_flat32']
compare = compare.sort_values('decay_current', ascending=False)
print(compare.head(args.top_n).to_string())

print(f'\nLargest rank shifts (decay_current vs flat_32):')
compare_sorted = compare.reindex(compare['rank_delta'].abs().sort_values(ascending=False).index)
print(compare_sorted.head(args.top_n).to_string())

# ── 2. Shuffle test ────────────────────────────────────────────────────────────

if args.no_shuffle:
    print('\n[Shuffle test skipped — use without --no-shuffle to run it.]')
    sys.exit(0)

N = args.shuffles
print('\n' + '=' * 72)
print(f'2. Shuffle test  ({N} shuffles)')
print('=' * 72)
print('Measures how much vote-ordering affects each card\'s final rating.\n')

# For each scheme, collect N replays under shuffled vote order.
# Store per-card ratings across shuffles in a list-of-dicts, then compute std.

shuffle_std = {}   # label → {card: std_dev_across_shuffles}

votes_list = list(votes)

for scheme in SCHEMES:
    label  = scheme['label']
    k_fn   = scheme['k_fn']
    # ratings_by_card[card] = list of final ratings across shuffles
    ratings_by_card = defaultdict(list)

    for i in range(N):
        shuffled = votes_list.copy()
        random.shuffle(shuffled)
        result = replay(shuffled, k_fn)
        for card, rating in result.items():
            ratings_by_card[card].append(rating)

    shuffle_std[label] = {
        card: float(np.std(ratings)) for card, ratings in ratings_by_card.items()
    }
    print(f'  {label}: done ({N} shuffles)')

# Summary: median / 90th pct / max std per scheme
print('\nPer-card rating std dev across shuffles (lower = more stable):')
std_summary = []
for scheme in SCHEMES:
    label = scheme['label']
    stds  = list(shuffle_std[label].values())
    std_summary.append({
        'scheme':      label,
        'median_std':  round(float(np.median(stds)), 2),
        'p90_std':     round(float(np.percentile(stds, 90)), 2),
        'max_std':     round(float(np.max(stds)), 2),
        'n_cards':     len(stds),
    })

print(pd.DataFrame(std_summary).to_string(index=False))

# Most order-sensitive cards under decay_current
print(f'\nMost order-sensitive cards under decay_current (top {args.top_n} by rating std dev):')
decay_stds = shuffle_std['decay_current']
flat_stds  = shuffle_std['flat_32']

sensitive = pd.DataFrame([
    {
        'card_name':        card,
        'decay_std':        round(decay_stds.get(card, 0), 2),
        'flat32_std':       round(flat_stds.get(card, 0), 2),
        'extra_sensitivity': round(decay_stds.get(card, 0) - flat_stds.get(card, 0), 2),
        'decay_rating':     round(chron_results['decay_current'].get(card, INITIAL_ELO), 1),
        'flat32_rating':    round(chron_results['flat_32'].get(card, INITIAL_ELO), 1),
        'games':            sum(1 for _, cb, ch in votes if _ == card or cb == card),
    }
    for card in all_cards
], columns=['card_name', 'decay_std', 'flat32_std', 'extra_sensitivity',
            'decay_rating', 'flat32_rating', 'games'])

sensitive = sensitive.sort_values('decay_std', ascending=False)
print(sensitive.head(args.top_n).to_string(index=False))

print(f'\nLeast order-sensitive cards under decay_current (bottom {args.top_n}):')
print(sensitive.tail(args.top_n).sort_values('decay_std').to_string(index=False))
