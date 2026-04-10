"""
set_report.py — Per-set Elo summary for all finished queues.

"Finished" means any queue whose ID is not in PENDING_QUEUES (default: 16, 17).
Only sets with at least MIN_SET_SIZE cards in finished queues are shown.

Usage:
    DATABASE_URL=postgresql://... python set_report.py
    DATABASE_URL=postgresql://... python set_report.py --pending 16 17 18
    DATABASE_URL=postgresql://... python set_report.py --min-cards 10
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_data import load_processed_cards

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--pending',   type=int, nargs='+', default=[16, 17],
                    help='Queue IDs not yet finished (excluded from counts)')
parser.add_argument('--min-cards', type=int, default=20,
                    help='Minimum cards in bracket for a set to appear (default 20)')
args = parser.parse_args()

PENDING_QUEUES = set(args.pending)
MIN_SET_SIZE   = args.min_cards

# ── Load queues ───────────────────────────────────────────────────────────────

QUEUES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queues.json')
with open(QUEUES_PATH, encoding='utf-8') as f:
    queues_data = json.load(f)

finished_cards = set()
for q in queues_data['queues']:
    if q['id'] not in PENDING_QUEUES:
        finished_cards.update(q['cards'])

print(f'Finished queues: {len(finished_cards):,} cards '
      f'(excluding queues {sorted(PENDING_QUEUES)})')

# ── Load Elo ratings ──────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)
if not DATABASE_URL:
    raise RuntimeError('DATABASE_URL environment variable is not set.')

conn = psycopg2.connect(DATABASE_URL)
elo_df = pd.read_sql(
    'SELECT card_name, rating FROM elo_ratings WHERE card_name = ANY(%s)',
    conn, params=(list(finished_cards),)
)
conn.close()

elo_map = dict(zip(elo_df['card_name'], elo_df['rating']))

# ── Load card metadata ────────────────────────────────────────────────────────

cards = load_processed_cards()
cards = cards[cards['name'].isin(finished_cards)].copy()
cards['elo'] = cards['name'].map(elo_map)

# ── Compute global top-10% threshold ─────────────────────────────────────────

top10_threshold = cards['elo'].quantile(0.90)

# ── Aggregate by set ──────────────────────────────────────────────────────────

def set_stats(g):
    n          = len(g)
    median_elo = round(g['elo'].median(), 1)
    top10_n    = int((g['elo'] >= top10_threshold).sum())
    top10_pct  = round(top10_n / n * 100, 1)
    best_row   = g.loc[g['elo'].idxmax()]
    best_card  = f'{best_row["name"]} ({round(best_row["elo"], 0):.0f})'
    return pd.Series({
        'cards_in_bracket': n,
        'median_elo':       median_elo,
        'top10_count':      top10_n,
        'top10_pct':        top10_pct,
        'highest_elo_card': best_card,
    })

report = (
    cards.groupby(['set_name', 'set'])
         .apply(set_stats, include_groups=False)
         .reset_index()
         .rename(columns={'set_name': 'Set Name', 'set': 'Set Code'})
)

report = report[report['cards_in_bracket'] >= MIN_SET_SIZE].copy()
report = report.sort_values('median_elo', ascending=False).reset_index(drop=True)

# ── Set card counts ───────────────────────────────────────────────────────────

def set_card_counts():
    """
    Returns a DataFrame with one row per set, sorted by release date descending.
    Columns: set_name, set, released_at, new_cards.
    'new_cards' is the number of unique cards first printed in that set
    (i.e. all cards in the processed dataset, which deduplicates by first printing).
    """
    all_cards = load_processed_cards()
    counts = (
        all_cards.groupby(['set_name', 'set', 'released_at'], sort=False)
                 .size()
                 .reset_index(name='new_cards')
                 .sort_values('released_at', ascending=False)
                 .reset_index(drop=True)
    )
    return counts

# ── Graph ─────────────────────────────────────────────────────────────────────

def plot_median_elo(report, min_cards=20, save=True):
    '''
    Horizontal bar chart of median Elo by set, filtered to sets with at least
    min_cards cards in the bracket. Sets sorted by median Elo ascending (highest at top).
    '''
    df = report[report['cards_in_bracket'] >= min_cards].copy()
    df = df.sort_values('median_elo', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.28)))
    ax.barh(df['Set Name'], df['median_elo'], color='steelblue', height=0.7)
    ax.axvline(1500, color='#888', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Median Elo')
    ax.set_title(f'Median Elo by Set (>={min_cards} cards in bracket)')
    ax.tick_params(axis='y', labelsize=7)
    fig.tight_layout()

    if save:
        out = f'set_median_elo_min{min_cards}.png'
        fig.savefig(out, dpi=150)
        print(f'Saved to {out}')

    plt.show()

# ── Print ─────────────────────────────────────────────────────────────────────

print(f'\nTop-10% threshold (global): {top10_threshold:.1f}')
print(f'Sets shown (>={MIN_SET_SIZE} cards): {len(report)}\n')

pd.set_option('display.max_rows',    200)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width',       160)
pd.set_option('display.max_colwidth', 45)

print(report[['Set Name', 'Set Code', 'cards_in_bracket',
              'median_elo', 'top10_count', 'top10_pct',
              'highest_elo_card']].to_string(index=False))
