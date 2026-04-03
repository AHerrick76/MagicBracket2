"""
sanity_check_shuffle.py — Verify Elo-weighted shuffle is biasing correctly.

Simulates 100 card_a shuffles using the live Elo ratings and queue, then
reports the distribution of mean Elo across the first 100 slots of each shuffle,
vs the expected mean if the shuffle were uniform.

Usage:
    DATABASE_URL=postgresql://... python sanity_check_shuffle.py --queue 2
"""

import argparse
import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)
if not DATABASE_URL:
    raise RuntimeError('DATABASE_URL environment variable is not set.')

QUEUES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queues.json')

parser = argparse.ArgumentParser()
parser.add_argument('--queue', type=int, default=2)
parser.add_argument('--n-sim', type=int, default=100)
parser.add_argument('--prefix', type=int, default=100, help='Number of leading slots to evaluate')
args = parser.parse_args()

# ── Load queue cards ─────────────────────────────────────────────────────────

with open(QUEUES_PATH) as f:
    queues_data = json.load(f)

queue = next((q for q in queues_data['queues'] if q['id'] == args.queue), None)
if queue is None:
    raise ValueError(f'Queue {args.queue} not found')
cards = queue['cards']
print(f'Queue {args.queue}: {len(cards)} cards')

# ── Load Elo ratings ─────────────────────────────────────────────────────────

conn = psycopg2.connect(DATABASE_URL)
elo_df = pd.read_sql(
    'SELECT card_name, rating FROM elo_ratings WHERE card_name = ANY(%s)',
    conn, params=(cards,)
)
conn.close()

INITIAL_ELO = 1500.0
elo_map = dict(zip(elo_df['card_name'], elo_df['rating']))
elos = np.array([elo_map.get(c, INITIAL_ELO) for c in cards])

print(f'Elo loaded: {len(elo_df)} cards  |  mean={elos.mean():.1f}  min={elos.min():.1f}  max={elos.max():.1f}')

# ── Replicate the weighting logic from app.py ─────────────────────────────────

_ELO_PCT_XP = [0.00, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 0.99, 1.00]
_ELO_PCT_FP = [0.45, 0.45, 0.65, 0.75, 1.00, 1.10, 1.25, 1.35, 1.35]

n = len(cards)
order = np.argsort(elos)
pct = np.empty(n, dtype=float)
pct[order] = np.arange(n) / max(n - 1, 1)
weights = np.interp(pct, _ELO_PCT_XP, _ELO_PCT_FP)

print(f'\nWeight sanity check (should increase with Elo):')
for p in [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]:
    idx = int(p * (n - 1))
    rank_card = cards[order[idx]]
    print(f'  pct={p:.2f}  elo={elos[order[idx]]:.1f}  weight={weights[order[idx]]:.3f}  card={rank_card}')

# ── Run simulations ───────────────────────────────────────────────────────────

def weighted_shuffle(card_indices, w):
    keys = [random.random() ** (1.0 / w[i]) for i in card_indices]
    return [card_indices[j] for j in np.argsort(keys)[::-1]]

prefix = args.prefix
mean_elos_weighted = []
mean_elos_uniform  = []

for _ in range(args.n_sim):
    idx = list(range(n))

    shuffled_w = weighted_shuffle(idx, weights)
    mean_elos_weighted.append(elos[shuffled_w[:prefix]].mean())

    random.shuffle(idx)
    mean_elos_uniform.append(elos[idx[:prefix]].mean())

mean_elos_weighted = np.array(mean_elos_weighted)
mean_elos_uniform  = np.array(mean_elos_uniform)

overall_mean = elos.mean()

print(f'\n── First-{prefix} mean Elo across {args.n_sim} simulations ──────────────────')
print(f'  Overall pool mean       : {overall_mean:.1f}')
print(f'  Weighted shuffle mean   : {mean_elos_weighted.mean():.1f}  (std {mean_elos_weighted.std():.1f})')
print(f'  Uniform shuffle mean    : {mean_elos_uniform.mean():.1f}  (std {mean_elos_uniform.std():.1f})')
print(f'  Uplift vs uniform       : {mean_elos_weighted.mean() - mean_elos_uniform.mean():+.1f} Elo points')

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(
    min(mean_elos_weighted.min(), mean_elos_uniform.min()) - 5,
    max(mean_elos_weighted.max(), mean_elos_uniform.max()) + 5,
    30
)
ax.hist(mean_elos_uniform,  bins=bins, alpha=0.55, color='steelblue', label='Uniform shuffle')
ax.hist(mean_elos_weighted, bins=bins, alpha=0.55, color='tomato',    label='Elo-weighted shuffle')
ax.axvline(overall_mean,                color='black',     linestyle=':',  linewidth=1.5, label=f'Pool mean ({overall_mean:.0f})')
ax.axvline(mean_elos_uniform.mean(),    color='steelblue', linestyle='--', linewidth=1.5, label=f'Uniform mean ({mean_elos_uniform.mean():.0f})')
ax.axvline(mean_elos_weighted.mean(),   color='tomato',    linestyle='--', linewidth=1.5, label=f'Weighted mean ({mean_elos_weighted.mean():.0f})')

ax.set_xlabel(f'Mean Elo in first {prefix} slots')
ax.set_ylabel('Simulations')
ax.set_title(f'Queue {args.queue} — Elo-weighted vs uniform shuffle  ({args.n_sim} simulations)')
ax.legend()
fig.tight_layout()
plt.show()
