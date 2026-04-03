"""
votes_analysis.py — Analyse Elo ratings by Universes Beyond status for a given queue.

Usage:
    DATABASE_URL=postgresql://... python votes_analysis.py
    DATABASE_URL=postgresql://... python votes_analysis.py --queue 2
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
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.weightstats import DescrStatsW, ttest_ind

load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)
if not DATABASE_URL:
    raise RuntimeError('DATABASE_URL environment variable is not set.')

QUEUES_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queues.json')
UB_SETS      = {'spm', 'spe', 'tla', 'tle', 'fic', 'tmt', 'tmc', 'msh', 'msc'}
UB_SLD_DATE  = '2025-06-13'

parser = argparse.ArgumentParser()
parser.add_argument('--queue', type=int, default=2, help='Queue ID to analyse (default: 2)')
args = parser.parse_args()

# ── Load queue card list ────────────────────────────────────────────────────

with open(QUEUES_PATH) as f:
    queues_data = json.load(f)

queue = next((q for q in queues_data['queues'] if q['id'] == args.queue), None)
if queue is None:
    raise ValueError(f'Queue {args.queue} not found in {QUEUES_PATH}')

queue_cards = set(queue['cards'])
print(f'Queue {args.queue}: {len(queue_cards)} cards')

# ── Load card metadata and tag UB ───────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_data import load_processed_cards

df_cards = load_processed_cards()
df_queue = df_cards[df_cards['name'].isin(queue_cards)].copy()


def _is_ub(row):
    if row.get('security_stamp') == 'triangle':
        return True
    if row['set'] in UB_SETS:
        return True
    if row['set'] == 'sld' and str(row['released_at'])[:10] >= UB_SLD_DATE:
        return True
    return False


df_queue['is_ub'] = df_queue.apply(_is_ub, axis=1)

# ── Load Elo ratings from DB ────────────────────────────────────────────────

conn = psycopg2.connect(DATABASE_URL)
elo  = pd.read_sql(
    'SELECT card_name, rating, wins, losses FROM elo_ratings',
    conn,
)
conn.close()

df = (
    df_queue[['name', 'set', 'set_name', 'released_at', 'is_ub']]
    .merge(elo, left_on='name', right_on='card_name', how='left')
)
df['rating']       = df['rating'].fillna(1500.0)
df['wins']         = df['wins'].fillna(0).astype(int)
df['losses']       = df['losses'].fillna(0).astype(int)
df['games_played'] = df['wins'] + df['losses']

# Only cards that have actually played; pure-1500 cards carry no signal
df_played = df[df['games_played'] > 0].copy()
print(f'Cards with ≥1 game played: {len(df_played)} / {len(df)}')

# ── Summary table ───────────────────────────────────────────────────────────

summary = (
    df_played
    .groupby('is_ub')
    .agg(
        n=('rating', 'count'),
        mean_elo=('rating', 'mean'),
        median_elo=('rating', 'median'),
        std_elo=('rating', 'std'),
        mean_games=('games_played', 'mean'),
    )
    .rename(index={False: 'Non-UB', True: 'Universes Beyond'})
    .round(1)
)
print()
print(summary.to_string())

# ── t-test ──────────────────────────────────────────────────────────────────

ub_elo     = df_played[df_played['is_ub']]['rating'].values
non_ub_elo = df_played[~df_played['is_ub']]['rating'].values

t_stat, p_val, dof = ttest_ind(ub_elo, non_ub_elo, usevar='unequal')

print()
print('── Welch t-test: UB vs Non-UB Elo ─────────────────────────────────')
print(f'  t-statistic : {t_stat:+.4f}')
print(f'  p-value     : {p_val:.4f}')
print(f'  dof         : {dof:.1f}')
print(f'  n(UB)       : {len(ub_elo)}')
print(f'  n(non-UB)   : {len(non_ub_elo)}')
print()

# ── Visualisation ───────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 5))

bins = np.linspace(df_played['rating'].min() - 10, df_played['rating'].max() + 10, 40)

ax.hist(non_ub_elo, bins=bins, alpha=0.55, color='steelblue', label='Non-UB')
ax.hist(ub_elo,     bins=bins, alpha=0.55, color='tomato',    label='Universes Beyond')

ax.axvline(non_ub_elo.mean(), color='steelblue', linestyle='--', linewidth=1.8,
           label=f'Non-UB mean ({non_ub_elo.mean():.0f})')
ax.axvline(ub_elo.mean(),     color='tomato',    linestyle='--', linewidth=1.8,
           label=f'UB mean ({ub_elo.mean():.0f})')

sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
ax.set_title(
    f'Queue {args.queue} — Elo distribution: UB vs Non-UB  '
    f'(t={t_stat:+.2f}, p={p_val:.3f} {sig})'
)
ax.set_xlabel('Elo Rating')
ax.set_ylabel('Number of Cards')
ax.legend()
fig.tight_layout()
plt.show()

# ── Positional bias: card_a vs card_b win rates ─────────────────────────────

conn = psycopg2.connect(DATABASE_URL)
votes = pd.read_sql(
    'SELECT card_a, card_b, chosen FROM votes WHERE queue_id = %s',
    conn, params=(args.queue,),
)
conn.close()

print(f'\nQueue {args.queue} votes loaded: {len(votes):,}')

# Overall: what fraction of votes did card_a win?
card_a_wins  = (votes['chosen'] == votes['card_a']).sum()
total_votes  = len(votes)
overall_rate = card_a_wins / total_votes

z_stat, p_overall = proportions_ztest(card_a_wins, total_votes, value=0.5)

print()
print('── Overall positional bias (card_a win rate) ───────────────────────')
print(f'  card_a wins : {card_a_wins:,} / {total_votes:,}  ({overall_rate:.1%})')
print(f'  z-statistic : {z_stat:+.4f}')
print(f'  p-value     : {p_overall:.4f}')

# Per-card: winrate as card_a vs winrate as card_b
records = []
for card in queue_cards:
    as_a = votes[votes['card_a'] == card]
    as_b = votes[votes['card_b'] == card]
    n_a  = len(as_a)
    n_b  = len(as_b)
    records.append({
        'card':       card,
        'n_a':        n_a,
        'n_b':        n_b,
        'winrate_a':  (as_a['chosen'] == card).sum() / n_a if n_a > 0 else np.nan,
        'winrate_b':  (as_b['chosen'] == card).sum() / n_b if n_b > 0 else np.nan,
    })

df_pos = pd.DataFrame(records)

# Paired test: cards that appeared in both positions
paired = df_pos.dropna(subset=['winrate_a', 'winrate_b']).copy()
paired['diff'] = paired['winrate_a'] - paired['winrate_b']

t_stat_p, p_paired, _ = DescrStatsW(paired['diff']).ttest_mean(0.0)
sig_p = '***' if p_paired < 0.001 else '**' if p_paired < 0.01 else '*' if p_paired < 0.05 else 'n.s.'

print()
print('── Paired test: per-card (winrate_a − winrate_b) ───────────────────')
print(f'  cards in both positions : {len(paired)}')
print(f'  mean difference         : {paired["diff"].mean():+.4f}')
print(f'  t-statistic             : {t_stat_p:+.4f}')
print(f'  p-value                 : {p_paired:.4f}  {sig_p}')
print()

# Visualisation: scatter winrate_a vs winrate_b per card
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

ax_s, ax_h = axes2

ax_s.scatter(paired['winrate_b'], paired['winrate_a'], alpha=0.35, s=18, color='steelblue')
lims = [0, 1]
ax_s.plot(lims, lims, 'k--', linewidth=1, label='y = x (no bias)')
ax_s.set_xlabel('Win rate as card_b')
ax_s.set_ylabel('Win rate as card_a')
ax_s.set_title(f'Queue {args.queue} — Per-card positional win rates (n={len(paired)})')
ax_s.legend()

ax_h.hist(paired['diff'], bins=30, color='mediumpurple', alpha=0.75)
ax_h.axvline(0,                    color='black',      linestyle='--', linewidth=1.2, label='No bias')
ax_h.axvline(paired['diff'].mean(), color='mediumpurple', linestyle='-',  linewidth=1.8,
             label=f'Mean diff ({paired["diff"].mean():+.3f})')
ax_h.set_xlabel('winrate_a − winrate_b')
ax_h.set_ylabel('Number of cards')
ax_h.set_title(
    f'Queue {args.queue} — Positional win-rate difference  '
    f'(t={t_stat_p:+.2f}, p={p_paired:.3f} {sig_p})'
)
ax_h.legend()

fig2.tight_layout()
plt.show()
