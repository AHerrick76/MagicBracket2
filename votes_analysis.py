"""
votes_analysis.py — Analyse Elo ratings by card category (UB / Funny / Normal) for one or more queues.

Usage:
    DATABASE_URL=postgresql://... python votes_analysis.py
    DATABASE_URL=postgresql://... python votes_analysis.py --queue 2
    DATABASE_URL=postgresql://... python votes_analysis.py --queue 2 3 4
    DATABASE_URL=postgresql://... python votes_analysis.py --phase top10
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
from statsmodels.stats.weightstats import ttest_ind

load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)
if not DATABASE_URL:
    raise RuntimeError('DATABASE_URL environment variable is not set.')

QUEUES_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queues.json')
UB_SETS      = {'spm', 'spe', 'tla', 'tle', 'fic', 'tmt', 'tmc', 'msh', 'msc'}
UB_SLD_DATE  = '2025-06-13'

parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['full', 'top10'], default='full',
                    help='Which phase to analyse: full (default) or top10')
parser.add_argument('--queue',  type=int, nargs='+', default=[2], help='Queue ID(s) to analyse (default: 2)')
parser.add_argument('--rarity', type=str, nargs='+', default=None,
                    help='Restrict Elo analysis to one or more rarities: common uncommon rare mythic (default: all)')
args = parser.parse_args()

VOTES_TABLE = 'votes_top10'       if args.phase == 'top10' else 'votes'
ELO_TABLE   = 'elo_ratings_top10' if args.phase == 'top10' else 'elo_ratings'
print(f'Phase: {args.phase}  (votes={VOTES_TABLE}, elo={ELO_TABLE})')

queue_ids = args.queue
label = f'Queue{"s" if len(queue_ids) > 1 else ""} {", ".join(str(q) for q in queue_ids)}'
rarity_filter = [r.lower() for r in args.rarity] if args.rarity else None
if rarity_filter:
    label += f' ({", ".join(rarity_filter)})'

# ── Load queue card list(s) ─────────────────────────────────────────────────

with open(QUEUES_PATH) as f:
    queues_data = json.load(f)

queue_cards = set()
for qid in queue_ids:
    q = next((q for q in queues_data['queues'] if q['id'] == qid), None)
    if q is None:
        raise ValueError(f'Queue {qid} not found in {QUEUES_PATH}')
    queue_cards.update(q['cards'])
    print(f'Queue {qid}: {len(q["cards"])} cards')

print(f'Total unique cards across {label}: {len(queue_cards)}')

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


df_queue['is_ub']       = df_queue.apply(_is_ub, axis=1)
df_queue['is_funny']    = (df_queue['set_type'] == 'funny') & ~df_queue['is_playtest'].fillna(False)
df_queue['is_playtest'] = df_queue['is_playtest'].fillna(False).astype(bool)

# ── Load Elo ratings from DB ────────────────────────────────────────────────

conn = psycopg2.connect(DATABASE_URL)
elo  = pd.read_sql(
    f'SELECT card_name, rating, wins, losses FROM {ELO_TABLE}',
    conn,
)
conn.close()

df = (
    df_queue[['name', 'set', 'set_name', 'released_at', 'set_type', 'rarity', 'is_ub', 'is_funny', 'is_playtest']]
    .merge(elo, left_on='name', right_on='card_name', how='left')
)
df['rating']       = df['rating'].fillna(1500.0)
df['wins']         = df['wins'].fillna(0).astype(int)
df['losses']       = df['losses'].fillna(0).astype(int)
df['games_played'] = df['wins'] + df['losses']

# Assign mutually exclusive category; priority: UB > Playtest > Funny > Commander > Normal
def _category(row):
    if row['is_ub']:
        return 'Universes Beyond'
    if row['is_playtest']:
        return 'Playtest'
    if row['is_funny']:
        return 'Funny'
    if row['set_type'] == 'commander':
        return 'Commander'
    return 'Normal'

df['category'] = df.apply(_category, axis=1)

# Optional rarity filter (applies to Elo analysis only; positional bias uses full vote log)
if rarity_filter:
    df = df[df['rarity'].str.lower().isin(rarity_filter)].copy()
    print(f'Rarity filter: {rarity_filter}  →  {len(df)} cards remaining')

# Only cards that have actually played; pure-1500 cards carry no signal
df_played = df[df['games_played'] > 0].copy()
print(f'Cards with ≥1 game played: {len(df_played)} / {len(df)}')

# ── Summary table ───────────────────────────────────────────────────────────

summary = (
    df_played
    .groupby('category')
    .agg(
        n=('rating', 'count'),
        mean_elo=('rating', 'mean'),
        median_elo=('rating', 'median'),
        std_elo=('rating', 'std'),
        mean_games=('games_played', 'mean'),
    )
    .round(1)
)
print()
print(summary.to_string())

# ── Pairwise Welch t-tests ───────────────────────────────────────────────────

groups = {cat: df_played[df_played['category'] == cat]['rating'].values
          for cat in ['Normal', 'Universes Beyond', 'Playtest', 'Funny', 'Commander']}

pairs = [
    ('Normal', 'Universes Beyond'),
    ('Normal', 'Playtest'),
    ('Normal', 'Funny'),
    ('Normal', 'Commander'),
    ('Universes Beyond', 'Playtest'),
    ('Universes Beyond', 'Funny'),
    ('Universes Beyond', 'Commander'),
    ('Playtest', 'Funny'),
    ('Playtest', 'Commander'),
    ('Funny', 'Commander'),
]

print()
print('── Pairwise Welch t-tests ───────────────────────────────────────────')
for a, b in pairs:
    if len(groups[a]) < 2 or len(groups[b]) < 2:
        print(f'  {a} vs {b}: insufficient data')
        continue
    t_stat, p_val, dof = ttest_ind(groups[a], groups[b], usevar='unequal')
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    print(f'  {a} vs {b}:')
    print(f'    t={t_stat:+.3f}  p={p_val:.4f} {sig}  dof={dof:.0f}'
          f'  means: {groups[a].mean():.1f} vs {groups[b].mean():.1f}'
          f'  n={len(groups[a])} vs {len(groups[b])}')
print()

# ── Visualisation ───────────────────────────────────────────────────────────

COLORS = {'Normal': 'steelblue', 'Universes Beyond': 'tomato', 'Playtest': 'goldenrod', 'Funny': 'mediumseagreen', 'Commander': 'mediumpurple'}

fig, ax = plt.subplots(figsize=(11, 5))
bins = np.linspace(df_played['rating'].min() - 10, df_played['rating'].max() + 10, 40)

for cat, color in COLORS.items():
    vals = groups[cat]
    if len(vals) == 0:
        continue
    ax.hist(vals, bins=bins, alpha=0.50, color=color, label=f'{cat} (n={len(vals)})')
    ax.axvline(vals.mean(), color=color, linestyle='--', linewidth=1.8,
               label=f'{cat} mean ({vals.mean():.0f})')

ax.set_title(f'{label} — Elo distribution by category')
ax.set_xlabel('Elo Rating')
ax.set_ylabel('Number of Cards')
ax.legend()
fig.tight_layout()
plt.show()

# ── Positional bias: card_a vs card_b win rates ─────────────────────────────
# (commented out for now)

# conn = psycopg2.connect(DATABASE_URL)
# votes = pd.read_sql(
#     'SELECT card_a, card_b, chosen FROM votes WHERE queue_id = ANY(%s)',
#     conn, params=(queue_ids,),
# )
# conn.close()
#
# print(f'\n{label} votes loaded: {len(votes):,}')
#
# # Overall: what fraction of votes did card_a win?
# card_a_wins  = (votes['chosen'] == votes['card_a']).sum()
# total_votes  = len(votes)
# overall_rate = card_a_wins / total_votes
#
# z_stat, p_overall = proportions_ztest(card_a_wins, total_votes, value=0.5)
#
# print()
# print('── Overall positional bias (card_a win rate) ───────────────────────')
# print(f'  card_a wins : {card_a_wins:,} / {total_votes:,}  ({overall_rate:.1%})')
# print(f'  z-statistic : {z_stat:+.4f}')
# print(f'  p-value     : {p_overall:.4f}')
#
# # Per-card: winrate as card_a vs winrate as card_b
# records = []
# for card in queue_cards:
#     as_a = votes[votes['card_a'] == card]
#     as_b = votes[votes['card_b'] == card]
#     n_a  = len(as_a)
#     n_b  = len(as_b)
#     records.append({
#         'card':       card,
#         'n_a':        n_a,
#         'n_b':        n_b,
#         'winrate_a':  (as_a['chosen'] == card).sum() / n_a if n_a > 0 else np.nan,
#         'winrate_b':  (as_b['chosen'] == card).sum() / n_b if n_b > 0 else np.nan,
#     })
#
# df_pos = pd.DataFrame(records)
#
# # Paired test: cards that appeared in both positions
# paired = df_pos.dropna(subset=['winrate_a', 'winrate_b']).copy()
# paired['diff'] = paired['winrate_a'] - paired['winrate_b']
#
# t_stat_p, p_paired, _ = DescrStatsW(paired['diff']).ttest_mean(0.0)
# sig_p = '***' if p_paired < 0.001 else '**' if p_paired < 0.01 else '*' if p_paired < 0.05 else 'n.s.'
#
# print()
# print('── Paired test: per-card (winrate_a − winrate_b) ───────────────────')
# print(f'  cards in both positions : {len(paired)}')
# print(f'  mean difference         : {paired["diff"].mean():+.4f}')
# print(f'  t-statistic             : {t_stat_p:+.4f}')
# print(f'  p-value                 : {p_paired:.4f}  {sig_p}')
# print()
#
# # Visualisation: scatter winrate_a vs winrate_b per card
# fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
#
# ax_s, ax_h = axes2
#
# ax_s.scatter(paired['winrate_b'], paired['winrate_a'], alpha=0.35, s=18, color='steelblue')
# lims = [0, 1]
# ax_s.plot(lims, lims, 'k--', linewidth=1, label='y = x (no bias)')
# ax_s.set_xlabel('Win rate as card_b')
# ax_s.set_ylabel('Win rate as card_a')
# ax_s.set_title(f'{label} — Per-card positional win rates (n={len(paired)})')
# ax_s.legend()
#
# ax_h.hist(paired['diff'], bins=30, color='mediumpurple', alpha=0.75)
# ax_h.axvline(0,                    color='black',      linestyle='--', linewidth=1.2, label='No bias')
# ax_h.axvline(paired['diff'].mean(), color='mediumpurple', linestyle='-',  linewidth=1.8,
#              label=f'Mean diff ({paired["diff"].mean():+.3f})')
# ax_h.set_xlabel('winrate_a − winrate_b')
# ax_h.set_ylabel('Number of cards')
# ax_h.set_title(
#     f'{label} — Positional win-rate difference  '
#     f'(t={t_stat_p:+.2f}, p={p_paired:.3f} {sig_p})'
# )
# ax_h.legend()
#
# fig2.tight_layout()
# plt.show()
