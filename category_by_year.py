"""
category_by_year.py — Stacked bar chart + table of card category mix by year (post-C16).

Categories (mutually exclusive, in priority order):
  UB        — Universes Beyond
  Playtest  — Mystery Booster playtest cards and similar
  Funny     — Un-sets / holiday cards (excluding playtest sets)
  Normal    — everything else

Usage:
    python category_by_year.py
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_data import load_processed_cards

PLAYTEST_SETS = {'unk', 'cmb1', 'cmb2', 'mb2', 'pf25', 'pf24', 'punk'}
UB_SETS       = {'spm', 'spe', 'tla', 'tle', 'fic', 'tmt', 'tmc', 'msh', 'msc'}
UB_SLD_DATE   = '2025-06-13'

COLORS = {
    'Normal':           'steelblue',
    'Universes Beyond': 'tomato',
    'Playtest':         'goldenrod',
    'Funny':            'mediumseagreen',
    'Commander':        'mediumpurple',
}

# ── Load & tag ───────────────────────────────────────────────────────────────

df = load_processed_cards()
df = df[df['released_at'] > '2016-11-11'].copy()
df['year'] = pd.to_datetime(df['released_at']).dt.year

def _is_ub(row):
    if row.get('security_stamp') == 'triangle': return True
    if row['set'] in UB_SETS: return True
    if row['set'] == 'sld' and str(row['released_at'])[:10] >= UB_SLD_DATE: return True
    return False

df['is_playtest'] = df['set'].isin(PLAYTEST_SETS)
df['is_ub']       = df.apply(_is_ub, axis=1)
df['is_funny']    = (df['set_type'] == 'funny') & ~df['is_playtest']

def _category(row):
    if row['is_ub']:                          return 'Universes Beyond'
    if row['is_playtest']:                    return 'Playtest'
    if row['is_funny']:                       return 'Funny'
    if row['set_type'] == 'commander':        return 'Commander'
    return 'Normal'

df['category'] = df.apply(_category, axis=1)

# ── Build year × category counts & percentages ───────────────────────────────

counts = (
    df.groupby(['year', 'category'])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=list(COLORS.keys()), fill_value=0)
)
pct = counts.div(counts.sum(axis=1), axis=0) * 100

print(pct.round(1).to_string())

# ── Plot: stacked bar chart (left) + table (right) ───────────────────────────

fig = plt.figure(figsize=(16, 6))
ax_bar = fig.add_subplot(1, 2, 1)
ax_tbl = fig.add_subplot(1, 2, 2)

years  = pct.index.tolist()
bottom = pd.Series(0.0, index=pct.index)
for cat, color in COLORS.items():
    ax_bar.bar(pct.index, pct[cat], bottom=bottom, color=color, label=cat, width=0.7)
    bottom += pct[cat]

ax_bar.set_title('Post-C16 cards by category — relative frequency')
ax_bar.set_xlabel('Year')
ax_bar.set_ylabel('% of cards')
ax_bar.set_ylim(0, 100)
ax_bar.set_xticks(years)
ax_bar.tick_params(axis='x', rotation=45)
ax_bar.legend()

# ── Table ─────────────────────────────────────────────────────────────────────

ax_tbl.axis('off')

table_data = [
    [str(year)] + [f'{pct.loc[year, cat]:.1f}%  ({int(counts.loc[year, cat])})'
                   for cat in COLORS]
    for year in years
]
col_labels = ['Year'] + list(COLORS.keys())

tbl = ax_tbl.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.auto_set_column_width(list(range(len(col_labels))))

# Colour header cells to match line chart
for col_idx, cat in enumerate(COLORS.keys(), start=1):
    tbl[(0, col_idx)].set_facecolor(COLORS[cat])
    tbl[(0, col_idx)].set_text_props(color='white', fontweight='bold')

ax_tbl.set_title('Category % by year', pad=12)

fig.tight_layout()
plt.show()
