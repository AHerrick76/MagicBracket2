"""
simulate_elo.py — Elo convergence simulation for the Magic Bracket.

Samples 500 cards at random, assigns them 'true' Elo ratings drawn from a
chess-like normal distribution, then simulates votes using Mode 3 (Broad)
matchup selection: balanced similarity model, 25%-pool candidate draw, with
the same Elo-bracket filtering and Elo-weighted card_a selection as app.py.

Tracks how quickly the simulated Elo percentile rankings converge to the
true rankings, plotting 3 convergence thresholds (10% / 5% / 1%) against
total vote count.

Usage:
    python simulate_elo.py
"""

import random
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_data import load_processed_cards
from similarity import build_candidate_models, get_candidates


# ── Parameters ─────────────────────────────────────────────────────────────────

SEED              = 42
N_CARDS           = 500
N_VOTES           = 20_000
SNAPSHOT_INTERVAL = 250     # take a convergence reading every N votes

# True Elo distribution — stable chess ratings are roughly normal
TRUE_ELO_MEAN = 1500.0
TRUE_ELO_STD  = 200.0      # std of ~200 gives ~10:1 odds at the 2-sigma extremes

# Elo update parameters — must match app.py
INITIAL_ELO = 1500.0
ELO_K       = 32

# Mode 3 / Elo bracket parameters — must match app.py
MODE3_FRACTION          = 0.25
ELO_BRACKET_HALF_WIDTHS = [0.05, 0.20, 0.40, None]

# Convergence thresholds (percentile distance, 0–1 scale)
THRESHOLDS = [0.10, 0.05, 0.01]

# How often to rebuild the Elo-sorted list (votes between rebuilds)
ELO_SORT_INTERVAL = 25


# ── Data loading ───────────────────────────────────────────────────────────────

random.seed(SEED)
rng = np.random.default_rng(SEED)

print('Loading card data...')
df       = load_processed_cards()
post_c16 = df[df['released_at'] > pd.Timestamp('2016-11-11')].reset_index(drop=True)
print(f'Post-C16 cards: {len(post_c16)}')

sample_idx = sorted(random.sample(range(len(post_c16)), N_CARDS))
sample_df  = post_c16.iloc[sample_idx].reset_index(drop=True)
card_names = sample_df['name'].tolist()
print(f'Sampled {N_CARDS} cards')

# Assign true Elos
true_elos    = rng.normal(TRUE_ELO_MEAN, TRUE_ELO_STD, N_CARDS)
true_elo_map = dict(zip(card_names, true_elos))

# True percentile ranks (fixed for the whole simulation)
true_sorted = sorted(card_names, key=lambda n: true_elo_map[n])
true_pct    = {name: i / (N_CARDS - 1) for i, name in enumerate(true_sorted)}

print(f'True Elo range: [{true_elos.min():.0f}, {true_elos.max():.0f}]  '
      f'mean={true_elos.mean():.0f}  std={true_elos.std():.0f}')


# ── Similarity models ──────────────────────────────────────────────────────────

print(f'\nBuilding similarity models for {N_CARDS} cards...')
models   = build_candidate_models(sample_df, n_neighbors=5)
mode3_n  = max(1, int(N_CARDS * MODE3_FRACTION))
print(f'Models ready.  Mode 3 pool size: {mode3_n}')


# ── Simulation state ───────────────────────────────────────────────────────────

sim_elos  = {name: INITIAL_ELO for name in card_names}
sim_games = {name: 0 for name in card_names}


# ── Helper functions ───────────────────────────────────────────────────────────


def get_elo_sorted() -> list:
    return sorted(card_names, key=lambda n: sim_elos[n])


def get_elo_bracket_pool(card_a_name: str, elo_sorted: list):
    """
    Reproduce app.py's _get_elo_bracket_pool():
    pick a half-width uniformly at random from ELO_BRACKET_HALF_WIDTHS and
    return the subset of cards whose current Elo rank falls within that window.
    Returns None for the 'elo_any' case (no filter).
    """
    half_width = random.choice(ELO_BRACKET_HALF_WIDTHS)
    if half_width is None:
        return None

    n      = len(elo_sorted)
    rank   = {nm: i for i, nm in enumerate(elo_sorted)}
    a_pct  = rank.get(card_a_name, n // 2) / n
    lo_idx = int(max(0.0, a_pct - half_width) * n)
    hi_idx = int(min(1.0, a_pct + half_width) * n)
    hi_idx = max(lo_idx + 1, hi_idx)
    pool   = elo_sorted[lo_idx:hi_idx]
    return pool if pool else None


def pick_matchup(elo_sorted: list):
    """
    Reproduce app.py's pick_matchup() for mode=3 (Broad):
      - card_a: weighted by current simulated Elo
      - card_b: uniform draw from Mode 3 similarity candidates, filtered by
                a randomly chosen Elo bracket around card_a
    """
    # card_a: weighted by current Elo (slight bias toward higher-rated cards)
    weights = [sim_elos[n] for n in card_names]
    card_a  = random.choices(card_names, weights=weights, k=1)[0]

    # Elo bracket pre-filter
    elo_pool      = get_elo_bracket_pool(card_a, elo_sorted)
    fallback_pool = elo_pool if elo_pool is not None else card_names

    # Mode 3 similarity candidates (balanced model, up to 25% of pool)
    n3         = min(mode3_n, max(1, len(fallback_pool)))
    candidates = get_candidates(card_a, models, n_neighbors=n3, allowed_names=elo_pool)
    card_list  = candidates['balanced'] or fallback_pool
    card_b     = random.choice(card_list)

    # Guard: ensure card_b differs from card_a
    if card_b == card_a:
        others = [c for c in fallback_pool if c != card_a]
        if others:
            card_b = random.choice(others)

    return card_a, card_b


def simulate_vote(card_a: str, card_b: str) -> str:
    """Pick a winner probabilistically from true Elos (standard chess formula)."""
    p_a = 1.0 / (1.0 + 10.0 ** ((true_elo_map[card_b] - true_elo_map[card_a]) / 400.0))
    return card_a if random.random() < p_a else card_b


def update_elo(winner: str, loser: str) -> None:
    r_w = sim_elos[winner]
    r_l = sim_elos[loser]
    e_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))
    sim_elos[winner] = r_w + ELO_K * (1.0 - e_w)
    sim_elos[loser]  = r_l + ELO_K * (0.0 - (1.0 - e_w))
    sim_games[winner] += 1
    sim_games[loser]  += 1


def compute_convergence() -> list:
    """
    For each card compare its percentile in the current simulated Elos vs the
    fixed true Elos.  Returns fraction of cards within each threshold.
    """
    sim_sorted = sorted(card_names, key=lambda n: sim_elos[n])
    sim_pct    = {name: i / (N_CARDS - 1) for i, name in enumerate(sim_sorted)}
    return [
        sum(1 for nm in card_names if abs(sim_pct[nm] - true_pct[nm]) <= t) / N_CARDS
        for t in THRESHOLDS
    ]


# ── Run simulation ─────────────────────────────────────────────────────────────

snapshots  = []   # [(vote_count, [frac_10pct, frac_5pct, frac_1pct])]
elo_sorted = get_elo_sorted()

# Baseline: all Elos equal → percentile order is arbitrary
snapshots.append((0, compute_convergence()))
print(f'\nBaseline (0 votes): {snapshots[0][1]}')

print(f'\nSimulating {N_VOTES:,} votes...')
for vote_idx in range(N_VOTES):

    if vote_idx % ELO_SORT_INTERVAL == 0:
        elo_sorted = get_elo_sorted()

    card_a, card_b = pick_matchup(elo_sorted)
    winner         = simulate_vote(card_a, card_b)
    loser          = card_b if winner == card_a else card_a
    update_elo(winner, loser)

    if (vote_idx + 1) % SNAPSHOT_INTERVAL == 0:
        fracs = compute_convergence()
        snapshots.append((vote_idx + 1, fracs))

        if (vote_idx + 1) % 2_500 == 0:
            print(f'  {vote_idx + 1:>6,} votes — '
                  f'within 10%: {fracs[0]:.1%}  '
                  f'within  5%: {fracs[1]:.1%}  '
                  f'within  1%: {fracs[2]:.1%}')

print('Simulation complete.\n')

# Final game-count stats
games = list(sim_games.values())
print(f'Games per card — min: {min(games)}  median: {np.median(games):.0f}  max: {max(games)}')
print(f'Final Elo range: [{min(sim_elos.values()):.0f}, {max(sim_elos.values()):.0f}]')


# ── Plot ───────────────────────────────────────────────────────────────────────

xs  = [s[0] for s in snapshots]
y10 = [s[1][0] for s in snapshots]
y5  = [s[1][1] for s in snapshots]
y1  = [s[1][2] for s in snapshots]

# Random-baseline reference lines (what you'd expect from a uniform random assignment)
baseline_10 = 2 * 0.10   # ~20% of cards fall within ±10pp by chance
baseline_5  = 2 * 0.05
baseline_1  = 2 * 0.01

fig, ax = plt.subplots(figsize=(12, 6))

ax.axhline(baseline_10, color='steelblue',  linestyle=':', linewidth=1,   alpha=0.45,
           label=f'Random baseline ≈{baseline_10:.0%} (±10%)')
ax.axhline(baseline_5,  color='darkorange', linestyle=':', linewidth=1,   alpha=0.45,
           label=f'Random baseline ≈{baseline_5:.0%} (±5%)')
ax.axhline(baseline_1,  color='seagreen',   linestyle=':', linewidth=1,   alpha=0.45,
           label=f'Random baseline ≈{baseline_1:.0%} (±1%)')

ax.plot(xs, y10, label='Within 10% of true percentile', color='steelblue',  linewidth=2.2)
ax.plot(xs, y5,  label='Within 5% of true percentile',  color='darkorange', linewidth=2.2)
ax.plot(xs, y1,  label='Within 1% of true percentile',  color='seagreen',   linewidth=2.2)

ax.set_xlabel('Total votes', fontsize=12)
ax.set_ylabel('Fraction of cards converged', fontsize=12)
ax.set_title(
    f'Elo convergence: {N_CARDS} cards, Mode 3 (Broad) matchups\n'
    f'True Elo ~ N({TRUE_ELO_MEAN:.0f}, {TRUE_ELO_STD:.0f}²) · '
    f'K={ELO_K} (flat) · '
    f'snapshot every {SNAPSHOT_INTERVAL} votes',
    fontsize=12
)
ax.set_xlim(0, N_VOTES)
ax.set_ylim(0, 1.04)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.25)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elo_convergence.png')
plt.savefig(out_path, dpi=150)
plt.show()
print(f'Saved to {out_path}')
