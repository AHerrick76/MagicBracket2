"""
elo_stability.py — Simulate Elo stability across shuffled vote orderings for a queue.

Loads all votes for a target queue and replays them N times in random order under
the current Elo parameters (K-decay matching app.py). Reports how much each card's
final Elo varies depending on vote order — a measure of how reliable the ranking is.

Designed for use in an ipython session:

    %run elo_stability.py
    sim = QueueSimulator(queue_id=5)
    sim.run(n=200)

    # Card mode — distribution of final Elo for one card
    fig, df = sim.card_distribution('Sheoldred, the Apocalypse')

    # Top-10% mode — stability of the top 10% cutoff
    fig, df = sim.top_pct_stability()
    df.head(30)   # cards near the boundary, sorted by P(top 10%)
"""

import json
import os
import random
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

QUEUES_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queues.json')
DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)

INITIAL_ELO = 1500.0
ELO_K       = 32
ELO_K_DECAY = 30


# ── Core replay ───────────────────────────────────────────────────────────────

def _replay(vote_sequence):
    """
    Replay (card_a, card_b, chosen) tuples under the app.py K-decay formula.
    Returns dict {card_name: final_rating}.
    """
    ratings = defaultdict(lambda: INITIAL_ELO)
    games   = defaultdict(int)

    for card_a, card_b, chosen in vote_sequence:
        winner = chosen
        loser  = card_b if chosen == card_a else card_a

        r_w = ratings[winner]
        r_l = ratings[loser]
        e_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))

        k_w = ELO_K * ELO_K_DECAY / (ELO_K_DECAY + games[winner])
        k_l = ELO_K * ELO_K_DECAY / (ELO_K_DECAY + games[loser])

        ratings[winner] += k_w * (1.0 - e_w)
        ratings[loser]  += k_l * (0.0 - (1.0 - e_w))
        games[winner]   += 1
        games[loser]    += 1

    return dict(ratings)


# ── Simulator ─────────────────────────────────────────────────────────────────

class QueueSimulator:
    """
    Loads votes and live Elo for one queue, then runs N shuffle simulations.

    Parameters
    ----------
    queue_id : int
        The queue to analyse.

    Attributes (available after run())
    ------------------------------------
    results : dict[card_name, list[float]]
        Per-card list of final Elo values across all simulations.
    n_sims : int
        Number of simulations completed.
    live_elo : dict[card_name, float]
        Elo from chronological replay of queue votes — the fair baseline for comparisons.
    db_elo : dict[card_name, float]
        Raw Elo from the database (global vote history). Available for reference but
        not used in charts — it's computed from different inputs than the simulations.
    queue_cards : list[str]
        Card names in this queue.
    top_pct_cutoff : int
        Number of cards in the top 10% (ceil(len(queue_cards) * 0.10)).
    """

    def __init__(self, queue_id: int):
        self.queue_id = queue_id
        self.results:    dict[str, list[float]] = {}
        self.n_sims:     int  = 0
        self.live_elo:   dict[str, float] = {}   # chronological replay of queue votes (fair baseline)
        self.db_elo:     dict[str, float] = {}   # raw DB value (global history, for reference only)
        self.queue_cards: list[str] = []
        self.top_pct_cutoff: int = 0
        self._votes: list[tuple] = []

        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        """Load queue card list, votes, and live Elo from DB."""
        # Queue cards
        with open(QUEUES_PATH) as f:
            queues_data = json.load(f)
        queue = next((q for q in queues_data['queues'] if q['id'] == self.queue_id), None)
        if queue is None:
            raise ValueError(f'Queue {self.queue_id} not found in {QUEUES_PATH}')
        self.queue_cards     = queue['cards']
        self.top_pct_cutoff  = max(1, int(np.ceil(len(self.queue_cards) * 0.10)))

        print(f'Queue {self.queue_id}: {len(self.queue_cards)} cards, '
              f'top-10% cutoff = {self.top_pct_cutoff} cards')

        if not DATABASE_URL:
            raise RuntimeError('DATABASE_URL not set.')

        conn = psycopg2.connect(DATABASE_URL)

        # Votes scoped to this queue_id
        votes_df = pd.read_sql(
            'SELECT card_a, card_b, chosen FROM votes WHERE queue_id = %s ORDER BY id',
            conn, params=(self.queue_id,)
        )
        self._votes = list(votes_df.itertuples(index=False, name=None))
        print(f'Loaded {len(self._votes):,} votes for queue {self.queue_id}.')

        # DB Elo (global history — kept for reference only, not used as sim baseline)
        elo_df = pd.read_sql(
            'SELECT card_name, rating FROM elo_ratings WHERE card_name = ANY(%s)',
            conn, params=(self.queue_cards,)
        )
        conn.close()
        self.db_elo = dict(zip(elo_df['card_name'], elo_df['rating'].astype(float)))

        # Chronological replay of queue votes — the fair baseline to compare shuffles against.
        # Uses identical inputs/parameters as the simulations; only the order differs.
        self.live_elo = _replay(self._votes)
        print(f'Chronological baseline computed ({len(self.live_elo)} cards rated).')

    # ── Simulation ────────────────────────────────────────────────────────────

    def run(self, n: int = 100):
        """
        Run N shuffle simulations. Appends to any prior results.

        Prints elapsed time per simulation so you can judge feasibility.

        Parameters
        ----------
        n : int
            Number of shuffles to run (default 100).
        """
        votes_list  = list(self._votes)
        all_cards   = self.queue_cards

        # Initialise storage for cards not yet seen
        for card in all_cards:
            if card not in self.results:
                self.results[card] = []

        print(f'Running {n} simulations…')
        t_total = 0.0

        for i in range(1, n + 1):
            t0      = time.perf_counter()
            shuffled = votes_list.copy()
            random.shuffle(shuffled)
            sim_ratings = _replay(shuffled)
            elapsed = time.perf_counter() - t0
            t_total += elapsed

            for card in all_cards:
                self.results[card].append(sim_ratings.get(card, INITIAL_ELO))

            if i == 1 or i % max(1, n // 10) == 0 or i == n:
                print(f'  sim {i:>{len(str(n))}}/{n}  ({elapsed*1000:.0f} ms)')

        self.n_sims += n
        avg_ms = t_total / n * 1000
        print(f'\nDone. {n} simulations in {t_total:.1f}s  '
              f'(avg {avg_ms:.0f} ms/sim, {t_total/60:.1f} min total)')

    # ── Card mode ─────────────────────────────────────────────────────────────

    def card_distribution(self, card_name: str,
                          bins: int = 40) -> tuple[plt.Figure, pd.DataFrame]:
        """
        Plot the distribution of final Elo for one card across all simulations,
        with a vertical line at the card's live Elo.

        Parameters
        ----------
        card_name : str
            Exact card name (case-sensitive).
        bins : int
            Number of histogram bins (default 40).

        Returns
        -------
        fig : matplotlib.figure.Figure
        df  : pd.DataFrame
            Columns: sim_index, elo — one row per simulation.
        """
        if self.n_sims == 0:
            raise RuntimeError('No simulations run yet — call .run(n) first.')
        if card_name not in self.results:
            raise ValueError(f'{card_name!r} not found in queue {self.queue_id}.')

        elos     = np.array(self.results[card_name])
        live     = self.live_elo.get(card_name, INITIAL_ELO)
        mean_elo = elos.mean()
        std_elo  = elos.std()
        pct_rank = (elos <= live).mean() * 100  # percentile of live Elo within sim distribution

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(elos, bins=bins, color='steelblue', alpha=0.75, edgecolor='#1a1a1a', linewidth=0.4)
        ax.axvline(live, color='#c9a84c', linewidth=2.0, label=f'Chronological Elo: {live:.0f}')
        ax.axvline(mean_elo, color='#cc4444', linewidth=1.5, linestyle='--',
                   label=f'Sim mean: {mean_elo:.0f}')

        ax.set_title(f'{card_name} — Elo distribution across {self.n_sims} shuffles '
                     f'(Queue {self.queue_id})')
        ax.set_xlabel('Final Elo')
        ax.set_ylabel('Simulations')
        ax.legend()

        # Stats box
        stats = (f'Mean:  {mean_elo:.1f}\n'
                 f'Std:   {std_elo:.1f}\n'
                 f'Min:   {elos.min():.1f}\n'
                 f'Max:   {elos.max():.1f}\n'
                 f'Chron pct: {pct_rank:.1f}th')
        ax.text(0.02, 0.97, stats, transform=ax.transAxes,
                fontsize=8.5, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.7, edgecolor='#444'))

        fig.tight_layout()
        plt.show()

        df = pd.DataFrame({'sim_index': range(self.n_sims), 'elo': elos})
        return fig, df

    # ── Top-10% stability ─────────────────────────────────────────────────────

    def top_pct_stability(self,
                          boundary_margin: float = 0.20,
                          figsize: tuple = (12, 6)
                          ) -> tuple[plt.Figure, pd.DataFrame]:
        """
        For each card, compute the fraction of simulations in which it lands
        in the top 10% of its queue by Elo.

        Parameters
        ----------
        boundary_margin : float
            Cards shown in the chart are those with P(top 10%) in
            [boundary_margin, 1 - boundary_margin].  Default 0.20 means
            cards that are in the top 10% between 20% and 80% of the time.
        figsize : tuple
            Figure size for the bar chart.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Bar chart of boundary cards sorted by P(top 10%).
        df  : pd.DataFrame
            Full table of all queue cards, sorted by P(top 10%) descending.
            Columns: card_name, p_top10, times_top10, n_sims, live_elo,
                     live_rank, mean_sim_elo, std_sim_elo
        """
        if self.n_sims == 0:
            raise RuntimeError('No simulations run yet — call .run(n) first.')

        cutoff = self.top_pct_cutoff
        n      = self.n_sims
        cards  = self.queue_cards

        # Build array: (n_cards, n_sims) — each row is one card's Elo across sims
        card_order = cards
        matrix     = np.array([self.results[c] for c in card_order])  # shape (n_cards, n_sims)

        # For each simulation column, rank cards descending
        # rank 1 = best; top 10% = rank <= cutoff
        ranks_per_sim = np.argsort(-matrix, axis=0).argsort(axis=0) + 1  # (n_cards, n_sims)
        in_top        = (ranks_per_sim <= cutoff)                         # bool (n_cards, n_sims)
        p_top10       = in_top.mean(axis=1)                               # (n_cards,)
        times_top10   = in_top.sum(axis=1)                                # (n_cards,)

        mean_sim = matrix.mean(axis=1)
        std_sim  = matrix.std(axis=1)

        # Live rank within queue
        live_elos  = np.array([self.live_elo.get(c, INITIAL_ELO) for c in card_order])
        live_ranks = np.argsort(-live_elos).argsort() + 1

        df = pd.DataFrame({
            'card_name':    card_order,
            'p_top10':      p_top10,
            'times_top10':  times_top10,
            'n_sims':       n,
            'live_elo':     live_elos.round(1),
            'live_rank':    live_ranks,
            'mean_sim_elo': mean_sim.round(1),
            'std_sim_elo':  std_sim.round(1),
        }).sort_values('p_top10', ascending=False).reset_index(drop=True)

        # ── Chart: boundary cards ─────────────────────────────────────────────
        boundary = df[(df['p_top10'] >= boundary_margin) &
                      (df['p_top10'] <= 1 - boundary_margin)].copy()

        fig, ax = plt.subplots(figsize=figsize)

        if boundary.empty:
            ax.text(0.5, 0.5, 'No boundary cards\n(try a wider boundary_margin)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12, color='#888')
        else:
            colors = plt.cm.RdYlGn(boundary['p_top10'].values)
            bars   = ax.barh(boundary['card_name'], boundary['p_top10'],
                             color=colors, edgecolor='#1a1a1a', linewidth=0.3)
            ax.axvline(0.5, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_xlabel('P(top 10%)')
            ax.set_title(
                f'Queue {self.queue_id} — Top-10% boundary cards  '
                f'(cutoff = top {cutoff} of {len(cards)}, {self.n_sims} sims)\n'
                f'Showing {len(boundary)} cards with P in '
                f'[{boundary_margin:.0%}, {1-boundary_margin:.0%}]'
            )
            ax.invert_yaxis()

            # Annotate with live rank
            for bar, (_, row) in zip(bars, boundary.iterrows()):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'#{int(row["live_rank"])}',
                        va='center', fontsize=7, color='#aaa')

        fig.tight_layout()
        plt.show()

        return fig, df
