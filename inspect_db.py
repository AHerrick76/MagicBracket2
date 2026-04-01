'''
Load votes and elo_ratings from the database into pandas DataFrames.

Usage:
    DATABASE_URL=postgresql://... python inspect_db.py
'''

import json
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)

if not DATABASE_URL:
    raise RuntimeError('DATABASE_URL environment variable is not set.')

conn = psycopg2.connect(DATABASE_URL)

votes      = pd.read_sql('SELECT * FROM votes      ORDER BY id',          conn)
elo        = pd.read_sql('SELECT * FROM elo_ratings ORDER BY rating DESC', conn)

conn.close()

print(f'Votes:       {len(votes)} rows')
print(f'Elo ratings: {len(elo)} rows')
print()

# if len(votes):
#     print('=== Most recent votes ===')
#     print(votes.tail(10).to_string(index=False))
#     print()

# if len(elo):
#     print('=== Top 20 cards by Elo ===')
#     print(elo.head(20)[['card_name', 'rating', 'wins', 'losses']].to_string(index=False))


def _poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def distribution_in_queue(elo, queue_number, votes=None, queues_path='queues.json'):
    '''
    Restrict an elo DataFrame to cards in a given queue and compute vote distributions.

    When `votes` is provided, games_played and card_b appearances are computed from
    votes filtered to that queue_id (accurate per-queue counts). Without `votes`, falls
    back to cumulative wins+losses from the elo table and card_b_dist is None.

    Parameters
    ----------
    elo : pd.DataFrame
        Full elo_ratings DataFrame (columns: card_name, rating, wins, losses, ...).
    queue_number : int
        The queue id (1-indexed) to filter to.
    votes : pd.DataFrame or None
        Full votes DataFrame. If provided, per-queue vote counts are derived from it.
    queues_path : str
        Path to queues.json.

    Returns
    -------
    queue_elo : pd.DataFrame
        elo rows for cards in the queue, with an added `games_played` column.
    vote_distribution : dict[int, int]
        Maps games_played → number of cards with that many total appearances.
        Includes all queue cards, so 0-vote cards appear under key 0.
    card_b_dist : dict[int, int] or None
        Maps card_b_appearances → number of cards with that count.
        None when `votes` is not provided.
    '''
    with open(queues_path) as f:
        queues_data = json.load(f)

    queue = next((q for q in queues_data['queues'] if q['id'] == queue_number), None)
    if queue is None:
        raise ValueError(f'Queue {queue_number} not found in {queues_path}')

    queue_cards = set(queue['cards'])
    queue_elo = elo[elo['card_name'].isin(queue_cards)].copy()

    if votes is not None:
        qv = votes[votes['queue_id'] == queue_number]
        a_counts = Counter(qv['card_a'])
        b_counts = Counter(qv['card_b'])

        games_played_per_card = {card: a_counts.get(card, 0) + b_counts.get(card, 0)
                                  for card in queue_cards}
        card_b_per_card = {card: b_counts.get(card, 0) for card in queue_cards}

        queue_elo['games_played'] = queue_elo['card_name'].map(games_played_per_card).fillna(0).astype(int)
        card_b_dist = dict(sorted(Counter(card_b_per_card.values()).items()))
    else:
        elo_map = dict(zip(queue_elo['card_name'], queue_elo['wins'] + queue_elo['losses']))
        games_played_per_card = {card: elo_map.get(card, 0) for card in queue_cards}
        queue_elo['games_played'] = queue_elo['card_name'].map(games_played_per_card).fillna(0).astype(int)
        card_b_dist = None

    vote_distribution = dict(sorted(Counter(games_played_per_card.values()).items()))
    return queue_elo, vote_distribution, card_b_dist


def plot_queue_distribution(vote_dist, card_b_dist, queue_number, save=True):
    '''
    Plot actual vs expected (Poisson) distributions for all-vote appearances
    and card_b appearances in a queue.

    Parameters
    ----------
    vote_dist : dict[int, int]
        From distribution_in_queue: games_played → card count.
    card_b_dist : dict[int, int] or None
        From distribution_in_queue: card_b_appearances → card count.
        If None, only the all-votes panel is drawn.
    queue_number : int
        Used in the plot title and output filename.
    save : bool
        If True, saves the figure to queue_{queue_number}_distribution.png.
    '''
    has_b = card_b_dist is not None
    fig, axes = plt.subplots(1, 2 if has_b else 1,
                             figsize=(14 if has_b else 7, 5))
    if not has_b:
        axes = [axes]

    panels = [(axes[0], vote_dist, 'All Votes', 'Appearances per card (card A + card B)')]
    if has_b:
        panels.append((axes[1], card_b_dist, 'Card B Appearances', 'Card B appearances per card'))

    for ax, dist, title, xlabel in panels:
        N = sum(dist.values())
        lam = sum(k * v for k, v in dist.items()) / N if N else 0

        max_k = max(dist.keys()) if dist else 0
        ks = list(range(0, max_k + 1))
        actual_y   = [dist.get(k, 0) for k in ks]
        expected_y = [_poisson_pmf(k, lam) * N for k in ks]

        ax.bar(ks, actual_y, color='steelblue', alpha=0.65, label='Actual')
        ax.plot(ks, expected_y, 'r--o', markersize=4, linewidth=1.5,
                label=f'Expected (Poisson, λ={lam:.1f})')

        ax.set_title(f'Queue {queue_number} — {title}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Number of cards')
        ax.legend()

    fig.tight_layout()

    if save:
        out = f'queue_{queue_number}_distribution.png'
        fig.savefig(out, dpi=150)
        print(f'Saved to {out}')

    plt.show()
