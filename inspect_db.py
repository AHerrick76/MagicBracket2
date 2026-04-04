'''
Load votes and elo_ratings from the database into pandas DataFrames.

Usage:
    DATABASE_URL=postgresql://... python inspect_db.py
'''

import json
import math
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_data import load_processed_cards

DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)

if not DATABASE_URL:
    raise RuntimeError('DATABASE_URL environment variable is not set.')

conn = psycopg2.connect(DATABASE_URL)

votes      = pd.read_sql('SELECT * FROM votes      ORDER BY id',          conn)
elo        = pd.read_sql('SELECT * FROM elo_ratings ORDER BY rating DESC', conn)

conn.close()

# Merge card metadata (is_ub, is_playtest, is_funny, set_type, rarity, etc.) onto elo
_cards = load_processed_cards()

UB_SETS     = {'spm', 'spe', 'tla', 'tle', 'fic', 'tmt', 'tmc', 'msh', 'msc'}
UB_SLD_DATE = '2025-06-13'

def _is_ub(row):
    if row.get('security_stamp') == 'triangle': return True
    if row['set'] in UB_SETS: return True
    if row['set'] == 'sld' and str(row['released_at'])[:10] >= UB_SLD_DATE: return True
    return False

_cards['is_ub']    = _cards.apply(_is_ub, axis=1)
_cards['is_funny'] = (_cards['set_type'] == 'funny') & ~_cards['is_playtest'].fillna(False)

def _category(row):
    if row['is_ub']:                   return 'Universes Beyond'
    if row['is_playtest']:             return 'Playtest'
    if row['is_funny']:                return 'Funny'
    if row['set_type'] == 'commander': return 'Commander'
    return 'Normal'

_cards['category'] = _cards.apply(_category, axis=1)

_meta_cols = ['name', 'set', 'set_name', 'set_type', 'rarity', 'released_at',
              'is_ub', 'is_playtest', 'is_funny', 'category']
elo = elo.merge(_cards[_meta_cols], left_on='card_name', right_on='name', how='left').drop(columns='name')

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
    Restrict an elo DataFrame to cards in one or more queues and compute vote distributions.

    When `votes` is provided, games_played and card_b appearances are computed from
    votes filtered to those queue_id(s) (accurate per-queue counts). Without `votes`, falls
    back to cumulative wins+losses from the elo table and card_b_dist is None.

    Parameters
    ----------
    elo : pd.DataFrame
        Full elo_ratings DataFrame (columns: card_name, rating, wins, losses, ...).
    queue_number : int or list[int]
        The queue id(s) to filter to. A single int or a list of ints.
    votes : pd.DataFrame or None
        Full votes DataFrame. If provided, per-queue vote counts are derived from it.
    queues_path : str
        Path to queues.json.

    Returns
    -------
    queue_elo : pd.DataFrame
        elo rows for cards in the queue(s), with an added `games_played` column.
    vote_distribution : dict[int, int]
        Maps games_played → number of cards with that many total appearances.
        Includes all queue cards, so 0-vote cards appear under key 0.
    card_b_dist : dict[int, int] or None
        Maps card_b_appearances → number of cards with that count.
        None when `votes` is not provided.
    label : str
        Human-readable label for the queue(s), e.g. "Queue 2" or "Queues 2, 3".
    '''
    queue_ids = [queue_number] if isinstance(queue_number, int) else list(queue_number)

    with open(queues_path) as f:
        queues_data = json.load(f)

    queue_cards = set()
    card_to_queue = {}  # card_name -> queue_id (first queue wins if card appears in multiple)
    for qid in queue_ids:
        queue = next((q for q in queues_data['queues'] if q['id'] == qid), None)
        if queue is None:
            raise ValueError(f'Queue {qid} not found in {queues_path}')
        for card in queue['cards']:
            queue_cards.add(card)
            card_to_queue.setdefault(card, qid)

    label = f'Queue{"s" if len(queue_ids) > 1 else ""} {", ".join(str(q) for q in queue_ids)}'
    queue_elo = elo[elo['card_name'].isin(queue_cards)].copy()
    queue_elo['queue_id'] = queue_elo['card_name'].map(card_to_queue)

    if votes is not None:
        qv = votes[votes['queue_id'].isin(queue_ids)]
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

    queue_elo['overall_rank']     = queue_elo['rating'].rank(ascending=False, method='min').astype(int)
    queue_elo['intra_queue_rank'] = queue_elo.groupby('queue_id')['rating'].rank(ascending=False, method='min').astype(int)

    vote_distribution = dict(sorted(Counter(games_played_per_card.values()).items()))
    return queue_elo, vote_distribution, card_b_dist, label


def plot_queue_distribution(vote_dist, card_b_dist, label, save=True):
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
    label : str
        Human-readable label used in the plot title and output filename.
    save : bool
        If True, saves the figure to a .png named after the label.
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

        ax.set_title(f'{label} — {title}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Number of cards')
        ax.legend()

    fig.tight_layout()

    if save:
        slug = label.replace(' ', '_').replace(',', '')
        out  = f'{slug}_distribution.png'
        fig.savefig(out, dpi=150)
        print(f'Saved to {out}')

    plt.show()


def longest_win_streak(votes, top_n=10):
    '''
    Find the longest consecutive win streak for each card across all votes.

    Parameters
    ----------
    votes : pd.DataFrame
        Full votes DataFrame (columns: id, card_a, card_b, chosen, ...).
    top_n : int
        Number of top cards to return (default 10).

    Returns
    -------
    pd.DataFrame
        Columns: card, max_win_streak, total_games, sorted by max_win_streak descending.
    '''
    # Build a flat list of (vote_id, card, won) for every appearance
    a = votes[['id', 'card_a', 'chosen']].copy()
    a.columns = ['vote_id', 'card', 'chosen']
    a['won'] = a['card'] == a['chosen']

    b = votes[['id', 'card_b', 'chosen']].copy()
    b.columns = ['vote_id', 'card', 'chosen']
    b['won'] = b['card'] == b['chosen']

    appearances = pd.concat([a, b], ignore_index=True).sort_values('vote_id')

    results = []
    for card, group in appearances.groupby('card'):
        max_streak = current = 0
        for won in group['won']:
            if won:
                current += 1
                if current > max_streak:
                    max_streak = current
            else:
                current = 0
        results.append({'card': card, 'max_win_streak': max_streak, 'total_games': len(group)})

    return (
        pd.DataFrame(results)
        .sort_values('max_win_streak', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def votes_per_card_per_queue(votes, queues_path='queues.json'):
    '''
    Return the average number of vote appearances per card for each queue.

    Parameters
    ----------
    votes : pd.DataFrame
        Full votes DataFrame.
    queues_path : str
        Path to queues.json.

    Returns
    -------
    pd.DataFrame
        Columns: queue_id, queue_size, total_votes, avg_votes_per_card.
        Sorted by queue_id ascending.
    '''
    with open(queues_path) as f:
        queues_data = json.load(f)

    rows = []
    for q in queues_data['queues']:
        qid   = q['id']
        qsize = len(q['cards'])
        qv    = votes[votes['queue_id'] == qid]
        total = len(qv) * 2  # each vote = 2 card appearances
        rows.append({
            'queue_id':          qid,
            'queue_size':        qsize,
            'total_votes':       len(qv),
            'avg_votes_per_card': round(total / qsize, 2) if qsize else 0,
        })

    return pd.DataFrame(rows).sort_values('queue_id').reset_index(drop=True)
