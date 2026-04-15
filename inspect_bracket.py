'''
Load bracket_votes and bracket_results from the database into pandas DataFrames.

Usage:
    DATABASE_URL=postgresql://... python inspect_bracket.py
'''

import json
import os
import sys

import pandas as pd
import psycopg2
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)
if not DATABASE_URL:
    raise RuntimeError('DATABASE_URL environment variable is not set.')

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
BRACKET_JSON = os.path.join(BASE_DIR, 'bracket.json')

conn = psycopg2.connect(DATABASE_URL)

votes   = pd.read_sql('SELECT * FROM bracket_votes   ORDER BY id',          conn)
results = pd.read_sql('SELECT * FROM bracket_results ORDER BY matchup_id',  conn)

conn.close()

# ── Merge bracket.json context onto votes and results ──────────────────────────

_bracket      = None
_matchup_by_id = {}

if os.path.exists(BRACKET_JSON):
    with open(BRACKET_JSON, encoding='utf-8') as _f:
        _bracket = json.load(_f)
    _matchup_by_id = {m['id']: m for m in _bracket['matchups']}

ROUND_LABELS = {1: 'R64', 2: 'R32', 3: 'R16', 4: 'QF', 5: 'SF', 6: 'F'}

if _matchup_by_id and len(votes):
    votes['round_label'] = votes['round'].map(ROUND_LABELS)

if _matchup_by_id and len(results):
    results['round_label'] = results['round'].map(ROUND_LABELS)

# ── Summary ────────────────────────────────────────────────────────────────────

print(f'bracket_votes:   {len(votes)} rows')
print(f'bracket_results: {len(results)} rows')

if len(votes):
    unique_ballots = votes['ballot_id'].nunique()
    unique_ips     = votes['ip_address'].nunique()
    print(f'\nUnique ballots:  {unique_ballots}')
    print(f'Unique IPs:      {unique_ips}')

    by_day = votes.groupby('day').agg(
        ballots=('ballot_id', 'nunique'),
        individual_votes=('id', 'count'),
        unique_ips=('ip_address', 'nunique'),
    ).reset_index()
    print(f'\nVotes by day:\n{by_day.to_string(index=False)}')

# if len(results):
#     print(f'\nResults so far:\n'
#           + results[['matchup_id', 'round_label', 'day', 'bracket_date',
#                       'card_a', 'card_b', 'votes_a', 'votes_b', 'winner']]
#           .to_string(index=False))

# temp_results is built after the helper functions are defined (see below)


# ── Helper functions ───────────────────────────────────────────────────────────

def votes_per_matchup(votes=votes):
    '''
    Return vote counts per matchup, merged with card names.

    Returns
    -------
    pd.DataFrame
        Columns: matchup_id, round, day, card_a, card_b, votes_a, votes_b, total_votes.
        Sorted by matchup_id.
    '''
    if votes.empty:
        return pd.DataFrame()

    v = votes.copy()
    v['is_a'] = (v['chosen'] == v['card_a']).astype(int)
    v['is_b'] = (v['chosen'] == v['card_b']).astype(int)
    tally = (
        v.groupby(['matchup_id', 'round', 'day', 'card_a', 'card_b'])
        .agg(votes_a=('is_a', 'sum'), votes_b=('is_b', 'sum'))
        .reset_index()
    )
    tally['total_votes'] = tally['votes_a'] + tally['votes_b']
    return tally.sort_values('matchup_id').reset_index(drop=True)


def ballot_summary(votes=votes):
    '''
    One row per ballot: how many matchups were voted on and which IP/device submitted it.

    Returns
    -------
    pd.DataFrame
        Columns: ballot_id, ip_address, device, day, timestamp, matchups_voted.
    '''
    if votes.empty:
        return pd.DataFrame()

    return (
        votes.groupby('ballot_id').agg(
            ip_address=('ip_address', 'first'),
            device=('device', 'first'),
            day=('day', 'first'),
            timestamp=('timestamp', 'first'),
            matchups_voted=('matchup_id', 'count'),
        )
        .reset_index()
        .sort_values('timestamp')
        .reset_index(drop=True)
    )


def votes_per_ip(votes=votes):
    '''
    Return the number of ballots submitted per IP address, useful for spotting multi-voters.

    Returns
    -------
    pd.DataFrame
        Columns: ip_address, ballots, days_voted. Sorted by ballots descending.
    '''
    if votes.empty:
        return pd.DataFrame()

    return (
        votes.groupby('ip_address').agg(
            ballots=('ballot_id', 'nunique'),
            days_voted=('day', 'nunique'),
        )
        .reset_index()
        .sort_values('ballots', ascending=False)
        .reset_index(drop=True)
    )


def temp_results_df(votes=votes, results=results):
    '''
    Return a bracket_results-shaped DataFrame for matchups that are currently
    in progress (i.e. have votes but no finalised result yet), using the current
    vote tallies to determine a provisional winner.

    Returns
    -------
    pd.DataFrame
        Same columns as bracket_results: matchup_id, round, round_label, day,
        card_a, card_b, votes_a, votes_b, winner.  winner is None when tied.
        card_a and card_b include the seed in parentheses, e.g. "Ragavan (11)".
        Sorted by matchup_id.
    '''
    tally = votes_per_matchup(votes)
    if tally.empty:
        return pd.DataFrame()

    concluded_ids = set(results['matchup_id']) if not results.empty else set()
    pending = tally[~tally['matchup_id'].isin(concluded_ids)].copy()
    if pending.empty:
        return pd.DataFrame()

    # Build name -> seed lookup from bracket.json matchups
    _seed = {}
    for m in _matchup_by_id.values():
        if 'seed_a' in m:
            _seed[m['name_a']] = m['seed_a']
            _seed[m['name_b']] = m['seed_b']

    def _with_seed(name):
        s = _seed.get(name)
        return f'{name} ({s})' if s is not None else name

    # Attach raw seed numbers before renaming cards (needed for upset calc)
    pending['seed_a'] = pending['card_a'].map(lambda n: _seed.get(n))
    pending['seed_b'] = pending['card_b'].map(lambda n: _seed.get(n))

    pending['card_a'] = pending['card_a'].map(_with_seed)
    pending['card_b'] = pending['card_b'].map(_with_seed)

    def _winner(row):
        if row['votes_a'] > row['votes_b']:
            return row['card_a']
        if row['votes_b'] > row['votes_a']:
            return row['card_b']
        return None  # tied

    def _upset(row):
        if row['votes_a'] == row['votes_b']:
            return -1  # tied
        a_winning = row['votes_a'] > row['votes_b']
        # lower seed number = higher seed; upset if lower seed loses
        a_is_higher_seed = row['seed_a'] < row['seed_b']
        higher_seed_winning = a_winning == a_is_higher_seed
        return 0 if higher_seed_winning else 1

    pending['winner']      = pending.apply(_winner, axis=1)
    pending['upset']       = pending.apply(_upset, axis=1)
    pending['round_label'] = pending['round'].map(ROUND_LABELS)

    total = pending['votes_a'] + pending['votes_b']
    pending['winning_margin'] = (
        (pending[['votes_a', 'votes_b']].max(axis=1) / total * 100)
        .where(total > 0)
        .round(1)
    )

    cols = ['matchup_id', 'round', 'round_label', 'day',
            'card_a', 'card_b', 'votes_a', 'votes_b', 'winner', 'upset', 'winning_margin']
    return pending[cols].sort_values('matchup_id').reset_index(drop=True)


temp_results = temp_results_df()


def matchup_margin(results=results):
    '''
    Return each completed matchup with its vote margin and percentage.

    Returns
    -------
    pd.DataFrame
        Columns: matchup_id, round_label, card_a, card_b, votes_a, votes_b,
                 total, winner, margin, pct_winner. Sorted by pct_winner descending.
    '''
    if results.empty:
        return pd.DataFrame()

    df = results.copy()
    df['total']      = df['votes_a'] + df['votes_b']
    df['margin']     = (df['votes_a'] - df['votes_b']).abs()
    df['pct_winner'] = df.apply(
        lambda r: round(100 * max(r['votes_a'], r['votes_b']) / r['total'], 1)
        if r['total'] else None, axis=1
    )
    cols = ['matchup_id', 'round_label', 'card_a', 'card_b',
            'votes_a', 'votes_b', 'total', 'winner', 'margin', 'pct_winner']
    return df[cols].sort_values('pct_winner', ascending=False).reset_index(drop=True)


# if not temp_results.empty:
#     print(f'\nTemp results (current round if concluded now):\n'
#           + temp_results.to_string(index=False))
