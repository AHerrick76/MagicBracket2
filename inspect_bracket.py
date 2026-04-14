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

if len(results):
    print(f'\nResults so far:\n'
          + results[['matchup_id', 'round_label', 'day', 'bracket_date',
                      'card_a', 'card_b', 'votes_a', 'votes_b', 'winner']]
          .to_string(index=False))


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

    tally = (
        votes.groupby(['matchup_id', 'round', 'day', 'card_a', 'card_b'])
        .apply(lambda g: pd.Series({
            'votes_a': (g['chosen'] == g['card_a']).sum(),
            'votes_b': (g['chosen'] == g['card_b']).sum(),
        }), include_groups=False)
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
