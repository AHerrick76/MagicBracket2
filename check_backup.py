'''
Sanity-check the db_backup/ CSV files.

Checks performed:
  1. Row counts for every table
  2. Null-fraction per column — flags any column with unexpected nulls
  3. Card-name consistency: elo card set vs vote card set for each phase

Usage:
    python check_backup.py [--backup-dir <path>]
'''

import argparse
import os
import sys

import pandas as pd

# Columns that should never contain nulls
REQUIRED_NON_NULL = {
    'votes':               ['id', 'timestamp', 'card_a', 'card_b', 'chosen'],
    'elo_ratings':         ['card_name', 'rating', 'wins', 'losses'],
    'votes_top10':         ['id', 'timestamp', 'card_a', 'card_b', 'chosen'],
    'elo_ratings_top10':   ['card_name', 'rating', 'wins', 'losses'],
    'bracket_votes':       ['id', 'timestamp', 'ip_address', 'round', 'day',
                            'matchup_id', 'ballot_id', 'card_a', 'card_b'],
    'bracket_results':     ['matchup_id', 'round', 'day', 'card_a', 'card_b',
                            'votes_a', 'votes_b', 'winner'],
    'bracket_state':       ['id', 'current_day'],
    'page_views':          ['id', 'timestamp', 'page'],
    'finals_favorite_cards': ['id', 'timestamp', 'ip_address', 'ballot_id'],
    'queue_transitions':     ['id', 'queue_id', 'activated_at'],
    'personal_picks':        ['id'],
}

ALL_TABLES = list(REQUIRED_NON_NULL.keys())  # order matters for reporting

NULL_WARN_THRESHOLD = 0.05  # flag any column with >5% nulls


def check_nulls(df, table):
    required = set(REQUIRED_NON_NULL.get(table, []))
    n = len(df)
    if n == 0:
        print('    (empty — skipping null check)')
        return

    issues = []
    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count == 0:
            continue
        frac = null_count / n
        tag = ''
        if col in required:
            tag = '  <-- REQUIRED COLUMN HAS NULLS'
        elif frac > NULL_WARN_THRESHOLD:
            tag = f'  <-- {frac:.0%} nulls'
        issues.append((col, null_count, frac, tag))

    if not issues:
        print('    nulls: none')
        return

    for col, count, frac, tag in issues:
        print(f'    {col}: {count:,} nulls ({frac:.1%}){tag}')


def check_card_consistency(votes_df, elo_df, votes_table, elo_table):
    vote_cards = set(votes_df['card_a'].dropna()) | set(votes_df['card_b'].dropna())
    elo_cards  = set(elo_df['card_name'].dropna())

    in_votes_not_elo = vote_cards - elo_cards
    in_elo_not_votes = elo_cards - vote_cards

    print(f'    cards in {votes_table} (card_a|card_b): {len(vote_cards):,}')
    print(f'    cards in {elo_table} (card_name):       {len(elo_cards):,}')

    if in_votes_not_elo:
        sample = sorted(in_votes_not_elo)[:5]
        print(f'    IN VOTES but NOT in elo ({len(in_votes_not_elo)}): {sample}{"..." if len(in_votes_not_elo) > 5 else ""}')
    else:
        print(f'    all vote cards present in elo table')

    if in_elo_not_votes:
        sample = sorted(in_elo_not_votes)[:5]
        print(f'    IN ELO but NOT in votes ({len(in_elo_not_votes)}): {sample}{"..." if len(in_elo_not_votes) > 5 else ""}')
    else:
        print(f'    all elo cards appeared in votes')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backup-dir', default='db_backup')
    args = parser.parse_args()

    if not os.path.isdir(args.backup_dir):
        print(f'ERROR: backup directory not found: {args.backup_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'Backup directory: {os.path.abspath(args.backup_dir)}\n')
    print('=' * 60)

    tables = {}
    for table in ALL_TABLES:
        path = os.path.join(args.backup_dir, f'{table}.csv')
        print(f'{table}')
        if not os.path.exists(path):
            print('    FILE MISSING\n')
            tables[table] = None
            continue
        df = pd.read_csv(path)
        tables[table] = df
        print(f'    rows: {len(df):,}')
        check_nulls(df, table)
        print()

    print('=' * 60)
    print('Card-name consistency checks\n')

    pairs = [
        ('votes',      'elo_ratings',       'Full phase'),
        ('votes_top10','elo_ratings_top10',  'Top-10% phase'),
    ]
    for votes_t, elo_t, label in pairs:
        print(f'{label}: {votes_t} vs {elo_t}')
        vdf = tables.get(votes_t)
        edf = tables.get(elo_t)
        if vdf is None or edf is None:
            print('    skipped (missing table)')
        else:
            check_card_consistency(vdf, edf, votes_t, elo_t)
        print()


if __name__ == '__main__':
    main()
