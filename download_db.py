'''
Download all PostgreSQL tables to local CSV files.

Usage:
    python download_db.py [--output-dir <path>]

Connects via DATABASE_URL from .env and dumps every table in the public
schema to a CSV in the output directory (default: db_backup/).
'''

import argparse
import csv
import os
import sys
from datetime import datetime

import psycopg2
from dotenv import load_dotenv

load_dotenv()

TABLES = [
    # Full voting phase
    'votes',
    'elo_ratings',
    # Top-10% voting phase
    'votes_top10',
    'elo_ratings_top10',
    'page_views',
    # Top-64 bracket phase
    'bracket_votes',
    'bracket_results',
    'bracket_state',
    'finals_favorite_cards',
    # Operational / misc
    'queue_transitions',
    'personal_picks',
]


def dump_table(cur, table, out_dir):
    try:
        cur.execute(f'SELECT * FROM {table}')
    except psycopg2.errors.UndefinedTable:
        print(f'  {table}: table not found — skipped')
        return 0

    rows = cur.fetchall()
    col_names = [desc[0] for desc in cur.description]
    out_path = os.path.join(out_dir, f'{table}.csv')

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(col_names)
        writer.writerows(rows)

    print(f'  {table}: {len(rows):,} rows -> {out_path}')
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description='Download PostgreSQL tables to CSV.')
    parser.add_argument('--output-dir', default='db_backup',
                        help='Directory to write CSV files (default: db_backup/)')
    args = parser.parse_args()

    database_url = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)
    if not database_url:
        print('ERROR: DATABASE_URL is not set.', file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Connecting to database...')
    conn = psycopg2.connect(database_url)
    conn.autocommit = True
    cur = conn.cursor()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Download started: {timestamp}')
    print(f'Output directory: {os.path.abspath(args.output_dir)}\n')

    total_rows = 0
    for table in TABLES:
        total_rows += dump_table(cur, table, args.output_dir)

    cur.close()
    conn.close()

    print(f'\nDone. {total_rows:,} total rows across {len(TABLES)} tables.')


if __name__ == '__main__':
    main()
