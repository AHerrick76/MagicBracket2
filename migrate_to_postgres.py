'''
One-time migration: copies votes.db (SQLite) → PostgreSQL.

Usage:
    DATABASE_URL=postgresql://user:pass@host:5432/dbname python migrate_to_postgres.py

The script:
  1. Creates tables in PostgreSQL (if they don't already exist)
  2. Copies all rows from `votes` and `elo_ratings`
  3. Advances the votes ID sequence to avoid future collisions
  4. Prints a summary of rows migrated / skipped

Safe to re-run: existing rows (matched by primary key) are silently skipped.
'''

import os
import sqlite3
import sys

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
load_dotenv()

SQLITE_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'votes.db')
DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)
DATABASE_URL = 'postgresql://postgres:fAjGtguifUugXovxeMZREGrfrWQYNLyB@crossover.proxy.rlwy.net:41626/railway'


def main():
    if not DATABASE_URL:
        print('ERROR: DATABASE_URL environment variable is not set.', file=sys.stderr)
        print('  Example: DATABASE_URL=postgresql://user:pass@host:5432/db python migrate_to_postgres.py')
        sys.exit(1)

    if not os.path.exists(SQLITE_PATH):
        print(f'ERROR: SQLite database not found at {SQLITE_PATH}', file=sys.stderr)
        sys.exit(1)

    print(f'Source:      {SQLITE_PATH}')
    print(f'Destination: {DATABASE_URL[:DATABASE_URL.index("@") + 1]}...')  # hide password

    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = sqlite3.Row

    pg_conn = psycopg2.connect(DATABASE_URL)
    pg_conn.autocommit = False
    cur = pg_conn.cursor()

    # ── Create tables ──────────────────────────────────────────────────────────
    print('\nCreating tables (if not exist)...')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS votes (
            id          SERIAL  PRIMARY KEY,
            timestamp   TEXT    NOT NULL,
            ip_address  TEXT,
            session_id  TEXT,
            card_a      TEXT    NOT NULL,
            card_b      TEXT    NOT NULL,
            chosen      TEXT    NOT NULL,
            config_name TEXT
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS elo_ratings (
            card_name    TEXT    PRIMARY KEY,
            rating       REAL    NOT NULL DEFAULT 1500.0,
            wins         INTEGER NOT NULL DEFAULT 0,
            losses       INTEGER NOT NULL DEFAULT 0,
            last_updated TEXT
        )
    ''')
    pg_conn.commit()

    # ── Migrate votes ──────────────────────────────────────────────────────────
    print('Reading votes from SQLite...')
    votes = sqlite_conn.execute('SELECT * FROM votes ORDER BY id').fetchall()
    print(f'  Found {len(votes)} rows.')

    if votes:
        rows = [
            (row['id'], row['timestamp'], row['ip_address'], row['session_id'],
             row['card_a'], row['card_b'], row['chosen'], row['config_name'])
            for row in votes
        ]
        execute_values(
            cur,
            '''INSERT INTO votes (id, timestamp, ip_address, session_id,
                                  card_a, card_b, chosen, config_name)
               VALUES %s ON CONFLICT (id) DO NOTHING''',
            rows,
        )
        inserted_votes = cur.rowcount
        pg_conn.commit()

        # Advance the SERIAL sequence so new inserts don't collide with migrated IDs
        max_id = max(row['id'] for row in votes)
        cur.execute(
            "SELECT setval(pg_get_serial_sequence('votes', 'id'), %s, true)", (max_id,)
        )
        pg_conn.commit()
        print(f'  Inserted: {inserted_votes}  Skipped: {len(votes) - inserted_votes}')
    else:
        print('  Nothing to migrate.')

    # ── Migrate elo_ratings ────────────────────────────────────────────────────
    print('Reading elo_ratings from SQLite...')
    ratings = sqlite_conn.execute('SELECT * FROM elo_ratings').fetchall()
    print(f'  Found {len(ratings)} rows.')

    if ratings:
        rows = [
            (row['card_name'], row['rating'], row['wins'], row['losses'], row['last_updated'])
            for row in ratings
        ]
        execute_values(
            cur,
            '''INSERT INTO elo_ratings (card_name, rating, wins, losses, last_updated)
               VALUES %s ON CONFLICT (card_name) DO UPDATE SET
                   rating       = EXCLUDED.rating,
                   wins         = EXCLUDED.wins,
                   losses       = EXCLUDED.losses,
                   last_updated = EXCLUDED.last_updated''',
            rows,
        )
        inserted_ratings = cur.rowcount
        pg_conn.commit()
        print(f'  Inserted: {inserted_ratings}  Skipped: {len(ratings) - inserted_ratings}')
    else:
        print('  Nothing to migrate.')

    sqlite_conn.close()
    cur.close()
    pg_conn.close()
    print('\nMigration complete.')


if __name__ == '__main__':
    main()
