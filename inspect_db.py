'''
Load votes and elo_ratings from the database into pandas DataFrames.

Usage:
    DATABASE_URL=postgresql://... python inspect_db.py
'''

import os
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

if len(votes):
    print('=== Most recent votes ===')
    print(votes.tail(10).to_string(index=False))
    print()

if len(elo):
    print('=== Top 20 cards by Elo ===')
    print(elo.head(20)[['card_name', 'rating', 'wins', 'losses']].to_string(index=False))
