'''
Load votes.db into pandas DataFrames for interactive exploration.

Run with:
    python -i load_votes.py

Available after loading:
    votes       — all vote records
    elo         — current Elo ratings (sorted descending)
'''

import sqlite3
import pandas as pd

DB_PATH = 'votes.db'

with sqlite3.connect(DB_PATH) as conn:
    votes = pd.read_sql('SELECT * FROM votes', conn, parse_dates=['timestamp'])
    elo   = pd.read_sql('SELECT * FROM elo_ratings ORDER BY rating DESC', conn,
                        parse_dates=['last_updated'])

print(f'votes : {len(votes):,} rows')
print(f'elo   : {len(elo):,} rows')
print()
print(votes.head())
