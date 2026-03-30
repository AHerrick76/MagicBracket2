"""
advance_queue.py — Advance the bracket to the next daily card queue.

Writes the new queue_id to the queue_transitions table in the database.
After running this, trigger a reload via:
    POST /api/reload_queue   (no restart needed)
or restart the app (Railway redeploy).

Usage:
    python advance_queue.py               # advance to next queue
    python advance_queue.py --id 3        # activate a specific queue by ID
    python advance_queue.py --status      # show current queue without changing it
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import psycopg2
from dotenv import load_dotenv

load_dotenv()

QUEUES_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queues.json')
DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)

if not DATABASE_URL:
    print('ERROR: DATABASE_URL not set.', file=sys.stderr)
    sys.exit(1)

if not os.path.exists(QUEUES_PATH):
    print('ERROR: queues.json not found — run generate_queues.py first.', file=sys.stderr)
    sys.exit(1)

with open(QUEUES_PATH, encoding='utf-8') as f:
    queue_data = json.load(f)

total_queues = queue_data['total_queues']
queue_index  = {q['id']: q for q in queue_data['queues']}

parser = argparse.ArgumentParser()
parser.add_argument('--id',     type=int, default=None, help='Activate a specific queue ID')
parser.add_argument('--status', action='store_true',    help='Show current queue, no change')
args = parser.parse_args()

conn = psycopg2.connect(DATABASE_URL)
conn.autocommit = False
cur  = conn.cursor()

# Ensure table exists
cur.execute('''
    CREATE TABLE IF NOT EXISTS queue_transitions (
        id           SERIAL  PRIMARY KEY,
        queue_id     INTEGER NOT NULL,
        activated_at TEXT    NOT NULL
    )
''')
conn.commit()

# Current active queue
cur.execute('SELECT queue_id, activated_at FROM queue_transitions ORDER BY id DESC LIMIT 1')
row        = cur.fetchone()
current_id = row[0] if row else 0
activated  = row[1] if row else '(never)'

print(f'Current queue: {current_id} / {total_queues}  (activated: {activated})')
if current_id > 0 and current_id in queue_index:
    print(f'  Cards in current queue: {len(queue_index[current_id]["cards"])}')

if args.status:
    cur.close()
    conn.close()
    sys.exit(0)

# Determine target
next_id = args.id if args.id is not None else current_id + 1

if next_id < 1 or next_id > total_queues:
    print(f'ERROR: queue {next_id} out of range (1–{total_queues}).', file=sys.stderr)
    cur.close()
    conn.close()
    sys.exit(1)

# Activate
ts = datetime.now(timezone.utc).isoformat()
cur.execute(
    'INSERT INTO queue_transitions (queue_id, activated_at) VALUES (%s, %s)',
    (next_id, ts),
)
conn.commit()

q = queue_index[next_id]
print(f'\nActivated queue {next_id}: {len(q["cards"])} cards.')
print(f'First 5: {q["cards"][:5]}')
print(f'\nNow reload the app:  POST /api/reload_queue  (or restart the server).')

cur.close()
conn.close()
