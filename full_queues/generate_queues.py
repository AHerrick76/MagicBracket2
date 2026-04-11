"""
generate_queues.py — Generate the daily card queue schedule.

Shuffles all post-C16 cards and chunks them into batches of QUEUE_SIZE (500).
The last batch will be smaller than 500; all others are exactly 500.
Saves the result to queues.json.

Run this once before starting the bracket, or re-run to regenerate with a
different shuffle (pass --seed for a reproducible result).

Usage:
    python generate_queues.py
    python generate_queues.py --seed 42
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_data import load_processed_cards

QUEUE_SIZE  = 500
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queues.json')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None, help='RNG seed for reproducible shuffle')
args = parser.parse_args()

print('Loading card data...')
df       = load_processed_cards()
post_c16 = df[df['released_at'] > pd.Timestamp('2016-11-11')].reset_index(drop=True)
cards    = post_c16['name'].tolist()
print(f'{len(cards)} post-C16 cards loaded.')

rng = random.Random(args.seed)
rng.shuffle(cards)

chunks = [cards[i:i + QUEUE_SIZE] for i in range(0, len(cards), QUEUE_SIZE)]

output = {
    'generated_at': datetime.now().isoformat(),
    'seed':         args.seed,
    'queue_size':   QUEUE_SIZE,
    'total_queues': len(chunks),
    'total_cards':  len(cards),
    'queues': [{'id': i + 1, 'cards': chunk} for i, chunk in enumerate(chunks)],
}

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)

print(f'Generated {len(chunks)} queues.')
print(f'  Queues 1–{len(chunks)-1}: {QUEUE_SIZE} cards each')
print(f'  Queue {len(chunks)}: {len(chunks[-1])} cards')
print(f'Saved to {OUTPUT_PATH}')
