"""
generate_bracket.py — Build the top-64 bracket in two phases.

Phase 1 (default): Queries elo_ratings_top10 for current live Elos, picks the
top 64 cards, enriches with image/display metadata, and writes top_64_candidates.json
for review before committing to a bracket.

Phase 2 (--build-bracket): Reads top_64_candidates.json, assigns seeds 1–64 by
rank, builds the 63-matchup single-elimination bracket, and writes bracket.json.

Usage:
    # Phase 1 — generate candidates
    python generate_bracket.py
    python generate_bracket.py --dry-run
    python generate_bracket.py --top-n 32

    # Phase 2 — build bracket
    python generate_bracket.py --build-bracket
    python generate_bracket.py --build-bracket --dry-run
    python generate_bracket.py --build-bracket --candidates-in my_candidates.json --bracket-out my_bracket.json
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_data import load_processed_cards

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--build-bracket', action='store_true',
                    help='Phase 2: read candidates JSON and write bracket.json')
parser.add_argument('--dry-run', action='store_true',
                    help='Print plan without writing output file')
parser.add_argument('--top-n', type=int, default=64,
                    help='(Phase 1) Number of cards to select (default: 64)')
parser.add_argument('--out', default=None,
                    help='(Phase 1) Output path (default: top_64_candidates.json)')
parser.add_argument('--candidates-in', default=None,
                    help='(Phase 2) Input candidates JSON (default: top_64_candidates.json)')
parser.add_argument('--bracket-out', default=None,
                    help='(Phase 2) Output bracket JSON (default: bracket.json)')
args = parser.parse_args()

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CANDIDATES_PATH = args.candidates_in or os.path.join(BASE_DIR, 'top_64_candidates.json')
BRACKET_OUT     = args.bracket_out  or os.path.join(BASE_DIR, 'bracket.json')
CANDIDATES_OUT  = args.out          or os.path.join(BASE_DIR, 'top_64_candidates.json')

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Build bracket structure from candidates JSON
# ══════════════════════════════════════════════════════════════════════════════

if args.build_bracket:

    # ── Load candidates ────────────────────────────────────────────────────────

    if not os.path.exists(CANDIDATES_PATH):
        sys.exit(f'{CANDIDATES_PATH} not found. Run Phase 1 first (without --build-bracket).')

    with open(CANDIDATES_PATH, encoding='utf-8') as f:
        candidates_data = json.load(f)

    cards = candidates_data['cards']
    n = len(cards)

    if n == 0 or (n & (n - 1)) != 0:
        sys.exit(f'Number of candidates must be a power of 2 (got {n}). '
                 'Re-run Phase 1 with --top-n set to a power of 2 (e.g. 64).')

    num_rounds = n.bit_length() - 1   # log2(n)
    print(f'Building {n}-card bracket ({num_rounds} rounds) from {CANDIDATES_PATH}')

    # ── Assign seeds ───────────────────────────────────────────────────────────
    # Cards are already rank-ordered (rank 1 = highest Elo). Seed == rank.

    for c in cards:
        c['seed'] = c['rank']

    seed_to_card = {c['seed']: c for c in cards}

    # ── Bracket slot order ─────────────────────────────────────────────────────

    def bracket_seed_order(size):
        """
        Return seed numbers in bracket slot order for a field of `size`.
        Adjacent pairs in the result are Round-1 matchups.
        Guarantees #1 and #2 can only meet in the final, etc.
        """
        if size == 1:
            return [1]
        prev = bracket_seed_order(size // 2)
        result = []
        for s in prev:
            result.append(s)
            result.append(size + 1 - s)
        return result

    slots = bracket_seed_order(n)   # n seed numbers in bracket order

    # ── Build matchups ─────────────────────────────────────────────────────────

    # Round 1: n//2 matchups, IDs 1 … n//2
    # Round 2: n//4 matchups, IDs n//2+1 … n//2+n//4
    # ...
    # Each round's matchup count = n // 2^round
    # ID ranges are computed cumulatively.

    # Day schedule:
    #   Day 1: first half of R1 matchups  (slots 0 .. n//4-1)
    #   Day 2: second half of R1 matchups (slots n//4 .. n//2-1)
    #   Day 3: Round 2
    #   Day d: Round d-1  (for d >= 3)

    matchups = []
    matchup_id = 1

    # Track the start ID of each round so we can build feeder references
    round_start_id = {}   # round_num → first matchup ID in that round

    # Round 1
    r1_count = n // 2
    round_start_id[1] = 1
    half = r1_count // 2
    for i in range(r1_count):
        seed_a = slots[i * 2]
        seed_b = slots[i * 2 + 1]
        card_a = seed_to_card[seed_a]
        card_b = seed_to_card[seed_b]
        matchups.append({
            'id':     matchup_id,
            'round':  1,
            'day':    1 if i < half else 2,
            'seed_a': seed_a,
            'seed_b': seed_b,
            'name_a': card_a['name'],
            'name_b': card_b['name'],
        })
        matchup_id += 1

    # Rounds 2 through num_rounds
    for rnd in range(2, num_rounds + 1):
        round_start_id[rnd] = matchup_id
        n_matchups = n // (2 ** rnd)
        prev_start = round_start_id[rnd - 1]
        day = rnd + 1   # R2→day3, R3→day4, …, R6→day7
        for i in range(n_matchups):
            matchups.append({
                'id':       matchup_id,
                'round':    rnd,
                'day':      day,
                'feeder_a': prev_start + i * 2,
                'feeder_b': prev_start + i * 2 + 1,
            })
            matchup_id += 1

    assert len(matchups) == n - 1, f'Expected {n-1} matchups, got {len(matchups)}'

    # ── Cards list for bracket.json ────────────────────────────────────────────
    # Keep only the fields app_bracket.py reads from _cards_by_name.

    cards_out = [
        {
            'name':      c['name'],
            'seed':      c['seed'],
            'img_front': c.get('img_front'),
            'img_back':  c.get('img_back'),
            'set_name':  c.get('set_name', ''),
            'year':      c.get('year'),
        }
        for c in cards
    ]

    # ── Print summary ──────────────────────────────────────────────────────────

    print(f'\nMatchup schedule:')
    for rnd in range(1, num_rounds + 1):
        rnd_matchups = [m for m in matchups if m['round'] == rnd]
        days = sorted({m['day'] for m in rnd_matchups})
        day_str = f'day{"s" if len(days) > 1 else ""} {", ".join(str(d) for d in days)}'
        label = {1: 'Round of 64', 2: 'Round of 32', 3: 'Round of 16',
                 4: 'Quarterfinals', 5: 'Semifinals', 6: 'Final'}.get(rnd, f'Round {rnd}')
        print(f'  Round {rnd} ({label:15s}): {len(rnd_matchups):2d} matchup(s), {day_str}')

    print(f'\nTotal matchups: {len(matchups)}')
    print(f'\nRound 1 first four matchups:')
    for m in matchups[:4]:
        print(f'  #{m["seed_a"]:2d} {m["name_a"]}  vs  #{m["seed_b"]:2d} {m["name_b"]}')

    display_shuffle_seed = random.randint(0, 2**31 - 1)

    bracket = {
        'generated_at':        datetime.now(timezone.utc).isoformat(),
        'num_cards':           n,
        'num_rounds':          num_rounds,
        'display_shuffle_seed': display_shuffle_seed,
        'cards':               cards_out,
        'matchups':            matchups,
    }

    if args.dry_run:
        print(f'\n--dry-run: {BRACKET_OUT} NOT written.')
    else:
        with open(BRACKET_OUT, 'w', encoding='utf-8') as f:
            json.dump(bracket, f, indent=2, ensure_ascii=False)
        print(f'\nWrote {BRACKET_OUT}')

    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Generate top-N candidates from live Elo ratings
# ══════════════════════════════════════════════════════════════════════════════

NUM_CARDS = args.top_n
TOP10_JSON_PATH = os.path.join(BASE_DIR, 'top_10_queue.json')

DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)
if not DATABASE_URL:
    sys.exit('DATABASE_URL environment variable is not set.')

DOUBLE_FACED_LAYOUTS               = {'transform', 'modal_dfc'}
SIDEWAYS_TYPES                     = {'Battle', 'Room'}
SIDEWAYS_LAYOUTS                   = {'split'}
SIDEWAYS_LAYOUT_KEYWORD_EXCLUSIONS = {'Aftermath'}

IMAGE_OVERRIDES = {
    'Sothera, the Supervoid': 'https://cards.scryfall.io/large/front/e/9/e99d6fc0-dcf2-4b25-81c2-02c230a36246.jpg',
    'The Wandering Emperor':  'https://cards.scryfall.io/large/front/f/a/fab2d8a9-ab4c-4225-a570-22636293c17d.jpg',
}

# ── Load queue_id lookup from top_10_queue.json ────────────────────────────────

print(f'Loading {TOP10_JSON_PATH}...')
with open(TOP10_JSON_PATH, encoding='utf-8') as f:
    top10_data = json.load(f)
name_to_queue = {e['name']: e.get('queue_id') for e in top10_data['cards']}
print(f'{len(name_to_queue)} cards in top-10% pool.')

# ── Query live Elos from DB ────────────────────────────────────────────────────

print('Querying elo_ratings_top10...')
conn = psycopg2.connect(DATABASE_URL)
try:
    cur = conn.cursor()
    cur.execute('''
        SELECT card_name, rating, wins, losses
        FROM elo_ratings_top10
        ORDER BY rating DESC
    ''')
    rows = cur.fetchall()
finally:
    conn.close()

print(f'{len(rows)} cards found in elo_ratings_top10.')

all_cards = [
    {'name': name, 'elo': rating, 'wins': wins, 'losses': losses,
     'games': wins + losses, 'queue_id': name_to_queue.get(name)}
    for name, rating, wins, losses in rows
]

top_cards = all_cards[:NUM_CARDS]

print(f'\nTop {NUM_CARDS} cards selected.')
print(f'  Elo range: {top_cards[-1]["elo"]:.1f} – {top_cards[0]["elo"]:.1f}')
print(f'  #{NUM_CARDS}: "{top_cards[-1]["name"]}"  (Elo {top_cards[-1]["elo"]:.1f}, {top_cards[-1]["games"]} games)')
if len(all_cards) > NUM_CARDS:
    first_out = all_cards[NUM_CARDS]
    print(f'  First out: "{first_out["name"]}"  (Elo {first_out["elo"]:.1f}, {first_out["games"]} games)')

# ── Load card metadata ─────────────────────────────────────────────────────────

print('\nLoading card metadata...')
df = load_processed_cards()
top_names = {e['name'] for e in top_cards}
meta = df[df['name'].isin(top_names)].drop_duplicates('name').set_index('name')

def _meta(name, col, default=''):
    try:
        v = meta.at[name, col]
        return default if (isinstance(v, float) and pd.isna(v)) else v
    except KeyError:
        return default

def _is_sideways(name):
    layout    = str(_meta(name, 'layout'))
    type_line = str(_meta(name, 'type_line'))
    keywords  = _meta(name, 'keywords', None)
    kw_list   = list(keywords) if keywords is not None and not isinstance(keywords, float) else []
    vintage   = _meta(name, 'legal_vintage', 'not_legal') in {'legal', 'restricted'}
    if any(t in type_line for t in SIDEWAYS_TYPES):
        return True
    if layout in SIDEWAYS_LAYOUTS and vintage and not any(k in kw_list for k in SIDEWAYS_LAYOUT_KEYWORD_EXCLUSIONS):
        return True
    return False

def _img(name, col):
    v = _meta(name, col, None)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v)

# ── Build card records ─────────────────────────────────────────────────────────

cards_out = []
for rank, entry in enumerate(top_cards, start=1):
    name = entry['name']
    img_front = IMAGE_OVERRIDES.get(name) or _img(name, 'img_front')
    img_back  = _img(name, 'img_back')
    set_name  = str(_meta(name, 'set_name', ''))
    released  = _meta(name, 'released_at', None)
    try:
        year = int(pd.to_datetime(released).year) if released else None
    except Exception:
        year = None

    cards_out.append({
        'rank':        rank,
        'name':        name,
        'elo':         round(entry['elo'], 2),
        'wins':        entry['wins'],
        'losses':      entry['losses'],
        'games':       entry['games'],
        'queue_id':    entry['queue_id'],
        'img_front':   img_front,
        'img_back':    img_back,
        'is_sideways': _is_sideways(name),
        'set_name':    set_name,
        'year':        year,
    })

# ── Print summary ──────────────────────────────────────────────────────────────

print(f'\nTop {NUM_CARDS} candidates:')
for c in cards_out:
    flags = ''
    if c['img_back']:    flags += '  [DFC]'
    if c['is_sideways']: flags += '  [sideways]'
    print(f'  #{c["rank"]:2d}  {c["elo"]:7.1f}  ({c["games"]:4d}g)  {c["name"]}{flags}')

# ── Write output ───────────────────────────────────────────────────────────────

out = {
    'generated_at': datetime.now(timezone.utc).isoformat(),
    'num_cards':    NUM_CARDS,
    'cards':        cards_out,
}

if args.dry_run:
    print(f'\n--dry-run: {CANDIDATES_OUT} NOT written.')
else:
    with open(CANDIDATES_OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\nWrote {CANDIDATES_OUT}')
