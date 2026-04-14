"""
advance_bracket.py — Close the current voting round and advance winners.

For each matchup scheduled on the current day:
  1. Tallies votes from bracket_votes
  2. Determines the winner (most votes; ties go to the higher seed / lower seed number)
  3. Writes results to bracket_results
  4. Advances bracket_state.current_day by 1

Run this script once per round after voting closes.

Usage:
    python advance_bracket.py              # close current day
    python advance_bracket.py --dry-run    # preview without writing
    python advance_bracket.py --day 2      # close a specific day (leaves bracket_state unchanged)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import psycopg2
from dotenv import load_dotenv

load_dotenv()

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true',
                    help='Preview results without writing to DB')
parser.add_argument('--day', type=int, default=None,
                    help='Override which day to close (default: current bracket_state day)')
args = parser.parse_args()

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
BRACKET_JSON = os.path.join(BASE_DIR, 'bracket.json')
DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)

if not DATABASE_URL:
    sys.exit('DATABASE_URL environment variable is not set.')
if not os.path.exists(BRACKET_JSON):
    sys.exit(f'{BRACKET_JSON} not found. Run generate_bracket.py to build the bracket first.')

with open(BRACKET_JSON, encoding='utf-8') as f:
    bracket = json.load(f)

matchup_by_id = {m['id']: m for m in bracket['matchups']}

# ── DB ─────────────────────────────────────────────────────────────────────────

conn = psycopg2.connect(DATABASE_URL)
conn.autocommit = False
cur = conn.cursor()

try:
    # Determine which day to close
    cur.execute('SELECT current_day FROM bracket_state WHERE id = 1')
    row = cur.fetchone()
    if row is None:
        sys.exit('bracket_state has no row. Start app_bracket.py at least once to initialise the DB.')

    state_day = row[0]
    close_day = args.day if args.day is not None else state_day

    print(f'Closing day {close_day} (bracket_state is currently day {state_day}).')

    # Matchups scheduled for this day
    day_matchups = [m for m in bracket['matchups'] if m['day'] == close_day]
    if not day_matchups:
        sys.exit(f'No matchups found for day {close_day} in bracket.json.')
    print(f'{len(day_matchups)} matchup(s) to close.')

    # Load existing bracket_results to resolve feeder winners for R2+ matchups
    cur.execute('SELECT matchup_id, winner, winner_seed FROM bracket_results')
    results_by_id = {mid: {'winner': w, 'winner_seed': ws} for mid, w, ws in cur.fetchall()}

    def resolve_card(matchup, side):
        """Return (name, seed) for 'a' or 'b' side; looks up feeder results for R2+ matchups."""
        if matchup['round'] == 1:
            return matchup[f'name_{side}'], matchup[f'seed_{side}']
        feeder_id = matchup[f'feeder_{side}']
        res = results_by_id.get(feeder_id)
        if res:
            return res['winner'], res['winner_seed']
        return None, None

    # Tally votes per matchup
    cur.execute(
        'SELECT matchup_id, chosen, COUNT(*) FROM bracket_votes '
        'WHERE day = %s GROUP BY matchup_id, chosen',
        (close_day,)
    )
    votes_map = {}   # {matchup_id: {card_name: count}}
    for mid, chosen, cnt in cur.fetchall():
        votes_map.setdefault(mid, {})[chosen] = cnt

    # ── Process each matchup ───────────────────────────────────────────────────

    now          = datetime.now(timezone.utc).isoformat()
    bracket_date = datetime.now(timezone.utc).date().isoformat()
    results_to_write = []

    for m in day_matchups:
        name_a, seed_a = resolve_card(m, 'a')
        name_b, seed_b = resolve_card(m, 'b')

        if name_a is None or name_b is None:
            print(f'  [SKIP]  Matchup {m["id"]:2d}: cards not resolved '
                  f'(feeder matchups not yet closed). Skipping.')
            continue

        mv      = votes_map.get(m['id'], {})
        votes_a = mv.get(name_a, 0)
        votes_b = mv.get(name_b, 0)
        total   = votes_a + votes_b

        # Winner: most votes; tie → lower seed number (better seeded card)
        if votes_a > votes_b:
            winner, winner_seed = name_a, seed_a
            tie = False
        elif votes_b > votes_a:
            winner, winner_seed = name_b, seed_b
            tie = False
        else:
            tie = True
            if seed_a is not None and seed_b is not None and seed_a < seed_b:
                winner, winner_seed = name_a, seed_a
            else:
                winner, winner_seed = name_b, seed_b

        pct_a = f'{100 * votes_a / total:.1f}%' if total else 'no votes'
        pct_b = f'{100 * votes_b / total:.1f}%' if total else 'no votes'
        tie_tag = '  [TIE — seed decides]' if tie else ''
        print(f'  [{m["id"]:2d}] R{m["round"]:d}  '
              f'{name_a} ({votes_a}, {pct_a})  vs  '
              f'{name_b} ({votes_b}, {pct_b})  →  {winner}{tie_tag}')

        results_to_write.append((
            m['id'], m['round'], m['day'], bracket_date,
            name_a, name_b, seed_a, seed_b,
            votes_a, votes_b,
            winner, winner_seed,
            now,
        ))

    if not results_to_write:
        print('\nNothing to write.')
        conn.rollback()
        sys.exit(0)

    if args.dry_run:
        print(f'\n--dry-run: {len(results_to_write)} result(s) NOT written.')
        conn.rollback()
        sys.exit(0)

    # ── Write bracket_results ──────────────────────────────────────────────────

    for row in results_to_write:
        cur.execute('''
            INSERT INTO bracket_results
                (matchup_id, round, day, bracket_date, card_a, card_b, seed_a, seed_b,
                 votes_a, votes_b, winner, winner_seed, closed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (matchup_id) DO UPDATE SET
                votes_a      = EXCLUDED.votes_a,
                votes_b      = EXCLUDED.votes_b,
                winner       = EXCLUDED.winner,
                winner_seed  = EXCLUDED.winner_seed,
                closed_at    = EXCLUDED.closed_at,
                bracket_date = EXCLUDED.bracket_date
        ''', row)

    # ── Advance bracket_state ──────────────────────────────────────────────────

    next_day = close_day + 1
    if args.day is None:
        cur.execute('UPDATE bracket_state SET current_day = %s WHERE id = 1', (next_day,))
        print(f'\nAdvanced bracket_state: day {close_day} → day {next_day}.')
    else:
        print(f'\n--day flag used: bracket_state remains at day {state_day}.')

    conn.commit()
    print(f'Done. {len(results_to_write)} matchup(s) recorded.')

except Exception:
    conn.rollback()
    raise
finally:
    conn.close()
