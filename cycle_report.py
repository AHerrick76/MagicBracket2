"""
cycle_report.py — Elo performance summary for card cycles, scored within their queue.

Usage (ipython):
    from cycle_report import cycle_report
    df = cycle_report()

Top-10% membership is read directly from top_10_queue.json (pre-computed).
Elo ratings are still fetched from the database for best/worst/average stats
across all cycle members, not just those that made the top 10%.
"""

import json
import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

_DIR = os.path.dirname(os.path.abspath(__file__))


def cycle_report() -> pd.DataFrame:
    """
    Returns a DataFrame with one row per (set, cycle), columns:
        set, cycle, total_cards, top_10_pct, average_elo, best_card, worst_card, worst_card_elo
    where best_card / worst_card are formatted as "Name (1523)" etc.
    Sorted by set order in cycles.json, then cycle order within each set.
    Top-10% membership is read from top_10_queue.json.
    """
    # ── Load cycles ───────────────────────────────────────────────────────────
    with open(os.path.join(_DIR, 'cycles.json'), encoding='utf-8') as f:
        cycles = json.load(f)

    # ── Load top-10% card set from top_10_queue.json ──────────────────────────
    with open(os.path.join(_DIR, 'top_10_queue.json'), encoding='utf-8') as f:
        top10_data = json.load(f)

    top10_set: set[str] = {entry['name'] for entry in top10_data['cards']}

    # ── Load Elo ratings ──────────────────────────────────────────────────────
    db_url = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)
    if not db_url:
        raise RuntimeError('DATABASE_URL environment variable is not set.')

    conn = psycopg2.connect(db_url)
    elo_df = pd.read_sql('SELECT card_name, rating FROM elo_ratings', conn)
    conn.close()
    elo_map: dict[str, float] = dict(zip(elo_df['card_name'], elo_df['rating']))

    # ── Build report rows ─────────────────────────────────────────────────────
    rows = []
    for set_name, cycle_dict in cycles.items():
        for cycle_name, cards in cycle_dict.items():
            if not cards:
                continue

            rated = [(card, elo_map[card]) for card in cards if card in elo_map]

            if not rated:
                rows.append({
                    'set':           set_name,
                    'cycle':         cycle_name,
                    'total_cards':   len(cards),
                    'top_10_pct':    0,
                    'average_elo':   None,
                    'best_card':     '',
                    'worst_card':    '',
                    'worst_card_elo': None,
                })
                continue

            top10 = sum(1 for card, _ in rated if card in top10_set)

            best  = max(rated, key=lambda x: x[1])
            worst = min(rated, key=lambda x: x[1])

            rows.append({
                'set':           set_name,
                'cycle':         cycle_name,
                'total_cards':   len(cards),
                'top_10_pct':    top10,
                'average_elo':   round(sum(e for _, e in rated) / len(rated), 1),
                'best_card':     f'{best[0]} ({round(best[1])})',
                'worst_card':    f'{worst[0]} ({round(worst[1])})',
                'worst_card_elo': round(worst[1], 1),
            })

    return pd.DataFrame(rows)
