"""
cycle_report.py — Elo performance summary for card cycles, scored within their queue.

Usage (ipython):
    from cycle_report import cycle_report
    df = cycle_report()

The "top 10%" threshold is computed per queue: a cycle card is counted as top-10%
if its Elo exceeds the 90th percentile of all cards in the same queue.
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
    The top-10% threshold is computed per queue.
    """
    # ── Load cycles ───────────────────────────────────────────────────────────
    with open(os.path.join(_DIR, 'cycles.json'), encoding='utf-8') as f:
        cycles = json.load(f)

    # ── Load queues — build card → queue_id map ───────────────────────────────
    with open(os.path.join(_DIR, 'queues.json'), encoding='utf-8') as f:
        queues_data = json.load(f)

    card_to_queue: dict[str, int] = {}
    for q in queues_data['queues']:
        for card in q['cards']:
            card_to_queue[card] = q['id']

    # ── Load Elo ratings ──────────────────────────────────────────────────────
    db_url = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)
    if not db_url:
        raise RuntimeError('DATABASE_URL environment variable is not set.')

    conn = psycopg2.connect(db_url)
    elo_df = pd.read_sql('SELECT card_name, rating FROM elo_ratings', conn)
    conn.close()
    elo_map: dict[str, float] = dict(zip(elo_df['card_name'], elo_df['rating']))

    # ── Compute per-queue 90th-percentile threshold ───────────────────────────
    queue_threshold: dict[int, float] = {}
    for q in queues_data['queues']:
        qid = q['id']
        ratings = [elo_map[c] for c in q['cards'] if c in elo_map]
        if ratings:
            queue_threshold[qid] = pd.Series(ratings).quantile(0.90)

    # ── Build report rows ─────────────────────────────────────────────────────
    rows = []
    for set_name, cycle_dict in cycles.items():
        for cycle_name, cards in cycle_dict.items():
            if not cards:
                continue

            rated = []
            for card in cards:
                qid = card_to_queue.get(card)
                elo = elo_map.get(card)
                if qid is None or elo is None:
                    continue
                rated.append((card, elo, qid))

            if not rated:
                rows.append({
                    'set':         set_name,
                    'cycle':       cycle_name,
                    'total_cards': len(cards),
                    'top_10_pct':   0,
                    'average_elo':  None,
                    'best_card':    '',
                    'worst_card':   '',
                    'worst_card_elo': None,
                })
                continue

            top10 = sum(
                1 for card, elo, qid in rated
                if qid in queue_threshold and elo >= queue_threshold[qid]
            )

            best  = max(rated, key=lambda x: x[1])
            worst = min(rated, key=lambda x: x[1])

            rows.append({
                'set':         set_name,
                'cycle':       cycle_name,
                'total_cards': len(cards),
                'top_10_pct':     top10,
                'average_elo':    round(sum(e for _, e, _ in rated) / len(rated), 1),
                'best_card':      f'{best[0]} ({round(best[1])})',
                'worst_card':     f'{worst[0]} ({round(worst[1])})',
                'worst_card_elo': round(worst[1], 1),
            })

    return pd.DataFrame(rows)
