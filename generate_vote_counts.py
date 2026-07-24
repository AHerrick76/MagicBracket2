'''
generate_vote_counts.py — Build a static vote-counts page from CSV backups.

Reads db_backup/ CSVs (no database connection needed) and writes
templates/vote_counts_static.html, which app_bracket.py then serves directly.

Usage:
    python generate_vote_counts.py [--backup-dir <path>] [--out <path>]
'''

import argparse
import json
import os

import pandas as pd

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKUP_DIR  = os.path.join(BASE_DIR, 'db_backup')
OUT_DEFAULT = os.path.join(BASE_DIR, 'templates', 'vote_counts_static.html')

ROUND_LABELS = {1: 'Round of 64', 2: 'Round of 32', 3: 'Round of 16',
                4: 'Quarterfinals', 5: 'Semifinals', 6: 'Final'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backup-dir', default=BACKUP_DIR)
    parser.add_argument('--out',        default=OUT_DEFAULT)
    args = parser.parse_args()

    # ── Load bracket structure ─────────────────────────────────────────────────
    with open(os.path.join(BASE_DIR, 'bracket.json'), encoding='utf-8') as f:
        bracket = json.load(f)

    # ── Load queue metadata ────────────────────────────────────────────────────
    with open(os.path.join(BASE_DIR, 'queues.json'), encoding='utf-8') as f:
        queues_data = json.load(f)
    queues_meta = {q['id']: q for q in queues_data.get('queues', []) if q['id'] != 1}

    # ── Load top-10% size ──────────────────────────────────────────────────────
    top10_json_path = os.path.join(BASE_DIR, 'top_10_queue.json')
    try:
        with open(top10_json_path, encoding='utf-8') as f:
            t10 = json.load(f)
        top10_size = t10.get('total_cards') or len(t10.get('cards', []))
    except Exception:
        top10_size = None

    # ── Aggregate from CSVs ────────────────────────────────────────────────────
    bv = pd.read_csv(os.path.join(args.backup_dir, 'bracket_votes.csv'))
    day_rows = (
        bv.groupby('day')
          .agg(votes=('id', 'count'), ballots=('ballot_id', 'nunique'))
          .to_dict('index')
    )
    # day_rows: {day: {'votes': N, 'ballots': M}}

    v = pd.read_csv(os.path.join(args.backup_dir, 'votes.csv'))
    v_filtered = v[v['queue_id'].notna() & (v['queue_id'] != 1)]
    queue_vote_rows = v_filtered.groupby('queue_id')['id'].count().to_dict()
    queue_vote_rows = {int(k): int(v) for k, v in queue_vote_rows.items()}

    vt10 = pd.read_csv(os.path.join(args.backup_dir, 'votes_top10.csv'))
    top10_votes = len(vt10)

    # ── Build bracket HTML table ───────────────────────────────────────────────
    day_info = {}
    for m in bracket['matchups']:
        d, r = m['day'], m['round']
        if d not in day_info:
            day_info[d] = {'round': r}

    round_days = {}
    for d, info in day_info.items():
        round_days.setdefault(info['round'], []).append(d)
    for r in round_days:
        round_days[r].sort()

    bracket_rows_html = ''
    for d in sorted(day_info.keys()):
        r = day_info[d]['round']
        days_in_round = round_days[r]
        if len(days_in_round) > 1:
            day_within = days_in_round.index(d) + 1
            label = f'{ROUND_LABELS.get(r, f"Round {r}")} — Day {day_within}'
        else:
            label = ROUND_LABELS.get(r, f'Round {r}')
        data = day_rows.get(d, {'votes': 0, 'ballots': 0})
        bracket_rows_html += (
            f'<tr><td>{label}</td>'
            f'<td class="num">{data["ballots"]:,}</td>'
            f'<td class="num">{data["votes"]:,}</td></tr>\n'
        )

    total_ballots = sum(v['ballots'] for v in day_rows.values())
    total_votes   = sum(v['votes']   for v in day_rows.values())

    # ── Build queue HTML table ─────────────────────────────────────────────────
    all_qids = sorted(set(list(queues_meta.keys()) + list(queue_vote_rows.keys())))
    queue_rows_html = ''
    for qid in all_qids:
        votes   = queue_vote_rows.get(qid, 0)
        q_meta  = queues_meta.get(qid, {})
        qsize   = len(q_meta.get('cards', [])) if q_meta else None
        qsize_str = f'{qsize:,}' if qsize is not None else '—'
        queue_rows_html += (
            f'<tr><td>Queue {qid}</td>'
            f'<td class="num">{qsize_str}</td>'
            f'<td class="num">{votes:,}</td></tr>\n'
        )

    queue_total_votes = sum(queue_vote_rows.values())
    top10_size_str = f'{top10_size:,}' if top10_size is not None else '—'

    # ── Render HTML ────────────────────────────────────────────────────────────
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vote Counts — Magic Bracket</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #1a1a1a; color: #ddd; font-family: sans-serif; }}
nav {{
  display: flex; justify-content: center; gap: 2rem;
  padding: 0.75rem 1rem;
  background: #13132a; border-bottom: 1px solid #2a2a4a;
  flex-wrap: wrap;
}}
nav a {{ color: #c8a96e; text-decoration: none; font-size: 0.85rem; letter-spacing: 0.03em; }}
nav a:hover {{ text-decoration: underline; }}
.page-header {{
  background: #111; padding: 14px 20px 12px;
  border-bottom: 1px solid #2a2a2a;
  text-align: center;
}}
.page-header h1 {{
  font-family: Georgia, serif;
  font-size: 1.4rem; color: #c9a84c;
  letter-spacing: 0.12em; text-transform: uppercase; font-weight: normal;
}}
.content {{ padding: 24px 20px; max-width: 560px; margin: 0 auto; }}
h2 {{ color: #c9a84c; margin: 28px 0 10px; font-size: 1.05rem; letter-spacing: 0.05em; text-transform: uppercase; font-weight: normal; }}
h2:first-child {{ margin-top: 0; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
th {{ text-align: left; color: #888; border-bottom: 1px solid #444; padding: 5px 10px; font-weight: normal; }}
td {{ padding: 5px 10px; border-bottom: 1px solid #222; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
th.num {{ text-align: right; }}
.total {{ color: #c9a84c; font-weight: bold; margin: 8px 0 0; font-size: 0.88rem; }}
</style>
</head>
<body>

<nav>
  <a href="/">Home</a>
  <a href="/bracket">Full Bracket</a>
  <a href="/honorable-mentions">Honorable Mentions</a>
  <a href="/community-favorites">Community Favorites</a>
  <a href="/universe">Card Browser</a>
  <a href="/vote-counts">Vote Counts</a>
  <a href="/faq">FAQ</a>
  <a href="/share">Share</a>
</nav>

<div class="page-header">
  <h1>Vote Counts</h1>
</div>

<div class="content">

<h2>Top 64 Bracket</h2>
<table>
  <tr><th>Round</th><th class="num">Ballots</th><th class="num">Total Votes</th></tr>
  {bracket_rows_html}
</table>
<p class="total">Total: {total_ballots:,} ballots &nbsp;/&nbsp; {total_votes:,} individual votes</p>

<h2>Top 10% Tournament</h2>
<table>
  <tr><th>Phase</th><th class="num">Cards</th><th class="num">Votes</th></tr>
  <tr><td>Tournament</td><td class="num">{top10_size_str}</td><td class="num">{top10_votes:,}</td></tr>
</table>

<h2>Regular Queues</h2>
<table>
  <tr><th>Queue</th><th class="num">Cards</th><th class="num">Votes</th></tr>
  {queue_rows_html}
</table>
<p class="total">Total: {queue_total_votes:,} votes</p>

</div>
</body>
</html>'''

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Written to {args.out}')


if __name__ == '__main__':
    main()
