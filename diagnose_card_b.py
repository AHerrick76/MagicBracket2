"""
diagnose_card_b.py — Measure card_b exposure bias in the similarity graph.

For each card, computes how many other cards in the same queue list it among
their top-N intra-queue similarity candidates (in-degree). Cards with low
in-degree are rarely selected as card_b regardless of how many votes are cast.

In both modes, neighbor lists are composed of intra-queue cards only (matching
what the app should produce). The feature matrices are built once on all
post-C16 cards; only the KNN index is rebuilt per queue.

Modes
-----
Single queue (default):
    Reports in-degree stats and top/bottom cards for one queue.

All queues (--all-queues):
    Iterates over every queue, builds an intra-queue KNN index for each,
    then aggregates in-degree results across the full ~15,800-card set.

Usage
-----
    python diagnose_card_b.py                   # active queue from DB
    python diagnose_card_b.py --queue 2         # specific queue
    python diagnose_card_b.py --all-queues      # full analysis across all queues
    python diagnose_card_b.py --n 5             # neighbors per card per config (default 5)
    python diagnose_card_b.py --n 125           # approximate Mode 3 (25% of 500)
    python diagnose_card_b.py --top 15          # cards shown at each extreme (default 15)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_data import load_processed_cards
from similarity import (
    build_structured_matrix, build_text_matrix,
    ALPHA_CONFIGS, TYPE_ALPHA_OVERRIDES, SAME_SET_PENALTY, SAME_SET_FETCH_BUFFER,
    extract_keyword_abilities,
)

QUEUES_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queues.json')
DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)

parser = argparse.ArgumentParser()
parser.add_argument('--queue',      type=int,             help='Queue ID to analyse (default: latest active from DB)')
parser.add_argument('--all-queues', action='store_true',  help='Analyse every queue and aggregate across the full card set')
parser.add_argument('--n',          type=int, default=5,  help='Neighbors per card per config (default 5; use 125 to approximate Mode 3)')
parser.add_argument('--top',        type=int, default=15, help='Cards to show at each extreme (default 15)')
args = parser.parse_args()

# ── Load queues.json ──────────────────────────────────────────────────────────

if not os.path.exists(QUEUES_PATH):
    print('ERROR: queues.json not found — run generate_queues.py first.', file=sys.stderr)
    sys.exit(1)

with open(QUEUES_PATH, encoding='utf-8') as f:
    queue_data = json.load(f)
queue_index   = {q['id']: q for q in queue_data['queues']}
card_to_queue = {card: q['id'] for q in queue_data['queues'] for card in q['cards']}

# ── Load card data and build feature matrices once ───────────────────────────

print('Loading card data...')
df_all   = load_processed_cards()
post_c16 = df_all[df_all['released_at'] > pd.Timestamp('2016-11-11')].reset_index(drop=True)
print(f'{len(post_c16)} post-C16 cards loaded.')

all_names    = post_c16['name'].values
name_to_idx  = {name: i for i, name in enumerate(all_names)}
name_to_set  = dict(zip(post_c16['name'], post_c16['set'].fillna('')))

print('Building feature matrices (once for all cards)...')
keyword_abilities = extract_keyword_abilities(post_c16)
struct_mat        = build_structured_matrix(post_c16, keyword_abilities)
text_mat, _       = build_text_matrix(post_c16)

# Pre-combine into one sparse matrix per alpha so queue slicing is fast.
all_alphas = set(ALPHA_CONFIGS.values())
for ov in TYPE_ALPHA_OVERRIDES.values():
    all_alphas.update(ov.values())

combined_by_alpha = {}
for alpha in sorted(all_alphas):
    combined_by_alpha[alpha] = sp.hstack([
        sp.csr_matrix(struct_mat * np.sqrt(alpha)),
        text_mat * np.sqrt(1.0 - alpha),
    ], format='csr')
    print(f'  alpha={alpha} combined.')

# ── Core: compute intra-queue in-degrees for a single queue ──────────────────

def queue_indegrees(queue_cards, N):
    """
    Build a KNN index on the queue subset and return per-card in-degree counts.

    Neighbor lists are computed within the queue only (intra-queue), matching
    how the app should generate card_b candidates.

    Returns dict: card_name -> {config_name: int, 'mean': float}
    """
    # Resolve to row indices in the global matrices; drop unknown names.
    indices = [name_to_idx[c] for c in queue_cards if c in name_to_idx]
    if len(indices) < 2:
        return {}

    q_names   = np.array([all_names[i] for i in indices])
    fetch_n   = min(len(indices), N + SAME_SET_FETCH_BUFFER + 1)
    indegrees = {name: {} for name in q_names}

    for config_name, alpha in ALPHA_CONFIGS.items():
        sub_mat = combined_by_alpha[alpha][indices]   # slice rows for this queue

        nn = NearestNeighbors(n_neighbors=fetch_n, metric='cosine',
                              algorithm='brute', n_jobs=-1)
        nn.fit(sub_mat)
        dists_all, idx_all = nn.kneighbors(sub_mat)

        counts = np.zeros(len(q_names), dtype=int)
        for local_i, (dists, local_neighbors) in enumerate(zip(dists_all, idx_all)):
            card_set_i = name_to_set.get(q_names[local_i], '')
            candidates = []
            for d, local_j in zip(dists, local_neighbors):
                if local_j == local_i:
                    continue
                if card_set_i and name_to_set.get(q_names[local_j]) == card_set_i:
                    d *= SAME_SET_PENALTY
                candidates.append((d, local_j))
            candidates.sort(key=lambda x: x[0])
            for _, local_j in candidates[:N]:
                counts[local_j] += 1

        for local_i, name in enumerate(q_names):
            indegrees[name][config_name] = int(counts[local_i])

    for name in q_names:
        indegrees[name]['mean'] = float(np.mean([
            indegrees[name][c] for c in ALPHA_CONFIGS
        ]))
    return indegrees


# ── Reporting helpers ─────────────────────────────────────────────────────────

def gini(values):
    arr = np.sort(np.array(values, dtype=float))
    n   = len(arr)
    if arr.sum() == 0:
        return 0.0
    return (2 * np.dot(np.arange(1, n + 1), arr) - (n + 1) * arr.sum()) / (n * arr.sum())


def print_report(indegrees, top_k, show_queue=False, label=''):
    names = list(indegrees.keys())
    sorted_names = sorted(names, key=lambda n: indegrees[n]['mean'])

    print(f'\n── In-degree summary{" — " + label if label else ""} {"─" * max(0, 55 - len(label))}')
    for key in list(ALPHA_CONFIGS) + ['mean']:
        vals = [indegrees[n][key] for n in names]
        arr  = np.array(vals)
        print(f'  {key:<14}: zero={100*(arr==0).mean():5.1f}%  '
              f'p10={np.percentile(arr,10):.0f}  '
              f'median={np.median(arr):.1f}  '
              f'p90={np.percentile(arr,90):.0f}  '
              f'max={arr.max():.0f}  '
              f'Gini={gini(vals):.3f}')

    hdr = f'  {"Card":<50} {"text_heavy":>10} {"balanced":>10} {"struct_heavy":>12} {"mean":>6}'
    if show_queue:
        hdr += f'  {"queue":>5}'

    print(f'\n── Bottom {top_k} — least likely to appear as card_b ─────────────────────')
    print(hdr)
    for name in sorted_names[:top_k]:
        d = indegrees[name]
        row = f'  {name:<50} {d["text_heavy"]:>10} {d["balanced"]:>10} {d["struct_heavy"]:>12} {d["mean"]:>6.1f}'
        if show_queue:
            row += f'  {card_to_queue.get(name, "?"):>5}'
        print(row)

    print(f'\n── Top {top_k} — most likely to appear as card_b ───────────────────────────')
    print(hdr)
    for name in sorted_names[-top_k:][::-1]:
        d = indegrees[name]
        row = f'  {name:<50} {d["text_heavy"]:>10} {d["balanced"]:>10} {d["struct_heavy"]:>12} {d["mean"]:>6.1f}'
        if show_queue:
            row += f'  {card_to_queue.get(name, "?"):>5}'
        print(row)


# ═════════════════════════════════════════════════════════════════════════════
# ALL-QUEUES MODE
# ═════════════════════════════════════════════════════════════════════════════

if args.all_queues:
    print(f'\nAll-queues mode: N={args.n}, {queue_data["total_queues"]} queues to process.')
    all_indegrees = {}
    for q in queue_data['queues']:
        qid   = q['id']
        cards = q['cards']
        print(f'  Queue {qid:>3} / {queue_data["total_queues"]}  ({len(cards)} cards)...', end='', flush=True)
        result = queue_indegrees(cards, args.n)
        all_indegrees.update(result)
        print(f'  done.')

    print(f'\n{len(all_indegrees)} cards processed across {queue_data["total_queues"]} queues.')
    print_report(all_indegrees, args.top, show_queue=True, label='all queues')

# ═════════════════════════════════════════════════════════════════════════════
# SINGLE-QUEUE MODE
# ═════════════════════════════════════════════════════════════════════════════

else:
    queue_id = args.queue
    if queue_id is None:
        if DATABASE_URL:
            try:
                import psycopg2
                conn = psycopg2.connect(DATABASE_URL)
                cur  = conn.cursor()
                cur.execute('SELECT queue_id FROM queue_transitions ORDER BY id DESC LIMIT 1')
                row = cur.fetchone()
                queue_id = row[0] if row else 1
                cur.close(); conn.close()
                print(f'Active queue from DB: {queue_id}')
            except Exception as e:
                print(f'Could not read DB ({e}); defaulting to queue 1.')
                queue_id = 1
        else:
            queue_id = 1
            print('No DATABASE_URL — defaulting to queue 1.')

    if queue_id not in queue_index:
        print(f'ERROR: queue {queue_id} not in queues.json.', file=sys.stderr)
        sys.exit(1)

    queue_cards = queue_index[queue_id]['cards']
    print(f'\nQueue {queue_id}: {len(queue_cards)} cards. Computing in-degrees (N={args.n})...')
    indegrees = queue_indegrees(queue_cards, args.n)
    print_report(indegrees, args.top, label=f'queue {queue_id}')

    # ── Empirical card_b counts from votes DB ─────────────────────────────────
    if DATABASE_URL:
        try:
            import psycopg2
            conn = psycopg2.connect(DATABASE_URL)
            cur  = conn.cursor()
            cur.execute('''
                SELECT card_b, COUNT(*) AS cnt
                FROM votes
                WHERE queue_id = %s
                GROUP BY card_b
                ORDER BY cnt DESC
            ''', (queue_id,))
            rows = cur.fetchall()
            cur.close(); conn.close()

            if rows:
                n_cards = len(indegrees)
                print(f'\n── Empirical card_b counts from votes (queue {queue_id}) ─────────────────')
                total_votes = sum(r[1] for r in rows)
                print(f'  Total votes: {total_votes}  |  Seen as card_b: {len(rows)} / {n_cards}  |  Never: {n_cards - len(rows)}')
                print(f'\n  Most frequent as card_b:')
                for name, cnt in rows[:10]:
                    print(f'    {name:<50} {cnt:>5}')
                if len(rows) >= 10:
                    print(f'\n  Least frequent as card_b (among those seen):')
                    for name, cnt in rows[-10:]:
                        print(f'    {name:<50} {cnt:>5}')
            else:
                print(f'\nNo votes found for queue {queue_id} yet.')
        except Exception as e:
            print(f'\nCould not query votes DB: {e}')
    else:
        print('\nNo DATABASE_URL — skipping empirical vote analysis.')
