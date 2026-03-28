'''
Precomputes UMAP 2D embeddings for the Similarity Map page.

Run once (or whenever card data changes):
    python mapgen.py

Output (written to same directory as this script):
    map_text_heavy.json
    map_balanced.json
    map_struct_heavy.json

Each file: {"points": [[name, x, y, colorClass], ...], "n_cards": N}
  x, y       : float, normalised to [0, 1]
  colorClass : one of W U B R G M(ulticolor) C(olorless)

Same-set penalty (SAME_SET_PENALTY from similarity.py) is applied to
k-NN distances before UMAP runs, so same-set cards are nudged apart in
the final layout.  TYPE_ALPHA_OVERRIDES are intentionally ignored here —
we build one global distance matrix per alpha config, not per-card.

Runtime estimate: 10-30 minutes for all three models.
Note: the first run triggers numba JIT compilation (~60s extra).
'''

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_data import load_card_file, process_cards, FILENAME
from similarity import (
    ALPHA_CONFIGS,
    SAME_SET_PENALTY,
    SAME_SET_FETCH_BUFFER,
    build_combined_matrix,
    extract_keyword_abilities,
)

try:
    import umap as umap_lib
except ImportError:
    print('umap-learn is required.  Install with:  pip install umap-learn')
    sys.exit(1)

OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
K         = 15                           # neighbors kept per card for UMAP
FETCH_K   = K + SAME_SET_FETCH_BUFFER   # fetch extra to absorb same-set reranking
MIN_DIST  = 0.05
N_EPOCHS  = 200
RAND_SEED = 42


# ── Color classification ───────────────────────────────────────────────────

def _color_class(color_identity):
    '''Return single-char color class: W/U/B/R/G/M/C.'''
    if not isinstance(color_identity, list) or len(color_identity) == 0:
        return 'C'
    if len(color_identity) > 1:
        return 'M'
    return color_identity[0]


# ── k-NN graph with same-set penalty ──────────────────────────────────────

def _build_penalised_knn(combined_matrix, card_sets, n_cards):
    '''
    Compute approximate k-NN, apply SAME_SET_PENALTY to same-set neighbors,
    re-sort, keep top-K, and return a symmetric sparse distance matrix.
    '''
    from sklearn.neighbors import NearestNeighbors

    print(f'  Computing {FETCH_K + 1}-NN graph...', flush=True)
    nn = NearestNeighbors(
        n_neighbors=FETCH_K + 1,   # +1 to include self (always at dist≈0)
        metric='cosine',
        algorithm='brute',
        n_jobs=-1,
    )
    nn.fit(combined_matrix)
    distances, indices = nn.kneighbors(combined_matrix)

    print(f'  Applying same-set penalty (×{SAME_SET_PENALTY}) and re-ranking...', flush=True)
    row_list, col_list, dist_list = [], [], []

    for i in range(n_cards):
        neighbors = []
        for rank in range(FETCH_K + 1):
            j = int(indices[i, rank])
            if j == i:
                continue
            d = float(distances[i, rank])
            if card_sets[i] and card_sets[i] == card_sets[j]:
                d *= SAME_SET_PENALTY
            neighbors.append((d, j))
        neighbors.sort()
        for d, j in neighbors[:K]:
            row_list.append(i)
            col_list.append(j)
            dist_list.append(d)

    dist_sparse = sp.csr_matrix(
        (dist_list, (row_list, col_list)),
        shape=(n_cards, n_cards),
    )
    # Symmetrize: take element-wise maximum (conservative — larger distance wins)
    return dist_sparse.maximum(dist_sparse.T)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print('Loading card data...', flush=True)
    raw      = load_card_file(FILENAME)
    df       = process_cards(raw)
    post_c16 = df[df['released_at'] > pd.Timestamp('2016-11-11')].reset_index(drop=True)
    n_cards  = len(post_c16)
    print(f'Post-C16 cards: {n_cards}', flush=True)

    names         = post_c16['name'].values
    card_sets     = post_c16['set'].fillna('').values
    color_classes = [_color_class(ci) for ci in post_c16['color_identity'].tolist()]

    print('Extracting keyword abilities...', flush=True)
    keyword_abilities = extract_keyword_abilities(post_c16)

    for config_name, alpha in ALPHA_CONFIGS.items():
        out_path = os.path.join(OUT_DIR, f'map_{config_name}.json')
        print(f'\n{"─"*55}', flush=True)
        print(f'Config: {config_name}  (alpha={alpha})', flush=True)
        t0 = time.time()

        print('  Building combined feature matrix...', flush=True)
        combined, _ = build_combined_matrix(
            post_c16, alpha=alpha, keyword_abilities=keyword_abilities
        )

        dist_sym = _build_penalised_knn(combined, card_sets, n_cards)

        print(
            f'  Running UMAP  '
            f'(n_neighbors={K}, min_dist={MIN_DIST}, n_epochs={N_EPOCHS})...',
            flush=True,
        )
        reducer = umap_lib.UMAP(
            n_components=2,
            metric='precomputed',
            n_neighbors=K,
            min_dist=MIN_DIST,
            n_epochs=N_EPOCHS,
            random_state=RAND_SEED,
            low_memory=True,
            n_jobs=1,       # more stable with sparse precomputed input
            verbose=True,
        )
        embedding = reducer.fit_transform(dist_sym)

        # Normalise each axis independently to [0, 1]
        xy_min = embedding.min(axis=0)
        xy_max = embedding.max(axis=0)
        embedding = (embedding - xy_min) / (xy_max - xy_min)

        points = [
            [
                names[i],
                round(float(embedding[i, 0]), 4),
                round(float(embedding[i, 1]), 4),
                color_classes[i],
            ]
            for i in range(n_cards)
        ]

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'points': points, 'n_cards': n_cards}, f, separators=(',', ':'))

        elapsed = time.time() - t0
        print(f'  Saved → {out_path}  ({n_cards} cards, {elapsed / 60:.1f} min)', flush=True)

    print('\nAll done.', flush=True)


if __name__ == '__main__':
    main()
