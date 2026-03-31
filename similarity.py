'''
Card similarity model: weighted combination of structured features and TF-IDF oracle text.

Combined similarity = alpha * struct_cosine + (1-alpha) * text_cosine

The combination is implemented via feature concatenation:
    combined_vec = [ sqrt(alpha)*struct_L2, sqrt(1-alpha)*text_L2 ]
so that cosine(combined_i, combined_j) = alpha*s_cos + (1-alpha)*t_cos exactly,
because both halves are L2-normalised before scaling.

Structured features (per card):
    - CMC                        (1,  normalised 0-1)
    - Color identity             (5,  binary WUBRG)
    - Card types                 (8,  binary)
    - Power / toughness          (2,  normalised 0-1)
    - Keyword abilities          (top N from dataset, binary)
    - Keyword actions in text    (fixed list, binary presence in oracle_text)
'''

import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ── Constants ─────────────────────────────────────────────────────────────────

COLORS     = ['W', 'U', 'B', 'R', 'G']
CARD_TYPES = ['Creature', 'Instant', 'Sorcery', 'Enchantment',
              'Artifact', 'Planeswalker', 'Land', 'Battle']

# Canonical keyword actions (CR 701). These are verbs that appear in oracle text
# and signal what a card *does*, making them strong discriminators for text-light models.
KEYWORD_ACTIONS = [
    'destroy', 'exile', 'counter', 'sacrifice', 'discard', 'draw', 'search',
    'reveal', 'shuffle', 'tap', 'untap', 'scry', 'surveil', 'mill', 'fight',
    'create', 'proliferate', 'investigate', 'explore', 'populate', 'adapt',
    'amass', 'connive', 'foretell', 'bolster', 'support', 'goad', 'exert',
    'manifest', 'transform', 'meld', 'regenerate', 'monstrosity', 'detain',
    'fateseal', 'vote', 'cipher', 'planeswalk', 'venture', 'learn', 'double',
]

# Same-set downranking: fetch extra candidates and apply a distance penalty to
# cards sharing a set with the query card, then re-sort before taking top-n.
SAME_SET_PENALTY = 1.35   # multiply distance by this for same-set candidates
SAME_SET_FETCH_BUFFER = 5 # extra neighbors fetched to give reranking headroom

# Three pre-defined calibrations used by get_candidates()
ALPHA_CONFIGS = {
    'text_heavy':   0.15,
    'balanced':     0.50,
    'struct_heavy': 0.85,
}

# Card types whose oracle text is too sparse to be useful at low alpha values.
# For these, all three tiers are shifted toward structured features.
# Keys must be substrings of type_line (case-sensitive, as Scryfall writes them).
TYPE_ALPHA_OVERRIDES = {
    'Planeswalker': {
        'text_heavy':   0.40,
        'balanced':     0.70,
        'struct_heavy': 0.90,
    },
}


# ── Keyword extraction ────────────────────────────────────────────────────────

def extract_keyword_abilities(df, n=60):
    '''
    Return the n most common keyword abilities present in df, drawn from
    Scryfall's already-parsed `keywords` column (e.g. Flying, Trample, etc.).
    '''
    counts = Counter()
    for kws in df['keywords'].dropna():
        if isinstance(kws, list):
            counts.update(kws)
    return [kw for kw, _ in counts.most_common(n)]


# ── Feature builders ──────────────────────────────────────────────────────────

def _parse_pt_series(series):
    '''Convert power/toughness strings to float; NaN or non-numeric (*, X) → 0.'''
    def _p(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0.0
        try:
            return min(float(v), 20) / 20
        except (ValueError, TypeError):
            return 0.0
    return series.apply(_p).values.reshape(-1, 1).astype(np.float32)


def build_structured_matrix(df, keyword_abilities=None):
    '''
    Build an L2-normalised (n_cards x n_features) float32 array.

    keyword_abilities : list of keyword ability strings to include as binary
                        features. If None, extracted automatically from df
                        (top 60 by frequency).
    '''
    if keyword_abilities is None:
        keyword_abilities = extract_keyword_abilities(df)

    parts = []

    # CMC (capped at 16)
    cmc = df['cmc'].fillna(0).clip(upper=16).astype(float) / 16
    parts.append(cmc.values.reshape(-1, 1).astype(np.float32))

    # Color identity — binary WUBRG
    for c in COLORS:
        col = df['color_identity'].apply(
            lambda x, _c=c: 1.0 if isinstance(x, list) and _c in x else 0.0
        ).values.reshape(-1, 1).astype(np.float32)
        parts.append(col)

    # Card type flags
    for t in CARD_TYPES:
        col = df['type_line'].fillna('').str.contains(t, case=True, regex=False) \
                             .astype(np.float32).values.reshape(-1, 1)
        parts.append(col)

    # Power / toughness (capped at 20)
    parts.append(_parse_pt_series(df['power']))
    parts.append(_parse_pt_series(df['toughness']))

    # Keyword abilities — binary flags from Scryfall's parsed `keywords` column
    for kw in keyword_abilities:
        col = df['keywords'].apply(
            lambda x, _kw=kw: 1.0 if isinstance(x, list) and _kw in x else 0.0
        ).values.reshape(-1, 1).astype(np.float32)
        parts.append(col)

    # Keyword actions — binary presence in oracle text
    oracle = df['oracle_text'].fillna('')
    for ka in KEYWORD_ACTIONS:
        col = oracle.str.contains(r'\b' + ka + r'\b', case=False, regex=True) \
                    .astype(np.float32).values.reshape(-1, 1)
        parts.append(col)

    mat = np.hstack(parts)
    return normalize(mat, norm='l2')


def build_text_matrix(df):
    '''
    Build an L2-normalised TF-IDF sparse matrix from oracle text.
    Returns (matrix, fitted_vectorizer).
    '''
    texts = df['oracle_text'].fillna('').tolist()
    vectorizer = TfidfVectorizer(
        strip_accents='unicode',
        lowercase=True,
        token_pattern=r'[a-z]{2,}',
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    mat = vectorizer.fit_transform(texts)
    return normalize(mat, norm='l2'), vectorizer


def build_combined_matrix(df, alpha=0.5, keyword_abilities=None):
    '''
    Combine structured and text matrices.
    alpha : weight for structured features (0 = text-only, 1 = struct-only).
    Returns (combined_sparse_matrix, fitted_tfidf_vectorizer).
    '''
    beta   = 1.0 - alpha
    struct = build_structured_matrix(df, keyword_abilities)
    text, vectorizer = build_text_matrix(df)

    combined = sp.hstack([
        sp.csr_matrix(struct * np.sqrt(alpha)),
        text * np.sqrt(beta),
    ], format='csr')

    return combined, vectorizer


# ── Candidate model (pre-built for fast lookup) ───────────────────────────────

def build_candidate_models(df, n_neighbors=5, alphas=None, type_overrides=None):
    '''
    Pre-fit one NearestNeighbors model per unique alpha value so that
    get_candidates() can do fast index lookups without rebuilding anything.
    Models are keyed by alpha value and shared across configs that happen to
    use the same alpha.

    Parameters
    ----------
    df             : processed, deduplicated cards DataFrame
    n_neighbors    : candidates to return per card per config (default 5)
    alphas         : dict of {config_name: alpha}; defaults to ALPHA_CONFIGS
    type_overrides : dict of {type_substring: {config_name: alpha}};
                     defaults to TYPE_ALPHA_OVERRIDES

    Returns
    -------
    A models dict suitable for passing to get_candidates().
    '''
    if alphas is None:
        alphas = ALPHA_CONFIGS
    if type_overrides is None:
        type_overrides = TYPE_ALPHA_OVERRIDES

    df = df.reset_index(drop=True)
    keyword_abilities = extract_keyword_abilities(df)
    names             = df['name'].values
    name_to_idx       = {name: i for i, name in enumerate(names)}
    name_to_type      = dict(zip(df['name'], df['type_line'].fillna('')))
    name_to_set       = dict(zip(df['name'], df['set'].fillna('')))

    # Collect every unique alpha value across all configs + overrides
    all_alphas = set(alphas.values())
    for override_alphas in type_overrides.values():
        all_alphas.update(override_alphas.values())

    # Build exactly one NearestNeighbors model per unique alpha
    alpha_models = {}
    for alpha in sorted(all_alphas):
        print(f'  Building model: alpha={alpha}...')
        combined, _ = build_combined_matrix(df, alpha=alpha,
                                            keyword_abilities=keyword_abilities)
        nn = NearestNeighbors(
            n_neighbors=n_neighbors + 1,
            metric='cosine',
            algorithm='brute',
            n_jobs=-1,
        )
        nn.fit(combined)
        alpha_models[alpha] = {'nn': nn, 'matrix': combined}

    return {
        '_names':             names,
        '_name_to_idx':       name_to_idx,
        '_name_to_type':      name_to_type,
        '_name_to_set':       name_to_set,
        '_keyword_abilities': keyword_abilities,
        '_n_neighbors':       n_neighbors,
        '_alpha_configs':     alphas,
        '_type_overrides':    type_overrides,
        '_alpha_models':      alpha_models,
    }


def get_candidates(card_name, models, n_neighbors=None, allowed_names=None):
    '''
    Return candidate matchups for a card under each alpha calibration.
    If the card's type_line matches a TYPE_ALPHA_OVERRIDES entry, the
    override alphas are used instead of the standard ALPHA_CONFIGS.

    Parameters
    ----------
    card_name     : str       — must be an exact name present in the model
    models        : dict      — output of build_candidate_models()
    n_neighbors   : int|None  — overrides model default if provided
    allowed_names : list|None — if provided, only candidates in this set are
                                returned (used for Elo-bracket pre-filtering)

    Returns
    -------
    dict with one key per config (e.g. 'text_heavy', 'balanced', 'struct_heavy'),
    each mapping to a list of up to n_neighbors card name strings.

    Example
    -------
    >>> models = build_candidate_models(post_c16)
    >>> get_candidates('Solitude', models)
    {
        'text_heavy':   ['Werefox Bodyguard', 'Skyclave Apparition', ...],
        'balanced':     ['Loran of the Third Path', ...],
        'struct_heavy': ['Bastion Enforcer', ...],
    }
    '''
    name_to_idx  = models['_name_to_idx']
    names        = models['_names']
    n            = n_neighbors if n_neighbors is not None else models['_n_neighbors']
    alpha_models = models['_alpha_models']
    name_to_set  = models.get('_name_to_set', {})

    if card_name not in name_to_idx:
        raise KeyError(f"'{card_name}' not found in candidate models.")

    idx       = name_to_idx[card_name]
    type_line = models['_name_to_type'].get(card_name, '')
    card_set  = name_to_set.get(card_name, '')

    allowed_set = set(allowed_names) if allowed_names is not None else None

    # Select alpha configs: use a type-specific override if one applies
    alpha_configs = models['_alpha_configs']
    for type_key, override in models['_type_overrides'].items():
        if type_key in type_line:
            alpha_configs = override
            break

    result = {}
    for config_name, alpha in alpha_configs.items():
        model = alpha_models[alpha]

        if allowed_set is not None:
            # Fetch enough extra neighbors so we can expect to find n within
            # the allowed set after filtering.  ratio = fraction of all cards
            # that are eligible; oversample by 3x plus the usual buffer.
            ratio = max(0.01, len(allowed_set) / len(names))
            fetch_n = min(len(names), int(n / ratio * 3) + SAME_SET_FETCH_BUFFER + 1)
        else:
            fetch_n = n + SAME_SET_FETCH_BUFFER + 1  # +1 to exclude self

        dists, indices = model['nn'].kneighbors(
            model['matrix'][idx], n_neighbors=fetch_n
        )
        # Apply allowed-set filter, same-set distance penalty, then re-sort and take top-n
        candidates = []
        for d, j in zip(dists[0], indices[0]):
            if j == idx:
                continue
            if allowed_set is not None and names[j] not in allowed_set:
                continue
            if card_set and name_to_set.get(names[j]) == card_set:
                d = d * SAME_SET_PENALTY
            candidates.append((d, j))
        candidates.sort(key=lambda x: x[0])
        result[config_name] = [names[j] for _, j in candidates[:n]]

    return result


# ── Queue-scoped candidate models ────────────────────────────────────────────

def build_queue_models(base_models, card_names):
    '''
    Build a models dict restricted to card_names by slicing the precomputed
    feature matrices from base_models and re-fitting the KNN indices.
    Feature computation is not repeated — only the small KNN re-fit runs,
    which takes a few seconds for ~500 cards.

    Parameters
    ----------
    base_models : dict — output of build_candidate_models() over the full card set
    card_names  : list of card name strings (must all be present in base_models)

    Returns
    -------
    A models dict with the same structure as build_candidate_models(), but
    with KNN indices fit on the restricted card set only.
    '''
    full_name_to_idx = base_models['_name_to_idx']

    valid       = [(n, full_name_to_idx[n]) for n in card_names if n in full_name_to_idx]
    names_arr   = np.array([n for n, _ in valid])
    row_indices = [i for _, i in valid]
    name_to_idx = {name: i for i, name in enumerate(names_arr)}

    n_neighbors = base_models['_n_neighbors']
    n_cards     = len(names_arr)

    alpha_models = {}
    for alpha, model in base_models['_alpha_models'].items():
        sub_matrix = model['matrix'][row_indices]
        nn = NearestNeighbors(
            n_neighbors=min(n_neighbors + 1, n_cards),
            metric='cosine',
            algorithm='brute',
            n_jobs=-1,
        )
        nn.fit(sub_matrix)
        alpha_models[alpha] = {'nn': nn, 'matrix': sub_matrix}

    return {
        '_names':             names_arr,
        '_name_to_idx':       name_to_idx,
        '_name_to_type':      base_models['_name_to_type'],
        '_name_to_set':       base_models['_name_to_set'],
        '_keyword_abilities': base_models['_keyword_abilities'],
        '_n_neighbors':       n_neighbors,
        '_alpha_configs':     base_models['_alpha_configs'],
        '_type_overrides':    base_models['_type_overrides'],
        '_alpha_models':      alpha_models,
    }


# ── Display helpers ───────────────────────────────────────────────────────────

def show_matches(matches, card_name, n=5):
    '''Print the top-n matches for a card from a find_matches() DataFrame.'''
    hits = matches[matches['card'] == card_name].head(n)
    if hits.empty:
        print(f"  '{card_name}' not found in match table.")
        return
    print(f"\nTop {n} matches for '{card_name}':")
    for _, row in hits.iterrows():
        print(f"  {row['rank']}. {row['match']:<45}  sim={row['similarity']:.4f}")


def show_candidates(card_name, models):
    '''Pretty-print the output of get_candidates() for a single card.'''
    type_line = models['_name_to_type'].get(card_name, '')
    alpha_configs = models['_alpha_configs']
    override_label = ''
    for type_key, override in models['_type_overrides'].items():
        if type_key in type_line:
            alpha_configs = override
            override_label = f' [{type_key} override]'
            break

    candidates = get_candidates(card_name, models)
    print(f"\nCandidates for '{card_name}'{override_label}:")
    for config, cards in candidates.items():
        alpha = alpha_configs[config]
        label = f"{config} (alpha={alpha})"
        print(f"  {label:<32} {cards}")


def find_matches(df, alpha=0.5, n_neighbors=5):
    '''
    For every card in df, find the n_neighbors most similar cards.
    Returns a DataFrame with columns: card, match, similarity, rank.
    (Convenience wrapper used for exploratory analysis.)
    '''
    df = df.reset_index(drop=True)
    combined, _ = build_combined_matrix(df, alpha=alpha)

    nn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric='cosine',
        algorithm='brute',
        n_jobs=-1,
    )
    nn.fit(combined)
    distances, indices = nn.kneighbors(combined)

    names = df['name'].values
    rows  = []
    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        for rank, (d, j) in enumerate(zip(dists[1:], idxs[1:]), start=1):
            rows.append({
                'card':       names[i],
                'match':      names[j],
                'similarity': round(float(1 - d), 4),
                'rank':       rank,
            })
    return pd.DataFrame(rows)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from parse_data import load_card_file, process_cards, FILENAME

    print('Loading and processing cards...')
    raw      = load_card_file(FILENAME)
    df       = process_cards(raw)
    post_c16 = df[df['released_at'] > pd.Timestamp('2016-11-11')].reset_index(drop=True)
    print(f'Post-C16 cards: {len(post_c16)}')

    SAMPLE_CARDS = [
        'Ragavan, Nimble Pilferer',
        'Unholy Heat',
        'Solitude',
        'Expressive Iteration',
        'Wrenn and Six',
    ]

    print('\nBuilding candidate models...')
    models = build_candidate_models(post_c16)

    print('\n' + '='*65)
    for card in SAMPLE_CARDS:
        show_candidates(card, models)
