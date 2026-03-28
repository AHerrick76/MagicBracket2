'''
matchup.py — prototype matchup display.

Call show_matchup() to display two cards side by side:
  - Card A: chosen at random from post-Commander 2016 cards
  - Card B: chosen from get_candidates(Card A), with a weighted random pick

The models and card data are initialised once on first call and cached
in module-level globals, so subsequent calls are fast.
'''

import io
import sys
import time
import random
from urllib.parse import quote

import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image

sys.path.insert(0, '.')
from parse_data import load_card_file, process_cards, FILENAME
from similarity import build_candidate_models, get_candidates


# ── Config ────────────────────────────────────────────────────────────────────

C16_DATE = pd.Timestamp('2016-11-11')

# Weights applied to the 5-card candidate list (must sum to 1)
CANDIDATE_WEIGHTS = [0.45, 0.30, 0.15, 0.07, 0.03]

# Scryfall asks for a small delay between requests
_SCRYFALL_DELAY = 0.1   # seconds


# ── Module-level cache (lazy-initialised on first show_matchup() call) ────────

_post_c16    = None
_models      = None
_image_cache = {}


def _init():
    global _post_c16, _models
    if _models is not None:
        return
    print('Initialising data and models (first call only)...')
    raw      = load_card_file(FILENAME)
    df       = process_cards(raw)
    _post_c16 = df[df['released_at'] > C16_DATE].reset_index(drop=True)
    _models   = build_candidate_models(_post_c16)
    print(f'Ready. {len(_post_c16)} post-C16 cards loaded.')


def _fetch_card_image(card_name):
    '''
    Return a PIL Image for card_name, using an in-memory cache.
    Fetches card metadata from Scryfall to resolve the image URL,
    then fetches the image itself.
    '''
    if card_name in _image_cache:
        return _image_cache[card_name]

    # Step 1: resolve image URL from card metadata
    meta_url = f'https://api.scryfall.com/cards/named?exact={quote(card_name)}'
    time.sleep(_SCRYFALL_DELAY)
    meta = requests.get(meta_url, timeout=10)
    meta.raise_for_status()
    card_data = meta.json()

    if 'image_uris' in card_data:
        img_url = card_data['image_uris']['normal']
    elif 'card_faces' in card_data:
        # DFCs store image_uris per face; use the front face
        img_url = card_data['card_faces'][0]['image_uris']['normal']
    else:
        raise ValueError(f"No image URI found for '{card_name}'")

    # Step 2: download the image
    time.sleep(_SCRYFALL_DELAY)
    img_resp = requests.get(img_url, timeout=10)
    img_resp.raise_for_status()

    img = Image.open(io.BytesIO(img_resp.content))
    _image_cache[card_name] = img
    return img


# ── Main function ─────────────────────────────────────────────────────────────

def show_matchup():
    '''
    Display two cards side by side and return their names and the
    similarity config that produced the pairing.

    Card A is picked uniformly at random from post-C16 cards.
    Card B is picked from one of the three candidate lists returned
    by get_candidates(), with position-weighted sampling:
        rank 1: 45%, rank 2: 30%, rank 3: 15%, rank 4: 7%, rank 5: 3%
    The config list (text_heavy / balanced / struct_heavy) is chosen
    uniformly at random before the weighted pick.

    Returns
    -------
    (card_a_name, card_b_name, config_name) — str, str, str
    '''
    _init()

    # ── Pick Card A ───────────────────────────────────────────────────────────
    card_a = _post_c16['name'].sample(1).iloc[0]

    # ── Pick Card B ───────────────────────────────────────────────────────────
    candidates  = get_candidates(card_a, _models)
    config_name = random.choice(list(candidates.keys()))
    card_list   = candidates[config_name]
    card_b      = random.choices(card_list, weights=CANDIDATE_WEIGHTS, k=1)[0]

    print(f'Matchup: {card_a!r}  vs  {card_b!r}  (source: {config_name})')

    # ── Fetch images ──────────────────────────────────────────────────────────
    img_a = _fetch_card_image(card_a)
    img_b = _fetch_card_image(card_b)

    # ── Display ───────────────────────────────────────────────────────────────
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 8))

    ax_a.imshow(img_a)
    ax_a.set_title(card_a, fontsize=12, pad=8)
    ax_a.axis('off')

    ax_b.imshow(img_b)
    ax_b.set_title(card_b, fontsize=12, pad=8)
    ax_b.axis('off')

    fig.suptitle(f'Similarity source: {config_name}', fontsize=9, color='grey', y=0.02)
    plt.tight_layout()
    plt.show()

    return card_a, card_b, config_name
