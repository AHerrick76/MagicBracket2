'''
Loads card file into a pandas dataframe
'''

import os
import pandas as pd
import json
import matplotlib.pyplot as plt

FILENAME     = "default-cards-20260323090744.json"
PARQUET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cards.parquet')

def load_card_file(filename):

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    return df


def _extract_img_front(row):
    '''Return the large image URL for the front face of a card.'''
    iu = row.get('image_uris')
    if isinstance(iu, dict):
        return iu.get('large')
    cf = row.get('card_faces')
    if isinstance(cf, list) and cf:
        face0 = cf[0]
        if isinstance(face0, dict):
            return face0.get('image_uris', {}).get('large')
    return None


def _extract_img_back(row):
    '''Return the large image URL for the back face of a double-faced card, or None.'''
    cf = row.get('card_faces')
    if isinstance(cf, list) and len(cf) >= 2:
        face1 = cf[1]
        if isinstance(face1, dict):
            return face1.get('image_uris', {}).get('large')
    return None


def process_cards(df):
    """Filter and reshape raw Scryfall data into analysis-ready form."""
    # Exclude digital-only cards (includes all Alchemy rebalanced 'A-' cards)
    df = df[~df['digital']].copy()

    # Keep only original printings
    df = df[~df['reprint']].copy()

    # Exclude non-playable "extras": tokens, emblems, art cards, memorabilia items, etc.
    # These are excluded by Scryfall's default search and are not Magic cards proper.
    EXTRA_LAYOUTS = {'token', 'double_faced_token', 'emblem', 'art_series', 'reversible_card',
                     'sticker', 'planar'}   # 'planar' covers both Plane and Phenomenon cards
    df = df[~df['layout'].isin(EXTRA_LAYOUTS)]
    df = df[~df['set_type'].isin({'memorabilia', 'token'})]

    # Exclude Un-set and non-standard gimmick subtypes by type_line.
    # Schemes are Archenemy cards; Contraptions/Attractions/Stickers are Un-set cards.
    UN_SUBTYPES = ['Contraption', 'Attraction', 'Sticker', 'Scheme']
    pattern = '|'.join(UN_SUBTYPES)
    df = df[~df['type_line'].fillna('').str.contains(pattern, regex=True)]

    # Some cards have multiple non-reprint rows (e.g. a set release and a simultaneous promo,
    # or a normal-frame version alongside a showcase/borderless/extended-art version).
    # Sort so the canonical printing wins:
    #   1. Earliest release date
    #   2. Non-promo before promo
    #   3. Normal frame before alternate treatments (showcase, borderless, extended art, etc.)
    #   4. Card id as a deterministic final tie-break
    ALT_FRAME_EFFECTS = {'showcase', 'extendedart', 'borderless', 'etched',
                         'shatteredglass', 'inverted'}

    def _is_alt_frame(fe):
        return int(isinstance(fe, list) and bool(set(fe) & ALT_FRAME_EFFECTS))

    df['released_at']   = pd.to_datetime(df['released_at'])
    df['_is_promo']     = (df['set_type'] == 'promo').astype(int)
    df['_is_alt_frame'] = (
        df['frame_effects'].apply(_is_alt_frame) if 'frame_effects' in df.columns
        else 0
    )
    df = df.sort_values(['released_at', '_is_promo', '_is_alt_frame', 'id']) \
           .drop_duplicates(subset='name', keep='first')
    df = df.drop(columns=['_is_promo', '_is_alt_frame'])

    # Extract image URLs from bulk data so we don't need live Scryfall API calls at runtime.
    # Single-faced cards have image_uris at top level; double-faced cards use card_faces.
    df['img_front'] = df.apply(_extract_img_front, axis=1)
    df['img_back']  = df.apply(_extract_img_back,  axis=1)

    # Expand legalities dict into one column per format (e.g. legal_standard, legal_modern, ...)
    legalities_df = pd.json_normalize(df['legalities'].tolist())
    legalities_df.index = df.index
    legalities_df = legalities_df.add_prefix('legal_')

    # Columns requested, plus two extras flagged below
    desired_cols = [
        'id', 'name', 'released_at', 'mana_cost', 'cmc', 'type_line',
        'oracle_text', 'colors', 'color_identity', 'keywords', 'produced_mana',
        'reprint', 'set_name', 'set', 'set_type', 'rarity', 'artist', 'frame',
        'power', 'toughness', 'flavor_text', 'loyalty', 'printed_name',
        'layout',     # ADDED: card layout type (normal, transform, split, modal_dfc, etc.)
        'reserved',   # ADDED: whether the card is on the Reserved List
        'img_front',  # ADDED: large image URL for front face (extracted from bulk data)
        'img_back',   # ADDED: large image URL for back face (double-faced cards only)
    ]
    existing_cols = [col for col in desired_cols if col in df.columns]

    df = pd.concat(
        [df[existing_cols].reset_index(drop=True), legalities_df.reset_index(drop=True)],
        axis=1
    )
    return df


def load_processed_cards():
    '''
    Load the processed card DataFrame.
    Uses cards.parquet if available (fast, ~seconds); falls back to the
    raw Scryfall JSON + process_cards() if not (slow, ~minutes).
    '''
    if os.path.exists(PARQUET_PATH):
        return pd.read_parquet(PARQUET_PATH)
    print(f'cards.parquet not found — loading from {FILENAME} (this is slow)...')
    return process_cards(load_card_file(FILENAME))


def cards_per_year(df):
    """Return yearly first-printing counts and display a bar chart."""
    year_counts = (
        df.assign(year=pd.to_datetime(df['released_at']).dt.year)
          .groupby('year')
          .size()
          .rename('count')
          .reset_index()
    )

    _, ax = plt.subplots(figsize=(14, 6))
    ax.bar(year_counts['year'], year_counts['count'], color='steelblue', edgecolor='white', width=0.8)
    ax.set_xlabel('Year')
    ax.set_ylabel('Unique Cards First Printed')
    ax.set_title('Unique Magic: The Gathering Cards First Printed Per Year')
    ax.set_xticks(year_counts['year'])
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

    return year_counts




