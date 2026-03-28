"""
Investigates the discrepancy between our 35,383 unique card count
and Scryfall's web search count of 32,226.
"""

import sys
sys.path.insert(0, r"c:\Users\atnhe\Documents\Scripting\Card Similarity")

from parse_data import load_card_file, process_cards, FILENAME
import pandas as pd

print("Loading raw data...")
raw = load_card_file(FILENAME)
print(f"Raw rows: {len(raw)}")

# ── Step 1: what does process_cards give us? ──────────────────────────────────
processed = process_cards(raw)
print(f"\nAfter process_cards (our number): {len(processed)}")

# ── Step 2: profile the raw non-digital, non-reprint population ───────────────
base = raw[~raw['digital'] & ~raw['reprint']].copy()
base['released_at'] = pd.to_datetime(base['released_at'])
base = base.sort_values(['released_at', 'id']).drop_duplicates(subset='name', keep='first')
print(f"\nBase (non-digital, non-reprint, deduped by name): {len(base)}")

# ── Step 3: break down by layout ──────────────────────────────────────────────
print("\n--- layout value_counts ---")
print(base['layout'].value_counts().to_string())

# ── Step 4: break down by set_type ───────────────────────────────────────────
print("\n--- set_type value_counts ---")
print(base['set_type'].value_counts().to_string())

# ── Step 5: Scryfall web search excludes 'extras' by default ─────────────────
# 'Extras' = tokens, art_series, double_faced_token, emblem, and
# set_types like 'memorabilia'. Let's filter those out and see where we land.

EXTRA_LAYOUTS = {'token', 'double_faced_token', 'emblem', 'art_series',
                 'reversible_card'}
EXTRA_SET_TYPES = {'memorabilia'}

# Layouts considered 'extra'
mask_extra_layout = base['layout'].isin(EXTRA_LAYOUTS)
mask_extra_set    = base['set_type'].isin(EXTRA_SET_TYPES)

print(f"\nRows with extra layout:   {mask_extra_layout.sum()}")
print(f"Rows with extra set_type: {mask_extra_set.sum()}")
print(f"Rows with either:         {(mask_extra_layout | mask_extra_set).sum()}")

no_extras = base[~mask_extra_layout & ~mask_extra_set]
print(f"\nAfter removing extras: {len(no_extras)}")

# ── Step 6: check for other unusual set_types or layouts in the remainder ─────
print("\n--- Remaining layout breakdown (no extras) ---")
print(no_extras['layout'].value_counts().to_string())

print("\n--- Remaining set_type breakdown (no extras) ---")
print(no_extras['set_type'].value_counts().to_string())

# ── Step 7: try also excluding 'minigame' layout if it exists ─────────────────
if 'minigame' in no_extras['layout'].values:
    no_extras2 = no_extras[no_extras['layout'] != 'minigame']
    print(f"\nAfter also removing minigame layout: {len(no_extras2)}")

# ── Step 8: show some example 'extra' cards to sanity-check ──────────────────
print("\n--- Sample rows removed as extras (layout) ---")
print(base[mask_extra_layout][['name', 'layout', 'set_type', 'set_name']].head(10).to_string())

print("\n--- Sample rows removed as extras (set_type) ---")
print(base[mask_extra_set][['name', 'layout', 'set_type', 'set_name']].head(10).to_string())
