'''
One-time preprocessing step: converts the Scryfall bulk JSON into cards.parquet.

Run whenever you download a new bulk data file:
    python preprocess.py

Output: cards.parquet (~15 MB) in the same directory.
This file is committed to git and used by app.py and mapgen.py at startup,
replacing the slow JSON load + process_cards() pipeline (~60s → ~3s).
'''

from parse_data import load_card_file, process_cards, FILENAME, PARQUET_PATH

print(f'Loading {FILENAME}...')
raw = load_card_file(FILENAME)

print('Processing cards...')
df = process_cards(raw)

print(f'Saving to {PARQUET_PATH}...')
df.to_parquet(PARQUET_PATH, index=False)

print(f'Done. {len(df):,} cards saved.')
