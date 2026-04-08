"""
browse_queue.py — Generate a standalone HTML gallery for a given queue.

Opens all card images in a grid; double-faced cards show both faces on hover.

Usage:
    python browse_queue.py --queue 2
    python browse_queue.py --queue 2 --out my_gallery.html
"""

import argparse
import json
import os
import sys
import webbrowser

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_data import load_processed_cards

DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)

QUEUES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queues.json')

parser = argparse.ArgumentParser()
parser.add_argument('--queue', type=int, required=True, help='Queue ID to browse')
parser.add_argument('--out',   type=str, default=None,  help='Output HTML path (default: queue_N_gallery.html)')
args = parser.parse_args()

# ── Load queue ───────────────────────────────────────────────────────────────

with open(QUEUES_PATH) as f:
    queues_data = json.load(f)

queue = next((q for q in queues_data['queues'] if q['id'] == args.queue), None)
if queue is None:
    raise ValueError(f'Queue {args.queue} not found in {QUEUES_PATH}')

queue_cards = set(queue['cards'])
print(f'Queue {args.queue}: {len(queue_cards)} cards')

# ── Load card metadata ───────────────────────────────────────────────────────

df = load_processed_cards()
df = df[df['name'].isin(queue_cards)].copy()
df['year'] = pd.to_datetime(df['released_at']).dt.year
df = df.sort_values('released_at')

# ── Load Elo ratings from DB ─────────────────────────────────────────────────

elo_map = {}  # card_name -> (rating, games_played)
if DATABASE_URL:
    try:
        conn = psycopg2.connect(DATABASE_URL)
        elo_df = pd.read_sql(
            'SELECT card_name, rating, wins, losses FROM elo_ratings WHERE card_name = ANY(%s)',
            conn, params=(list(queue_cards),)
        )
        conn.close()
        for _, r in elo_df.iterrows():
            elo_map[r['card_name']] = (round(r['rating'], 1), int(r['wins']) + int(r['losses']))
        print(f'Elo loaded for {len(elo_map)} cards')
    except Exception as e:
        print(f'Warning: could not load Elo from DB ({e}); ratings will be omitted')
else:
    print('Warning: DATABASE_URL not set; ratings will be omitted')

# ── Build card data for template ─────────────────────────────────────────────

cards_js = []
for _, row in df.iterrows():
    front = row.get('img_front') or ''
    back  = row.get('img_back')  or ''
    rating, games = elo_map.get(row['name'], (None, 0))
    cards_js.append({
        'name':     row['name'],
        'set_name': row.get('set_name', ''),
        'year':     int(row['year']) if pd.notna(row['year']) else '',
        'front':    front,
        'back':     back,
        'elo':      rating,
        'games':    games,
    })

import json as _json
cards_json = _json.dumps(cards_js, ensure_ascii=False)

# ── HTML template ─────────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Queue {args.queue} Gallery ({len(df)} cards)</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #1a1a1a;
    color: #ddd;
    font-family: sans-serif;
    padding: 16px;
  }}
  h1 {{
    text-align: center;
    margin-bottom: 4px;
    font-size: 1.3rem;
    color: #c9a84c;
  }}
  #subtitle {{
    text-align: center;
    font-size: 0.85rem;
    color: #888;
    margin-bottom: 12px;
  }}
  #search {{
    display: block;
    margin: 0 auto 16px;
    width: min(400px, 90vw);
    padding: 7px 12px;
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 6px;
    color: #ddd;
    font-size: 0.95rem;
  }}
  #grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
  }}
  .card-wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 280px;
    cursor: default;
  }}
  .card-wrap img {{
    width: 280px;
    border-radius: 7px;
    display: block;
  }}
  .card-wrap img.dfc {{
    cursor: crosshair;
    outline: 2px solid #c9a84c;
    outline-offset: 2px;
  }}
  .card-label {{
    font-size: 0.68rem;
    color: #999;
    text-align: center;
    margin-top: 4px;
    line-height: 1.3;
    word-break: break-word;
  }}
  .card-label strong {{
    color: #ddd;
    font-size: 0.72rem;
  }}

  /* hover preview */
  #preview {{
    display: none;
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    gap: 12px;
    background: rgba(0,0,0,0.88);
    padding: 18px;
    border-radius: 14px;
    z-index: 1000;
    pointer-events: none;
  }}
  #preview.visible {{ display: flex; }}
  #preview img {{ width: 280px; border-radius: 10px; }}
</style>
</head>
<body>

<h1>Queue {args.queue} — Card Gallery</h1>
<div id="subtitle">{len(df)} cards &nbsp;·&nbsp; double-faced cards outlined in gold; hover to see both faces</div>

<input id="search" type="text" placeholder="Filter by name or set…">

<div id="grid"></div>

<div id="preview">
  <img id="prev-front" src="" alt="Front">
  <img id="prev-back"  src="" alt="Back">
</div>

<script>
const CARDS = {cards_json};

const grid    = document.getElementById('grid');
const preview = document.getElementById('preview');
const prevF   = document.getElementById('prev-front');
const prevB   = document.getElementById('prev-back');
const search  = document.getElementById('search');

function render(cards) {{
  grid.innerHTML = '';
  cards.forEach(c => {{
    const wrap = document.createElement('div');
    wrap.className = 'card-wrap';
    wrap.dataset.name = c.name.toLowerCase();
    wrap.dataset.set  = c.set_name.toLowerCase();

    const img = document.createElement('img');
    img.src   = c.front;
    img.alt   = c.name;
    img.loading = 'lazy';

    if (c.back) {{
      img.classList.add('dfc');
      img.addEventListener('mouseenter', () => {{
        prevF.src = c.front;
        prevB.src = c.back;
        preview.classList.add('visible');
      }});
      img.addEventListener('mouseleave', () => {{
        preview.classList.remove('visible');
      }});
    }}

    const label = document.createElement('div');
    label.className = 'card-label';
    const eloStr   = c.elo  != null ? `Elo ${{Math.round(c.elo)}}` : 'Elo —';
    const gameStr  = `${{c.games}} vote${{c.games !== 1 ? 's' : ''}}`;
    label.innerHTML = `<strong>${{c.name}}</strong><br>${{c.set_name}} ${{c.year ? '(' + c.year + ')' : ''}}<br>${{eloStr}} &nbsp;·&nbsp; ${{gameStr}}`;

    wrap.appendChild(img);
    wrap.appendChild(label);
    grid.appendChild(wrap);
  }});
}}

render(CARDS);

search.addEventListener('input', () => {{
  const q = search.value.toLowerCase().trim();
  if (!q) {{ render(CARDS); return; }}
  render(CARDS.filter(c =>
    c.name.toLowerCase().includes(q) || c.set_name.toLowerCase().includes(q)
  ));
}});
</script>
</body>
</html>
"""

out_path = args.out or os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f'queue_{args.queue}_gallery.html'
)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f'Saved to {out_path}')
webbrowser.open(f'file:///{out_path.replace(os.sep, "/")}')
