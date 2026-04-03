"""
vote_tempo.py — Plot vote frequency in 10-minute windows over the last 25 hours.

Usage:
    DATABASE_URL=postgresql://... python vote_tempo.py
"""

import os
from datetime import datetime, timedelta, timezone
import zoneinfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)
if not DATABASE_URL:
    raise RuntimeError('DATABASE_URL environment variable is not set.')

ET    = zoneinfo.ZoneInfo('America/New_York')
today = datetime.now(ET).date()
yesterday = today - timedelta(days=1)
start = datetime(yesterday.year, yesterday.month, yesterday.day, 17, 0, tzinfo=ET)

conn = psycopg2.connect(DATABASE_URL)
votes = pd.read_sql(
    "SELECT timestamp FROM votes WHERE timestamp >= %s",
    conn, params=(start.isoformat(),), parse_dates=['timestamp']
)
all_votes = pd.read_sql(
    "SELECT timestamp, ip_address FROM votes ORDER BY timestamp",
    conn, parse_dates=['timestamp']
)
conn.close()

print(f'Votes in window: {len(votes):,}')

if votes.empty:
    print('No votes found in this window.')
    raise SystemExit

# Ensure timezone-aware
votes['timestamp'] = pd.to_datetime(votes['timestamp'], utc=True).dt.tz_convert('America/New_York')

votes = votes.set_index('timestamp').sort_index()
counts = votes.resample('10min').size().rename('votes')

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(counts.index, counts.values, width=pd.Timedelta(minutes=9), align='edge',
       color='steelblue', alpha=0.8)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%#I%p', tz='America/New_York'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz='America/New_York'))
fig.autofmt_xdate(rotation=45)

ax.set_xlabel('Time (ET)')
ax.set_ylabel('Votes per 10-minute window')
ax.set_title(
    f'Vote tempo — {start.strftime("%Y-%m-%d %#I:%M %p")} ET to now\n'
    f'Total: {len(votes):,} votes  ·  Peak: {counts.max()} votes/window  ·  '
    f'Mean: {counts[counts > 0].mean():.1f} votes/window (active windows only)'
)

fig.tight_layout()
plt.show()

# ── Cumulative unique IPs over all time ──────────────────────────────────────

all_votes['timestamp'] = pd.to_datetime(all_votes['timestamp'], utc=True).dt.tz_convert('America/New_York')
all_votes = all_votes.dropna(subset=['ip_address']).sort_values('timestamp')

# First appearance of each IP
first_seen = all_votes.groupby('ip_address')['timestamp'].min().sort_values()
cumulative = pd.Series(range(1, len(first_seen) + 1), index=first_seen.values)

fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(cumulative.index, cumulative.values, color='steelblue', linewidth=1.8)

locator = mdates.HourLocator(interval=6, tz='America/New_York')
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz='America/New_York'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

ax2.set_xlabel('Time (ET)')
ax2.set_ylabel('Cumulative unique IPs')
ax2.set_title(
    f'Cumulative unique visitors (by IP) — all time\n'
    f'Total: {len(first_seen):,} unique IPs'
)
fig2.tight_layout()
plt.show()
