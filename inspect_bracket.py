'''
Load bracket_votes and bracket_results from the database into pandas DataFrames.

Usage:
    DATABASE_URL=postgresql://... python inspect_bracket.py
'''

import json
import os
import sys

import pandas as pd
import psycopg2
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://', 1)
if not DATABASE_URL:
    raise RuntimeError('DATABASE_URL environment variable is not set.')

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
BRACKET_JSON = os.path.join(BASE_DIR, 'bracket.json')

conn = psycopg2.connect(DATABASE_URL)

votes     = pd.read_sql('SELECT * FROM bracket_votes         ORDER BY id',         conn)
results   = pd.read_sql('SELECT * FROM bracket_results     ORDER BY matchup_id', conn)
favorites = pd.read_sql('SELECT * FROM finals_favorite_cards ORDER BY id',        conn)

# Load only page_views that fall within the bracket's own timestamp range.
# This avoids pulling in views from earlier phases or future events.
if not votes.empty:
    _ts = pd.to_datetime(votes['timestamp'])
    _pv_start = _ts.min().floor('D')          # start of the first bracket day
    _pv_end   = _ts.max().ceil('D')           # end of the last bracket day
    page_views = pd.read_sql(
        'SELECT * FROM page_views WHERE timestamp::timestamptz >= %s AND timestamp::timestamptz <= %s ORDER BY id',
        conn, params=(_pv_start, _pv_end),
    )
else:
    page_views = pd.DataFrame()

conn.close()

# ── Merge bracket.json context onto votes and results ──────────────────────────

_bracket      = None
_matchup_by_id = {}

if os.path.exists(BRACKET_JSON):
    with open(BRACKET_JSON, encoding='utf-8') as _f:
        _bracket = json.load(_f)
    _matchup_by_id = {m['id']: m for m in _bracket['matchups']}

ROUND_LABELS = {1: 'R64', 2: 'R32', 3: 'R16', 4: 'QF', 5: 'SF', 6: 'F'}

if _matchup_by_id and len(votes):
    votes['round_label'] = votes['round'].map(ROUND_LABELS)

if _matchup_by_id and len(results):
    results['round_label'] = results['round'].map(ROUND_LABELS)

# ── Summary ────────────────────────────────────────────────────────────────────

print(f'bracket_votes:        {len(votes)} rows')
print(f'bracket_results:      {len(results)} rows')
print(f'finals_favorite_cards: {len(favorites)} rows')

if len(votes):
    unique_ballots = votes['ballot_id'].nunique()
    unique_ips     = votes['ip_address'].nunique()
    print(f'\nUnique ballots:  {unique_ballots}')
    print(f'Unique IPs:      {unique_ips}')

    by_day = votes.groupby('day').agg(
        ballots=('ballot_id', 'nunique'),
        individual_votes=('id', 'count'),
        unique_ips=('ip_address', 'nunique'),
    ).reset_index()
    print(f'\nVotes by day:\n{by_day.to_string(index=False)}')

# if len(results):
#     print(f'\nResults so far:\n'
#           + results[['matchup_id', 'round_label', 'day', 'bracket_date',
#                       'card_a', 'card_b', 'votes_a', 'votes_b', 'winner']]
#           .to_string(index=False))

# temp_results is built after the helper functions are defined (see below)


# ── Helper functions ───────────────────────────────────────────────────────────

def votes_per_matchup(votes=votes):
    '''
    Return vote counts per matchup, merged with card names.

    Returns
    -------
    pd.DataFrame
        Columns: matchup_id, round, day, card_a, card_b, votes_a, votes_b, total_votes.
        Sorted by matchup_id.
    '''
    if votes.empty:
        return pd.DataFrame()

    v = votes.copy()
    v['is_a'] = (v['chosen'] == v['card_a']).astype(int)
    v['is_b'] = (v['chosen'] == v['card_b']).astype(int)
    tally = (
        v.groupby(['matchup_id', 'round', 'day', 'card_a', 'card_b'])
        .agg(votes_a=('is_a', 'sum'), votes_b=('is_b', 'sum'))
        .reset_index()
    )
    tally['total_votes'] = tally['votes_a'] + tally['votes_b']
    return tally.sort_values('matchup_id').reset_index(drop=True)


def ballot_summary(votes=votes):
    '''
    One row per ballot: how many matchups were voted on and which IP/device submitted it.

    Returns
    -------
    pd.DataFrame
        Columns: ballot_id, ip_address, device, day, timestamp, matchups_voted.
    '''
    if votes.empty:
        return pd.DataFrame()

    return (
        votes.groupby('ballot_id').agg(
            ip_address=('ip_address', 'first'),
            device=('device', 'first'),
            day=('day', 'first'),
            timestamp=('timestamp', 'first'),
            matchups_voted=('matchup_id', 'count'),
        )
        .reset_index()
        .sort_values('timestamp')
        .reset_index(drop=True)
    )


def votes_per_ip(votes=votes):
    '''
    Return the number of ballots submitted per IP address, useful for spotting multi-voters.

    Returns
    -------
    pd.DataFrame
        Columns: ip_address, ballots, days_voted. Sorted by ballots descending.
    '''
    if votes.empty:
        return pd.DataFrame()

    return (
        votes.groupby('ip_address').agg(
            ballots=('ballot_id', 'nunique'),
            days_voted=('day', 'nunique'),
        )
        .reset_index()
        .sort_values('ballots', ascending=False)
        .reset_index(drop=True)
    )


def temp_results_df(votes=votes, results=results):
    '''
    Return a bracket_results-shaped DataFrame for matchups that are currently
    in progress (i.e. have votes but no finalised result yet), using the current
    vote tallies to determine a provisional winner.

    Returns
    -------
    pd.DataFrame
        Same columns as bracket_results: matchup_id, round, round_label, day,
        card_a, card_b, votes_a, votes_b, winner.  winner is None when tied.
        card_a and card_b include the seed in parentheses, e.g. "Ragavan (11)".
        Sorted by matchup_id.
    '''
    tally = votes_per_matchup(votes)
    if tally.empty:
        return pd.DataFrame()

    concluded_ids = set(results['matchup_id']) if not results.empty else set()
    pending = tally[~tally['matchup_id'].isin(concluded_ids)].copy()
    if pending.empty:
        return pd.DataFrame()

    # Build name -> seed lookup from bracket.json matchups
    _seed = {}
    for m in _matchup_by_id.values():
        if 'seed_a' in m:
            _seed[m['name_a']] = m['seed_a']
            _seed[m['name_b']] = m['seed_b']

    def _with_seed(name):
        s = _seed.get(name)
        return f'{name} ({s})' if s is not None else name

    # Attach raw seed numbers before renaming cards (needed for upset calc)
    pending['seed_a'] = pending['card_a'].map(lambda n: _seed.get(n))
    pending['seed_b'] = pending['card_b'].map(lambda n: _seed.get(n))

    pending['card_a'] = pending['card_a'].map(_with_seed)
    pending['card_b'] = pending['card_b'].map(_with_seed)

    def _winner(row):
        if row['votes_a'] > row['votes_b']:
            return row['card_a']
        if row['votes_b'] > row['votes_a']:
            return row['card_b']
        return None  # tied

    def _upset(row):
        if row['votes_a'] == row['votes_b']:
            return -1  # tied
        a_winning = row['votes_a'] > row['votes_b']
        # lower seed number = higher seed; upset if lower seed loses
        a_is_higher_seed = row['seed_a'] < row['seed_b']
        higher_seed_winning = a_winning == a_is_higher_seed
        return 0 if higher_seed_winning else 1

    pending['winner']      = pending.apply(_winner, axis=1)
    pending['upset']       = pending.apply(_upset, axis=1)
    pending['round_label'] = pending['round'].map(ROUND_LABELS)

    total = pending['votes_a'] + pending['votes_b']
    pending['winning_margin'] = (
        (pending[['votes_a', 'votes_b']].max(axis=1) / total * 100)
        .where(total > 0)
        .round(2)
    )

    cols = ['matchup_id', 'round', 'round_label', 'day',
            'card_a', 'card_b', 'votes_a', 'votes_b', 'winner', 'upset', 'winning_margin']
    return pending[cols].sort_values('matchup_id').reset_index(drop=True)


temp_results = temp_results_df()


def matchup_margin(results=results):
    '''
    Return each completed matchup with its vote margin and percentage.

    Returns
    -------
    pd.DataFrame
        Columns: matchup_id, round_label, card_a, card_b, votes_a, votes_b,
                 total, winner, margin, pct_winner. Sorted by pct_winner descending.
    '''
    if results.empty:
        return pd.DataFrame()

    df = results.copy()
    df['total']      = df['votes_a'] + df['votes_b']
    df['margin']     = (df['votes_a'] - df['votes_b']).abs()
    df['pct_winner'] = df.apply(
        lambda r: round(100 * max(r['votes_a'], r['votes_b']) / r['total'], 1)
        if r['total'] else None, axis=1
    )
    cols = ['matchup_id', 'round_label', 'card_a', 'card_b',
            'votes_a', 'votes_b', 'total', 'winner', 'margin', 'pct_winner']
    return df[cols].sort_values('pct_winner', ascending=False).reset_index(drop=True)


# if not temp_results.empty:
#     print(f'\nTemp results (current round if concluded now):\n'
#           + temp_results.to_string(index=False))


def top64_entry_stats():
    '''
    Return one row per top-64 bracket card with their entry statistics.

    Pulls from:
      - bracket.json          → seed and card name
      - top_10_queue.json     → normalized Elo at bracket entry
      - queues.json           → full queue membership and size
      - elo_ratings (DB)      → Elo used to rank cards within their queue

    Returns
    -------
    pd.DataFrame
        Columns: seed, name, normalized_elo, queue_rank
        where queue_rank is a string like "11 / 750", ordered by seed.
    '''
    TOP10_JSON  = os.path.join(BASE_DIR, 'top_10_queue.json')
    QUEUES_JSON = os.path.join(BASE_DIR, 'queues.json')

    with open(BRACKET_JSON, encoding='utf-8') as f:
        bracket_data = json.load(f)
    with open(TOP10_JSON, encoding='utf-8') as f:
        top10_data = json.load(f)
    with open(QUEUES_JSON, encoding='utf-8') as f:
        queues_data = json.load(f)

    # seed and name from bracket
    bracket_cards = {c['name']: c['seed'] for c in bracket_data['cards']}

    # normalized entry Elo from top_10_queue.json
    entry_elo = {c['name']: c['normalized_elo'] for c in top10_data['cards']}

    # full queue membership from queues.json
    queue_members = {}  # card_name → list of all card names in its queue
    queue_sizes   = {}  # card_name → int
    for q in queues_data['queues']:
        for name in q['cards']:
            queue_members[name] = q['cards']
            queue_sizes[name]   = len(q['cards'])

    # current Elo ratings for ranking within queue
    _conn = psycopg2.connect(DATABASE_URL)
    elo_df = pd.read_sql('SELECT card_name, rating FROM elo_ratings', _conn)
    _conn.close()
    elo_map = dict(zip(elo_df['card_name'], elo_df['rating'].astype(float)))

    rows = []
    for name, seed in bracket_cards.items():
        members = queue_members.get(name, [])
        # rank 1 = highest Elo; use 1500 for any card with no recorded rating
        ranked = sorted(members, key=lambda c: elo_map.get(c, 1500.0), reverse=True)
        rank = ranked.index(name) + 1 if name in ranked else None
        size = queue_sizes.get(name)
        queue_rank = f'{rank} / {size}' if rank is not None else 'N/A'
        rows.append({
            'seed':           seed,
            'name':           name,
            'normalized_elo': entry_elo.get(name),
            'queue_rank':     queue_rank,
        })

    df = pd.DataFrame(rows).sort_values('seed').reset_index(drop=True)
    return df


# ── Integrity / fraud checks ──────────────────────────────────────────────────

# Populated by vote_share_timeline_interactive on each redraw.
# matchup_id -> list of (x0, x1, z) sorted by |z| descending.
# x0/x1 are 1-based vote-count positions matching the plot's x-axis.
_flagged_chunks: dict = {}


def duplicate_ballots(votes=votes, min_shared_matchups=5):
    '''
    Identify ballots whose complete vote sequence is identical to at least one
    other ballot.

    Two independent voters agreeing by chance on N binary matchups has
    probability 0.5^N — roughly 1-in-4-billion for a 32-matchup ballot — so
    any genuine duplicates deserve scrutiny.

    Parameters
    ----------
    min_shared_matchups : int
        Ignore ballots that voted on fewer than this many matchups; very short
        partial ballots can match by chance.

    Returns
    -------
    pd.DataFrame
        Columns: ballot_id, ip_address, day, matchups_voted, duplicate_count.
        Only ballots sharing a fingerprint with at least one other are returned,
        sorted by duplicate_count desc, then ballot_id.
    '''
    if votes.empty:
        return pd.DataFrame()

    grp = votes.sort_values(['ballot_id', 'matchup_id'])
    fps = (
        grp.groupby('ballot_id')
        .apply(lambda g: tuple(zip(g['matchup_id'].tolist(), g['chosen'].tolist())))
        .reset_index()
    )
    fps.columns = ['ballot_id', 'fingerprint']
    fps['matchups_voted'] = fps['fingerprint'].map(len)
    fps = fps[fps['matchups_voted'] >= min_shared_matchups]

    fp_counts = fps['fingerprint'].value_counts()
    fps['duplicate_count'] = fps['fingerprint'].map(fp_counts)
    dupes = fps[fps['duplicate_count'] > 1].copy()

    if dupes.empty:
        print('No duplicate ballots found.')
        return pd.DataFrame()

    meta = (
        votes.groupby('ballot_id')
        .agg(ip_address=('ip_address', 'first'), day=('day', 'first'))
        .reset_index()
    )
    dupes = dupes.merge(meta, on='ballot_id')
    cols  = ['ballot_id', 'ip_address', 'day', 'matchups_voted', 'duplicate_count']
    return (
        dupes[cols]
        .sort_values(['duplicate_count', 'ballot_id'], ascending=[False, True])
        .reset_index(drop=True)
    )


def vote_share_timeline(day, matchup_id=None, votes=votes, smoothing=100,
                        rolling=False, x_start=100):
    '''
    Plot the running vote-share for card_a over time (by vote-arrival order)
    for every matchup on a given day, all on one axes.

    Two modes:
      rolling=False (default) — expanding (cumulative) mean.  Naturally
          stabilises as votes accumulate, which can mask late-stage stuffing.
      rolling=True — pure rolling window of width `smoothing`.  Every block of
          votes has equal weight regardless of when it arrives, so manipulation
          at vote #3000 is just as visible as at vote #300.

    Results before ~100 votes are noisy, so the x-axis starts at x_start.
    A stable line is normal; a sudden shift is a red flag.

    Parameters
    ----------
    day : str or int
        Filter to this day value (matches the `day` column).
    matchup_id : int or None
        If given, plot only that matchup instead of all matchups on the day.
    smoothing : int
        Window size in votes.  In rolling mode this is the only smoothing
        applied; in cumulative mode it is a light cosmetic smooth on top of
        the expanding mean (default 100).
    rolling : bool
        If True, use a rolling window instead of the cumulative mean.
    x_start : int
        First vote number shown on the x-axis; earlier votes are computed but
        not displayed (default 100).

    Shows a matplotlib figure; does not return a value.
    '''
    if votes.empty:
        return

    import matplotlib.pyplot as plt

    subset = votes[votes['day'] == day]
    if matchup_id is not None:
        subset = subset[subset['matchup_id'] == matchup_id]
    mids = sorted(subset['matchup_id'].unique())
    if not mids:
        print(f'No matchups found for day={day!r}.')
        return

    _, ax = plt.subplots(figsize=(12, 5))

    for mid in mids:
        grp    = subset[subset['matchup_id'] == mid].sort_values('id')
        card_a = grp['card_a'].iloc[0]
        card_b = grp['card_b'].iloc[0]

        is_a = (grp['chosen'] == grp['card_a']).astype(float).reset_index(drop=True)

        if rolling:
            # Pure rolling window — constant sensitivity throughout the poll
            smoothed = is_a.rolling(smoothing, min_periods=smoothing).mean() * 100
        else:
            # Expanding (cumulative) mean, with light cosmetic smoothing
            cumulative = is_a.expanding().mean() * 100
            smoothed   = cumulative.rolling(smoothing, min_periods=1).mean()

        # x is 1-based vote count; only plot from x_start onward
        x    = smoothed.index + 1
        mask = x >= x_start
        ax.plot(x[mask], smoothed[mask], linewidth=1.2, label=f'#{mid} {card_a} vs {card_b}')

    ax.axhline(50, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_ylim(25, 85)
    ax.set_xlabel('Vote # (within matchup)', fontsize=9)
    mode_label = f'rolling {smoothing}-vote window' if rolling else 'cumulative mean'
    ax.set_ylabel(f'% for card_a ({mode_label})', fontsize=9)

    round_label = subset['round_label'].iloc[0] if 'round_label' in subset.columns else ''
    title_parts = [f'Vote-share timeline — day {day}']
    if round_label:
        title_parts.append(round_label)
    ax.set_title(' · '.join(title_parts), fontsize=11)

    ax.legend(fontsize=7, loc='lower right')
    plt.tight_layout()
    plt.show()


def vote_share_timeline_interactive(day, matchup_id=None, votes=votes,
                                    init_smoothing=200, init_chunk=100,
                                    alpha=0.05, x_start=100):
    '''
    Interactive version of vote_share_timeline with two sliders.

    Smoothing slider  — rolling window width for the vote-share lines.
    Chunk slider      — size of non-overlapping windows used for the
                        Bonferroni-corrected binomial suspicion test.

    Suspicious chunks are highlighted in the matching line colour.  The
    annotation shows the total chunk count and the z-score threshold so
    you can judge whether flagged windows are genuinely surprising or are
    expected false positives.

    Suspicion test
    --------------
    For each matchup, votes are divided into non-overlapping chunks of
    `chunk_size`.  Under H0 the chunk vote-share ~ Normal(p̂, SE) where
    p̂ is the matchup's overall rate and SE = √(p̂(1-p̂)/chunk_size).
    z_crit is set via a Bonferroni correction across ALL chunks on the
    day, so the family-wise false-positive rate is held at `alpha`.

    Parameters
    ----------
    day : str or int
    matchup_id : int or None
    init_smoothing : int   Starting smoothing window (default 200).
    init_chunk     : int   Starting chunk size (default 100).
    alpha          : float Family-wise significance level (default 0.05).
    x_start        : int   First vote shown on x-axis (default 100).
    '''
    try:
        from scipy.stats import norm as _norm
        _ppf = _norm.ppf
    except ImportError:
        # Fallback table for common alpha values if scipy unavailable
        _ppf = lambda p: {0.9995: 3.29, 0.999: 3.09, 0.995: 2.81,
                          0.99: 2.58, 0.975: 1.96}.get(round(p, 4), 3.0)

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    if votes.empty:
        return

    subset = votes[votes['day'] == day]
    if matchup_id is not None:
        subset = subset[subset['matchup_id'] == matchup_id]
    mids = sorted(subset['matchup_id'].unique())
    if not mids:
        print(f'No matchups found for day={day!r}.')
        return

    # Pre-sort once; store as plain arrays for speed inside redraw()
    matchup_data = {}
    for mid in mids:
        grp = subset[subset['matchup_id'] == mid].sort_values('id')
        is_a = (grp['chosen'] == grp['card_a']).astype(float).reset_index(drop=True)

        ts_raw = pd.to_datetime(grp['timestamp']).reset_index(drop=True)
        try:
            ts_est = ts_raw.dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        except TypeError:
            ts_est = ts_raw.dt.tz_convert('America/New_York')  # already tz-aware

        matchup_data[mid] = {
            'is_a':       is_a,
            'timestamps': ts_est,
            'p_hat':      float(is_a.mean()),
            'card_a':     grp['card_a'].iloc[0],
            'card_b':     grp['card_b'].iloc[0],
        }

    def _fmt_time(ts):
        '''Cross-platform 12-hour time without a leading zero, e.g. "3:07 PM".'''
        h = ts.hour % 12 or 12
        return f'{h}:{ts.strftime("%M %p")}'

    colors    = plt.rcParams['axes.prop_cycle'].by_key()['color']
    mid_color = {mid: colors[i % len(colors)] for i, mid in enumerate(mids)}

    round_label = subset['round_label'].iloc[0] if 'round_label' in subset.columns else ''
    title_str   = ' · '.join(filter(None, [f'Vote-share timeline — day {day}', round_label]))

    fig, ax = plt.subplots(figsize=(13, 6))
    plt.subplots_adjust(bottom=0.20)

    ax_s_smooth = fig.add_axes([0.12, 0.10, 0.60, 0.03])
    ax_s_chunk  = fig.add_axes([0.12, 0.05, 0.60, 0.03])
    s_smooth = Slider(ax_s_smooth, 'Smoothing', 10,  600, valinit=init_smoothing, valstep=10)
    s_chunk  = Slider(ax_s_chunk,  'Chunk',     20,  600, valinit=init_chunk,     valstep=10)

    def redraw(*_):
        smoothing  = int(s_smooth.val)
        chunk_size = int(s_chunk.val)

        ax.cla()
        ax.set_ylim(25, 85)
        ax.axhline(50, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Vote # (within matchup)', fontsize=9)
        ax.set_ylabel(f'% for card_a (rolling {smoothing}-vote window)', fontsize=9)
        ax.set_title(title_str, fontsize=11)

        # Total complete chunks across all matchups → Bonferroni denominator
        n_chunks_total = sum(len(d['is_a']) // chunk_size for d in matchup_data.values())
        z_crit = _ppf(1 - alpha / (2 * max(n_chunks_total, 1)))

        n_flagged = 0
        _flagged_chunks.clear()
        for mid in mids:
            d      = matchup_data[mid]
            is_a   = d['is_a']
            p_hat  = d['p_hat']
            color  = mid_color[mid]
            se     = (p_hat * (1 - p_hat) / chunk_size) ** 0.5

            # Rolling vote-share line
            smoothed = is_a.rolling(smoothing, min_periods=smoothing).mean() * 100
            x        = smoothed.index + 1
            mask     = x >= x_start
            ax.plot(x[mask], smoothed[mask], linewidth=1.2, color=color,
                    label=f'#{mid} {d["card_a"]} vs {d["card_b"]}')

            # Chunk suspicion test — highlight flagged windows
            flagged_for_mid = []
            n_complete = len(is_a) // chunk_size
            for k in range(n_complete):
                chunk_share = is_a.iloc[k * chunk_size:(k + 1) * chunk_size].mean()
                z = (chunk_share - p_hat) / se if se > 0 else 0.0
                if abs(z) >= z_crit:
                    x0 = k * chunk_size + 1
                    x1 = (k + 1) * chunk_size
                    flagged_for_mid.append((x0, x1, round(z, 2)))
                    ax.axvspan(x0, x1, alpha=0.25, color=color, linewidth=0)

                    # Time range label at the bottom of the span
                    t0 = d['timestamps'].iloc[k * chunk_size]
                    t1 = d['timestamps'].iloc[(k + 1) * chunk_size - 1]
                    time_str = f'{_fmt_time(t0)}–{_fmt_time(t1)}'
                    ax.text(
                        (x0 + x1) / 2, 0.01, time_str,
                        transform=ax.get_xaxis_transform(),
                        ha='center', va='bottom', fontsize=6,
                        rotation=90, color=color,
                    )
                    n_flagged += 1

            if flagged_for_mid:
                _flagged_chunks[mid] = sorted(
                    flagged_for_mid, key=lambda t: abs(t[2]), reverse=True
                )

        ax.text(
            0.02, 0.98,
            (f'Chunks: {n_chunks_total}   '
             f'α={alpha:.2f}   '
             f'z_crit={z_crit:.2f}   '
             f'Flagged: {n_flagged}   '
             f'Expected false positives: {alpha:.2f}'),
            transform=ax.transAxes, va='top', ha='left', fontsize=7.5,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
        )

        ax.legend(fontsize=7, loc='lower right')
        fig.canvas.draw_idle()

    s_smooth.on_changed(redraw)
    s_chunk.on_changed(redraw)
    redraw()
    plt.show()


def investigate_flagged_chunk(matchup_id, chunk_index=0, votes=votes):
    '''
    Investigate a chunk flagged by the most recent vote_share_timeline_interactive
    run, without any manual slice arithmetic.

    Reads chunk positions from _flagged_chunks (populated on each slider redraw),
    extracts the ballot_ids from those vote rows, and passes them to
    suspicious_window_bias for cross-matchup analysis.

    Parameters
    ----------
    matchup_id : int
        The matchup to investigate.  Must have at least one flagged chunk
        from the last interactive plot.
    chunk_index : int or 'all'
        Which flagged chunk to use, ordered by |z| descending so 0 is always
        the most suspicious.  Pass 'all' to pool ballot_ids from every flagged
        chunk for this matchup.
    votes : pd.DataFrame

    Returns
    -------
    pd.DataFrame from suspicious_window_bias, or an empty DataFrame on error.
    '''
    if matchup_id not in _flagged_chunks or not _flagged_chunks[matchup_id]:
        print(f'No flagged chunks for matchup {matchup_id}. '
              f'Run vote_share_timeline_interactive first.')
        return pd.DataFrame()

    chunks = _flagged_chunks[matchup_id]   # sorted by |z| desc
    grp    = votes[votes['matchup_id'] == matchup_id].sort_values('id').reset_index(drop=True)

    if chunk_index == 'all':
        ballot_ids = set()
        for x0, x1, z in chunks:
            ballot_ids.update(grp.iloc[x0 - 1 : x1]['ballot_id'].tolist())
        print(f'Matchup {matchup_id}: pooling {len(chunks)} flagged chunk(s), '
              f'{len(ballot_ids)} unique ballots.')
    else:
        if chunk_index >= len(chunks):
            print(f'chunk_index {chunk_index} out of range — '
                  f'only {len(chunks)} flagged chunk(s) for matchup {matchup_id}.')
            return pd.DataFrame()
        x0, x1, z = chunks[chunk_index]
        ballot_ids = set(grp.iloc[x0 - 1 : x1]['ballot_id'].tolist())
        print(f'Matchup {matchup_id}, chunk {chunk_index}: '
              f'votes {x0}–{x1}  z={z:+.2f}  ({len(ballot_ids)} ballots)')

    return suspicious_window_bias(ballot_ids, target_matchup_id=matchup_id, votes=votes)


def ballot_speed(votes=votes, page_views=page_views, fast_threshold_seconds=60):
    '''
    Measure how quickly each ballot was completed, using page_views to anchor
    the start time.

    Three durations are computed:
      page_to_first  — seconds from the voter's last page load before their
                       first vote to that first vote.  Measures how long they
                       spent reading the ballot before starting.
      first_to_last  — seconds from first vote to last vote.  Measures how
                       fast they clicked through the matchups.
      total          — page_to_first + first_to_last (full session length).

    The join uses the latest page_views row for that IP whose timestamp is
    <= the ballot's first vote (i.e. the page load that actually preceded
    this session, ignoring any later revisits).  Ballots with no matching
    page view get NaN for page_to_first and total.

    Parameters
    ----------
    fast_threshold_seconds : int
        Ballots whose first_to_last duration falls below this are flagged
        is_fast (default 60).

    Returns
    -------
    pd.DataFrame
        Columns: ballot_id, ip_address, day, matchups_voted,
                 page_load, first_vote, last_vote,
                 page_to_first, first_to_last, total, is_fast.
        Sorted by first_to_last ascending (fastest first).
    '''
    if votes.empty:
        return pd.DataFrame()

    ts = votes.copy()
    ts['timestamp'] = pd.to_datetime(ts['timestamp'])

    agg = (
        ts.groupby('ballot_id')
        .agg(
            ip_address    =('ip_address', 'first'),
            day           =('day',        'first'),
            matchups_voted=('matchup_id', 'count'),
            first_vote    =('timestamp',  'min'),
            last_vote     =('timestamp',  'max'),
        )
        .reset_index()
    )

    # Join page_views: for each ballot IP, find the latest page load <= first_vote
    if not page_views.empty:
        pv = page_views.copy()
        pv['timestamp'] = pd.to_datetime(pv['timestamp'])

        # Narrow page_views to the time window spanned by these ballots before
        # the cross-join; avoids matching views from unrelated days when
        # ballot_speed is called on a filtered votes slice (e.g. one day only).
        pv = pv[
            (pv['timestamp'] >= agg['first_vote'].min().floor('D')) &
            (pv['timestamp'] <= agg['first_vote'].max().ceil('D'))
        ]

        # Cross-join ballot meta with page views on IP, then filter and take max
        merged = agg[['ballot_id', 'ip_address', 'first_vote']].merge(
            pv[['ip_address', 'timestamp']].rename(columns={'timestamp': 'page_load'}),
            on='ip_address',
            how='left',
        )
        merged = merged[merged['page_load'] <= merged['first_vote']]
        latest_load = (
            merged.groupby('ballot_id')['page_load'].max().reset_index()
        )
        agg = agg.merge(latest_load, on='ballot_id', how='left')
    else:
        agg['page_load'] = pd.NaT

    agg['page_to_first'] = (agg['first_vote'] - agg['page_load']).dt.total_seconds().round(1)
    agg['first_to_last'] = (agg['last_vote']  - agg['first_vote']).dt.total_seconds().round(1)
    agg['total']         = (agg['page_to_first'].fillna(0) + agg['first_to_last']).round(1)
    agg['is_fast']       = agg['first_to_last'] < fast_threshold_seconds

    cols = ['ballot_id', 'ip_address', 'day', 'matchups_voted',
            'page_load', 'first_vote', 'last_vote',
            'page_to_first', 'first_to_last', 'total', 'is_fast']
    return agg[cols].sort_values('first_to_last').reset_index(drop=True)


def vote_burst_detection(votes=votes, window_seconds=60, burst_threshold=20):
    '''
    Find time windows where an unusually high number of votes arrived.

    Uses a sliding window anchored at each vote: for every vote i, count how
    many other votes arrived within the next `window_seconds`.  Windows that
    exceed `burst_threshold` are non-overlapping (a new window is only kept if
    its start is at least `window_seconds` after the previous kept window).

    A concentrated burst of votes from few IPs or few ballots is a strong
    signal of coordinated or automated voting.

    Parameters
    ----------
    window_seconds : int
        Width of the sliding window in seconds.
    burst_threshold : int
        Minimum votes within the window to be reported.

    Returns
    -------
    pd.DataFrame
        Columns: window_start, window_end, vote_count, ballot_count,
                 unique_ips, votes_per_second.
        Sorted by vote_count descending.
    '''
    if votes.empty:
        return pd.DataFrame()

    import numpy as np

    ts = votes.copy()
    ts['timestamp'] = pd.to_datetime(ts['timestamp'])
    ts = ts.sort_values('timestamp').reset_index(drop=True)

    times_ns  = ts['timestamp'].astype('int64').values
    window_ns = int(window_seconds * 1e9)

    # For each vote i, find the index of the first vote outside the window
    right_idx     = np.searchsorted(times_ns, times_ns + window_ns, side='right')
    window_counts = right_idx - np.arange(len(ts))

    rows           = []
    last_kept_time = None
    for i in np.where(window_counts >= burst_threshold)[0]:
        t_start = ts['timestamp'].iloc[i]
        if last_kept_time is not None and (t_start - last_kept_time).total_seconds() < window_seconds:
            continue
        window_votes = ts.iloc[i : int(right_idx[i])]
        rows.append({
            'window_start':     t_start,
            'window_end':       t_start + pd.Timedelta(seconds=window_seconds),
            'vote_count':       len(window_votes),
            'ballot_count':     window_votes['ballot_id'].nunique(),
            'unique_ips':       window_votes['ip_address'].nunique(),
            'votes_per_second': round(len(window_votes) / window_seconds, 2),
        })
        last_kept_time = t_start

    if not rows:
        print(f'No bursts ≥ {burst_threshold} votes in any {window_seconds}s window.')
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values('vote_count', ascending=False)
        .reset_index(drop=True)
    )


def suspicious_window_bias(window_ballot_ids, target_matchup_id=None, votes=votes):
    '''
    Given a set of ballot_ids from a suspicious voting window, compare how
    those voters chose across every matchup they participated in versus the
    overall population.

    For each matchup the function computes:
      - overall vote share for card_a across all voters
      - the suspicious subset's vote share for card_a
      - which card is the underdog (lower overall share)
      - whether the suspicious votes favoured the underdog or the favourite
      - a z-score measuring how far the subset deviates from the population

    The z-score uses the overall rate as the null hypothesis and the
    suspicious sample size as n.  A Bonferroni threshold is printed so you
    can judge whether any individual z is genuinely surprising given how
    many matchups are tested.

    Parameters
    ----------
    window_ballot_ids : array-like
        ballot_ids to treat as the suspicious subset (e.g. from window['ballot_id'].unique()).
    target_matchup_id : int or None
        If given, that matchup's row is marked with an asterisk in the output
        to flag it as the one that triggered the investigation.
    votes : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Columns: matchup_id, card_a, card_b, underdog, n_sus,
                 pct_a_overall, pct_a_sus, pct_underdog_overall,
                 pct_underdog_sus, underdog_delta, z, target.
        Sorted by abs(z) descending.
    '''
    ballot_set  = set(window_ballot_ids)
    sus_votes   = votes[votes['ballot_id'].isin(ballot_set)].copy()

    if sus_votes.empty:
        print('No votes found for these ballot_ids.')
        return pd.DataFrame()

    mids = sorted(sus_votes['matchup_id'].unique())
    n_matchups = len(mids)

    # Bonferroni threshold at family-wise α=0.05
    try:
        from scipy.stats import norm as _norm
        z_bonf = _norm.ppf(1 - 0.05 / (2 * n_matchups))
    except ImportError:
        z_bonf = 3.0  # conservative fallback

    rows = []
    for mid in mids:
        all_m = votes[votes['matchup_id'] == mid]
        sus_m = sus_votes[sus_votes['matchup_id'] == mid]

        card_a = all_m['card_a'].iloc[0]
        card_b = all_m['card_b'].iloc[0]

        p_all = (all_m['chosen'] == all_m['card_a']).mean()   # overall % for card_a
        p_sus = (sus_m['chosen'] == sus_m['card_a']).mean()   # suspicious % for card_a
        n_sus = len(sus_m)

        # Underdog = card with the lower overall vote share
        a_is_underdog = p_all < 0.5
        underdog = card_a if a_is_underdog else card_b

        # Underdog share: how often did each group pick the underdog?
        pct_underdog_all = (1 - p_all) if not a_is_underdog else p_all
        pct_underdog_sus = (1 - p_sus) if not a_is_underdog else p_sus

        se = (p_all * (1 - p_all) / n_sus) ** 0.5 if n_sus > 0 else 0
        z  = round((p_sus - p_all) / se, 2) if se > 0 else 0.0

        rows.append({
            'matchup_id':          mid,
            'card_a':              card_a,
            'card_b':              card_b,
            'underdog':            underdog,
            'n_sus':               n_sus,
            'pct_a_overall':       round(p_all * 100, 1),
            'pct_a_sus':           round(p_sus * 100, 1),
            'pct_underdog_overall': round(pct_underdog_all * 100, 1),
            'pct_underdog_sus':    round(pct_underdog_sus * 100, 1),
            'underdog_delta':      round((pct_underdog_sus - pct_underdog_all) * 100, 1),
            'z':                   z,
            'target':              mid == target_matchup_id,
        })

    df = (
        pd.DataFrame(rows)
        .assign(abs_z=lambda d: d['z'].abs())
        .sort_values('abs_z', ascending=False)
        .drop(columns='abs_z')
        .reset_index(drop=True)
    )

    print(f'Matchups tested: {n_matchups}   '
          f'Bonferroni z_crit (α=0.05): {z_bonf:.2f}   '
          f'Flagged: {(df["z"].abs() >= z_bonf).sum()}')
    return df


def full_chunk_investigation(matchup_id, chunk_index=0, fast_threshold_seconds=None,
                             fast_percentile=15, z_threshold=3.0,
                             votes=votes, page_views=page_views):
    '''
    One-call post-investigation for a chunk flagged by vote_share_timeline_interactive.

    Produces three outputs:

    1. bias_table     — full suspicious_window_bias result for all matchups those
                        voters participated in, sorted by |z| descending.
    2. speed_summary  — page_to_first distribution comparing the window ballots to
                        the full day, in five fixed time buckets.
    3. flagged_detail — bias_table rows with |z| > z_threshold, augmented with
                        fast-voter counts and vote shares (fast = page_to_first
                        below the resolved threshold AND in the suspicious window).

    Fast-voter threshold
    --------------------
    If fast_threshold_seconds is given, that value is used directly.
    Otherwise the threshold is the `fast_percentile`-th percentile of
    page_to_first for the day being investigated (default p15).  Using a
    per-day percentile keeps the "fast" fraction constant across days that
    differ in ballot length — later rounds have fewer matchups and naturally
    faster completions, so a bracket-wide cutoff would over-flag them.

    Parameters
    ----------
    matchup_id : int
        The matchup whose flagged chunk to investigate. Must have at least one
        entry in _flagged_chunks (populated by vote_share_timeline_interactive).
    chunk_index : int or 'all'
        Which flagged chunk to use, ordered by |z| descending so 0 is the most
        suspicious. Pass 'all' to pool every flagged chunk for this matchup.
    fast_threshold_seconds : float or None
        Explicit page_to_first cutoff in seconds. Overrides fast_percentile when set.
    fast_percentile : int
        Percentile of the full-bracket page_to_first distribution used as the
        fast cutoff when fast_threshold_seconds is None (default 15).
    z_threshold : float
        |z| threshold for the flagged_detail section (default 3.0).
    votes : pd.DataFrame
    page_views : pd.DataFrame

    Returns
    -------
    types.SimpleNamespace with attributes:
        .bias_table     pd.DataFrame  (all matchups tested)
        .speed_summary  pd.DataFrame  (speed-bucket comparison)
        .flagged_detail pd.DataFrame  (high-z matchups with fast-voter breakdown)
        .fast_threshold float         (resolved seconds cutoff used)
    '''
    import types

    # ── Step 1: extract window ballot IDs ─────────────────────────────────────
    if matchup_id not in _flagged_chunks or not _flagged_chunks[matchup_id]:
        print(f'No flagged chunks for matchup {matchup_id}. '
              f'Run vote_share_timeline_interactive first.')
        return None

    chunks = _flagged_chunks[matchup_id]
    grp    = votes[votes['matchup_id'] == matchup_id].sort_values('id').reset_index(drop=True)

    if chunk_index == 'all':
        window_ballot_ids = set()
        for x0, x1, _ in chunks:
            window_ballot_ids.update(grp.iloc[x0 - 1 : x1]['ballot_id'].tolist())
        print(f'Matchup {matchup_id}: pooling {len(chunks)} flagged chunk(s), '
              f'{len(window_ballot_ids)} unique window ballots.\n')
    else:
        if chunk_index >= len(chunks):
            print(f'chunk_index {chunk_index} out of range — '
                  f'only {len(chunks)} flagged chunk(s) for matchup {matchup_id}.')
            return None
        x0, x1, z = chunks[chunk_index]
        window_ballot_ids = set(grp.iloc[x0 - 1 : x1]['ballot_id'].tolist())
        print(f'Matchup {matchup_id}, chunk {chunk_index}: '
              f'votes {x0}–{x1}  z={z:+.2f}  ({len(window_ballot_ids)} window ballots)\n')

    day = votes[votes['matchup_id'] == matchup_id]['day'].iloc[0]

    # ── Step 2: cross-matchup bias table ──────────────────────────────────────
    print('── Cross-matchup bias ────────────────────────────────────────────────')
    bias_table = suspicious_window_bias(window_ballot_ids,
                                        target_matchup_id=matchup_id,
                                        votes=votes)
    if not bias_table.empty:
        print(bias_table.to_string(index=False))

    # ── Step 3: vote-timing distribution ──────────────────────────────────────
    print('\n── Vote timing ───────────────────────────────────────────────────────')

    # Compute ballot_speed across all days so the percentile is bracket-wide.
    all_speed    = ballot_speed(votes, page_views=page_views)
    day_speed    = all_speed[all_speed['day'] == day].copy()
    window_speed = day_speed[day_speed['ballot_id'].isin(window_ballot_ids)].copy()

    # Resolve the fast threshold
    if fast_threshold_seconds is not None:
        threshold = float(fast_threshold_seconds)
        print(f'Fast threshold: {threshold:.1f}s  (explicit)')
    else:
        raw = day_speed['page_to_first'].quantile(fast_percentile / 100)
        if pd.isna(raw):
            threshold = 30.0
            print(f'Fast threshold: {threshold:.1f}s  '
                  f'(fallback — no page_to_first data available)')
        else:
            threshold = round(float(raw), 1)
            print(f'Fast threshold: {threshold:.1f}s  '
                  f'(p{fast_percentile} of day-{day} page_to_first)')

    bins   = [0, 30, 60, 120, 600, float('inf')]
    labels = ['<30s', '30–60s', '60–120s', '2–10min', '>10min']

    def _bucket_dist(df, col='page_to_first'):
        valid = df[col].dropna()
        total = len(valid)
        counts = (
            pd.cut(valid, bins=bins, labels=labels, right=False)
            .value_counts()
            .reindex(labels, fill_value=0)
        )
        pct = (counts / total * 100).round(1) if total else counts * 0.0
        return counts, pct

    win_n, win_pct = _bucket_dist(window_speed)
    day_n, day_pct = _bucket_dist(day_speed)

    speed_summary = pd.DataFrame({
        'window_n':   win_n,
        'window_pct': win_pct,
        'day_n':      day_n,
        'day_pct':    day_pct,
    })

    n_win_valid = window_speed['page_to_first'].notna().sum()
    n_day_valid = day_speed['page_to_first'].notna().sum()
    print(f'page_to_first distribution  '
          f'(window n={n_win_valid}/{len(window_speed)} with page-view match, '
          f'day n={n_day_valid}/{len(day_speed)}):')
    print(speed_summary.to_string())

    n_fast_win   = (window_speed['page_to_first'] < threshold).sum()
    n_fast_day   = (day_speed['page_to_first']    < threshold).sum()
    pct_fast_win = n_fast_win / n_win_valid * 100 if n_win_valid else 0.0
    pct_fast_day = n_fast_day / n_day_valid * 100 if n_day_valid else 0.0
    print(f'\nFast voters (<{threshold:.1f}s):  '
          f'window {n_fast_win}/{n_win_valid} ({pct_fast_win:.1f}%)  '
          f'vs  full day {n_fast_day}/{n_day_valid} ({pct_fast_day:.1f}%)')

    # ── Step 4: per-matchup detail for high-z matchups ────────────────────────
    print(f'\n── Matchup detail for |z| > {z_threshold} '
          f'─────────────────────────────────────────')

    if bias_table.empty:
        flagged_detail = pd.DataFrame()
    else:
        flagged_rows = bias_table[bias_table['z'].abs() > z_threshold].copy()
        if flagged_rows.empty:
            print(f'No matchups with |z| > {z_threshold}.')
            flagged_detail = pd.DataFrame()
        else:
            # Fast voters = window ballots whose page_to_first < resolved threshold
            fast_ballot_ids = set(
                window_speed[window_speed['page_to_first'] < threshold]
                ['ballot_id'].tolist()
            )

            fast_stats = []
            for mid in flagged_rows['matchup_id']:
                all_m  = votes[votes['matchup_id'] == mid]
                fast_m = all_m[all_m['ballot_id'].isin(fast_ballot_ids)]
                pct_a_fast = (
                    round((fast_m['chosen'] == fast_m['card_a']).mean() * 100, 1)
                    if not fast_m.empty else None
                )
                fast_stats.append({
                    'matchup_id': mid,
                    'n_fast':     len(fast_m),
                    'pct_a_fast': pct_a_fast,
                })

            flagged_detail = flagged_rows.merge(
                pd.DataFrame(fast_stats), on='matchup_id'
            )

            show_cols = [
                'matchup_id', 'card_a', 'card_b', 'underdog',
                'n_sus', 'n_fast',
                'pct_a_overall', 'pct_a_sus', 'pct_a_fast',
                'pct_underdog_overall', 'pct_underdog_sus',
                'underdog_delta', 'z', 'target',
            ]
            show_cols = [c for c in show_cols if c in flagged_detail.columns]
            flagged_detail = flagged_detail[show_cols].reset_index(drop=True)
            print(flagged_detail.to_string(index=False))

    return types.SimpleNamespace(
        bias_table=bias_table,
        speed_summary=speed_summary,
        flagged_detail=flagged_detail,
        fast_threshold=threshold,
    )


def favorites_summary(favorites=favorites):
    '''
    Summarise the personal-favorites submissions from finals_favorite_cards.

    Returns
    -------
    pd.DataFrame
        One row per unique card name that was nominated, with columns:
        card_name, nominations (count of times it appeared across card_a/b/c).
        Sorted by nominations descending.

    Also prints:
        - total submissions and how many included a written response
        - the full favorites DataFrame for quick inspection
    '''
    if favorites.empty:
        print('No favorites submissions yet.')
        return pd.DataFrame()

    has_response = favorites['response_text'].notna() & (favorites['response_text'].str.strip() != '')
    print(f'Favorites submissions:  {len(favorites)}')
    print(f'With written response:  {has_response.sum()}')

    # Tally each card across the three pick slots
    picks = pd.concat([
        favorites['card_a'].dropna().rename('card_name'),
        favorites['card_b'].dropna().rename('card_name'),
        favorites['card_c'].dropna().rename('card_name'),
    ], ignore_index=True)

    if picks.empty:
        print('No card picks recorded.')
        return pd.DataFrame()

    tally = (
        picks.value_counts()
        .reset_index()
        .rename(columns={'count': 'nominations'})
        .sort_values('nominations', ascending=False)
        .reset_index(drop=True)
    )
    return tally


def top1pct_entry_stats():
    '''
    Return one row per card in the top 1% of each queue (by final queue Elo),
    enriched with their performance in the top-10% bracket.

    Columns
    -------
    name            card name
    queue_id        which daily queue the card came from
    queue_size      total cards in that queue
    queue_rank      intra-queue rank by final queue Elo (1 = best in queue)
    initial_elo     final Elo from the daily queue phase (elo_ratings)
    final_elo       Elo after the top-10% bracket (elo_ratings_top10)
    bracket_rank    rank among all top-10% entrants by final_elo (1 = best overall)

    Rows are sorted by bracket_rank.
    '''
    import math

    QUEUES_JSON = os.path.join(BASE_DIR, 'queues.json')

    with open(QUEUES_JSON, encoding='utf-8') as f:
        queues_data = json.load(f)

    _conn = psycopg2.connect(DATABASE_URL)
    queue_elo_df  = pd.read_sql('SELECT card_name, rating FROM elo_ratings',       _conn)
    bracket_elo_df = pd.read_sql('SELECT card_name, rating FROM elo_ratings_top10', _conn)
    _conn.close()

    queue_elo   = dict(zip(queue_elo_df['card_name'],   queue_elo_df['rating'].astype(float)))
    bracket_elo = dict(zip(bracket_elo_df['card_name'], bracket_elo_df['rating'].astype(float)))

    # bracket_rank: rank all top-10% entrants by final bracket Elo
    all_bracket = sorted(bracket_elo.items(), key=lambda x: x[1], reverse=True)
    bracket_rank_map = {name: rank for rank, (name, _) in enumerate(all_bracket, start=1)}

    rows = []
    for q in queues_data['queues']:
        qid   = q['id']
        cards = q['cards']
        size  = len(cards)

        cutoff = math.ceil(size * 0.01)
        ranked = sorted(cards, key=lambda c: queue_elo.get(c, 1500.0), reverse=True)

        for rank_1based, name in enumerate(ranked[:cutoff], start=1):
            rows.append({
                'name':         name,
                'queue_id':     qid,
                'queue_size':   size,
                'queue_rank':   rank_1based,
                'initial_elo':  round(queue_elo.get(name, 1500.0), 2),
                'final_elo':    round(bracket_elo.get(name), 2) if name in bracket_elo else None,
                'bracket_rank': bracket_rank_map.get(name),
            })

    df = pd.DataFrame(rows).sort_values('bracket_rank').reset_index(drop=True)
    return df
