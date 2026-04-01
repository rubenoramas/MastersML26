"""
predict_2026_corrected.py
=========================
2026 Masters predictions with corrected l6 feature proxy.

L6 proxy design (deliberate decision):
  l6_sg_total_avg  = sg_total from skill-ratings  (current rolling form)
  l6_sg_app_avg    = sg_app from skill-ratings
  l6_sg_putt_avg   = sg_putt from skill-ratings
  l6_top10_rate    = from global sg_total rank: top-10 → 0.50, top-25 → 0.30, else → 0.15
  l6_cut_rate      = 0.85 for active tour players, 0.75 for kaggle-career, 0.65 limited
  l6_avg_finish    = from field sg_total rank: 1-10→8, 11-25→18, 26-50→30, else→45

Rationale: skill-ratings already reflect rolling recent form — the same signal l6
features capture from same-season data. This avoids the stale-2022-data bias.
"""

import sys, os, warnings, ctypes, json
warnings.filterwarnings('ignore')

_LIBOMP = (
    "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/"
    "site-packages/sklearn/.dylibs/libomp.dylib"
)
if os.path.exists(_LIBOMP):
    ctypes.CDLL(_LIBOMP)

import numpy as np
import pandas as pd
import joblib
from rapidfuzz import process as fz_process, fuzz

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Paths ─────────────────────────────────────────────────────────────────────
FIELD_CSV   = f"{ROOT}/data/raw/masters_2026_field.csv"
SKILL_JSON  = f"{ROOT}/data/raw/preds_skill-ratings.json"
RESULTS_CSV = f"{ROOT}/data/raw/masters_results_2023_2025.csv"
KAGGLE_CSV  = f"{ROOT}/data/raw/ASA All PGA Raw Data - Tourn Level.csv"
FM_CSV      = f"{ROOT}/data/features/feature_matrix.csv"
MODEL_DIR   = f"{ROOT}/outputs/models"
PREDS_OUT   = f"{ROOT}/outputs/csv/predictions_2026.csv"
BETTING_OUT = f"{ROOT}/outputs/csv/betting_value.csv"

FEATURE_COLS = [
    'sg_total_weighted', 'sg_app_weighted', 'sg_putt_weighted',
    'sg_arg_weighted',   'sg_ott_weighted',  'sg_t2g_weighted',
    'top10_rate', 'cut_rate', 'augusta_fit',
    'masters_appearances', 'masters_avg_finish', 'masters_best_finish',
    'masters_wins', 'recent_form_bonus',
    'l6_sg_total_avg', 'l6_sg_app_avg', 'l6_sg_putt_avg',
    'l6_top10_rate', 'l6_cut_rate', 'l6_avg_finish',
]

NOV2020_ID = 401219478

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize(name):
    return str(name).lower().strip()

BLOCK_FUZZY = {'jordan smith'}

def fuzzy_match(name, candidates, threshold=85):
    n = normalize(name)
    if n in BLOCK_FUZZY:
        return None
    cands_lower = [normalize(c) for c in candidates]
    res = fz_process.extractOne(n, cands_lower, scorer=fuzz.token_sort_ratio)
    if res and res[1] >= threshold:
        return candidates[cands_lower.index(res[0])]
    return None

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
CLF_TARGETS = ['won', 'top5', 'top10', 'top16', 'top32', 'made_cut']
REG_TARGET  = 'finish_position'
models = {}
for tgt in CLF_TARGETS + [REG_TARGET]:
    path = f"{MODEL_DIR}/xgb_{tgt.replace('_','-')}.joblib"
    models[tgt] = joblib.load(path)
print(f"  Loaded {len(models)} models.")

# ── Load skill-ratings (primary SG source + global ranking) ──────────────────
with open(SKILL_JSON) as f:
    sr_data = json.load(f)
sr_players = sr_data['players']

# Build "First Last" → record lookup
def dg_name_to_lower(raw):
    if ', ' in raw:
        parts = raw.split(', ', 1)
        return normalize(parts[1] + ' ' + parts[0])
    return normalize(raw)

sr_lookup  = {dg_name_to_lower(p['player_name']): p for p in sr_players}
sr_raw_lk  = {normalize(p['player_name']): p for p in sr_players}

# Global sg_total rank across ALL 434 skill-rated players
sr_sorted = sorted(sr_players, key=lambda x: float(x.get('sg_total', -99)), reverse=True)
sr_global_rank = {dg_name_to_lower(p['player_name']): i+1 for i, p in enumerate(sr_sorted)}

def get_l6_top10_rate(global_rank):
    """Convert DataGolf global sg_total rank to l6_top10_rate proxy."""
    if global_rank is None:  return 0.15
    if global_rank <= 10:    return 0.50
    if global_rank <= 25:    return 0.30
    return 0.15

def get_l6_avg_finish(field_rank):
    """Convert Masters field sg_total rank to l6_avg_finish proxy."""
    if field_rank is None:   return 45.0
    if field_rank <= 10:     return 8.0
    if field_rank <= 25:     return 18.0
    if field_rank <= 50:     return 30.0
    return 45.0

def find_sr(name_lower):
    """Return skill-ratings record for a field player, or None."""
    if name_lower in sr_lookup:
        return sr_lookup[name_lower]
    # Fuzzy fallback against "Last, First" keys
    all_keys = list(sr_raw_lk.keys())
    parts = name_lower.split()
    target = f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) >= 2 else name_lower
    res = fz_process.extractOne(target, all_keys, scorer=fuzz.token_sort_ratio)
    if res and res[1] >= 85:
        return sr_raw_lk[res[0]]
    return None

# ── Load Kaggle CSV (veteran SG + Augusta history base) ──────────────────────
print("Loading Kaggle CSV...")
raw = pd.read_csv(KAGGLE_CSV)
raw = raw.loc[:, ~raw.columns.str.startswith('Unnamed')]
raw.loc[raw['tournament id'] == NOV2020_ID, 'season'] = 2020
raw['player_lower'] = raw['player'].str.lower().str.strip()

# Augusta rows and non-Augusta rows
masters_raw = raw[raw['tournament name'].str.contains('Masters', case=False, na=False)].copy()
non_masters  = raw[~raw['tournament name'].str.contains('Masters', case=False, na=False)].copy()

# Veteran SG career averages (players NOT in skill-ratings)
def kaggle_career_sg(plow):
    rows = non_masters[non_masters['player_lower'] == plow]
    if len(rows) == 0:
        return None
    return {c: rows[c].mean() for c in ['sg_total','sg_app','sg_putt','sg_arg','sg_ott','sg_t2g']
            if c in rows.columns}

# ── Augusta win history (Kaggle 2015-2022 + 2023-2025) ───────────────────────
kaggle_wins = masters_raw[masters_raw['pos'] == 1.0][['player_lower','season']].copy()
kaggle_wins.columns = ['player_lower', 'win_year']

res2325 = pd.read_csv(RESULTS_CSV)
res2325['player_lower'] = res2325['player_name'].str.lower().str.strip()
recent_wins = res2325[res2325['won'] == 1][['player_lower','year']].copy()
recent_wins.columns = ['player_lower', 'win_year']

all_wins = pd.concat([kaggle_wins, recent_wins], ignore_index=True)
all_wins['win_year'] = all_wins['win_year'].astype(int)

# ── Feature matrix (Augusta history lookup) ───────────────────────────────────
fm = pd.read_csv(FM_CSV)
fm_latest = fm.sort_values('season').groupby('player').last().reset_index()
fm_lk = fm_latest.set_index('player').to_dict('index')

MEAN_TOP10  = fm_latest['top10_rate'].mean()
MEAN_CUT    = fm_latest['cut_rate'].mean()
MEAN_AVG_F  = fm_latest['masters_avg_finish'].dropna().mean()
MEAN_BEST_F = fm_latest['masters_best_finish'].dropna().mean()

# ── Load field ────────────────────────────────────────────────────────────────
field = pd.read_csv(FIELD_CSV)
field['player_lower'] = field['player_name'].str.lower().str.strip()

# ── Compute field-level sg_total ranks (for l6_avg_finish proxy) ──────────────
# First pass: get sg_total for each field player
sg_totals_field = {}
for _, frow in field.iterrows():
    plow = frow['player_lower']
    sr = find_sr(plow)
    if sr:
        sg_totals_field[plow] = float(sr.get('sg_total', 0) or 0)
    else:
        kg = kaggle_career_sg(plow)
        sg_totals_field[plow] = float(kg['sg_total']) if (kg and kg.get('sg_total')) else 0.0

# Rank within field
field_sorted = sorted(sg_totals_field.items(), key=lambda x: x[1], reverse=True)
field_rank   = {plow: i+1 for i, (plow, _) in enumerate(field_sorted)}

# ── Build 2026 feature vectors ────────────────────────────────────────────────
print("Building 2026 feature vectors...")
rows_2026 = []

for _, frow in field.iterrows():
    pname = frow['player_name']
    plow  = frow['player_lower']

    # ── SG features ───────────────────────────────────────────────────────
    sr = find_sr(plow)
    if sr:
        sg_total = float(sr.get('sg_total', 0) or 0)
        sg_app   = float(sr.get('sg_app',   0) or 0)
        sg_putt  = float(sr.get('sg_putt',  0) or 0)
        sg_arg   = float(sr.get('sg_arg',   0) or 0)
        sg_ott   = float(sr.get('sg_ott',   0) or 0)
        sg_t2g   = sg_total - sg_putt
        sg_source = 'skill-ratings'
        limited   = False
    else:
        kg = kaggle_career_sg(plow)
        if kg and any(v != 0 for v in kg.values() if v == v):
            sg_total = float(kg.get('sg_total', 0) or 0)
            sg_app   = float(kg.get('sg_app',   0) or 0)
            sg_putt  = float(kg.get('sg_putt',  0) or 0)
            sg_arg   = float(kg.get('sg_arg',   0) or 0)
            sg_ott   = float(kg.get('sg_ott',   0) or 0)
            sg_t2g   = float(kg.get('sg_t2g',   sg_total - sg_putt) or 0)
            sg_source = 'kaggle-career'
            limited   = False
        else:
            sg_total = sg_app = sg_putt = sg_arg = sg_ott = sg_t2g = 0.0
            sg_source = 'limited-data'
            limited   = True

    # ── Augusta history ────────────────────────────────────────────────────
    fm_key = fuzzy_match(plow, list(fm_lk.keys()))
    if fm_key:
        src     = fm_lk[fm_key]
        top10_r = float(src['top10_rate'])
        cut_r   = float(src['cut_rate'])
        apprnc  = int(src['masters_appearances']) + 1
        avg_f   = float(src['masters_avg_finish']) if pd.notna(src.get('masters_avg_finish')) else MEAN_AVG_F
        best_f  = float(src['masters_best_finish']) if pd.notna(src.get('masters_best_finish')) else MEAN_BEST_F
    else:
        top10_r = MEAN_TOP10
        cut_r   = MEAN_CUT
        apprnc  = 0
        avg_f   = MEAN_AVG_F
        best_f  = MEAN_BEST_F

    # Augusta fit composite
    aug_fit = (2.0*sg_app + 2.0*sg_putt + 1.0*sg_total + 0.5*sg_ott) / 5.5

    # ── masters_wins / recent_form_bonus ──────────────────────────────────
    p_wins = all_wins[all_wins['player_lower'] == plow]
    masters_wins_val      = int((p_wins['win_year'] < 2026).sum())
    win_yrs               = p_wins['win_year'].tolist()
    recent_form_bonus_val = 1 if any(2020 <= y < 2026 for y in win_yrs) else 0

    # ── L6 proxy — CORRECTED (skill-ratings based, not stale fm_latest) ───
    # SG: direct from skill-ratings (current rolling form = what l6 captures)
    l6_sg_total = sg_total
    l6_sg_app   = sg_app
    l6_sg_putt  = sg_putt

    # l6_top10_rate: global sg_total rank (across all 434 skill-rated players)
    global_rnk = sr_global_rank.get(plow)
    l6_top10 = get_l6_top10_rate(global_rnk)

    # l6_cut_rate: 0.85 active tour, 0.75 kaggle-career, 0.65 limited
    if sg_source == 'skill-ratings':
        l6_cut = 0.85
    elif sg_source == 'kaggle-career':
        l6_cut = 0.75
    else:
        l6_cut = 0.65

    # l6_avg_finish: field sg_total rank (within 93-player Masters field)
    fld_rnk  = field_rank.get(plow)
    l6_avg_f = get_l6_avg_finish(fld_rnk)

    rows_2026.append({
        'player_name':         pname,
        'player_lower':        plow,
        'sg_source':           sg_source,
        'limited_data':        limited,
        'sg_total_weighted':   sg_total,
        'sg_app_weighted':     sg_app,
        'sg_putt_weighted':    sg_putt,
        'sg_arg_weighted':     sg_arg,
        'sg_ott_weighted':     sg_ott,
        'sg_t2g_weighted':     sg_t2g,
        'top10_rate':          top10_r,
        'cut_rate':            cut_r,
        'augusta_fit':         aug_fit,
        'masters_appearances': apprnc,
        'masters_avg_finish':  avg_f,
        'masters_best_finish': best_f,
        'masters_wins':        masters_wins_val,
        'recent_form_bonus':   recent_form_bonus_val,
        'l6_sg_total_avg':     l6_sg_total,
        'l6_sg_app_avg':       l6_sg_app,
        'l6_sg_putt_avg':      l6_sg_putt,
        'l6_top10_rate':       l6_top10,
        'l6_cut_rate':         l6_cut,
        'l6_avg_finish':       l6_avg_f,
        # For blending reference
        '_sg_total_raw':       sg_total,
        '_field_rank':         fld_rnk or 93,
    })

df26 = pd.DataFrame(rows_2026)
print(f"Built feature vectors: {df26.shape}")

# NaN guard
for col in FEATURE_COLS:
    df26[col] = pd.to_numeric(df26[col], errors='coerce')
    if df26[col].isna().any():
        df26[col] = df26[col].fillna(df26[col].mean())

# ── Run all 7 models ──────────────────────────────────────────────────────────
print("Running models...")
X = df26[FEATURE_COLS].values

raw_won   = models['won'].predict_proba(X)[:, 1]
raw_top5  = models['top5'].predict_proba(X)[:, 1]
raw_top10 = models['top10'].predict_proba(X)[:, 1]
raw_top16 = models['top16'].predict_proba(X)[:, 1]
raw_top32 = models['top32'].predict_proba(X)[:, 1]
raw_mc    = models['made_cut'].predict_proba(X)[:, 1]
raw_fp    = models['finish_position'].predict(X)

# Normalize win probability to sum to 100%
model_win_pct = raw_won / raw_won.sum() * 100

df26['model_win_pct'] = model_win_pct
df26['top5_pct']      = raw_top5  * 100
df26['top10_pct']     = raw_top10 * 100
df26['top16_pct']     = raw_top16 * 100
df26['top32_pct']     = raw_top32 * 100
df26['made_cut_pct']  = raw_mc    * 100
df26['proj_finish']   = raw_fp

# ── DataGolf skill-derived win probabilities (softmax on sg_total, τ=1.5) ─────
tau = 1.5
exp_vals = np.exp(df26['sg_total_weighted'].values.astype(float) / tau)
dg_win_pct = (exp_vals / exp_vals.sum()) * 100
df26['dg_win_pct'] = dg_win_pct

# ── Blend: 60% model + 40% DataGolf ──────────────────────────────────────────
blended = 0.6 * df26['model_win_pct'] + 0.4 * df26['dg_win_pct']
df26['blended_win_pct'] = blended / blended.sum() * 100

# ── Print top 20 ──────────────────────────────────────────────────────────────
print()
print("=" * 80)
print("2026 Masters Predictions — Top 20 (with corrected l6 proxy)")
print(f"{'Model':>7} = model-only win%  |  {'DG':>5} = DataGolf sg_total softmax win%")
print(f"{'Blend':>7} = 0.6×Model + 0.4×DG  (re-normalized)")
print("=" * 80)
hdr = (f"{'Rk':<4} {'Player':<28} {'Blend%':>7} {'Model%':>7} {'DG%':>6} "
       f"{'Top10%':>7} {'MadeCut%':>9} {'ProjFin':>8} {'Wins':>5} {'RFB':>4}")
print(hdr)
print("-" * len(hdr))

top20 = df26.nlargest(20, 'blended_win_pct').reset_index(drop=True)
for i, r in top20.iterrows():
    print(
        f"{i+1:<4} {r['player_name']:<28} "
        f"{r['blended_win_pct']:>7.2f} {r['model_win_pct']:>7.2f} {r['dg_win_pct']:>6.2f} "
        f"{r['top10_pct']:>7.1f} {r['made_cut_pct']:>9.1f} {r['proj_finish']:>8.1f} "
        f"{int(r['masters_wins']):>5} {int(r['recent_form_bonus']):>4}"
    )

# ── Key player call-outs ──────────────────────────────────────────────────────
print()
print("Key players:")
for name in ['scottie scheffler', 'rory mcilroy', 'jon rahm', 'justin thomas']:
    row = df26[df26['player_lower'] == name]
    if len(row):
        r = row.iloc[0]
        rank = int(df26['blended_win_pct'].rank(ascending=False)[row.index[0]])
        print(f"  #{rank:<3} {r['player_name']:<22}  blend={r['blended_win_pct']:.2f}%  "
              f"model={r['model_win_pct']:.2f}%  DG={r['dg_win_pct']:.2f}%  "
              f"l6_top10={r['l6_top10_rate']:.2f}  l6_avg_fin={r['l6_avg_finish']:.0f}  "
              f"wins={int(r['masters_wins'])}  rfb={int(r['recent_form_bonus'])}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_cols = ['player_name', 'blended_win_pct', 'model_win_pct', 'dg_win_pct',
            'top5_pct', 'top10_pct', 'top16_pct', 'top32_pct', 'made_cut_pct',
            'proj_finish', 'masters_appearances', 'masters_avg_finish',
            'masters_best_finish', 'masters_wins', 'recent_form_bonus',
            'l6_sg_total_avg', 'l6_top10_rate', 'l6_cut_rate', 'l6_avg_finish',
            'sg_source', 'limited_data']
df26[out_cols].sort_values('blended_win_pct', ascending=False).to_csv(PREDS_OUT, index=False)
print(f"\nSaved -> {PREDS_OUT}")
print("Done.")
