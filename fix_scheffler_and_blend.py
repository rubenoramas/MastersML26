"""
fix_scheffler_and_blend.py
==========================
1. Add masters_wins + recent_form_bonus to training feature matrix
2. Retrain all 7 XGBoost models
3. Build 2026 feature vectors with new features
4. Print Scheffler's full 2026 feature vector
5. Blend model predictions with DataGolf skill-derived win probabilities
6. Print blended top 20
"""

import sys, os, warnings, ctypes, json
warnings.filterwarnings('ignore')

# ── libomp fix for macOS ──────────────────────────────────────────────────────
_LIBOMP = (
    "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/"
    "site-packages/sklearn/.dylibs/libomp.dylib"
)
if os.path.exists(_LIBOMP):
    ctypes.CDLL(_LIBOMP)

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier, XGBRegressor
from sklearn.impute import SimpleImputer
from rapidfuzz import process as fz_process, fuzz

sys.path.insert(0, "src")
from api_client import pull_endpoint

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Paths ─────────────────────────────────────────────────────────────────────
FM_PATH       = f"{ROOT}/data/features/feature_matrix.csv"
RAW_CSV       = f"{ROOT}/data/raw/ASA All PGA Raw Data - Tourn Level.csv"
RESULTS_CSV   = f"{ROOT}/data/raw/masters_results_2023_2025.csv"
FIELD_CSV     = f"{ROOT}/data/raw/masters_2026_field.csv"
SKILL_JSON    = f"{ROOT}/data/raw/preds_skill-ratings.json"
PREDS_OUT     = f"{ROOT}/outputs/csv/predictions_2026.csv"
BETTING_OUT   = f"{ROOT}/outputs/csv/betting_value.csv"
MODEL_DIR     = f"{ROOT}/outputs/models"

NOV2020_ID    = 401219478
DECAY         = 0.85

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize(name: str) -> str:
    return str(name).lower().strip()

BLOCK_FUZZY = {'jordan smith'}

def fuzzy_match(name: str, candidates: list[str], threshold: int = 85) -> str | None:
    n = normalize(name)
    if n in BLOCK_FUZZY:
        return None
    cands_lower = [normalize(c) for c in candidates]
    res = fz_process.extractOne(n, cands_lower, scorer=fuzz.token_sort_ratio)
    if res and res[1] >= threshold:
        return candidates[cands_lower.index(res[0])]
    return None

def exp_weighted_avg(series, decay):
    vals = series.dropna().values
    if len(vals) == 0:
        return np.nan
    weights = np.array([decay ** i for i in range(len(vals) - 1, -1, -1)])
    return np.dot(vals, weights) / weights.sum()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load & enhance feature matrix with masters_wins + recent_form_bonus
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Adding masters_wins + recent_form_bonus to training data")
print("=" * 60)

# Load Kaggle raw data (Augusta rows only)
raw = pd.read_csv(RAW_CSV)
raw.loc[raw['tournament id'] == NOV2020_ID, 'season'] = 2020
masters_raw = raw[raw['course'].str.contains('Augusta', na=False)].copy()
masters_raw['player_lower'] = masters_raw['player'].str.lower().str.strip()

# Load 2023-2025 results
res2325 = pd.read_csv(RESULTS_CSV)
res2325['player_lower'] = res2325['player_name'].str.lower().str.strip()

# Build comprehensive Augusta win history: who won in which year?
# From Kaggle: season col and pos == 1.0 (winner)
kaggle_wins = masters_raw[masters_raw['pos'] == 1.0][['player_lower', 'season']].copy()
kaggle_wins.columns = ['player_lower', 'win_year']

# From 2023-2025
res_wins = res2325[res2325['won'] == 1][['player_lower', 'year']].copy()
res_wins.columns = ['player_lower', 'win_year']

all_wins = pd.concat([kaggle_wins, res_wins], ignore_index=True)
all_wins['win_year'] = all_wins['win_year'].astype(int)

print("All Augusta winners 2015-2025:")
print(all_wins.sort_values('win_year').to_string(index=False))

# For each row in feature matrix: compute masters_wins and recent_form_bonus
fm = pd.read_csv(FM_PATH)
fm['player_lower'] = fm['player'].str.lower().str.strip()

# masters_wins = number of Masters wins BEFORE this season
# recent_form_bonus = 1 if player won Masters in 2020 or any year < current season (and >= 2020)
def add_win_features(fm, all_wins):
    masters_wins_col = []
    rfb_col = []
    for _, row in fm.iterrows():
        p = row['player_lower']
        yr = int(row['season'])
        player_wins = all_wins[all_wins['player_lower'] == p]['win_year'].tolist()
        wins_before = [w for w in player_wins if w < yr]
        masters_wins_col.append(len(wins_before))
        rfb = 1 if any(w >= 2020 for w in wins_before) else 0
        rfb_col.append(rfb)
    fm = fm.copy()
    fm['masters_wins'] = masters_wins_col
    fm['recent_form_bonus'] = rfb_col
    return fm

fm = add_win_features(fm, all_wins)
print("\nFeature matrix masters_wins distribution:")
print(fm['masters_wins'].value_counts().sort_index())
print("\nFeature matrix recent_form_bonus distribution:")
print(fm['recent_form_bonus'].value_counts().sort_index())

# Check who in training data has recent_form_bonus=1
rfb_players = fm[fm['recent_form_bonus'] == 1][['player', 'season', 'masters_wins', 'recent_form_bonus', 'won']]
print("\nRows with recent_form_bonus=1:")
print(rfb_players.to_string(index=False))

# Save updated feature matrix
fm.to_csv(FM_PATH, index=False)
print(f"\nSaved updated feature matrix: {fm.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Retrain all 7 XGBoost models
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Retraining all 7 models with new features")
print("=" * 60)

FEATURE_COLS = [
    'sg_total_weighted', 'sg_app_weighted', 'sg_putt_weighted',
    'sg_arg_weighted', 'sg_ott_weighted', 'sg_t2g_weighted',
    'top10_rate', 'cut_rate', 'augusta_fit',
    'masters_appearances', 'masters_avg_finish', 'masters_best_finish',
    'masters_wins', 'recent_form_bonus',           # NEW
]

TARGETS = ['won', 'top5', 'top10', 'top16', 'top32', 'made_cut', 'finish_position']

XGB_CLF_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric='logloss', random_state=42, verbosity=0
)
XGB_REG_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0
)

train = fm[fm['season'] < 2022].copy()
val   = fm[fm['season'] == 2022].copy()

imp = SimpleImputer(strategy='mean')
X_train = imp.fit_transform(train[FEATURE_COLS])
X_val   = imp.transform(val[FEATURE_COLS])

models = {}
for tgt in TARGETS:
    y_train = train[tgt].values
    y_val   = val[tgt].values
    if tgt == 'finish_position':
        m = XGBRegressor(**XGB_REG_PARAMS)
        m.fit(X_train, y_train)
    else:
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw   = n_neg / max(n_pos, 1)
        m = XGBClassifier(scale_pos_weight=spw, **XGB_CLF_PARAMS)
        m.fit(X_train, y_train)
    models[tgt] = m
    path = f"{MODEL_DIR}/xgb_{tgt.replace('_','-')}.joblib"
    joblib.dump(m, path)
    print(f"  trained {tgt:20s} -> {path.split('/')[-1]}")

print("\nAll 7 models retrained and saved.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Build 2026 feature vectors
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Building 2026 feature vectors")
print("=" * 60)

# Load skill-ratings
with open(SKILL_JSON) as f:
    sr_data = json.load(f)
sr_players = sr_data['players']
# Build "First Last" lookup from "Last, First" DG names
sr_lookup = {}
for p in sr_players:
    raw_name = p['player_name']
    if ', ' in raw_name:
        parts = raw_name.split(', ', 1)
        first_last = normalize(parts[1] + ' ' + parts[0])
    else:
        first_last = normalize(raw_name)
    sr_lookup[first_last] = p

# Also build lookup by raw DG name format
sr_lookup_raw = {normalize(p['player_name']): p for p in sr_players}

def get_sg_from_skillratings(name_normalized):
    """Return SG dict from skill-ratings or None."""
    # Try direct "first last" match
    if name_normalized in sr_lookup:
        return sr_lookup[name_normalized]
    # Try "Last, First" format
    parts = name_normalized.split()
    if len(parts) == 2:
        alt = f"{parts[1]}, {parts[0]}"
        if alt in sr_lookup_raw:
            return sr_lookup_raw[alt]
    # Fuzzy match against "Last, First" keys
    all_dg_names = list(sr_lookup_raw.keys())
    # Convert name_normalized to "Last, First" for matching
    if len(parts) == 2:
        target = f"{parts[1]}, {parts[0]}"
    else:
        target = name_normalized
    res = fz_process.extractOne(target, all_dg_names, scorer=fuzz.token_sort_ratio)
    if res and res[1] >= 85:
        return sr_lookup_raw[res[0]]
    return None

# Load Kaggle Augusta rows for career SG fallback
kaggle_aug = masters_raw.copy()

# Build comprehensive Augusta history (Kaggle 2015-2022 + 2023-2025)
def aug_stats(player_name_lower, before_year=2026):
    # Kaggle rows
    k_rows = kaggle_aug[kaggle_aug['player_lower'] == player_name_lower]
    k_rows = k_rows[k_rows['season'] < before_year]
    # 2023-2025 rows
    r_rows = res2325[res2325['player_lower'] == player_name_lower]
    r_rows = r_rows[r_rows['year'] < before_year]

    appearances = len(k_rows) + len(r_rows)

    # Made-cut finishes from Kaggle (use numeric pos column; NaN = missed cut)
    k_made = k_rows[k_rows['made_cut'] == 1]['pos'].dropna().tolist()
    # Made-cut finishes from 2023-2025
    r_made = r_rows[r_rows['made_cut'] == 1]['finish_position'].tolist()
    mc_finishes = [f for f in (k_made + r_made) if f != 99 and not pd.isna(f)]

    avg_finish  = float(np.mean(mc_finishes)) if mc_finishes else np.nan
    best_finish = float(np.min(mc_finishes))  if mc_finishes else np.nan

    # Masters wins (before before_year)
    player_wins = all_wins[(all_wins['player_lower'] == player_name_lower) &
                           (all_wins['win_year'] < before_year)]
    masters_wins_val = len(player_wins)

    # recent_form_bonus
    rfb = 1 if (player_wins['win_year'] >= 2020).any() else 0

    # recent_masters_avg: last 3 made-cut appearances
    all_mc = []
    for _, kr in k_rows[k_rows['made_cut'] == 1].dropna(subset=['pos']).sort_values('season').iterrows():
        all_mc.append(float(kr['pos']))
    for _, rr in r_rows[r_rows['made_cut'] == 1].sort_values('year').iterrows():
        if rr['finish_position'] != 99:
            all_mc.append(float(rr['finish_position']))
    recent_mc_avg = float(np.mean(all_mc[-3:])) if all_mc else np.nan

    return {
        'masters_appearances': appearances,
        'masters_avg_finish':  avg_finish,
        'masters_best_finish': best_finish,
        'masters_wins':        masters_wins_val,
        'recent_form_bonus':   rfb,
        'recent_masters_avg':  recent_mc_avg,
    }

# Build Kaggle career SG averages for veterans
def kaggle_career_sg(player_name_lower):
    rows = raw[raw['player'].str.lower().str.strip() == player_name_lower]
    rows = rows[~rows['course'].str.contains('Augusta', na=False)]
    if len(rows) == 0:
        return None
    return {
        'sg_total': rows['sg_total'].mean() if 'sg_total' in rows.columns else 0,
        'sg_app':   rows['sg_app'].mean()   if 'sg_app' in rows.columns else 0,
        'sg_putt':  rows['sg_putt'].mean()  if 'sg_putt' in rows.columns else 0,
        'sg_arg':   rows['sg_arg'].mean()   if 'sg_arg' in rows.columns else 0,
        'sg_ott':   rows['sg_ott'].mean()   if 'sg_ott' in rows.columns else 0,
        'sg_t2g':   rows['sg_t2g'].mean()   if 'sg_t2g' in rows.columns else 0,
        'top10_rate': ((rows['pos'].fillna(999) <= 10).sum() / max(len(rows), 1)) if 'pos' in rows.columns else 0,
        'cut_rate':   (rows['made_cut'] == 1).mean() if 'made_cut' in rows.columns else 0,
    }

# Get form rates from feature matrix (2022 row if available, else career in FM)
fm_2022 = fm[fm['season'] == 2022].copy()
fm_2022.set_index('player_lower', inplace=True)

field = pd.read_csv(FIELD_CSV)
field['player_lower'] = field['player_name'].str.lower().str.strip()

MASTERS_DATE = pd.Timestamp('2026-04-10')

rows_2026 = []
for _, frow in field.iterrows():
    pname = frow['player_name']
    plow  = frow['player_lower']

    # --- SG features: prefer skill-ratings, fallback to Kaggle career ---
    sg_data = get_sg_from_skillratings(plow)
    limited_data = False
    sg_source = 'skill-ratings'

    if sg_data is None:
        # Try fuzzy match in Kaggle field
        kg_career = kaggle_career_sg(plow)
        if kg_career and kg_career['sg_total'] != 0:
            sg_data = {
                'sg_total': kg_career['sg_total'],
                'sg_app':   kg_career['sg_app'],
                'sg_putt':  kg_career['sg_putt'],
                'sg_arg':   kg_career['sg_arg'],
                'sg_ott':   kg_career['sg_ott'],
                'sg_t2g':   kg_career.get('sg_t2g', 0),
            }
            sg_source = 'kaggle-career'
        else:
            sg_data = {'sg_total':0,'sg_app':0,'sg_putt':0,'sg_arg':0,'sg_ott':0,'sg_t2g':0}
            sg_source = 'limited-data'
            limited_data = True

    sg_total = float(sg_data.get('sg_total', 0) or 0)
    sg_app   = float(sg_data.get('sg_app',   0) or 0)
    sg_putt  = float(sg_data.get('sg_putt',  0) or 0)
    sg_arg   = float(sg_data.get('sg_arg',   0) or 0)
    sg_ott   = float(sg_data.get('sg_ott',   0) or 0)
    sg_t2g   = float(sg_data.get('sg_t2g',   sg_app + sg_arg + sg_ott) or 0)

    # Form rates: from 2022 FM row if available
    if plow in fm_2022.index:
        fm_row = fm_2022.loc[plow]
        top10_rate = float(fm_row.get('top10_rate', 0.2))
        cut_rate   = float(fm_row.get('cut_rate',   0.75))
        sg_total_w = float(fm_row.get('sg_total_weighted', sg_total))
        sg_app_w   = float(fm_row.get('sg_app_weighted',   sg_app))
        sg_putt_w  = float(fm_row.get('sg_putt_weighted',  sg_putt))
        sg_arg_w   = float(fm_row.get('sg_arg_weighted',   sg_arg))
        sg_ott_w   = float(fm_row.get('sg_ott_weighted',   sg_ott))
        sg_t2g_w   = float(fm_row.get('sg_t2g_weighted',   sg_t2g))
    else:
        # Use current SG as weighted value (no historical Kaggle data)
        sg_total_w = sg_total; sg_app_w = sg_app; sg_putt_w = sg_putt
        sg_arg_w   = sg_arg;   sg_ott_w = sg_ott; sg_t2g_w  = sg_t2g
        top10_rate = 0.20; cut_rate = 0.75

    # Override weighted values with current skill-ratings (avoids stale 2022 values)
    if sg_source == 'skill-ratings':
        sg_total_w = sg_total; sg_app_w = sg_app; sg_putt_w = sg_putt
        sg_arg_w   = sg_arg;   sg_ott_w = sg_ott; sg_t2g_w  = sg_t2g

    # Augusta history
    astats = aug_stats(plow)
    appearances = astats['masters_appearances']
    avg_finish  = astats['masters_avg_finish']
    best_finish = astats['masters_best_finish']
    masters_wins_val = astats['masters_wins']
    rfb          = astats['recent_form_bonus']
    recent_avg   = astats['recent_masters_avg']

    # Impute first-timers / no history
    field_avg_finish  = 25.0
    field_best_finish = 20.0
    if pd.isna(avg_finish):
        avg_finish  = field_avg_finish
    if pd.isna(best_finish):
        best_finish = field_best_finish

    # Augusta fit composite
    aug_fit = (2.0*sg_app_w + 2.0*sg_putt_w + 1.0*sg_total_w + 0.5*sg_ott_w) / 5.5

    rows_2026.append({
        'player_name':         pname,
        'player_lower':        plow,
        'sg_total_weighted':   sg_total_w,
        'sg_app_weighted':     sg_app_w,
        'sg_putt_weighted':    sg_putt_w,
        'sg_arg_weighted':     sg_arg_w,
        'sg_ott_weighted':     sg_ott_w,
        'sg_t2g_weighted':     sg_t2g_w,
        'top10_rate':          top10_rate,
        'cut_rate':            cut_rate,
        'augusta_fit':         aug_fit,
        'masters_appearances': appearances,
        'masters_avg_finish':  avg_finish,
        'masters_best_finish': best_finish,
        'masters_wins':        masters_wins_val,
        'recent_form_bonus':   rfb,
        'recent_masters_avg':  recent_avg,
        'sg_source':           sg_source,
        'limited_data':        limited_data,
    })

df26 = pd.DataFrame(rows_2026)
print(f"Built 2026 feature vectors for {len(df26)} players.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Print Scheffler's full feature vector
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Scheffler's 2026 feature vector")
print("=" * 60)

sch = df26[df26['player_lower'].str.contains('scheffler')].iloc[0]
print(f"Player: {sch['player_name']}  (sg_source={sch['sg_source']})\n")
print(f"  {'Feature':<25} {'Value':>10}")
print(f"  {'-'*36}")
for feat in FEATURE_COLS + ['recent_masters_avg']:
    val = sch.get(feat, 'N/A')
    if isinstance(val, float):
        print(f"  {feat:<25} {val:>10.4f}")
    else:
        print(f"  {feat:<25} {str(val):>10}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Run all 7 models on 2026 field
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Running models on 2026 field")
print("=" * 60)

X26 = imp.transform(df26[FEATURE_COLS])

# Get raw predictions
raw_won   = models['won'].predict_proba(X26)[:, 1]
raw_top5  = models['top5'].predict_proba(X26)[:, 1]
raw_top10 = models['top10'].predict_proba(X26)[:, 1]
raw_top16 = models['top16'].predict_proba(X26)[:, 1]
raw_top32 = models['top32'].predict_proba(X26)[:, 1]
raw_mc    = models['made_cut'].predict_proba(X26)[:, 1]
raw_fp    = models['finish_position'].predict(X26)

# Normalize win probability to sum to 100%
win_pct = raw_won / raw_won.sum() * 100

df26['win_pct']      = win_pct
df26['top5_pct']     = raw_top5 * 100
df26['top10_pct']    = raw_top10 * 100
df26['top16_pct']    = raw_top16 * 100
df26['top32_pct']    = raw_top32 * 100
df26['made_cut_pct'] = raw_mc * 100
df26['proj_finish']  = raw_fp


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: DataGolf skill-derived win probabilities (softmax on sg_total)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: DataGolf skill-derived win probabilities")
print("=" * 60)
print("NOTE: preds/pre-tournament shows Valero Texas Open (Masters not yet")
print("current event; tournament starts April 10). Using DataGolf sg_total")
print("from preds/skill-ratings + softmax to derive DG win probabilities.")

# Softmax on sg_total from skill-ratings — restricted to 93-player field
# This converts DataGolf's current authoritative skill ratings to win probabilities
sg_totals = df26['sg_total_weighted'].values.astype(float)
# Temperature-scaled softmax (τ=1.5 spreads the distribution realistically)
tau = 1.5
exp_vals = np.exp(sg_totals / tau)
dg_win_pct = (exp_vals / exp_vals.sum()) * 100

df26['dg_win_pct'] = dg_win_pct

# Show top 10 DG-derived
dg_top = df26.nlargest(10, 'dg_win_pct')[['player_name', 'sg_total_weighted', 'dg_win_pct']]
print("\nDataGolf skill-derived win probability (top 10):")
for _, r in dg_top.iterrows():
    print(f"  {r['player_name']:<30} sg_total={r['sg_total_weighted']:.3f}  dg_win%={r['dg_win_pct']:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Blend predictions (60% model + 40% DataGolf)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Blended predictions (0.6 × model + 0.4 × DataGolf)")
print("=" * 60)

df26['blended_win_pct'] = 0.6 * df26['win_pct'] + 0.4 * df26['dg_win_pct']

# Re-normalize to exactly 100%
df26['blended_win_pct'] = df26['blended_win_pct'] / df26['blended_win_pct'].sum() * 100

print("\nScheffler blending detail:")
sch_row = df26[df26['player_lower'].str.contains('scheffler')].iloc[0]
print(f"  Model win%:          {sch_row['win_pct']:.2f}%")
print(f"  DataGolf win%:       {sch_row['dg_win_pct']:.2f}%")
print(f"  Blended win%:        {sch_row['blended_win_pct']:.2f}%")

print("\nMcIlroy blending detail:")
mc_row = df26[df26['player_lower'].str.contains('mcilroy')].iloc[0]
print(f"  Model win%:          {mc_row['win_pct']:.2f}%")
print(f"  DataGolf win%:       {mc_row['dg_win_pct']:.2f}%")
print(f"  Blended win%:        {mc_row['blended_win_pct']:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Print blended top 20
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Blended Top 20 — 2026 Masters Predictions")
print("=" * 60)

top20 = df26.nlargest(20, 'blended_win_pct').reset_index(drop=True)

header = (f"{'Rank':<5} {'Player':<28} {'Blend%':>7} {'Model%':>7} {'DG%':>7} "
          f"{'Top10%':>7} {'ProjFin':>8} {'AugWins':>8} {'RFB':>5}")
print(header)
print("-" * len(header))
for i, r in top20.iterrows():
    print(
        f"{i+1:<5} {r['player_name']:<28} {r['blended_win_pct']:>7.2f} "
        f"{r['win_pct']:>7.2f} {r['dg_win_pct']:>7.2f} "
        f"{r['top10_pct']:>7.1f} {r['proj_finish']:>8.1f} "
        f"{int(r['masters_wins']):>8} {int(r['recent_form_bonus']):>5}"
    )

# Save
df26_out = df26[['player_name', 'blended_win_pct', 'win_pct', 'dg_win_pct',
                  'top5_pct', 'top10_pct', 'top16_pct', 'top32_pct', 'made_cut_pct',
                  'proj_finish', 'masters_appearances', 'masters_avg_finish',
                  'masters_best_finish', 'masters_wins', 'recent_form_bonus',
                  'recent_masters_avg', 'sg_source', 'limited_data']].copy()
df26_out.sort_values('blended_win_pct', ascending=False, inplace=True)
df26_out.to_csv(PREDS_OUT, index=False)
print(f"\nSaved -> {PREDS_OUT}")

# Betting value
try:
    bv = pd.read_csv(BETTING_OUT)
    if 'player_name' in bv.columns and 'market_implied_pct' in bv.columns:
        bv2 = bv.merge(
            df26_out[['player_name', 'blended_win_pct']],
            on='player_name', how='left'
        )
        bv2['edge'] = bv2['blended_win_pct'] - bv2['market_implied_pct']
        bv2['value_flag'] = bv2['edge'].apply(lambda x: 'Value' if x > 0 else 'No Value')
        bv2.to_csv(BETTING_OUT, index=False)
        print(f"Updated betting value -> {BETTING_OUT}")
        value_bets = bv2[bv2['value_flag'] == 'Value'].sort_values('edge', ascending=False)
        if len(value_bets):
            print("\nValue bets (blended model vs market):")
            for _, r in value_bets.head(10).iterrows():
                print(f"  {r['player_name']:<30} edge={r['edge']:.2f}%")
        else:
            print("\nNo value bets found with blended probabilities.")
except Exception as e:
    print(f"Betting value update skipped: {e}")

print("\nDone.")
