# MastersML26 — Project Log

## Project Overview
- **Analytics Question:** "Which pre-tournament performance metrics best predict Masters Tournament outcomes, and can a data-driven model outperform betting market implied probabilities?"
- **Course:** Sports Data Analysis, University of Oregon
- **Due date:** May 3, 2026

## Data Sources & Decisions
- **Kaggle "ASA All PGA Raw Data - Tourn Level.csv"** — 36,864 rows × 37 columns, 2015–2022 historical PGA Tour tournament-level data including all SG categories, finish position, made_cut
- **DataGolf API `preds/skill-ratings`** — 434 active players, current rolling SG averages (sg_total, sg_app, sg_arg, sg_ott, sg_putt, driving_acc, driving_dist)
- **DataGolf API `preds/player-decompositions`** — 130 players (partial confirmed Masters field as of March 30), Augusta-specific adjustments including course_history_adjustment, final_pred (Augusta-adjusted SG per round)
- **`data/raw/masters_2026_field.csv`** — authoritative 93-player confirmed field (manually curated); used as the master player list in notebook 05
- **DataGolf API `betting-tools/outrights`** — current 2026 Masters live odds

## Problems & Workarounds
- **Problem:** DataGolf API historical endpoints (historical-raw-data/rounds, historical-odds/outrights) returned 403 — not included in base subscription tier
- **Workaround:** Using Kaggle CSV for historical training data instead
- **Problem:** Kaggle dataset does not cover 2023–2026
- **Workaround:** DataGolf preds/player-decompositions provides the full 2026 Masters field with Augusta-specific adjustments already computed, bridging the gap with current form data
- **Problem:** Kaggle CSV mislabels the November 2020 Masters (COVID-postponed from April 2020, played 2020-11-15, tournament_id=401219478) as season=2021, creating duplicate player-season entries that inflated the feature matrix 660 → 993 rows via Cartesian merges
- **Workaround:** Reassigned tournament_id 401219478 to season=2020 at load time, restoring correct 8-season structure (2015–2022 including 2020) and eliminating all duplicates
- **Problem:** Kaggle CSV cuts off at 2022 — Masters results for 2023, 2024, and 2025 are missing, leaving Augusta history statistics (appearances, avg finish, best finish, wins) stale for all returning players
- **Workaround:** Manually compiled `data/raw/masters_results_2023_2025.csv` from official PGA Tour records (170 rows across 3 years). Combined with Kaggle Augusta subset at prediction time to build comprehensive 2015–2025 Augusta history per player. Added two new features: `masters_wins` (career Augusta wins count) and `recent_masters_avg` (mean finish of last 3 made-cut appearances).

## API Endpoints — Accessible vs Blocked
- `preds/skill-ratings` — SUCCESS
- `preds/pre-tournament` — SUCCESS
- `field-updates` — SUCCESS
- `preds/player-decompositions` — SUCCESS (130 Masters field players)
- `betting-tools/outrights` — SUCCESS
- `historical-raw-data/rounds` — BLOCKED (403)
- `historical-event-stats/finishes` — ERROR (404)
- `historical-odds/outrights` — BLOCKED (403)

## Tools Used
Python, Jupyter Notebooks, DataGolf API, Kaggle, XGBoost, scikit-learn, pandas, seaborn, matplotlib, Tableau, VS Code, GitHub

## Key Findings

### Notebook 02 — EDA & Correlation (Masters historical data, 2015–2022)
- **Masters subset:** 660 rows, 213 unique players, 7 years (2015–2022; no 2020 due to COVID postponement)
- **Made-cut rate:** 65.9% of Masters starts result in making the cut
- **Top 5 predictors of Masters finish position** (Pearson r, all p < 0.001):
  1. `sg_total`  r = -0.823 — strongest single predictor; higher total SG → better finish
  2. `n_rounds`  r = -0.802 — proxy for making the cut; collinear with made_cut
  3. `made_cut`  r = -0.800 — nearly identical signal to n_rounds
  4. `sg_t2g`    r = -0.693 — tee-to-green dominates Augusta; course rewards ball-striking
  5. `sg_app`    r = -0.608 — approach play is the single most informative SG sub-category
- **SG sub-category ranking by predictive strength:** sg_app (-0.608) > sg_arg (-0.390) > sg_ott (-0.370) > sg_putt (-0.305)
- **Key insight:** Approach play (sg_app) outperforms putting (sg_putt) at Augusta — consistent with the course's reputation for rewarding iron play into undulating greens
- **Outputs:** `outputs/figures/sg_distributions.png`, `outputs/figures/correlation_heatmap.png`, `outputs/csv/correlation_matrix.csv`, `outputs/csv/sg_stats_summary.csv`

### Notebook 03 — Feature Engineering
- **Final feature matrix:** 621 player-year rows × 23 columns, saved to `data/features/feature_matrix.csv`
- **Seasons covered:** 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022 (8 Masters events)
- **Unique players:** 202; zero duplicate player-season rows
- **Dropped:** 39 players with no prior SG data in the dataset window (early-career players whose PGA Tour history predates the CSV coverage)
- **Rolling form features (decay=0.85, window=12):** sg_total_weighted, sg_app_weighted, sg_putt_weighted, sg_arg_weighted, sg_ott_weighted, sg_t2g_weighted, top10_rate, cut_rate
- **Augusta history features:** masters_appearances, masters_avg_finish, masters_best_finish (NaN for 197 first-time or early-career entrants — expected; will use imputation in modeling)
- **Augusta fit score:** composite of sg_app (×2), sg_putt (×2), sg_total (×1), sg_ott (×0.5), normalized by 5.5
- **Target distributions (621 rows):** made_cut 68.6%, top32 41.9%, top16 21.7%, top10 14.5%, top5 7.9%, won 1.3%
- **Data issue found & fixed:** Kaggle CSV misassigned the November 2020 Masters to season=2021, creating 65 duplicate player-season entries. Fixed by reassigning tournament_id 401219478 to season=2020.

### Notebook 04 — Modeling

**Architecture:** 6 XGBoost classifiers (one per target: won, top5, top10, top16, top32, made_cut) + 1 XGBoost regressor (finish_position). n_estimators=400, max_depth=4, lr=0.05, subsample=0.8. Class imbalance corrected via `scale_pos_weight = n_neg/n_pos` (ranges from ~1.4× for made_cut to ~76× for won). Train: 2015–2021 (540 rows, 7 seasons). Holdout: 2022 (81 rows).

**Final model performance on 2022 holdout (20-feature model, with l6 features):**

| Target | Type | ROC-AUC | Log Loss | MAE | RMSE |
|---|---|---|---|---|---|
| won | Classifier | 0.9625 | 0.0463 | — | — |
| top5 | Classifier | 0.9158 | 0.1783 | — | — |
| top10 | Classifier | 0.7221 | 0.4630 | — | — |
| top16 | Classifier | 0.6202 | 0.7230 | — | — |
| top32 | Classifier | 0.5638 | 0.8653 | — | — |
| made_cut | Classifier | 0.7440 | 0.5798 | — | — |
| finish_position | Regressor | — | — | 24.28 | 29.01 |

**Leave-one-year-out cross-validation (7 folds: 2015–2022 excl. 2020):**

| Target | CV Mean AUC | Std | Assessment |
|---|---|---|---|
| won | 0.5687 | ±0.446 | Volatile — inherently hard to predict winner in golf |
| top5 | 0.7534 | ±0.083 | Good, moderate variance |
| top10 | 0.7444 | ±0.047 | **Reliable — most consistent signal** |
| top16 | 0.7043 | ±0.061 | Reliable |
| top32 | 0.6471 | ±0.065 | Moderate |
| made_cut | 0.6535 | ±0.070 | Moderate |
| finish_position MAE | 23.30 | ±2.17 | Stable across folds |

- **Hardest year: 2017** (avg AUC 0.548) — Sergio Garcia's first Augusta win; no model signal
- **Easiest year: 2019** (avg AUC 0.778) — Tiger Woods' return win; strong SG + course history
- **Key finding:** The 2022 single-holdout AUC of 0.963 for `won` was an outlier. CV reveals that win prediction is inherently volatile in golf (a 79-player field where any player can win). The model's most reliable, actionable output is `top10`/`top16` probability, not win percentage. Win probabilities should be treated as relative orderings, not calibrated probabilities.

**2022 winner validation:** Scheffler ranked **#4 of 81** by predicted win probability (p=0.038). Top pick was Russell Henley (p=0.095), who finished T30.

**Top feature importances (won model — final 20-feature version):**
1. `sg_t2g_weighted` — 25.4% (dominant signal, consistent with EDA r=-0.693)
2. `sg_arg_weighted` — 11.6%
3. `l6_avg_finish` — 7.0% (new from l6 features)
4. `l6_top10_rate` — 6.4% (new from l6 features)
5. `sg_app_weighted` — 4.9%

**Infrastructure fix:** XGBoost requires `libomp.dylib` (missing from base macOS). Fixed by loading sklearn's bundled libomp via `ctypes.CDLL` before importing xgboost. Also set via `DYLD_LIBRARY_PATH` for `nbconvert` execution.

**Saved:** `outputs/models/xgb_*.joblib` (7 files), `outputs/csv/model_evaluation.csv`, `outputs/csv/feature_importance.csv`, `outputs/csv/cross_validation_results.csv`

## Modeling Decisions

### Notebook 04 — Modeling
- **Models trained:** 7 total — XGBoost Classifier for won, top5, top10, top16, top32, made_cut; XGBoost Regressor for finish_position
- **Train/validation split:** Training 2015–2021, Holdout 2022 (never seen during training)
- **Final holdout performance:** won AUC 0.9625, top5 AUC 0.9158, top10 AUC 0.7221, top16 AUC 0.6202, made_cut AUC 0.7440, finish MAE 24.28
- **Cross-validation (leave-one-year-out):** won CV mean 0.5687 ±0.446 (volatile — golf variance), top10 CV mean 0.7444 ±0.047 (stable and reliable), top16 CV mean 0.7043 ±0.061, finish CV MAE 23.30 ±2.17
- **Key finding:** Win prediction is inherently volatile in golf. Top10/16/32 prediction is the model's most reliable and academically defensible output. The 2022 holdout won AUC of 0.963 was an outlier — CV mean of 0.569 is the honest estimate.
- **Hardest year:** 2017 (Garcia win, avg AUC 0.548 — unpredicted outcome)
- **Easiest year:** 2019 (avg AUC 0.778)
- **Top feature:** sg_t2g_weighted at 25.4% importance in win model — consistent with EDA finding
- **Class imbalance handling:** scale_pos_weight parameter in XGBoost for all classifiers
- **Models saved:** 7 .joblib files in outputs/models/

### Notebook 05 — Predictions & Output
- **2026 field:** 93 players from manually compiled masters_2026_field.csv (official confirmed field as of March 30, 2026)
- **SG source for 2026:** DataGolf preds/skill-ratings (434 active players, updated weekly)
- **Augusta history:** Kaggle 2015–2022 + manually compiled masters_2023_2025_results.csv from official PGA Tour records
- **l6 feature proxy:** DataGolf skill-ratings used as 2026 l6 approximation — sg_total/app/putt mapped directly, l6_top10_rate and l6_avg_finish imputed from global SG rank buckets
- **Model stacking:** final_win_pct = 0.6 × model_win_pct + 0.4 × DataGolf_win_pct (softmax normalized)
- **Top prediction:** Tommy Fleetwood 16.49% blended win probability — model identifies as statistically undervalued vs market (~+2500 odds)
- **Known limitation:** Scheffler ranks #8 at 1.84% blended vs market implied ~20%. Root cause: training data (2015–2022) contains only 2 Scheffler Augusta appearances before his dominant 2022–2024 run. The blending step partially corrects this via DataGolf's 3.41% contribution.
- **Features tested and reverted:** sg_t2g_elite binary feature (Kyle Porter external research) — improved finish MAE but added noise to win classifier due to insufficient training examples meeting threshold. Documented as future improvement with more training data.
- **Tableau CSVs saved:**
  - outputs/csv/predictions_2026.csv
  - outputs/csv/correlation_matrix.csv
  - outputs/csv/betting_value.csv
  - outputs/csv/model_evaluation.csv
  - outputs/csv/cross_validation_results.csv
  - outputs/csv/feature_importance.csv

### L6 Recent Form Features (added 2026-03-31)

**New features added to training data and all three notebooks:**

6 last-6-events simple averages (same-season PGA Tour events before each Masters, no decay weighting):
- `l6_sg_total_avg` — simple avg sg_total over last 6 same-season events
- `l6_sg_app_avg` — simple avg sg_app
- `l6_sg_putt_avg` — simple avg sg_putt
- `l6_top10_rate` — fraction of last 6 finishing top 10
- `l6_cut_rate` — fraction of last 6 making the cut
- `l6_avg_finish` — simple avg finishing position (missed cuts penalized as 80)

**Implementation notes:**
- Minimum 2 same-season events required; fewer → impute from player's career non-Masters averages
- Career average imputation covers 66 of 660 training rows (10%) — mostly early-career players with sparse Jan–April schedules
- Feature matrix rebuilt: 621 × 31 columns (was 23 base → 26 after Scheffler fix → 31 with l6)
- All 7 XGBoost models retrained with 20-feature FEATURE_COLS (12 base + 2 aug wins + 6 l6)

**2022 holdout evaluation — before vs after l6 features:**

| Target | Before (no l6) | After (with l6) | Δ |
|---|---|---|---|
| won | ROC-AUC 0.9875 | 0.9625 | -0.025 |
| top5 | 0.9263 | 0.9158 | -0.011 |
| top10 | 0.7013 | **0.7221** | +0.021 |
| top16 | 0.6094 | 0.6202 | +0.011 |
| top32 | 0.5601 | 0.5638 | +0.004 |
| made_cut | 0.7062 | **0.7440** | +0.038 |
| finish_position MAE | 23.85 | 24.28 | +0.43 |

l6 features improved `top10` and `made_cut` classification while the `won` model saw slight AUC decrease — consistent with l6 providing incremental short-term form signal that helps broad field separation more than narrow winner identification.

**Feature importance (won model) — top 5 after l6 addition:**
1. `sg_t2g_weighted` — 25.4% (unchanged dominant signal)
2. `sg_arg_weighted` — 11.6%
3. `l6_avg_finish` — **7.0%** (new, 3rd most important)
4. `l6_top10_rate` — **6.4%** (new, 4th most important)
5. `sg_app_weighted` — 4.9%

`masters_wins` and `recent_form_bonus` show 0% importance in the won model — sparse training signal (only 3 rows with `recent_form_bonus=1` in training data 2015–2022).

**2026 predictions — known limitation of l6 proxy:**
- 2026 same-season pre-Masters events not available in the dataset
- Proxy: skill-ratings SG for l6 SG features; player's most recent fm_latest l6 rates (from 2022 or earlier) for l6_top10_rate, l6_cut_rate, l6_avg_finish
- Initial proxy (fm_latest l6 values) caused stale-data bias — Justin Thomas ranked #1 at 36.69% from his strong Jan–April 2022 form; McIlroy dropped to #8 at 2.89%
- **Fixed in `predict_2026_corrected.py`**: l6 features now use DataGolf skill-ratings as 2026 proxy since Kaggle data cutoff is 2022. This is a deliberate design decision — skill-ratings reflect current rolling form which is the underlying signal l6 features capture.

  Corrected l6 proxy rules:
  - `l6_sg_total_avg` = `sg_total` from skill-ratings (current rolling form)
  - `l6_sg_app_avg` = `sg_app` from skill-ratings
  - `l6_sg_putt_avg` = `sg_putt` from skill-ratings
  - `l6_top10_rate`: global sg_total rank → top-10 = 0.50, top-25 = 0.30, else 0.15
  - `l6_cut_rate`: 0.85 active tour / 0.75 kaggle-career / 0.65 limited-data
  - `l6_avg_finish`: field sg_total rank → 1–10 = 8, 11–25 = 18, 26–50 = 30, else 45

**Final blended top 10 (0.6 × model + 0.4 × DataGolf softmax, after l6 fix):**

| Rk | Player | Blend% | Model% | DG% | Top10% | Wins | RFB |
|---|---|---|---|---|---|---|---|
| 1 | Tommy Fleetwood | 16.49% | 26.20% | 1.93% | 3.9% | 0 | 0 |
| 2 | Hideki Matsuyama | 9.24% | 14.39% | 1.50% | 50.5% | 1 | 1 |
| 3 | Matt Fitzpatrick | 7.22% | 10.59% | 2.15% | 25.8% | 0 | 0 |
| 4 | Viktor Hovland | 6.40% | 9.79% | 1.32% | 8.3% | 0 | 0 |
| 5 | Patrick Cantlay | 4.95% | 7.25% | 1.50% | 58.6% | 0 | 0 |
| 6 | Rory McIlroy | 3.25% | 3.74% | 2.52% | 57.7% | 1 | 1 |
| 8 | Scottie Scheffler | 1.84% | 0.78% | 3.41% | 19.1% | 2 | 1 |
| 14 | Jon Rahm | 1.43% | 0.53% | 2.78% | 53.6% | 1 | 1 |
| 18 | Justin Thomas | 1.31% | 1.36% | 1.23% | 71.0% | 0 | 0 |

Justin Thomas falls to #18 (1.31%) with corrected l6 proxy (l6_top10=0.15, l6_avg_fin=30) since he's ranked outside the global top 25 by sg_total. Scheffler rises to #8 (1.84%) — still below market but blending with DG softmax (3.41%) helps. Fleetwood remains #1 driven by the model.

### Leave-One-Year-Out Cross-Validation (added 2026-04-01)

**Methodology:** 7 folds over years [2015, 2016, 2017, 2018, 2019, 2021, 2022]. Each fold trains all 7 XGBoost models on 6 years, evaluates on the held-out year. 2020 excluded from CV folds (COVID-postponed November Masters with non-standard field/calendar context; still used in final training). Final saved models unchanged — CV is evaluation only.

**Cross-validation results vs single 2022 holdout:**

| Target | CV Mean AUC | CV Std | CV Min | CV Max | 2022-only | Δ |
|---|---|---|---|---|---|---|
| won | 0.5687 | ±0.446 | 0.027 | 0.963 | 0.9625 | -0.394 |
| top5 | 0.7534 | ±0.083 | 0.673 | 0.916 | 0.9158 | -0.162 |
| top10 | 0.7444 | ±0.047 | 0.689 | 0.839 | 0.7221 | +0.022 |
| top16 | 0.7043 | ±0.061 | 0.620 | 0.778 | 0.6202 | +0.084 |
| top32 | 0.6471 | ±0.065 | 0.564 | 0.776 | 0.5638 | +0.083 |
| made_cut | 0.6535 | ±0.070 | 0.529 | 0.744 | 0.7440 | -0.091 |
| finish MAE | 23.30 | ±2.17 | 19.27 | 25.98 | 24.28 | -0.98 |
| finish RMSE | 28.31 | ±2.44 | 23.91 | 31.47 | 29.01 | -0.70 |

**Per-year average AUC across all 6 classifiers:**
- 2017: 0.5479 (hardest)
- 2021: 0.6048
- 2018: 0.6190
- 2016: 0.7007
- 2015: 0.7448
- 2022: 0.7547
- 2019: 0.7781 (easiest)

**Key findings:**

1. **`won` model is highly volatile** — mean AUC of 0.569 with std ±0.446 reveals extreme fold-to-fold variance. The 2022 holdout AUC of 0.963 was an unusually favorable year (Scheffler ranked #4). In 2017 (Sergio Garcia's first Augusta win) the model AUC was 0.027 — essentially random. In 2021 (Matsuyama) it was 0.198. The winner model should not be interpreted as reliably strong.

2. **Bracket targets are stable and credible** — `top10` (CV mean 0.744), `top16` (0.704), and `top32` (0.647) show low variance (std ±0.047–0.065), confirming consistent signal for field separation. These are more trustworthy for tournament analysis than the win probability output.

3. **2022 was not representative** — the single 2022 holdout overstated `won` and `top5` accuracy. CV gives a more honest picture: the model is a reliable field-separator but a noisy winner predictor.

4. **Hardest year: 2017** — Sergio Garcia's victory; model struggled across all targets (avg AUC 0.548). Garcia had limited Augusta history and the field was unpredictable. **Easiest year: 2019** — Tiger Woods' return win; his deep Augusta history and strong SG form made him a model-identifiable favorite.

5. **Finish position MAE is stable** — 23.3 ± 2.2 strokes across folds; the 2022 holdout (24.28) was representative.

**Implications for 2026 predictions:** Win probability rankings (Fleetwood #1 at 16.5%, etc.) should be treated as relative orderings rather than calibrated probabilities. The model's most actionable output is field segmentation (top10/top16 probabilities) rather than explicit win percentages. The DataGolf softmax blend partially corrects for winner-model instability by anchoring predictions to current SG skill ratings.

**Saved:** `outputs/csv/cross_validation_results.csv` (49 rows: 7 years × 7 targets)

### Scheffler Calibration Fix + Model Stacking (added 2026-03-31)

**Root cause of Scheffler underranking:** The XGBoost model was trained on 2015–2022 data where Scheffler had only 2 Augusta appearances (2020, 2021) with modest finishes (19th, 18th). His 2022 win was in the training holdout year (not trained on) and his 2023/2024 appearances were outside the Kaggle CSV window. So the model's Augusta history features painted him as a mid-tier debutant, not a two-time champion.

**Fix 1 — New features added to training data and 2026 vectors:**
- `masters_wins`: count of Augusta wins before the current season (0 for Scheffler in all training rows; **2** in his 2026 prediction vector)
- `recent_form_bonus`: binary flag, 1 if player won Masters in 2020 or later (captures the era-relevant winner signal; **1** for Scheffler in 2026)
- Training set rows with `recent_form_bonus=1`: Dustin Johnson (2021, 2022) and Hideki Matsuyama (2022) — those who won in 2020 then appeared in subsequent events
- All 7 models were retrained with these new features; `predictions_2026.csv` and `betting_value.csv` overwritten

**Fix 2 — Model stacking (DataGolf skill blend):**
- `preds/pre-tournament` was not available for the Masters at time of run (tournament starts April 10; DataGolf's live pre-tournament endpoint showed Valero Texas Open, the current week's event)
- **Workaround:** Applied temperature-scaled softmax (τ=1.5) on DataGolf `sg_total` from `preds/skill-ratings` across the 93-player field, converting current skill ratings to win probabilities. This serves as DataGolf's authoritative current-form signal.
- **Blend formula:** `final_win_pct = 0.6 × model_win_pct + 0.4 × datagolf_derived_win_pct`
- Re-normalized to sum to 100%

**Scheffler post-fix blending detail:**
- Model win%: 0.51% → still underweighted but `recent_form_bonus=1` and `masters_wins=2` are now in the feature vector
- DataGolf win% (softmax sg_total=2.570): 3.43%
- **Blended win%: 1.68% → rank #11** (up from #17)
- Still below market implied (~10%), reflecting residual model limitation from sparse Augusta training data

**Key blended predictions:**
- McIlroy: 17.45% (model 27.40%, DG 2.53%)
- Fitzpatrick: 8.75%, Cantlay: 5.77%, Rahm: 4.63%
- Scheffler: 1.68% (#11), Schauffele: 3.12% (#7)

### Notebook 05 — 2026 Predictions & Betting Value

**Field construction:**
- Authoritative 93-player field from `data/raw/masters_2026_field.csv` (manually curated confirmed entries as of 2026-03-30)
- DataGolf `preds/player-decompositions` covered only 24 of those 93 players at time of pull — the API was still tracking an incomplete field 10 days before the tournament; the major players (Scheffler, McIlroy, Rahm, etc.) had not yet been confirmed in DataGolf's system

**SG feature construction for 2026:**
- 79 players: current `preds/skill-ratings` (active tour players)
- 14 players: Kaggle CSV career averages (veterans: Woods, Couples, Weir, Singh, plus others matched)
- 10 players flagged `limited_data=True` (SG=0; predictions unreliable): Cabrera, Fang, Brennan, Herrington, Holtz, Howell, Kataoka, Laopakdee, Olazabal, Pulcini
- Augusta history (appearances, avg/best finish, form rates) from feature matrix for 52 matched players; field-mean imputation for first-timers and unmatched players
- DataGolf `course_history_adjustment` applied where available to fine-tune imputed avg/best finish for unmatched players

**Top 10 model predictions (win probability) — updated with 2023–2025 Augusta history:**

| Rank | Player | Win% | Top10% | ProjFinish | AugAppearances | AugAvgFinish | AugBestFinish | AugWins |
|---|---|---|---|---|---|---|---|---|
| 1 | Rory McIlroy | 31.52% | 71.5% | 14.0 | 11 | 8.6 | 1 | 1 |
| 2 | Matt Fitzpatrick | 21.41% | 16.2% | 26.3 | 5 | 18.2 | 10 | 0 |
| 3 | Patrick Cantlay | 8.55% | 14.9% | 29.6 | 5 | 17.4 | 14 | 0 |
| 4 | Jon Rahm | 6.34% | 42.7% | 24.1 | 8 | 11.3 | 1 | 1 |
| 5 | Tommy Fleetwood | 5.19% | 6.3% | 38.1 | 5 | 18.6 | 3 | 0 |
| 6 | Xander Schauffele | 4.48% | 84.2% | 31.5 | 6 | 10.8 | 8 | 0 |
| 7 | Collin Morikawa | 3.80% | 60.8% | 22.7 | 4 | 12.5 | 3 | 0 |
| 8 | Bryson DeChambeau | 2.70% | 8.3% | 42.7 | 5 | 20.4 | 5 | 0 |
| 9 | Keegan Bradley | 1.78% | 21.8% | 21.2 | 3 | 16.3 | 16 | 0 |
| 10 | Ludvig Aberg | 1.65% | — | — | 1 | 2.0 | 2 | 0 |

**Key prediction notes:**
- McIlroy's high win% driven by strong sg_total (2.113), sg_ott (0.936), 11 Augusta appearances, and 2025 win — model weights his course familiarity and recent win heavily
- Scheffler ranks **#17** (win%=0.59%) despite being world #1 (sg_total=2.570, 2 Augusta wins) — model undervalues him because his Augusta appearances (6) and avg finish (8.8) do not yet rank elite compared to the training distribution; 2023–2025 data raised his appearances from 2 to 6 but wasn't enough to overcome the model's historical weighting
- Scheffler's model undervaluation vs market (~10% implied) represents the most notable **buy-side market inefficiency** identified — flagged as a known model limitation
- Betting value: only Tommy Fleetwood identified as "Value" (edge +2.44%) vs market — limited coverage because `betting-tools/outrights` only covered the 130-player partial DataGolf field, leaving most elite players without odds data for edge calculation

**Augusta history updates from 2023–2025 data (key players):**
- Scheffler: 6 appearances, avg=8.8, best=1, wins=2, recent_avg=5.0
- McIlroy: 11 appearances, avg=8.6, best=1, wins=1, recent_avg=8.3
- Rahm: 8 appearances, avg=11.3, best=1, wins=1

**Outputs:** `outputs/csv/predictions_2026.csv`, `outputs/csv/betting_value.csv`, `outputs/figures/predictions_2026_top20.png`

### T2G Elite Feature — Tested and Reverted (2026-03-31)

**Research source:** Kyle Porter (@KylePorterNS) documented that 11 of the last 13 Masters winners (84.6%) had SG:Tee-to-Green ≥ 1.7 in the 3 months before the tournament.

**Features tested:**
- `sg_t2g_3mo_avg` — continuous: average sg_t2g from non-Masters events within 90 days before each Masters
- `sg_t2g_elite` — binary: 1 if `sg_t2g_3mo_avg >= 1.7`, else 0

**What the evaluation showed (22-feature model vs 20-feature model on 2022 holdout):**

| Target | 20-feat | 22-feat (with T2G) | Δ |
|---|---|---|---|
| won AUC | 0.9625 | 0.9467 | -0.016 |
| top5 AUC | 0.9158 | 0.9028 | -0.013 |
| top16 AUC | 0.6202 | 0.6464 | +0.026 |
| finish MAE | 24.28 | 23.27 | -1.01 |
| finish RMSE | 29.01 | 27.75 | -1.26 |

T2G features improved finish position regression and mid-field classification, but degraded the win and top5 classifiers — the models that matter most for predictions.

**Why it was reverted:** The 2026 win probability rankings were noticeably distorted. Cantlay jumped to #2 (18.86% model, no T2G elite flag), Scheffler fell from #8 to #13, Morikawa from #7 to #14, Aberg dropped out of the top 20 entirely. The root cause: the Kaggle 2015–2022 training data has very few examples of players with sg_t2g_3mo_avg ≥ 1.7 (the threshold was rarely reached in the window — mostly imputed from career averages), so the binary `sg_t2g_elite` signal had insufficient training signal to generalize reliably. The feature effectively added noise to the win classifier.

**Decision:** Reverted to 20-feature model (6 SG weighted + top10/cut/fit + 4 Augusta history + 6 l6 recent form). Feature matrix restored to 621 × 31. All 7 models retrained. 2026 predictions confirmed matching pre-T2G output (Fleetwood #1, Matsuyama #2, McIlroy #6, Scheffler #8).

**Lesson:** Externally validated domain rules (11/13 winners) don't automatically translate to useful ML features if the training set lacks sufficient positive examples within the computed window. The Kyle Porter finding remains contextually valid but cannot be reliably operationalized with this training data size.
