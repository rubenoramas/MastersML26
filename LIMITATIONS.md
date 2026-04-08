# MastersML26 — Model Limitations

## LIV Golf Player Limitations

### Overview
10 of the 93 players in the 2026 Masters field compete primarily on the LIV Golf League rather than the PGA Tour. This creates two specific data limitations in our model:

**1. Training Data Gap**
The Kaggle dataset (2015-2022) is PGA Tour data only. LIV Golf launched in 2022, meaning players who defected — primarily Rahm (left after 2023 Masters), DeChambeau (left 2022), Johnson (left 2022), Smith (left 2022) — have little or no recent PGA Tour data. The model's current form features for these players come entirely from DataGolf skill-ratings, which do track LIV performance, but the model was trained on PGA Tour performance patterns.

**2. Field Quality Normalization Problem**
LIV Golf fields are significantly weaker on average than PGA Tour fields. A player gaining +1.5 SG against a LIV field is not equivalent to gaining +1.5 SG against a PGA Tour field. DataGolf applies their own adjustments to normalize this, but residual inflation likely remains. This means LIV players' current form metrics may be overstated relative to PGA Tour players.

### The Two Most Affected Players

**Jon Rahm — Currently 2nd in betting odds (~+900), ranked #14 in our model**
- Won the 2023 Masters — documented in our training data
- Left for LIV after the 2023 Masters
- 2026 LIV season: won Hong Kong, three runner-up finishes, leads LIV in SG: Approach
- Our model sees strong Augusta history but current form metrics derived from LIV data
- Expected model underranking: Rahm's elite current form is not fully captured due to field quality normalization
- Documented as a known limitation — his true contender probability is likely higher than our model outputs

**Bryson DeChambeau — Currently ~+1000 odds, ranked #10 in our model**
- Left for LIV in 2022 — most recent PGA Tour data is 4 years old
- 2026 LIV season: won last two events (Singapore, South Africa) entering Augusta
- Has finished top 6 at the last two Masters (T6 in 2024, T5 in 2025)
- Our model partially captures his Augusta history but current form underweighted
- Expected model underranking: similar field quality normalization issue as Rahm

### All 10 LIV Players in the 2026 Masters Field
| Player | How Qualified | Our Rank | Known Limitation |
|--------|--------------|----------|-----------------|
| Jon Rahm | Past champion (2023) | #14 | Field quality normalization |
| Bryson DeChambeau | Past major winner (2024 US Open) | #10 | Field quality normalization |
| Dustin Johnson | Past champion (2020) | Outside top 20 | Minimal recent data |
| Cameron Smith | OWGR top 50 | Outside top 20 | Minimal recent data |
| Sergio Garcia | Past champion (2017) | Outside top 20 | Veteran, limited current data |
| Tyrrell Hatton | Top 4 at 2025 US Open | Outside top 20 | LIV normalization |
| Bubba Watson | Past champion (2012, 2014) | Outside top 20 | Age, limited current data |
| Charl Schwartzel | Past champion (2011) | Outside top 20 | Age, limited current data |
| Tom McKibbin | 2025 Hong Kong Open winner | Outside top 20 | Limited PGA Tour history |
| Carlos Ortiz | Top 4 at 2025 US Open | Outside top 20 | Limited Augusta history |

### Scottie Scheffler Limitation (separate but related)
Scheffler ranked #8 in our blended model (1.84%) vs market implied ~20% (currently +400 favorite).

Root cause: The Kaggle training data (2015-2022) only contains 2 Scheffler Augusta appearances before his dominant run began. The model never encountered a player with his profile — world #1 for nearly 3 consecutive years, 2 Augusta wins in 3 years, dominant SG metrics across all categories. This is not a LIV issue but a training data recency issue. The 40% DataGolf blend partially corrects this (DataGolf has him at 3.41%), producing a blended 1.84%, but this still significantly underestimates his true win probability.

**What this means for interpreting our predictions:**
The model's most reliable outputs are contender identification (top10/16 probabilities) rather than exact win percentages. Our cross-validation confirmed top10 prediction at 0.744 mean AUC with low variance (±0.047) across all folds. The win probability rankings should be treated as relative orderings of statistical fit to Augusta's course demands, not calibrated absolute probabilities.
