# Findings from overnight runs (2026-03-19 to 2026-03-20)

## Round results

| Round | Score | Rank | Approach | Notes |
|-------|-------|------|----------|-------|
| 1 | 49.56 | 29 | Heuristic-only (accidental) | Originally had full observation coverage (~95 est.), but auto_solve.py overwrote with heuristic-only predictions |
| 2 | 0.08 | 151 | Observation 100% coverage, floor=0.001 | Terrible score despite seeing every cell |
| 3 | 2.15 | 93 | Observation 100% coverage, floor=0.001 | Same approach, same bad result |
| 4 | 0.82 | 86 | Observation 100% coverage, floor=0.001 | Same |
| 5 | pending | - | Observation 100% coverage, floor=0.001 | Submitted ~6:15 UTC |

## Critical discovery: ground truth is a probability distribution

The ground truth is NOT a single deterministic outcome. The simulation runs ~200 times with different random seeds, and the ground truth is the **empirical distribution** over those runs.

This means:
- Each cell's ground truth is a 6-class probability vector (not a single class)
- Many cells have significant probability spread across multiple classes
- Putting 99.5% confidence on one observed outcome is catastrophically wrong

### Ground truth statistics (from round 2 analysis)

- Mean max-class probability per cell: **0.749** (significant spread)
- Cells with max prob >= 0.9: **242/1600** (15%) — only these are "nearly deterministic"
- Cells with max prob >= 0.5: **1564/1600** (98%)
- Cells with max prob < 0.5: **36/1600** (2%) — very uncertain cells

### Example of why observation-only fails

Cell (17,2) in round 2, seed 0:
- We observed: forest (class 4)
- Our prediction: [0.001, 0.001, 0.001, 0.001, 0.995, 0.001]
- Ground truth:   [0.070, 0.350, 0.000, 0.015, 0.565, 0.000]
- KL divergence explodes because we put 0.1% on settlement when it should be 35%

Cell (10,28) — most uncertain cell:
- Ground truth: [0.355, 0.360, 0.000, 0.065, 0.220, 0.000]
- No single class dominates — this cell could be plains, settlement, or forest

## Why observation-only scored well in round 1 (49.56)

Round 1's submission was accidentally heuristic-only (no observations), using broader probability distributions like:
- Plains: [0.72, 0.22, 0.01, 0.01, 0.03, 0.01]
- Forest: [0.03, 0.26, 0.08, 0.01, 0.61, 0.01]

These spread distributions are closer to the actual ground truth than putting 99.5% on one class. Ironically, the "worse" heuristic approach scored 49.56 while the "better" observation approach scored 0.08.

## Implications for strategy

### What doesn't work
- Observation-only with high confidence on single outcome (scores 0.08-2.15)
- The floor value (0.001 vs 0.01) barely matters when the fundamental approach is wrong

### What should work
1. **Monte Carlo simulation**: Run our simulator 200+ times per seed, compute empirical class distributions per cell. Our simulator has ~77% single-run accuracy, which should produce reasonable distributions.
2. **ML model**: Train on simulator data (unlimited supply) or ground truth data from analysis endpoints. Predict probability distributions directly.
3. **Hybrid**: Use observations to calibrate/validate, but predict distributions not point estimates.

### Observations are still useful for
- Calibrating the simulator (compare one sample against predicted distributions)
- Validating predictions (does our distribution include the observed outcome with reasonable probability?)
- Training data (each observation is one sample from the true distribution)

## Key discovery: each query returns a different random sample

Tested on round 5: querying the same viewport (0,0) for the same seed twice returns **different results**. ~25-29% of cells differ between any two queries. Each query is one random sample from the ~200 simulation runs.

### Implications for query strategy

The optimal strategy is NOT full map coverage with one sample. Options:

1. **Multi-sample estimation**: Query the same viewports repeatedly to build empirical distributions. With 50 queries / 5 seeds = 10 per seed. If we query 5 viewports x 2 samples each, we get partial coverage with 2-sample estimates. Not great.

2. **Simulator + observation calibration**: Use Monte Carlo simulator for base distributions (free, unlimited), then use observations to calibrate/correct. Each observation is one data point telling us "the true distribution at this cell includes this class."

3. **Focus queries on high-value cells**: Use simulator to identify uncertain regions, then spend queries sampling those specific areas.

4. **Fewer seeds, more samples**: Instead of 10 queries per seed across all 5, spend 25 queries on 2 seeds (more accurate) and use simulator for the other 3.

### Sample variance data (round 5, seed 0, viewport 0,0)

- Samples 1 vs 2: 66/225 cells differ (29%)
- Samples 1 vs 3: 55/225 cells differ (24%)
- Samples 2 vs 3: 63/225 cells differ (28%)

## Data saved

All round data is saved to `/home/penguin/astar/data/`:
- `round{N}/details.json` — initial states for all 5 seeds
- `round{N}/obs_seed{S}.json` — observation grids (what we saw)
- `round{N}/analysis_seed{S}.json` — ground truth distributions + our predictions + scores

This analysis data (20 seeds across 4 completed rounds) is valuable training data for an ML approach.

## Terrain transition probabilities (from round 1 single observation)

These are from ONE simulation run, so they represent one sample, not the true distribution:
- Plains: 75% stays plains, 22% becomes settlement, 3% becomes forest
- Forest: 63% stays forest, 27% becomes settlement, 8% becomes port
- Settlement: 50% stays, 50% becomes forest (small sample size)
- Ocean/Mountain: 100% static (excluded from scoring)

## Bugs fixed during the night

1. **auto_solve.py overwrote good submission**: The script didn't check if a round was already submitted before resubmitting with worse predictions. Fixed by adding a safety check that skips submission if queries are insufficient for full coverage AND already submitted.
2. **Simulator hardcoded collapse probability**: Line 328 used hardcoded 0.5 instead of `self.collapse_chance`. Fixed.
3. **Simulator too many ruins**: Tuned parameters to produce zero ruins (matching observations). Accuracy improved from 32% to 77%.
