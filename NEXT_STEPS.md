# Next Steps — After Round 6 (score 82.9)

## Current status
- Round 6 scored 82.9, rank TBD — big jump from previous best of 49.56
- ML model: RandomForest, 47 features, trained on 5 rounds (33,927 examples)
- CV score: ~93 average (leave-one-round-out)
- Global features (expansion level estimated from 9 queries on seed 0) boost score by ~4 points

## Where we're losing points
1. Cells near settlement boundaries — most uncertain, model may not capture expansion dynamics
2. Expansion level estimation noise — one observation sample is noisy
3. Rare class predictions (ruin, port) — few training examples
4. Spatial patterns the per-cell model can't see (e.g., settlement clusters, faction boundaries)

## Ideas to implement (priority order)

### 1. Ensemble (implementing now)
- Train multiple models with different hyperparameters
- Average their predictions
- Low risk, reliable 1-3 point improvement

### 2. More features (implementing now)
- Settlement cluster size (connected component the nearest settlement belongs to)
- Terrain texture: count of terrain transitions in neighborhood
- Distance to nearest ocean in each cardinal direction
- Number of settlements in radius 5, 10
- Ratio of plains to forest in neighborhood
- Whether cell is between two settlements (potential expansion corridor)

### 3. Train separate models per class
- 6 specialized binary models instead of one multi-output
- Each learns different feature importances
- Moderate effort, could help with rare classes

### 4. CNN on the grid
- Treat 40x40 grid as image, predict 40x40x6 probability map
- Could capture spatial patterns (clusters, boundaries)
- Risk: only 25 training examples — might overfit badly
- Could use data augmentation (rotations, flips)

### 5. Better query strategy
- 1 query per seed (5 total) for per-seed expansion estimate
- Remaining 45 queries: focus on high-uncertainty cells
- Or: multiple samples of same viewport for better distribution estimate

### 6. Cross-seed information sharing
- All 5 seeds share expansion level
- Could train model that jointly predicts all 5 seeds

## Key metrics
| Approach | CV Score | Notes |
|----------|----------|-------|
| Observation-only | 0.08-2.15 | Ground truth is distribution, not single class |
| Heuristic-only | 49.56 | Round 1 accidental submission |
| RF no global features | 89.3 | Good baseline |
| RF + perfect global features | 92.8 | Oracle |
| RF + 1-sample global features | 93.2 | What we deploy |
| Round 6 actual | 82.9 | First ML submission |
