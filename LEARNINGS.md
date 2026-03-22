# Learnings — Astar Island Competition (NM i AI 2026)

## Score progression

| Round | Score | Rank | Approach | Weight |
|-------|-------|------|----------|--------|
| 1 | 49.56 | 29 | Heuristic (accidental) | 1.05 |
| 2 | 0.08 | 151 | Observation-only | 1.10 |
| 3 | 2.15 | 93 | Observation-only | 1.16 |
| 4 | 0.82 | 86 | Observation-only | 1.22 |
| 5 | 0.12 | 144 | Observation-only | 1.28 |
| 6 | 82.88 | 11 | ML (RF, 47feat, 5 rounds) | 1.34 |
| 7 | 69.68 | 8 | ML (RF, 47feat, 5 rounds) | 1.41 |
| 8 | 85.65 | 46 | ML (RF, 61feat, 7 rounds) | 1.48 |
| 9 | 75.63 | 118 | ML (RF, 61feat, 8 rounds) | 1.55 |
| 10 | 20.57 | 209 | Simulator (forced gr=0.02) | 1.63 |
| 11 | 61.88 | 112 | Simulator (gr=0.20, high expansion) | 1.71 |
| 12 | 32.73 | 116 | Simulator (default gr) | 1.80 |
| 13 | 82.09 | 98 | Simulator (default gr) | 1.89 |
| 14 | 80.84 | 41 | ML (XGB, 61feat, 13 rounds) | 1.98 |
| 15 | **93.88** | **8** | ML (XGB, 61feat, 14 rounds) | 2.08 |
| 16 | 85.06 | 40 | ML (old offline metric mismatch exposed) | 2.18 |
| 17 | 91.93 | 35 | ML (entropy^2 + seed-0 obs + targeted smoothing) | 2.29 |

## CRITICAL DISCOVERY: Scoring formula (from docs)

The competition uses **entropy-weighted KL divergence**:
```
For each dynamic cell (not ocean/mountain):
  entropy = -sum(gt[c] * log(gt[c]))
  cell_kl = sum(gt[c] * log(gt[c] / pred[c]))

weighted_kl = sum(entropy * cell_kl) / sum(entropy)
score = 100 * exp(-weighted_kl)
```

Key implications:
- **High-entropy cells (uncertain outcomes) count MORE** — settlement boundary cells matter most
- **Static cells (ocean, mountain) are EXCLUDED** from scoring
- Analysis API returns 200-sim GT, but scoring may use fewer sims
- **floor=0.001 is optimal** — docs recommend 0.01 but that costs ~1.4 points

## Current best approach: XGBoost ML model

### Model specs
- **XGBoost** MultiOutputRegressor: 500 trees, max_depth=6, learning_rate=0.05
- 61 input features per cell, 6 output classes (probability distribution)
- Trained on 102,078 examples from 15 rounds
- **Leave-one-round-out CV: 93.46** (unweighted), **92.70** (entropy-weighted)
- Training time: ~10 seconds on 384-core machine

### Why XGBoost > RandomForest
- With 5 rounds of data: RF was better (93.3 vs 87.6) — XGB overfit
- With 13 rounds of data: XGB is better (93.89 vs 91.45) — enough data to learn
- XGB also trains 2x faster

### Feature list (61 features)
1. Own terrain type — one-hot for 8 terrain values (8)
2. Neighbor terrain counts at radius 1, 2, 3, normalized (24)
3. Distance to nearest settlement (1)
4. Settlement density at radius 2, 4, 8 (3)
5. Is settlement / has port (2)
6. Coastal / near mountain (2)
7. Position: x, y, dist to center, dist to edge (4)
8. Settlement density at radius 5, 10, 15 (3)
9. Total settlement count (1)
10. Distance to 2nd and 3rd nearest settlement (2)
11. Ocean distance in 4 cardinal directions (4)
12. Plains/forest ratio in radius 3 (1)
13. Terrain transitions (texture) (1)
14. Between two settlements indicator (1)
15. Nearby port count (1)
16. Global features: settlement/empty/forest ratio (3)

### ML experiments tested (2026-03-21)
- XGB+LGBM ensemble: 93.84 (+0.37 over baseline) — best in CV
- LightGBM alone (31 leaves): 93.73 (+0.26)
- XGB subsampling (0.8/0.8): 93.69 (+0.23)
- XGB lr=0.02, 1000 trees: 93.56 (+0.09)
- XGB regularization: no change
- XGB 1000 trees: -0.20 (worse)
- XGB depth 8: -0.39 (worse)
- XGB depth 8, 1500 trees, lr=0.03: -0.50 (worst)
- New spatial features (reachability, directional pressure): +0.05 (not worth it)
- Per-seed observations: no improvement over seed-0 obs
- ML+sim ensemble blending: HURTS at all blend weights

### What was tested and didn't help
- More features (76 total): -0.24 worse
- RF+XGB ensemble: -0.46 worse than XGB alone
- More RF trees (1000 vs 500): no change
- Deeper RF (depth 20 vs 15): no change
- Settlement stats features (pop/food/wealth from replay): -8.5 worse
- Bayesian observation correction (any pw): barely helps or hurts

## Query strategy

### Current (working well)
- 9 queries on seed 0 → compute global features (settlement/empty/forest ratios)
- 8 queries per seed for Bayesian correction (pw=50, marginal benefit)
- Total: 49/50 queries used

### Key finding: 1 observation sample is OPTIMAL for global features
- Perfect GT global features: 93.66
- 1 observation sample: **93.89** (BEST — captures seed-specific signal)
- 5 samples averaged: 92.95 (worse — averages out the signal)
- No global features: 90.25 (3.6 points worse)

### Per-seed observations don't help
- Per-seed obs (9 queries × 5 seeds): 93.69
- Seed-0 obs for all seeds: 93.75
- Delta: -0.06 (within noise)

## Game mechanics (confirmed from replay API analysis)

### Static terrain
- Mountains and ocean NEVER change (from docs, confirmed in data)

### Population growth
- CONSTANT per settlement (~0.05-0.11 per year), NOT proportional to food/pop
- Growth rate varies PER ROUND (0.047-0.111) — this drives expansion vs collapse
- Growth rate also varies PER SETTLEMENT within a round

### Defense growth
- Formula: `defense += pop * 0.0475`, capped at 1.0

### Expansion
- Child settlement gets fixed pop ~0.500, parent loses only ~0.010
- Expansion distance: 1-4 Manhattan, 81% within distance 1-2
- 73% from plains, 27% from forest

### Collapse mechanics
- Some rounds have near-zero surviving settlements (growth rate too low)
- Even strong settlements (pop 2.0+, food 0.9+) can die in combat
- Collapse rounds vary wildly between simulation runs of same round

### Ruins
- Very rare in GT: 0.06%-3.27% average per round
- ML model predicts ~1.5% avg, well calibrated

### Ports
- Only appear in coastal cells (distance 1 from ocean)
- ~10-18% probability on coastal cells

### Replay API
- `POST /replay` with cookie auth, only for completed rounds
- Each call returns different sim_seed — different simulation run
- Frame 0 settlement data identical across runs (same initial conditions)

## Simulator approach (in ~/astar-sim)

### Status
- Backtests at 84.58 avg (entropy-weighted) vs ML's 92.70
- ML wins 11/13 rounds head-to-head
- Sim peaks at 93.82 (R4) but crashes to 67.48 (R3)
- ML+sim blending HURTS — sim adds noise, not signal
- Settlement stats barely matter (±0.4 pts)

### Not used for submissions — ML dominates on all metrics

## Round weights
- Weights increase 5% per round: weight = 1.05^round_number
- R15 = 2.08, later rounds ≈ 2.5x+
- Later rounds matter MORE — reliability beats occasional high scores

## Critical mistakes to never repeat
1. **Don't override parameters from observations** — R10 disaster (20.57 vs 80.46 default)
2. **Don't calibrate** — backtest proved it hurts (66 vs 95)
3. **Don't resubmit unless CERTAIN it's better** — R1 and R10 were both resubmit disasters
4. **Don't use all-same owner_id** — breaks conflict mechanics

## Infrastructure

### Key files (ml branch)
- `auto_solve.py` — main solver, run this
- `ml_solver.py` — feature extraction + XGBoost training
- `model.pkl` — trained model (auto-retrains when new analysis data arrives)
- `benchmark.py` — strategy comparison benchmarks
- `ml_experiments.py` — hyperparameter/model experiments
- `data/round{N}/` — details, observations, analysis per round
- `CLAUDE.md` — instructions for automated invocation

### Machines
- Main ML: 384 cores, 708GB RAM (CPU, no GPU) — ~/astar on ml branch
- Simulator dev: ~/astar-sim — separate Claude Code session

### Discord webhook
- For significant findings only (5+ point CV improvement)
- URL in memory file reference_discord.md
