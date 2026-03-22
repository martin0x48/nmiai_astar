# Astar Island - NM i AI 2026

## Task
Predict the final state of a Norse civilization simulator on a 40x40 grid after 50 years.
Each round has 5 stochastic seeds. We submit probability distributions over 6 classes per cell.

## Classes
- 0: Empty/Ocean/Plains
- 1: Settlement
- 2: Port
- 3: Ruin
- 4: Forest
- 5: Mountain

## API
- Base: `https://api.ainm.no/astar-island`
- Auth: Bearer token (see solve.py)
- `GET /rounds` - list rounds
- `GET /rounds/{id}` - initial map states (free)
- `POST /simulate` - observe final state viewport (costs 1 query)
- `POST /submit` - submit predictions
- `GET /budget` - check remaining queries
- `GET /my-rounds` - scores
- `GET /analysis/{round_id}/{seed_index}` - post-round ground truth comparison

## Constraints
- 50 queries per round, shared across 5 seeds
- Max 15x15 viewport per query
- 15 queries/min rate limit
- Predictions: H x W x 6 probabilities, must sum to 1.0 per cell
- NEVER use 0.0 probability (KL divergence explodes) - use 0.01 floor

## Scoring
- Entropy-weighted KL divergence: `score = 100 * e^(-KL_avg)`
- Static cells (ocean, mountain) excluded
- Per-round = average of 5 seeds, leaderboard = best round ever

## Key observations from round 1
- Initial grids differ across seeds (only 374/1600 cells identical)
- Massive settlement expansion: 4 settlements -> 44 in a 15x15 viewport
- ZERO ruins in final state (settlements are very resilient)
- 2 dominant factions tend to control most territory
- Settlement populations range 0.35-2.87, food 0.13-0.98

## Terrain transition probabilities (from round 1 observations)
- Plains: 75% stays plains, 22% becomes settlement, 3% becomes forest
- Forest: 63% stays forest, 27% becomes settlement, 8% becomes port
- Settlement: 50% stays, 50% becomes forest (small sample)
- Ocean/Mountain: 100% static (excluded from scoring)

## Strategy
- **Observation is king**: 9 queries per seed (3x3 tiling of 15x15) covers entire 40x40 map
- Full coverage observation with floor=0.001 → estimated score ~99.5
- Full coverage observation with floor=0.01 → estimated score ~95 (round 1 submission)
- Monte Carlo simulator alone → estimated score ~65
- **Always save observations** for potential resubmission with lower floor

## Files
- `solve.py` - v1 solver (round 1 submission). Floor=0.01, no observation saving.
- `solve_v2.py` - Improved solver. Saves observations, floor=0.001, resubmit support.
- `simulator.py` - Norse simulator, tuned to ~75% cell accuracy. Zero ruins, good settlement count.
- `analyze_observations.py` - Analysis scripts.
- `round1_data.json` - Cached round 1 initial states.
- `sim_s0_0_0.json` - Actual API observation: seed 0, viewport (0,0), 15x15.

## Setup
```bash
export PATH="$HOME/.local/bin:$PATH"
uv run python solve_v2.py          # full run (observe + submit)
uv run python solve_v2.py resubmit # resubmit with saved obs
uv run python solve_v2.py scores   # check scores
uv run python solve_v2.py analysis # post-round analysis
```

## Next steps
1. Check round 1 analysis endpoint after round closes (20:42 UTC)
2. Use analysis data to further calibrate simulator and heuristic priors
3. For round 2: run solve_v2.py immediately with full observation coverage
4. Consider strategic query allocation if map is larger than 40x40
