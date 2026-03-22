# Astar Island - NM i AI 2026

## What this is

A prediction competition. A Norse civilization simulator runs on a 40x40 grid for 50 years. Each round has 5 random seeds. We submit probability distributions (H x W x 6 classes) predicting the final state of each cell. Scored by KL divergence: `score = 100 * e^(-KL_avg)`. Higher is better.

## Classes

0=Empty/Ocean/Plains, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain

## API

- Base: `https://api.ainm.no/astar-island`
- Auth: Bearer token in `solve.py` / `auto_solve.py`
- `GET /rounds` — list rounds (check `status: "active"`)
- `GET /rounds/{id}` — initial map states (free, no query cost)
- `POST /simulate` — observe a 15x15 viewport of the final state (costs 1 query)
- `POST /submit` — submit prediction for a seed
- `GET /budget` — remaining queries this round
- `GET /my-rounds` — scores and submission status
- `GET /analysis/{round_id}/{seed_index}` — ground truth (only after round closes)

## Constraints

- 50 queries per round, shared across 5 seeds
- Max 15x15 viewport per query
- 15 queries/min rate limit (wait 4.5s between queries)
- Predictions: H x W x 6 probabilities, must sum to 1.0 per cell
- NEVER use 0.0 probability — use 0.001 floor minimum

## Your job on each invocation

1. Run `export PATH="$HOME/.local/bin:$PATH" && cd /home/penguin/astar && uv run python auto_solve.py`
2. If the output says "already submitted, skipping" and "Done" — no new round. **Stop. Do nothing else.**
3. If a new round was found and solved — report what happened (round number, scores if available, any errors). **Then stop.**
4. If a new round was found but auto_solve.py failed — investigate the error, fix it, and retry. Save all round data to `/home/penguin/astar/data/`.

Do NOT loop, poll, or keep running. Just run once and stop.

## How the solver works

- `auto_solve.py` — single-run script that:
  - Checks for active rounds not yet submitted
  - **Primary: ML model** (XGBoost + LightGBM + CatBoost ensemble, v4)
  - **Avg5 observation strategy**: observes all 5 seeds (9 queries each = 45 total), computes global features from each, predicts 5 times per target seed, averages predictions. Benchmarked at +0.22 proxy points vs seed-0-only.
  - **Temperature scaling** (T=1.09): softens overconfident predictions. Benchmarked at +0.46 proxy points.
  - Auto-retrains ML model when new analysis data is available
  - Saves all data to `/home/penguin/astar/data/round{N}/`
  - Grabs analysis for completed rounds
  - Falls back to seed-0-only observation if budget < 45 queries

## Key files

- `auto_solve.py` — main solver (run this)
- `ml_solver.py` — ML model training, feature extraction, evaluation, temperature scaling
- `model.pkl` — trained XGB+LGBM+Cat ensemble (v4, with temperature metadata)
- `benchmark.py` — model variant comparison with N=10 proxy scorer
- `benchmark_obs.py` — observation strategy comparison
- `benchmark_avg5.py` — avg5 observation ensemble benchmark
- `data/` — saved round data, observations, analysis

## Important lessons learned

- Ground truth is a PROBABILITY DISTRIBUTION (from ~200 simulation runs), not deterministic
- Each query returns a DIFFERENT random sample (~29% cells differ between queries)
- Observation-only approach scores terribly (0.08-2.15) because it puts 99.5% on one class
- **Temperature scaling helps** — T=1.09 softens predictions to match noisy live scorer (+0.46 proxy)
- **Avg5 observation helps** — averaging predictions from 5 seed observations reduces variance (+0.22 proxy)
- **Per-seed observation HURTS vs seed-0-only** — using per-seed obs directly (without averaging) is -0.48. Seed-0-only even beats oracle GT globals because observation noise correlates with live scorer noise.
- **DO NOT override parameters based on observations** — R10 disaster (20.57) was caused by forcing growth=0.02 on a low-but-not-collapse round
- NEVER resubmit if you can't achieve full coverage — you'll overwrite a good submission with a bad one
- Rounds come every ~3 hours, last ~2.75 hours
- The leaderboard uses best round ever, so one great round is enough
