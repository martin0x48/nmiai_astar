# Astar Island - NM i AI 2026

Predict the final state of a Norse civilization simulator on a 40x40 grid after 50 years. Each round has 5 stochastic seeds. We submit probability distributions over 6 classes per cell, scored by entropy-weighted KL divergence.

## Setup

```bash
uv sync
uv run python auto_solve.py
```

## How it works

- Observes the final state through 15x15 viewports (50 queries per round)
- ML ensemble (XGBoost + LightGBM + CatBoost) predicts cell probability distributions
- Temperature scaling softens overconfident predictions
- Avg5 strategy: observes all 5 seeds, averages predictions per target seed

See [SUMMARY.md](SUMMARY.md) for full details on the task, API, and strategy.

## Contributors

- [martin0x48](https://github.com/martin0x48)
- [tobiasfremming](https://github.com/tobiasfremming)
- [rasmusjs](https://github.com/rasmusjs)
- [jmnorheim](https://github.com/jmnorheim)
