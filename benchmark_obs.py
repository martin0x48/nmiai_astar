#!/usr/bin/env python
"""
Benchmark: seed-0-only vs per-seed observation strategy.

Compares:
  1. seed0_only: observe seed 0, use its global features for all 5 seeds (current)
  2. per_seed: observe each seed separately, use per-seed global features
  3. oracle_gt: use GT global features per seed (upper bound)

Uses the current best model (XGB+LGBM+Cat with T=1.09).
"""

import json
import os
import sys
import time
import numpy as np

from ml_solver import (
    extract_features, compute_global_features_from_gt,
    normalize_predictions, score_predictions, apply_temperature,
    CELL_TO_CLASS, NUM_CLASSES, DATA_DIR, MODEL_SAMPLE_WEIGHT_POWER,
    MODEL_TEMPERATURE, get_training_sample_weights,
)


def load_all_rounds():
    """Load raw round data per round per seed."""
    round_dirs = sorted(
        [d for d in os.listdir(DATA_DIR)
         if d.startswith("round") and os.path.isdir(os.path.join(DATA_DIR, d))],
        key=lambda d: int(d.replace("round", ""))
    )
    rounds = {}
    for rname in round_dirs:
        rdir = os.path.join(DATA_DIR, rname)
        if not os.path.exists(os.path.join(rdir, "details.json")):
            continue
        if not os.path.exists(os.path.join(rdir, "analysis_seed0.json")):
            continue
        with open(os.path.join(rdir, "details.json")) as f:
            details = json.load(f)
        seeds = {}
        for si in range(5):
            ap = os.path.join(rdir, f"analysis_seed{si}.json")
            if not os.path.exists(ap):
                continue
            with open(ap) as f:
                analysis = json.load(f)
            state = details["initial_states"][si]
            grid = np.array(state["grid"])
            gt = np.array(analysis["ground_truth"])
            seeds[si] = {
                'grid': grid,
                'settlements': state["settlements"],
                'gt': gt,
                'gt_global': compute_global_features_from_gt(gt, grid),
            }
        if seeds:
            rounds[rname] = seeds
            print(f"  {rname}: {len(seeds)} seeds", flush=True)
    return rounds


def simulate_obs_global(sd, rng):
    """Simulate a single full-map observation from one seed's GT."""
    grid, gt = sd['grid'], sd['gt']
    h, w = grid.shape
    sett, empty, forest, total = 0, 0, 0, 0
    for y in range(h):
        for x in range(w):
            if grid[y, x] in (10, 5):
                continue
            gt_cell = np.maximum(gt[y, x], 1e-10)
            gt_cell /= gt_cell.sum()
            c = rng.choice(6, p=gt_cell)
            if c in (1, 2):
                sett += 1
            elif c == 0:
                empty += 1
            elif c == 4:
                forest += 1
            total += 1
    if total == 0:
        return {'settlement_ratio': 0.12, 'empty_ratio': 0.62, 'forest_ratio': 0.22}
    return {
        'settlement_ratio': sett / total,
        'empty_ratio': empty / total,
        'forest_ratio': forest / total,
    }


def extract_features_for_seeds(seeds, gf_per_seed):
    """Extract X, Y for all seeds in a round, with per-seed global features."""
    X, Y = [], []
    for si, sd in seeds.items():
        gf = gf_per_seed[si]
        grid = sd['grid']
        settlements = sd['settlements']
        gt = sd['gt']
        h, w = grid.shape
        for y in range(h):
            for x in range(w):
                if grid[y, x] in (10, 5):
                    continue
                X.append(extract_features(grid, settlements, y, x, gf))
                Y.append(gt[y, x])
    return np.array(X), np.array(Y)


def score_proxy_n10(Y_true, Y_pred, N=10, n_draws=20, floor=0.001):
    """N-sample live-like proxy scorer (vectorized)."""
    Y_pred_safe = normalize_predictions(Y_pred.copy(), floor=floor)
    gt_safe = np.maximum(Y_true, 1e-10)
    gt_safe /= gt_safe.sum(axis=1, keepdims=True)
    scores = []
    for draw in range(n_draws):
        rng = np.random.default_rng(draw * 1000 + 42)
        Y_emp = rng.multinomial(N, gt_safe).astype(np.float64) / N
        s, _ = score_predictions(Y_emp, Y_pred_safe, weighted=True, floor=floor)
        scores.append(s)
    return float(np.mean(scores))


def train_ensemble(X, Y):
    """Train the current best model (XGB+LGBM+Cat)."""
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from ml_solver import EnsembleModel

    sw = get_training_sample_weights(Y, power=MODEL_SAMPLE_WEIGHT_POWER)

    xgb = MultiOutputRegressor(
        XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                     n_jobs=32, random_state=42, verbosity=0)
    )
    lgbm = MultiOutputRegressor(
        LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                      num_leaves=31, n_jobs=32, random_state=42, verbose=-1)
    )
    cat = MultiOutputRegressor(
        CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                          thread_count=32, random_seed=42, verbose=0,
                          allow_const_label=True, bootstrap_type='No')
    )
    model = EnsembleModel([xgb, lgbm, cat])
    model.fit(X, Y, sample_weight=sw)
    return model


def main():
    np.random.seed(42)
    rng = np.random.RandomState(999)

    print("Loading round data...", flush=True)
    rounds = load_all_rounds()
    round_names = sorted(rounds.keys(), key=lambda r: int(r.replace("round", "")))
    print(f"\n{len(round_names)} rounds\n", flush=True)

    # Precompute features for all three strategies per round
    print("Extracting features for 3 strategies...", flush=True)

    # Training always uses GT global features
    gt_train = {}
    # Test features differ by strategy
    test_seed0 = {}    # seed-0 obs for all seeds
    test_perseed = {}  # per-seed obs
    test_oracle = {}   # GT per seed (upper bound)

    for rname in round_names:
        seeds = rounds[rname]

        # Training: GT globals
        gt_gf = {si: sd['gt_global'] for si, sd in seeds.items()}
        gt_train[rname] = extract_features_for_seeds(seeds, gt_gf)

        # Strategy 1: seed-0 only observation
        seed0_gf = simulate_obs_global(seeds[0], rng)
        test_seed0[rname] = extract_features_for_seeds(seeds, {si: seed0_gf for si in seeds})

        # Strategy 2: per-seed observation
        perseed_gf = {si: simulate_obs_global(sd, rng) for si, sd in seeds.items()}
        test_perseed[rname] = extract_features_for_seeds(seeds, perseed_gf)

        # Strategy 3: oracle GT per seed
        test_oracle[rname] = extract_features_for_seeds(seeds, gt_gf)

    print("Feature extraction done.\n", flush=True)

    results = {'seed0_only': [], 'per_seed': [], 'oracle_gt': []}

    print("=" * 80)
    print("LEAVE-ONE-ROUND-OUT CV: OBSERVATION STRATEGY COMPARISON")
    print("=" * 80, flush=True)

    for test_round in round_names:
        t0 = time.time()

        # Train on all other rounds (GT globals)
        train_X = np.concatenate([gt_train[r][0] for r in round_names if r != test_round])
        train_Y = np.concatenate([gt_train[r][1] for r in round_names if r != test_round])

        model = train_ensemble(train_X, train_Y)

        # Score each strategy
        for strategy, test_data, label in [
            ('seed0_only', test_seed0, 'seed0'),
            ('per_seed', test_perseed, 'perseed'),
            ('oracle_gt', test_oracle, 'oracle'),
        ]:
            tX, tY = test_data[test_round]
            pred = normalize_predictions(model.predict(tX))
            pred = normalize_predictions(apply_temperature(pred, MODEL_TEMPERATURE))
            proxy = score_proxy_n10(tY, pred)
            old, _ = score_predictions(tY, pred, weighted=True)
            results[strategy].append({
                'round': test_round, 'proxy': proxy, 'old': old,
            })

        elapsed = time.time() - t0
        s0 = results['seed0_only'][-1]['proxy']
        ps = results['per_seed'][-1]['proxy']
        orc = results['oracle_gt'][-1]['proxy']
        delta = ps - s0
        print(
            f"  {test_round:<10}  seed0={s0:6.2f}  perseed={ps:6.2f}  "
            f"oracle={orc:6.2f}  delta(ps-s0)={delta:+.2f}  [{elapsed:.0f}s]",
            flush=True,
        )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for strat in ['seed0_only', 'per_seed', 'oracle_gt']:
        proxies = [r['proxy'] for r in results[strat]]
        olds = [r['old'] for r in results[strat]]
        print(f"\n  {strat}:")
        print(f"    Proxy N=10 avg: {np.mean(proxies):.2f}  (std: {np.std(proxies):.2f})")
        print(f"    Old metric avg: {np.mean(olds):.2f}")

    # Per-round table
    print("\n" + "=" * 80)
    print("PER-ROUND PROXY SCORES")
    print("=" * 80)
    print(f"{'Round':<10}  {'seed0_only':>10}  {'per_seed':>10}  {'oracle_gt':>10}  {'delta':>8}")
    print("-" * 60)

    for i, rname in enumerate(round_names):
        s0 = results['seed0_only'][i]['proxy']
        ps = results['per_seed'][i]['proxy']
        orc = results['oracle_gt'][i]['proxy']
        print(f"{rname:<10}  {s0:>10.2f}  {ps:>10.2f}  {orc:>10.2f}  {ps-s0:>+8.2f}")

    print("-" * 60)
    s0_avg = np.mean([r['proxy'] for r in results['seed0_only']])
    ps_avg = np.mean([r['proxy'] for r in results['per_seed']])
    orc_avg = np.mean([r['proxy'] for r in results['oracle_gt']])
    print(f"{'AVERAGE':<10}  {s0_avg:>10.2f}  {ps_avg:>10.2f}  {orc_avg:>10.2f}  {ps_avg-s0_avg:>+8.2f}")

    # Win rate
    wins = sum(1 for i in range(len(round_names))
               if results['per_seed'][i]['proxy'] > results['seed0_only'][i]['proxy'])
    print(f"\nPer-seed wins on {wins}/{len(round_names)} rounds")


if __name__ == "__main__":
    main()
