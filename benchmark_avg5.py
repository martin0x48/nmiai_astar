#!/usr/bin/env python
"""
Benchmark: average predictions from 5 different observation-based runs.

Idea: instead of observing only seed 0, observe all 5 seeds. For each seed's
observation, run the full prediction pipeline (extract features with that
seed's global features → predict). Then average the 5 output distributions.

This ensembles over observation noise rather than committing to one sample.

Compares:
  1. seed0_only: current approach (1 observation, 9 queries)
  2. avg5_obs: observe all 5 seeds, predict 5 times, average (45 queries)
  3. avg5_obs_multi: same but with 3 independent draws per seed (15 obs averaged)
"""

import argparse
import json
import os
import time
import numpy as np

from ml_solver import (
    extract_features, compute_global_features_from_gt,
    normalize_predictions, score_predictions, apply_temperature,
    NUM_CLASSES, DATA_DIR, MODEL_SAMPLE_WEIGHT_POWER,
    MODEL_TEMPERATURE, get_training_sample_weights,
)

OBS_REPEATS = 5


def load_all_rounds():
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
            seeds[si] = {
                'grid': np.array(state["grid"]),
                'settlements': state["settlements"],
                'gt': np.array(analysis["ground_truth"]),
                'gt_global': compute_global_features_from_gt(
                    np.array(analysis["ground_truth"]), np.array(state["grid"])),
            }
        if seeds:
            rounds[rname] = seeds
            print(f"  {rname}: {len(seeds)} seeds", flush=True)
    return rounds


def simulate_obs_global(sd, rng):
    """Simulate one full-map observation and compute global features."""
    grid, gt = sd['grid'], sd['gt']
    h, w = grid.shape
    sett, empty, forest, total = 0, 0, 0, 0
    for y in range(h):
        for x in range(w):
            if grid[y, x] in (10, 5):
                continue
            p = np.maximum(gt[y, x], 1e-10)
            p /= p.sum()
            c = rng.choice(6, p=p)
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


def stable_round_seed(round_name, obs_seed=0, draw=0, repeat=0):
    """Deterministic seed independent of Python's randomized string hashing."""
    round_num = int(round_name.replace("round", ""))
    return (
        1_000_003 * round_num
        + 7_919 * obs_seed
        + 104_729 * draw
        + 65_537 * repeat
        + 42
    )


def extract_XY(seeds, gf_per_seed):
    """Extract features+targets for all seeds using given global features."""
    X, Y = [], []
    for si, sd in seeds.items():
        gf = gf_per_seed[si]
        grid, sett, gt = sd['grid'], sd['settlements'], sd['gt']
        h, w = grid.shape
        for y in range(h):
            for x in range(w):
                if grid[y, x] in (10, 5):
                    continue
                X.append(extract_features(grid, sett, y, x, gf))
                Y.append(gt[y, x])
    return np.array(X), np.array(Y)


def score_proxy_n10(Y_true, Y_pred, N=10, n_draws=20, floor=0.001):
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
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from ml_solver import EnsembleModel

    sw = get_training_sample_weights(Y, power=MODEL_SAMPLE_WEIGHT_POWER)
    xgb = MultiOutputRegressor(
        XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                     n_jobs=32, random_state=42, verbosity=0))
    lgbm = MultiOutputRegressor(
        LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                      num_leaves=31, n_jobs=32, random_state=42, verbose=-1))
    cat = MultiOutputRegressor(
        CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                          thread_count=32, random_seed=42, verbose=0,
                          allow_const_label=True, bootstrap_type='No'))
    model = EnsembleModel([xgb, lgbm, cat])
    model.fit(X, Y, sample_weight=sw)
    return model


def predict_and_calibrate(model, X):
    """Predict + normalize + temperature scale."""
    pred = normalize_predictions(model.predict(X))
    return normalize_predictions(apply_temperature(pred, MODEL_TEMPERATURE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rounds",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of round numbers to evaluate, e.g. --rounds 12 13 14",
    )
    args = parser.parse_args()

    np.random.seed(42)

    print("Loading round data...", flush=True)
    rounds = load_all_rounds()
    round_names = sorted(rounds.keys(), key=lambda r: int(r.replace("round", "")))
    if args.rounds is not None:
        wanted = {f"round{r}" for r in args.rounds}
        round_names = [r for r in round_names if r in wanted]
    print(f"\n{len(round_names)} rounds\n", flush=True)

    # Precompute GT training features (same for all strategies)
    print("Extracting GT training features...", flush=True)
    gt_train = {}
    for rname in round_names:
        seeds = rounds[rname]
        gt_gf = {si: sd['gt_global'] for si, sd in seeds.items()}
        gt_train[rname] = extract_XY(seeds, gt_gf)
    print("Done.\n", flush=True)

    results = {'seed0_only': [], 'avg5_obs': [], 'avg5_multi3': []}

    print("=" * 80)
    print("LEAVE-ONE-ROUND-OUT CV")
    print("=" * 80, flush=True)

    for test_round in round_names:
        t0 = time.time()
        seeds = rounds[test_round]

        # Train on all other rounds
        train_X = np.concatenate([gt_train[r][0] for r in round_names if r != test_round])
        train_Y = np.concatenate([gt_train[r][1] for r in round_names if r != test_round])
        model = train_ensemble(train_X, train_Y)

        # We need test Y (same regardless of global features used)
        _, test_Y = gt_train[test_round]

        # --- Strategy 1: seed-0-only (current) ---
        s0_scores = []
        for repeat in range(OBS_REPEATS):
            rng1 = np.random.default_rng(stable_round_seed(test_round, obs_seed=0, repeat=repeat))
            s0_gf = simulate_obs_global(seeds[0], rng1)
            test_X_s0, _ = extract_XY(seeds, {si: s0_gf for si in seeds})
            pred_s0 = predict_and_calibrate(model, test_X_s0)
            s0_scores.append(score_proxy_n10(test_Y, pred_s0))
        proxy_s0 = float(np.mean(s0_scores))
        results['seed0_only'].append({'round': test_round, 'proxy': proxy_s0})

        # --- Strategy 2: avg5_obs ---
        # Observe each seed once, predict with each seed's globals, average predictions
        avg5_scores = []
        for repeat in range(OBS_REPEATS):
            preds_list = []
            for obs_seed in range(5):
                rng_obs = np.random.default_rng(
                    stable_round_seed(test_round, obs_seed=obs_seed, repeat=repeat)
                )
                obs_gf = simulate_obs_global(seeds[obs_seed], rng_obs)
                test_X_obs, _ = extract_XY(seeds, {si: obs_gf for si in seeds})
                pred_obs = predict_and_calibrate(model, test_X_obs)
                preds_list.append(pred_obs)
            avg5_pred = normalize_predictions(np.mean(preds_list, axis=0))
            avg5_scores.append(score_proxy_n10(test_Y, avg5_pred))
        proxy_avg5 = float(np.mean(avg5_scores))
        results['avg5_obs'].append({'round': test_round, 'proxy': proxy_avg5})

        # --- Strategy 3: avg5 with 3 draws per seed (15 total obs) ---
        avg15_scores = []
        for repeat in range(OBS_REPEATS):
            preds_multi = []
            for obs_seed in range(5):
                for draw in range(3):
                    rng_m = np.random.default_rng(
                        stable_round_seed(test_round, obs_seed=obs_seed, draw=draw, repeat=repeat)
                    )
                    obs_gf = simulate_obs_global(seeds[obs_seed], rng_m)
                    test_X_m, _ = extract_XY(seeds, {si: obs_gf for si in seeds})
                    pred_m = predict_and_calibrate(model, test_X_m)
                    preds_multi.append(pred_m)
            avg15_pred = normalize_predictions(np.mean(preds_multi, axis=0))
            avg15_scores.append(score_proxy_n10(test_Y, avg15_pred))
        proxy_avg15 = float(np.mean(avg15_scores))
        results['avg5_multi3'].append({'round': test_round, 'proxy': proxy_avg15})

        elapsed = time.time() - t0
        d1 = proxy_avg5 - proxy_s0
        d2 = proxy_avg15 - proxy_s0
        print(
            f"  {test_round:<10}  s0={proxy_s0:6.2f}  avg5={proxy_avg5:6.2f}({d1:+.2f})  "
            f"avg15={proxy_avg15:6.2f}({d2:+.2f})  [{elapsed:.0f}s]",
            flush=True,
        )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for strat in ['seed0_only', 'avg5_obs', 'avg5_multi3']:
        proxies = [r['proxy'] for r in results[strat]]
        print(f"  {strat:<15}  proxy avg: {np.mean(proxies):.2f}  (std: {np.std(proxies):.2f})")

    s0_avg = np.mean([r['proxy'] for r in results['seed0_only']])
    print(f"\n  avg5_obs   vs seed0: {np.mean([r['proxy'] for r in results['avg5_obs']]) - s0_avg:+.2f}")
    print(f"  avg5_multi3 vs seed0: {np.mean([r['proxy'] for r in results['avg5_multi3']]) - s0_avg:+.2f}")

    wins5 = sum(1 for i in range(len(round_names))
                if results['avg5_obs'][i]['proxy'] > results['seed0_only'][i]['proxy'])
    wins15 = sum(1 for i in range(len(round_names))
                 if results['avg5_multi3'][i]['proxy'] > results['seed0_only'][i]['proxy'])
    print(f"\n  avg5  wins: {wins5}/{len(round_names)}")
    print(f"  avg15 wins: {wins15}/{len(round_names)}")


if __name__ == "__main__":
    main()
