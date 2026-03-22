#!/usr/bin/env python
"""
Benchmark ML model variants using N=10 live-like proxy.

Models compared:
  1. baseline: mean XGB+LGBM ensemble (current v3)
  2. temp_scaled: baseline + learned temperature
  3. catboost3: XGB+LGBM+CatBoost mean ensemble
  4. catboost3_temp: XGB+LGBM+CatBoost + temperature

Evaluation protocol:
  - Train with GT global features
  - Test with simulated seed-0 observation (realistic live scenario)
  - Score with both old metric and N=10 live-like proxy
"""

import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize_scalar

from ml_solver import (
    extract_features, compute_global_features_from_gt,
    compute_cell_entropy, normalize_predictions, score_predictions,
    CELL_TO_CLASS, NUM_CLASSES, DATA_DIR, MODEL_SAMPLE_WEIGHT_POWER,
    get_training_sample_weights,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_rounds():
    """Load raw round data (grids, settlements, GT) per round per seed."""
    round_dirs = sorted(
        [d for d in os.listdir(DATA_DIR)
         if d.startswith("round") and os.path.isdir(os.path.join(DATA_DIR, d))],
        key=lambda d: int(d.replace("round", ""))
    )

    rounds = {}
    for rname in round_dirs:
        rdir = os.path.join(DATA_DIR, rname)
        details_path = os.path.join(rdir, "details.json")
        if not os.path.exists(details_path):
            continue
        if not os.path.exists(os.path.join(rdir, "analysis_seed0.json")):
            continue

        with open(details_path) as f:
            details = json.load(f)

        seeds = {}
        for seed_idx in range(5):
            ap = os.path.join(rdir, f"analysis_seed{seed_idx}.json")
            if not os.path.exists(ap):
                continue
            with open(ap) as f:
                analysis = json.load(f)

            state = details["initial_states"][seed_idx]
            grid = np.array(state["grid"])
            settlements = state["settlements"]
            gt = np.array(analysis["ground_truth"])

            seeds[seed_idx] = {
                'grid': grid,
                'settlements': settlements,
                'gt': gt,
                'gt_global': compute_global_features_from_gt(gt, grid),
            }

        if seeds:
            rounds[rname] = seeds
            print(f"  {rname}: {len(seeds)} seeds", flush=True)

    return rounds


def extract_features_for_round(seeds, global_features_per_seed):
    """Extract X, Y for a round given global features per seed."""
    X, Y = [], []
    for seed_idx, sd in seeds.items():
        gf = global_features_per_seed[seed_idx]
        grid = sd['grid']
        settlements = sd['settlements']
        gt = sd['gt']
        h, w = grid.shape
        for y in range(h):
            for x in range(w):
                if grid[y, x] in (10, 5):
                    continue
                features = extract_features(grid, settlements, y, x, gf)
                X.append(features)
                Y.append(gt[y, x])
    return np.array(X), np.array(Y)


def simulate_seed0_obs_global(seeds, rng):
    """Simulate a single observation from seed 0 GT and compute global features."""
    if 0 not in seeds:
        return {'settlement_ratio': 0.12, 'empty_ratio': 0.62, 'forest_ratio': 0.22}

    sd = seeds[0]
    grid = sd['grid']
    gt = sd['gt']
    h, w = grid.shape

    sett, empty, forest, total = 0, 0, 0, 0
    for y in range(h):
        for x in range(w):
            if grid[y, x] in (10, 5):
                continue
            gt_cell = np.maximum(gt[y, x], 1e-10)
            gt_cell = gt_cell / gt_cell.sum()
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


# ---------------------------------------------------------------------------
# Proxy scorer
# ---------------------------------------------------------------------------

def score_proxy_n10(Y_true, Y_pred, N=10, n_draws=20, floor=0.001):
    """N-sample live-like proxy scorer (vectorized)."""
    Y_pred_safe = normalize_predictions(Y_pred.copy(), floor=floor)

    gt_safe = np.maximum(Y_true, 1e-10)
    gt_safe = gt_safe / gt_safe.sum(axis=1, keepdims=True)

    scores = []
    for draw in range(n_draws):
        rng = np.random.default_rng(draw * 1000 + 42)
        # Vectorized multinomial: one draw per row
        Y_emp = rng.multinomial(N, gt_safe).astype(np.float64) / N

        score, _ = score_predictions(Y_emp, Y_pred_safe, weighted=True, floor=floor)
        scores.append(score)

    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def apply_temperature(Y_pred, T):
    """Temperature scaling: p_new = p^(1/T) / Z."""
    log_p = np.log(np.maximum(Y_pred, 1e-12))
    scaled = log_p / T
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def learn_temperature(Y_true, Y_pred, metric='proxy'):
    """Find optimal temperature T via grid search + refinement."""
    def score_at(T, n_draws=5):
        Y_cal = apply_temperature(Y_pred, T)
        if metric == 'proxy':
            s, _ = score_proxy_n10(Y_true, Y_cal, n_draws=n_draws)
        else:
            s, _ = score_predictions(Y_true, Y_cal, weighted=True)
        return s

    # Coarse grid search
    coarse_Ts = np.arange(0.7, 2.5, 0.1)
    coarse_scores = [score_at(T, n_draws=5) for T in coarse_Ts]
    best_idx = np.argmax(coarse_scores)
    best_T = coarse_Ts[best_idx]

    # Fine refinement around best
    fine_Ts = np.arange(max(0.5, best_T - 0.15), best_T + 0.16, 0.02)
    fine_scores = [score_at(T, n_draws=8) for T in fine_Ts]
    best_idx = np.argmax(fine_scores)

    return float(fine_Ts[best_idx]), float(fine_scores[best_idx])


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_xgb(X, Y, sw):
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor
    return MultiOutputRegressor(
        XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                     n_jobs=32, random_state=42, verbosity=0)
    ).fit(X, Y, sample_weight=sw)


def train_lgbm(X, Y, sw):
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor
    return MultiOutputRegressor(
        LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                      num_leaves=31, n_jobs=32, random_state=42, verbose=-1)
    ).fit(X, Y, sample_weight=sw)


def train_catboost(X, Y, sw):
    from catboost import CatBoostRegressor
    from sklearn.multioutput import MultiOutputRegressor
    return MultiOutputRegressor(
        CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                          thread_count=32, random_seed=42, verbose=0,
                          allow_const_label=True,
                          bootstrap_type='No')
    ).fit(X, Y, sample_weight=sw)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)
    obs_rng = np.random.RandomState(999)

    print("Loading round data...")
    rounds = load_all_rounds()
    round_names = sorted(rounds.keys(), key=lambda r: int(r.replace("round", "")))
    print(f"\n{len(round_names)} rounds with analysis data\n", flush=True)

    # Precompute features for all rounds:
    # - Training uses GT global features
    # - Testing uses simulated seed-0 observation
    print("Extracting features (GT for train, sim-obs for test)...", flush=True)
    gt_data = {}   # round -> (X, Y) with GT global features
    obs_data = {}  # round -> (X, Y) with simulated seed-0 obs global features

    for rname in round_names:
        seeds = rounds[rname]
        # GT global features for training
        gt_gf = {si: sd['gt_global'] for si, sd in seeds.items()}
        gt_data[rname] = extract_features_for_round(seeds, gt_gf)

        # Simulated seed-0 observation for testing
        obs_gf_val = simulate_seed0_obs_global(seeds, obs_rng)
        obs_gf = {si: obs_gf_val for si in seeds}
        obs_data[rname] = extract_features_for_round(seeds, obs_gf)

    print("Feature extraction done.\n", flush=True)

    # Results storage
    results = {name: [] for name in [
        'baseline', 'temp_scaled', 'catboost3', 'catboost3_temp',
    ]}
    all_temps = []
    all_temps_cat = []
    has_catboost = True

    # Check CatBoost availability
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        print("CatBoost not installed, skipping catboost variants\n")
        has_catboost = False

    print("=" * 80)
    print("LEAVE-ONE-ROUND-OUT CROSS-VALIDATION")
    print("=" * 80)

    for test_round in round_names:
        t0 = time.time()

        # Build train set (GT global features)
        train_Xs = [gt_data[r][0] for r in round_names if r != test_round]
        train_Ys = [gt_data[r][1] for r in round_names if r != test_round]
        train_X = np.concatenate(train_Xs)
        train_Y = np.concatenate(train_Ys)
        sw = get_training_sample_weights(train_Y, power=MODEL_SAMPLE_WEIGHT_POWER)

        # Test set (simulated obs global features)
        test_X, test_Y = obs_data[test_round]

        # Train base models
        xgb = train_xgb(train_X, train_Y, sw)
        lgbm = train_lgbm(train_X, train_Y, sw)
        xgb_pred = xgb.predict(test_X)
        lgbm_pred = lgbm.predict(test_X)

        # --- 1. Baseline: mean XGB+LGBM ---
        base_pred = normalize_predictions((xgb_pred + lgbm_pred) / 2.0)
        old_base, _ = score_predictions(test_Y, base_pred, weighted=True)
        proxy_base, std_base = score_proxy_n10(test_Y, base_pred, n_draws=20)
        results['baseline'].append({
            'round': test_round, 'old': old_base, 'proxy': proxy_base,
        })

        # --- 2. Temperature scaling ---
        # Learn T on a subsample of training predictions (T is 1 param, low overfit risk)
        train_base_pred = normalize_predictions(
            (xgb.predict(train_X) + lgbm.predict(train_X)) / 2.0
        )
        # Subsample for faster T learning
        n_sub = min(15000, len(train_Y))
        sub_idx = np.random.choice(len(train_Y), n_sub, replace=False)
        T_opt, _ = learn_temperature(train_Y[sub_idx], train_base_pred[sub_idx], metric='proxy')
        all_temps.append(T_opt)

        temp_pred = normalize_predictions(apply_temperature(base_pred, T_opt))
        old_temp, _ = score_predictions(test_Y, temp_pred, weighted=True)
        proxy_temp, std_temp = score_proxy_n10(test_Y, temp_pred, n_draws=20)
        results['temp_scaled'].append({
            'round': test_round, 'old': old_temp, 'proxy': proxy_temp, 'T': T_opt,
        })

        # --- 3 & 4. CatBoost variants ---
        if has_catboost:
            try:
                cat = train_catboost(train_X, train_Y, sw)
                cat_pred = cat.predict(test_X)
                cat3_pred = normalize_predictions(
                    (xgb_pred + lgbm_pred + cat_pred) / 3.0
                )
                old_cat3, _ = score_predictions(test_Y, cat3_pred, weighted=True)
                proxy_cat3, _ = score_proxy_n10(test_Y, cat3_pred, n_draws=20)
                results['catboost3'].append({
                    'round': test_round, 'old': old_cat3, 'proxy': proxy_cat3,
                })

                # CatBoost + temp
                train_cat3_pred = normalize_predictions(
                    (xgb.predict(train_X) + lgbm.predict(train_X) + cat.predict(train_X)) / 3.0
                )
                T_cat, _ = learn_temperature(train_Y[sub_idx], train_cat3_pred[sub_idx], metric='proxy')
                all_temps_cat.append(T_cat)

                cat3t_pred = normalize_predictions(apply_temperature(cat3_pred, T_cat))
                old_cat3t, _ = score_predictions(test_Y, cat3t_pred, weighted=True)
                proxy_cat3t, _ = score_proxy_n10(test_Y, cat3t_pred, n_draws=20)
                results['catboost3_temp'].append({
                    'round': test_round, 'old': old_cat3t, 'proxy': proxy_cat3t, 'T': T_cat,
                })
            except Exception as e:
                print(f"  [CatBoost error on {test_round}: {e}]")

        elapsed = time.time() - t0
        line = (
            f"  {test_round:<10} "
            f"baseline={proxy_base:6.2f}  "
            f"temp(T={T_opt:.2f})={proxy_temp:6.2f}  "
        )
        if has_catboost and results['catboost3'] and results['catboost3'][-1]['round'] == test_round:
            line += (
                f"cat3={proxy_cat3:6.2f}  "
                f"cat3+T={proxy_cat3t:6.2f}  "
            )
        line += f"[{elapsed:.0f}s]"
        print(line, flush=True)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for model_name in ['baseline', 'temp_scaled', 'catboost3', 'catboost3_temp']:
        data = results[model_name]
        if not data:
            continue
        proxies = [r['proxy'] for r in data]
        olds = [r['old'] for r in data]
        print(f"\n  {model_name}:")
        print(f"    Proxy N=10 avg: {np.mean(proxies):.2f}  (std across rounds: {np.std(proxies):.2f})")
        print(f"    Old metric avg: {np.mean(olds):.2f}")
        if 'T' in data[0]:
            temps = [r['T'] for r in data]
            print(f"    Temperature avg: {np.mean(temps):.3f}  (range: {min(temps):.2f} - {max(temps):.2f})")

    # Per-round table
    print("\n" + "=" * 80)
    print("PER-ROUND PROXY N=10 SCORES")
    print("=" * 80)

    active_models = [m for m in ['baseline', 'temp_scaled', 'catboost3', 'catboost3_temp']
                     if results[m]]
    header = f"{'Round':<10} " + " ".join(f"{m:<16}" for m in active_models)
    print(header)
    print("-" * len(header))

    round_lookup = {m: {r['round']: r for r in results[m]} for m in active_models}
    for rname in round_names:
        row = f"{rname:<10} "
        for m in active_models:
            if rname in round_lookup[m]:
                row += f"{round_lookup[m][rname]['proxy']:>7.2f}          "
            else:
                row += f"{'N/A':>7}          "
        print(row)

    print("-" * len(header))
    row = f"{'AVERAGE':<10} "
    for m in active_models:
        avg = np.mean([r['proxy'] for r in results[m]])
        row += f"{avg:>7.2f}          "
    print(row)

    # Deltas
    print("\n--- Improvements over baseline ---")
    base_avg = np.mean([r['proxy'] for r in results['baseline']])
    for m in active_models:
        if m == 'baseline':
            continue
        avg = np.mean([r['proxy'] for r in results[m]])
        print(f"  {m}: {avg - base_avg:+.2f}")


if __name__ == "__main__":
    main()
