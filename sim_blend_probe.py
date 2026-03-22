#!/usr/bin/env python
"""Probe whether a fixed-parameter simulator blend helps the current avg5 ML path.

This evaluates recent hard rounds only (19-21) using deterministic simulated
observations and the same N=10 proxy used elsewhere in this repo.
"""

import argparse
import json
import os
import sys
import time
import numpy as np

from ml_solver import (
    DATA_DIR,
    predict_round,
    train_model,
    normalize_predictions,
    score_predictions,
    compute_global_features_from_gt,
)

SIM_SEARCH_DIR = "/home/penguin/astar-sim-bruteforce/sim_search"
if SIM_SEARCH_DIR not in sys.path:
    sys.path.insert(0, SIM_SEARCH_DIR)

from fast_sim_v2 import run_simulation  # noqa: E402
from proxy_full import run_sims  # noqa: E402

DEFAULT_ROUNDS = [19, 20, 21]
DEFAULT_OBS_REPEATS = 3
DEFAULT_SIM_SIMS = 60
BLEND_ALPHAS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 1.00]
BEST_PARAMS_PATH = os.path.join(SIM_SEARCH_DIR, "best_params_v2.npy")


def load_all_rounds():
    rounds = {}
    for d in sorted(os.listdir(DATA_DIR), key=lambda x: int(x.replace("round", ""))):
        if not d.startswith("round"):
            continue
        round_num = int(d.replace("round", ""))
        rdir = os.path.join(DATA_DIR, d)
        details_path = os.path.join(rdir, "details.json")
        if not os.path.exists(details_path):
            continue
        with open(details_path) as f:
            details = json.load(f)
        seeds = {}
        for seed_idx in range(5):
            analysis_path = os.path.join(rdir, f"analysis_seed{seed_idx}.json")
            if not os.path.exists(analysis_path):
                continue
            with open(analysis_path) as f:
                analysis = json.load(f)
            state = details["initial_states"][seed_idx]
            grid = np.array(state["grid"], dtype=np.int32)
            gt = np.array(analysis["ground_truth"], dtype=np.float64)
            settlements = state["settlements"]
            seeds[seed_idx] = {
                "grid": grid,
                "settlements": settlements,
                "gt": gt,
                "gt_global": compute_global_features_from_gt(gt, grid),
            }
        if len(seeds) == 5:
            rounds[round_num] = {"details": details, "seeds": seeds}
    return rounds


def stable_round_seed(round_num, obs_seed=0, repeat=0):
    return 1_000_003 * round_num + 7_919 * obs_seed + 65_537 * repeat + 42


def simulate_obs_global(seed_data, rng):
    grid = seed_data["grid"]
    gt = seed_data["gt"]
    h, w = grid.shape
    sett = empty = forest = total = 0
    for y in range(h):
        for x in range(w):
            if grid[y, x] in (10, 5):
                continue
            probs = np.maximum(gt[y, x], 1e-10)
            probs /= probs.sum()
            cls = rng.choice(6, p=probs)
            if cls in (1, 2):
                sett += 1
            elif cls == 0:
                empty += 1
            elif cls == 4:
                forest += 1
            total += 1
    if total == 0:
        return {"settlement_ratio": 0.12, "empty_ratio": 0.62, "forest_ratio": 0.22}
    return {
        "settlement_ratio": sett / total,
        "empty_ratio": empty / total,
        "forest_ratio": forest / total,
    }


def extract_training_xy(rounds, exclude_round):
    from ml_solver import extract_features

    X, Y = [], []
    for round_num, round_data in rounds.items():
        if round_num == exclude_round:
            continue
        for seed_data in round_data["seeds"].values():
            grid = seed_data["grid"]
            settlements = seed_data["settlements"]
            gt = seed_data["gt"]
            gf = seed_data["gt_global"]
            h, w = grid.shape
            for y in range(h):
                for x in range(w):
                    if grid[y, x] in (10, 5):
                        continue
                    X.append(extract_features(grid, settlements, y, x, gf))
                    Y.append(gt[y, x])
    return np.array(X), np.array(Y)


def flatten_dynamic(gt, pred, grid):
    mask = (grid != 10) & (grid != 5)
    Y_true = gt[mask].reshape(-1, 6)
    Y_pred = pred[mask].reshape(-1, 6)
    return Y_true, Y_pred


def score_proxy_n10(Y_true, Y_pred, n_samples=10, n_draws=20, floor=0.001):
    gt_safe = np.maximum(Y_true, 1e-10)
    gt_safe /= gt_safe.sum(axis=1, keepdims=True)
    pred_safe = normalize_predictions(Y_pred.copy(), floor=floor)
    scores = []
    for draw in range(n_draws):
        rng = np.random.default_rng(draw * 1000 + 42)
        empirical = rng.multinomial(n_samples, gt_safe).astype(np.float64) / n_samples
        score, _ = score_predictions(empirical, pred_safe, weighted=True, floor=floor)
        scores.append(score)
    return float(np.mean(scores))


def build_sim_seed_data(seed_data):
    settlements = [dict(s) for s in seed_data["settlements"]]
    for i, s in enumerate(settlements):
        s.setdefault("population", 1.0)
        s.setdefault("food", 0.5)
        s.setdefault("wealth", 0.3)
        s.setdefault("defense", 0.5)
        s.setdefault("owner_id", i)
        s.setdefault("has_port", False)
        s.setdefault("alive", True)
    return {
        "grid": seed_data["grid"],
        "sett_x": np.array([s["x"] for s in settlements], dtype=np.int32),
        "sett_y": np.array([s["y"] for s in settlements], dtype=np.int32),
        "sett_pop": np.array([s["population"] for s in settlements], dtype=np.float64),
        "sett_food": np.array([s["food"] for s in settlements], dtype=np.float64),
        "sett_wealth": np.array([s["wealth"] for s in settlements], dtype=np.float64),
        "sett_defense": np.array([s["defense"] for s in settlements], dtype=np.float64),
        "sett_owner": np.array([s["owner_id"] for s in settlements], dtype=np.int32),
        "sett_has_port": np.array([1 if s.get("has_port", False) else 0 for s in settlements], dtype=np.int32),
        "sett_alive": np.array([1 if s.get("alive", True) else 0 for s in settlements], dtype=np.int32),
        "ground_truth": seed_data["gt"],
    }


def ml_avg5_predictions(model, round_num, round_data, obs_repeats):
    details = round_data["details"]
    pred_by_seed = {seed_idx: [] for seed_idx in range(5)}
    for repeat in range(obs_repeats):
        for obs_seed in range(5):
            rng = np.random.default_rng(stable_round_seed(round_num, obs_seed=obs_seed, repeat=repeat))
            obs_gf = simulate_obs_global(round_data["seeds"][obs_seed], rng)
            for target_seed in range(5):
                pred = predict_round(model, details, target_seed, obs_gf)
                pred_by_seed[target_seed].append(pred)
    return {
        seed_idx: normalize_predictions(np.mean(preds, axis=0))
        for seed_idx, preds in pred_by_seed.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, nargs="+", default=DEFAULT_ROUNDS)
    parser.add_argument("--obs-repeats", type=int, default=DEFAULT_OBS_REPEATS)
    parser.add_argument("--sim-sims", type=int, default=DEFAULT_SIM_SIMS)
    args = parser.parse_args()

    rounds = load_all_rounds()
    params = np.load(BEST_PARAMS_PATH)

    # Warm up numba so timings are representative.
    warm_seed = build_sim_seed_data(rounds[19]["seeds"][0])
    _ = run_simulation(
        warm_seed["grid"], warm_seed["sett_x"], warm_seed["sett_y"],
        warm_seed["sett_pop"], warm_seed["sett_food"], warm_seed["sett_wealth"],
        warm_seed["sett_defense"], warm_seed["sett_owner"], warm_seed["sett_has_port"],
        warm_seed["sett_alive"], 42, params
    )

    totals = {alpha: [] for alpha in BLEND_ALPHAS}
    per_round = {alpha: {} for alpha in BLEND_ALPHAS}

    print(f"Recent-round ML+sim blend probe")
    print(f"  rounds={args.rounds}  obs_repeats={args.obs_repeats}  sim_sims={args.sim_sims}")
    print()

    for round_num in args.rounds:
        t0 = time.time()
        round_data = rounds[round_num]
        X_train, Y_train = extract_training_xy(rounds, exclude_round=round_num)
        model = train_model(X_train, Y_train)

        ml_preds = ml_avg5_predictions(model, round_num, round_data, args.obs_repeats)
        sim_preds = {}
        for seed_idx in range(5):
            sim_seed_data = build_sim_seed_data(round_data["seeds"][seed_idx])
            sim_preds[seed_idx] = run_sims(params, sim_seed_data, n_sims=args.sim_sims)

        for alpha in BLEND_ALPHAS:
            seed_scores = []
            for seed_idx in range(5):
                gt = round_data["seeds"][seed_idx]["gt"]
                grid = round_data["seeds"][seed_idx]["grid"]
                if alpha <= 0.0:
                    pred = ml_preds[seed_idx]
                elif alpha >= 1.0:
                    pred = sim_preds[seed_idx]
                else:
                    pred = normalize_predictions((1.0 - alpha) * ml_preds[seed_idx] + alpha * sim_preds[seed_idx])
                Y_true, Y_pred = flatten_dynamic(gt, pred, grid)
                seed_scores.append(score_proxy_n10(Y_true, Y_pred))
            round_score = float(np.mean(seed_scores))
            totals[alpha].append(round_score)
            per_round[alpha][round_num] = round_score

        elapsed = time.time() - t0
        baseline = per_round[0.0][round_num]
        best_alpha = max(BLEND_ALPHAS, key=lambda a: per_round[a][round_num])
        best_score = per_round[best_alpha][round_num]
        print(
            f"  round{round_num}: ml={baseline:5.2f}  best_alpha={best_alpha:>4.2f}"
            f" -> {best_score:5.2f}  delta={best_score - baseline:+.2f}  [{elapsed:.0f}s]"
        )

    print("\nSummary")
    for alpha in BLEND_ALPHAS:
        avg = float(np.mean(totals[alpha]))
        print(f"  alpha={alpha:>4.2f}  avg_proxy={avg:5.2f}")

    best_alpha = max(BLEND_ALPHAS, key=lambda a: float(np.mean(totals[a])))
    best_avg = float(np.mean(totals[best_alpha]))
    baseline_avg = float(np.mean(totals[0.0]))
    print(
        f"\nBest fixed blend: alpha={best_alpha:.2f}  avg={best_avg:.2f}"
        f"  delta_vs_ml={best_avg - baseline_avg:+.2f}"
    )


if __name__ == "__main__":
    main()
