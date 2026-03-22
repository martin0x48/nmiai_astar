#!/usr/bin/env python
"""Fit simple scorer families against actual completed-round seed scores.

Uses the exact submitted prediction tensors stored in analysis_seed*.json and
searches over low-sample empirical-KL variants to see whether the current N=10
proxy can be improved.
"""

import argparse
import json
import os
from itertools import product

import numpy as np

from ml_solver import DATA_DIR


def load_seed_records():
    records = []
    for d in sorted(os.listdir(DATA_DIR), key=lambda x: int(x.replace("round", ""))):
        if not d.startswith("round"):
            continue
        rdir = os.path.join(DATA_DIR, d)
        for seed_idx in range(5):
            path = os.path.join(rdir, f"analysis_seed{seed_idx}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                analysis = json.load(f)
            pred = analysis.get("prediction")
            gt = analysis.get("ground_truth")
            score = analysis.get("score")
            init_grid = analysis.get("initial_grid")
            if pred is None or gt is None or score is None or init_grid is None:
                continue
            pred = np.asarray(pred, dtype=np.float64)
            gt = np.asarray(gt, dtype=np.float64)
            init_grid = np.asarray(init_grid, dtype=np.int32)
            mask = (init_grid != 10) & (init_grid != 5)
            records.append({
                "round": int(d.replace("round", "")),
                "seed": seed_idx,
                "score": float(score),
                "pred": pred[mask].reshape(-1, 6),
                "gt": gt[mask].reshape(-1, 6),
            })
    return records


def normalize(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return arr / arr.sum(axis=1, keepdims=True)


def weighted_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    safe_true = np.clip(y_true, 1e-12, 1.0)
    kl = np.where(y_true > 0, y_true * (np.log(safe_true) - np.log(y_pred)), 0.0).sum(axis=1)
    entropy = -(np.where(y_true > 0, y_true * np.log(safe_true), 0.0)).sum(axis=1)
    total_weight = max(entropy.sum(), 1e-12)
    kl_avg = float((entropy * kl).sum() / total_weight)
    return float(100.0 * np.exp(-kl_avg))


def unweighted_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    safe_true = np.clip(y_true, 1e-12, 1.0)
    kl = np.where(y_true > 0, y_true * (np.log(safe_true) - np.log(y_pred)), 0.0).sum(axis=1)
    return float(100.0 * np.exp(-float(np.mean(kl))))


def score_old(record, weighted, pred_floor):
    pred = np.maximum(record["pred"], pred_floor)
    pred = normalize(pred)
    gt = normalize(np.maximum(record["gt"], 1e-12))
    if weighted:
        return weighted_score(gt, pred)
    return unweighted_score(gt, pred)


def score_empirical(record, n_samples, weighted, pred_floor, target_tau, n_draws=8):
    pred = np.maximum(record["pred"], pred_floor)
    pred = normalize(pred)
    gt = normalize(np.maximum(record["gt"], 1e-12))

    scores = []
    for draw in range(n_draws):
        rng = np.random.default_rng(record["round"] * 10000 + record["seed"] * 100 + draw + 42)
        sampled = rng.multinomial(n_samples, gt).astype(np.float64)
        if target_tau > 0:
            sampled += target_tau / 6.0
        emp = normalize(np.maximum(sampled, 1e-12))
        if weighted:
            scores.append(weighted_score(emp, pred))
        else:
            scores.append(unweighted_score(emp, pred))
    return float(np.mean(scores))


def evaluate(records, scorer):
    actual = np.array([r["score"] for r in records], dtype=np.float64)
    predicted = np.array([scorer(r) for r in records], dtype=np.float64)
    mae = float(np.mean(np.abs(predicted - actual)))
    corr = float(np.corrcoef(predicted, actual)[0, 1])
    bias = float(np.mean(predicted - actual))
    return mae, corr, bias, predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--full-grid", action="store_true")
    args = parser.parse_args()

    records = load_seed_records()
    print(f"Loaded {len(records)} scored seed submissions")

    results = []

    if args.full_grid:
        n_list = [3, 5, 8, 10, 12, 15, 20, 30]
        pred_floors = [1e-6, 1e-4, 1e-3, 1e-2]
        taus = [0.0, 0.1, 0.5, 1.0, 2.0]
    else:
        # Coarse but focused sweep around the already plausible low-sample family.
        n_list = [5, 8, 10, 12, 15, 20]
        pred_floors = [1e-6, 1e-4, 1e-3]
        taus = [0.0, 0.1, 0.5, 1.0]

    total = 8 + (len(n_list) * 2 * len(pred_floors) * len(taus))
    done = 0

    # Old-metric baselines
    for weighted, pred_floor in product([False, True], [1e-6, 1e-4, 1e-3, 1e-2]):
        name = f"old weighted={weighted} pred_floor={pred_floor:g}"
        mae, corr, bias, _ = evaluate(records, lambda r, w=weighted, pf=pred_floor: score_old(r, w, pf))
        results.append((mae, -corr, abs(bias), name, {"mae": mae, "corr": corr, "bias": bias}))
        done += 1
        print(f"[{done}/{total}] {name}  MAE={mae:.3f} corr={corr:.4f} bias={bias:+.3f}", flush=True)

    # Empirical low-sample families
    for n_samples, weighted, pred_floor, target_tau in product(
        n_list,
        [False, True],
        pred_floors,
        taus,
    ):
        name = (
            f"emp N={n_samples} weighted={weighted} "
            f"pred_floor={pred_floor:g} target_tau={target_tau:g}"
        )
        mae, corr, bias, _ = evaluate(
            records,
            lambda r, n=n_samples, w=weighted, pf=pred_floor, tt=target_tau:
                score_empirical(r, n, w, pf, tt, n_draws=args.draws),
        )
        results.append((mae, -corr, abs(bias), name, {"mae": mae, "corr": corr, "bias": bias}))
        done += 1
        print(f"[{done}/{total}] {name}  MAE={mae:.3f} corr={corr:.4f} bias={bias:+.3f}", flush=True)

    results.sort()

    print("\nTop 15 fits by MAE / corr / bias:")
    for _, _, _, name, metrics in results[:15]:
        print(
            f"  {name:<60}  "
            f"MAE={metrics['mae']:.3f}  corr={metrics['corr']:.4f}  bias={metrics['bias']:+.3f}"
        )


if __name__ == "__main__":
    main()
