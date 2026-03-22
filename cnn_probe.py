#!/usr/bin/env python
"""
Quick CNN probe on recent held-out rounds.

Purpose:
- test whether a small patch CNN can compete with the current tree ensemble
- use the live-like N=10 proxy, not the old offline metric
- focus on recent hard rounds only (19-21) for a fast signal
"""

import json
import math
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from benchmark import (
    extract_features_for_round,
    load_all_rounds,
    score_proxy_n10,
    simulate_seed0_obs_global,
)
from ml_solver import (
    MODEL_SAMPLE_WEIGHT_POWER,
    compute_cell_entropy,
    compute_global_features_from_gt,
    normalize_predictions,
    score_predictions,
    train_model,
)

warnings.filterwarnings("ignore", message="X does not have valid feature names")

PATCH_RADIUS = 3
PATCH_SIZE = 2 * PATCH_RADIUS + 1
PATCH_CHANNELS = 10  # 8 terrain one-hot + settlement mask + port mask
HOLDOUTS = ["round19", "round20", "round21"]
TRAIN_SUBSAMPLE = 120_000
BATCH_SIZE = 1024
EPOCHS = 8
LEARNING_RATE = 1e-3


def stable_round_seed(round_name):
    return 10_007 * int(round_name.replace("round", "")) + 42


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_static_channels(grid, settlements):
    """Channels that depend only on the initial state."""
    h, w = grid.shape
    channels = np.zeros((PATCH_CHANNELS, h, w), dtype=np.float32)

    terrain_vals = [0, 1, 2, 3, 4, 5, 10, 11]
    for i, val in enumerate(terrain_vals):
        channels[i] = (grid == val).astype(np.float32)

    for s in settlements:
        x, y = s["x"], s["y"]
        channels[8, y, x] = 1.0
        if s.get("has_port", False):
            channels[9, y, x] = 1.0

    return channels


def extract_patch_dataset(seeds, gf_per_seed):
    """Build patch tensors, aux features, and soft targets for a round."""
    patches = []
    aux = []
    targets = []

    for seed_idx, sd in seeds.items():
        grid = sd["grid"]
        settlements = sd["settlements"]
        gt = sd["gt"]
        gf = gf_per_seed[seed_idx]
        h, w = grid.shape

        base = build_static_channels(grid, settlements)
        padded = np.pad(
            base,
            ((0, 0), (PATCH_RADIUS, PATCH_RADIUS), (PATCH_RADIUS, PATCH_RADIUS)),
            mode="constant",
            constant_values=0.0,
        )

        for y in range(h):
            for x in range(w):
                if grid[y, x] in (10, 5):
                    continue
                py = y + PATCH_RADIUS
                px = x + PATCH_RADIUS
                patch = padded[:, py - PATCH_RADIUS:py + PATCH_RADIUS + 1, px - PATCH_RADIUS:px + PATCH_RADIUS + 1]
                patches.append(patch)
                aux.append(
                    [
                        gf["settlement_ratio"],
                        gf["empty_ratio"],
                        gf["forest_ratio"],
                        x / w,
                        y / h,
                    ]
                )
                targets.append(gt[y, x])

    return (
        np.asarray(patches, dtype=np.float32),
        np.asarray(aux, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


class PatchDataset(Dataset):
    def __init__(self, patches, aux, targets, weights):
        self.patches = torch.from_numpy(patches)
        self.aux = torch.from_numpy(aux)
        self.targets = torch.from_numpy(targets)
        self.weights = torch.from_numpy(weights.astype(np.float32))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.patches[idx], self.aux[idx], self.targets[idx], self.weights[idx]


class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(PATCH_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 5, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, patch, aux):
        x = self.conv(patch)
        x = x.mean(dim=(2, 3))
        x = torch.cat([x, aux], dim=1)
        return self.head(x)


def train_cnn(train_patches, train_aux, train_targets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = np.maximum(compute_cell_entropy(train_targets), 1e-6) ** MODEL_SAMPLE_WEIGHT_POWER

    if len(train_targets) > TRAIN_SUBSAMPLE:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(train_targets), TRAIN_SUBSAMPLE, replace=False)
        train_patches = train_patches[idx]
        train_aux = train_aux[idx]
        train_targets = train_targets[idx]
        weights = weights[idx]

    dataset = PatchDataset(train_patches, train_aux, train_targets, weights)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = PatchCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        total_w = 0.0
        for patch, aux, target, weight in loader:
            patch = patch.to(device)
            aux = aux.to(device)
            target = target.to(device)
            weight = weight.to(device)

            logits = model(patch, aux)
            log_probs = F.log_softmax(logits, dim=1)
            loss_per = -(target * log_probs).sum(dim=1)
            loss = (loss_per * weight).sum() / weight.sum()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float((loss_per * weight).sum().item())
            total_w += float(weight.sum().item())

        print(f"    epoch {epoch + 1}/{EPOCHS} loss={total / max(total_w, 1e-9):.4f}", flush=True)

    return model


def predict_cnn(model, patches, aux):
    device = next(model.parameters()).device
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(patches), BATCH_SIZE):
            patch = torch.from_numpy(patches[i:i + BATCH_SIZE]).to(device)
            aux_batch = torch.from_numpy(aux[i:i + BATCH_SIZE]).to(device)
            logits = model(patch, aux_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds.append(probs)
    return normalize_predictions(np.concatenate(preds, axis=0))


def normalize_targets(Y):
    Y = np.maximum(np.asarray(Y, dtype=np.float64), 1e-12)
    return Y / Y.sum(axis=1, keepdims=True)


def main():
    set_seeds(42)
    print("Loading rounds...", flush=True)
    rounds = load_all_rounds()
    round_names = sorted(rounds.keys(), key=lambda r: int(r.replace("round", "")))

    # Precompute GT and realistic seed-0 observation data
    gt_data = {}
    obs_data = {}
    patch_gt = {}
    patch_obs = {}
    for rname in round_names:
        seeds = rounds[rname]
        gt_gf = {si: sd["gt_global"] for si, sd in seeds.items()}
        gt_data[rname] = extract_features_for_round(seeds, gt_gf)
        patch_gt[rname] = extract_patch_dataset(seeds, gt_gf)

        rng = np.random.default_rng(stable_round_seed(rname))
        obs_gf_val = simulate_seed0_obs_global(seeds, rng)
        obs_gf = {si: obs_gf_val for si in seeds}
        obs_data[rname] = extract_features_for_round(seeds, obs_gf)
        patch_obs[rname] = extract_patch_dataset(seeds, obs_gf)

    print("\nRecent holdout comparison", flush=True)
    print("=" * 80, flush=True)
    tree_scores = []
    cnn_scores = []

    for test_round in HOLDOUTS:
        t0 = time.time()
        print(f"\n{test_round}", flush=True)

        train_X = np.concatenate([gt_data[r][0] for r in round_names if r != test_round])
        train_Y = np.concatenate([gt_data[r][1] for r in round_names if r != test_round])
        test_X, test_Y = obs_data[test_round]
        test_Y = normalize_targets(test_Y)

        tree = train_model(train_X, train_Y)
        tree_pred = normalize_predictions(tree.predict(test_X))
        tree_proxy, _ = score_proxy_n10(test_Y, tree_pred, n_draws=20)
        tree_old, _ = score_predictions(test_Y, tree_pred, weighted=True)

        train_patches = np.concatenate([patch_gt[r][0] for r in round_names if r != test_round])
        train_aux = np.concatenate([patch_gt[r][1] for r in round_names if r != test_round])
        train_targets = np.concatenate([patch_gt[r][2] for r in round_names if r != test_round])
        test_patches, test_aux, test_targets = patch_obs[test_round]
        test_targets = normalize_targets(test_targets)

        cnn = train_cnn(train_patches, train_aux, train_targets)
        cnn_pred = predict_cnn(cnn, test_patches, test_aux)
        cnn_proxy, _ = score_proxy_n10(test_targets, cnn_pred, n_draws=20)
        cnn_old, _ = score_predictions(test_targets, cnn_pred, weighted=True)

        tree_scores.append(tree_proxy)
        cnn_scores.append(cnn_proxy)

        print(
            f"  tree proxy={tree_proxy:.2f} old={tree_old:.2f} | "
            f"cnn proxy={cnn_proxy:.2f} old={cnn_old:.2f} | "
            f"delta={cnn_proxy - tree_proxy:+.2f} [{time.time() - t0:.0f}s]",
            flush=True,
        )

    print("\n" + "=" * 80)
    print(f"tree avg proxy: {np.mean(tree_scores):.2f}")
    print(f"cnn  avg proxy: {np.mean(cnn_scores):.2f}")
    print(f"cnn delta vs tree: {np.mean(cnn_scores) - np.mean(tree_scores):+.2f}")


if __name__ == "__main__":
    main()
