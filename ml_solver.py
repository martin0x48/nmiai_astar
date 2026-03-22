"""
ML-based solver for Astar Island.

Strategy: Learn the mapping from initial state features → outcome probability
distributions using ground truth analysis data from past rounds.

Features per cell:
- Initial terrain type (one-hot)
- Neighbor terrain counts at radius 1, 2, 3
- Settlement proximity features
- Coastal/mountain adjacency
- Position features

Target: 6-class probability distribution
"""

import json
import os
import time
import sys
import numpy as np
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from collections import Counter
import pickle

TOKEN = "YOUR_JWT_TOKEN_HERE"
BASE = "https://api.ainm.no/astar-island"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
DATA_DIR = "/home/penguin/astar/data"
MODEL_PATH = "/home/penguin/astar/model.pkl"
MODEL_VERSION = 4
MODEL_SAMPLE_WEIGHT_POWER = 2.0
MODEL_FAMILY = "xgb_lgbm_cat_ensemble"
MODEL_TEMPERATURE = 1.09  # learned via N=10 proxy CV: softens overconfident predictions

if __name__ == "__main__":
    sys.modules["ml_solver"] = sys.modules[__name__]

CELL_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
TERRAIN_TYPES = [0, 1, 2, 3, 4, 5, 10, 11]  # all possible terrain values
NUM_CLASSES = 6


def extract_features(grid, settlements, y, x, global_features=None):
    """Extract features for a single cell from the initial state.

    Args:
        grid: 2D numpy array of terrain values
        settlements: list of settlement dicts
        y, x: cell coordinates
        global_features: optional dict with round-level features like
            {'settlement_ratio': 0.15, 'empty_ratio': 0.60, 'forest_ratio': 0.22}
            Estimated from observations or ground truth.

    Returns a feature vector capturing local context.
    """
    h, w = grid.shape
    features = []

    # 1. Own terrain (one-hot for 8 terrain types)
    cell_val = grid[y, x]
    for t in TERRAIN_TYPES:
        features.append(1.0 if cell_val == t else 0.0)

    # 2. Neighbor terrain counts at radius 1, 2, 3
    for radius in [1, 2, 3]:
        counts = {}
        for t in TERRAIN_TYPES:
            counts[t] = 0
        n_cells = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    counts[grid[ny, nx]] += 1
                    n_cells += 1
        # Normalize by number of cells in neighborhood
        for t in TERRAIN_TYPES:
            features.append(counts[t] / max(n_cells, 1))

    # 3. Settlement features
    sett_positions = [(s['x'], s['y']) for s in settlements]

    # Distance to nearest settlement
    if sett_positions:
        min_dist = min(abs(sx - x) + abs(sy - y) for sx, sy in sett_positions)
        features.append(min(min_dist, 20) / 20.0)  # normalized, capped at 20
    else:
        features.append(1.0)

    # Number of settlements within radius 2, 4, 8
    for r in [2, 4, 8]:
        count = sum(1 for sx, sy in sett_positions
                    if abs(sx - x) <= r and abs(sy - y) <= r)
        features.append(count / max(len(sett_positions), 1))

    # Is this cell a settlement in initial state?
    is_sett = any(sx == x and sy == y for sx, sy in sett_positions)
    features.append(1.0 if is_sett else 0.0)

    # Has port?
    has_port = any(s['x'] == x and s['y'] == y and s.get('has_port', False)
                   for s in settlements)
    features.append(1.0 if has_port else 0.0)

    # 4. Coastal (adjacent to ocean)
    is_coastal = False
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 10:
                is_coastal = True
                break
    features.append(1.0 if is_coastal else 0.0)

    # Adjacent to mountain
    near_mountain = False
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 5:
                near_mountain = True
                break
    features.append(1.0 if near_mountain else 0.0)

    # 5. Position features (normalized)
    features.append(x / w)
    features.append(y / h)

    # Distance to map center
    features.append(((x - w/2)**2 + (y - h/2)**2)**0.5 / ((w/2)**2 + (h/2)**2)**0.5)

    # Distance to nearest edge (non-ocean land edge)
    features.append(min(x, y, w - 1 - x, h - 1 - y) / (min(w, h) / 2))

    # 6. Additional features

    # Number of settlements in larger radii
    for r in [5, 10, 15]:
        count = sum(1 for sx, sy in sett_positions
                    if abs(sx - x) <= r and abs(sy - y) <= r)
        features.append(count / max(len(sett_positions), 1))

    # Total settlement count (global)
    features.append(len(sett_positions) / 50.0)  # normalized

    # Distance to 2nd and 3rd nearest settlement
    if len(sett_positions) >= 2:
        dists = sorted(abs(sx - x) + abs(sy - y) for sx, sy in sett_positions)
        features.append(min(dists[1], 20) / 20.0)
        if len(dists) >= 3:
            features.append(min(dists[2], 20) / 20.0)
        else:
            features.append(1.0)
    else:
        features.append(1.0)
        features.append(1.0)

    # Distance to nearest ocean in each cardinal direction
    for dy_dir, dx_dir in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        dist = 0
        cy, cx = y, x
        while 0 <= cy < h and 0 <= cx < w and grid[cy, cx] != 10:
            cy += dy_dir
            cx += dx_dir
            dist += 1
        features.append(min(dist, 20) / 20.0)

    # Plains-to-forest ratio in radius 3
    plains_r3 = 0
    forest_r3 = 0
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if grid[ny, nx] == 11:
                    plains_r3 += 1
                elif grid[ny, nx] == 4:
                    forest_r3 += 1
    total_pf = plains_r3 + forest_r3
    features.append(plains_r3 / max(total_pf, 1))

    # Number of terrain transitions in immediate neighborhood (texture measure)
    transitions = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] != cell_val:
                transitions += 1
    features.append(transitions / 8.0)

    # Is cell between two settlements? (on the line connecting two nearest settlements)
    if len(sett_positions) >= 2:
        dists = [(abs(sx - x) + abs(sy - y), sx, sy) for sx, sy in sett_positions]
        dists.sort()
        _, s1x, s1y = dists[0]
        _, s2x, s2y = dists[1]
        # Check if cell is roughly between them
        between_x = min(s1x, s2x) <= x <= max(s1x, s2x)
        between_y = min(s1y, s2y) <= y <= max(s1y, s2y)
        features.append(1.0 if between_x and between_y else 0.0)
    else:
        features.append(0.0)

    # Number of port settlements nearby
    port_count = sum(1 for s in settlements
                     if s.get('has_port', False) and abs(s['x'] - x) <= 5 and abs(s['y'] - y) <= 5)
    features.append(port_count / max(len(settlements), 1))

    # 7. Global features (round-level expansion estimate)
    if global_features is not None:
        features.append(global_features.get('settlement_ratio', 0.12))
        features.append(global_features.get('empty_ratio', 0.62))
        features.append(global_features.get('forest_ratio', 0.22))
    else:
        # Default: average across training data
        features.append(0.12)
        features.append(0.62)
        features.append(0.22)

    return features


def compute_global_features_from_gt(ground_truth, initial_grid):
    """Compute round-level features from ground truth distribution."""
    mask = (initial_grid != 10) & (initial_grid != 5)
    gt = ground_truth[mask]
    return {
        'settlement_ratio': float(gt[:, 1].mean()),
        'empty_ratio': float(gt[:, 0].mean()),
        'forest_ratio': float(gt[:, 4].mean()),
    }


def compute_global_features_from_observation(obs_grid, initial_grid):
    """Estimate round-level features from a single observation sample."""
    h, w = initial_grid.shape
    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    total = 0
    for y in range(h):
        for x in range(w):
            if obs_grid[y, x] >= 0 and initial_grid[y, x] not in (10, 5):
                cls = CELL_TO_CLASS.get(obs_grid[y, x], 0)
                counts[cls] += 1
                total += 1

    if total == 0:
        return {'settlement_ratio': 0.12, 'empty_ratio': 0.62, 'forest_ratio': 0.22}

    return {
        'settlement_ratio': (counts[1] + counts[2]) / total,  # settlements + ports
        'empty_ratio': counts[0] / total,
        'forest_ratio': counts[4] / total,
    }


def extract_all_features(grid, settlements, global_features=None):
    """Extract features for all cells in the grid."""
    h, w = grid.shape
    all_features = []
    coords = []
    for y in range(h):
        for x in range(w):
            features = extract_features(grid, settlements, y, x, global_features)
            all_features.append(features)
            coords.append((y, x))
    return np.array(all_features), coords


def load_training_data():
    """Load ground truth data from analysis files."""
    X_all = []
    Y_all = []
    skipped = 0

    # Find all rounds with analysis data
    round_dirs = sorted([d for d in os.listdir(DATA_DIR)
                         if d.startswith("round") and os.path.isdir(os.path.join(DATA_DIR, d))])

    for round_dir_name in round_dirs:
        round_dir = os.path.join(DATA_DIR, round_dir_name)
        details_path = os.path.join(round_dir, "details.json")

        if not os.path.exists(details_path):
            continue

        with open(details_path) as f:
            details = json.load(f)

        for seed_idx in range(5):
            analysis_path = os.path.join(round_dir, f"analysis_seed{seed_idx}.json")
            if not os.path.exists(analysis_path):
                continue

            with open(analysis_path) as f:
                analysis = json.load(f)

            # Get initial state
            initial_state = details["initial_states"][seed_idx]
            grid = np.array(initial_state["grid"])
            settlements = initial_state["settlements"]
            gt = np.array(analysis["ground_truth"])

            # Note: replay frame 0 has richer settlement data (pop, food, wealth, defense, owner_id)
            # but adding those as features hurt CV scores. Kept simple for now.

            # Compute global features from ground truth
            gf = compute_global_features_from_gt(gt, grid)

            h, w = grid.shape

            # Extract features and targets for each cell
            for y in range(h):
                for x in range(w):
                    # Skip ocean and mountain (static, not scored)
                    if grid[y, x] == 10 or grid[y, x] == 5:
                        skipped += 1
                        continue

                    features = extract_features(grid, settlements, y, x, gf)
                    target = gt[y, x]  # 6-class probability distribution

                    X_all.append(features)
                    Y_all.append(target)

    print(f"Loaded {len(X_all)} training examples, skipped {skipped} static cells")
    return np.array(X_all), np.array(Y_all)


class EnsembleModel:
    """Ensemble of multiple models that averages predictions."""

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = [m.predict(X) for m in self.models]
        return np.mean(predictions, axis=0)

    def fit(self, X, Y, sample_weight=None):
        for m in self.models:
            if sample_weight is None:
                m.fit(X, Y)
            else:
                m.fit(X, Y, sample_weight=sample_weight)
        return self


# Ensure ensembles trained via `python ml_solver.py train` unpickle via `import ml_solver`.
EnsembleModel.__module__ = "ml_solver"


def compute_cell_entropy(Y):
    """Per-cell entropy for a batch of target distributions."""
    Y = np.asarray(Y, dtype=np.float64)
    safe_Y = np.clip(Y, 1e-12, 1.0)
    return -(np.where(Y > 0, Y * np.log(safe_Y), 0.0)).sum(axis=1)


def get_training_sample_weights(Y, power=MODEL_SAMPLE_WEIGHT_POWER):
    """Upweight uncertain cells that dominate the leaderboard metric."""
    entropy = compute_cell_entropy(Y)
    return np.maximum(entropy, 1e-6) ** power


def normalize_predictions(Y_pred, floor=0.001):
    """Clip, floor, and renormalize predictions into valid distributions."""
    Y_pred = np.clip(Y_pred, 0, 1)
    Y_pred = np.maximum(Y_pred, floor)
    return Y_pred / Y_pred.sum(axis=-1, keepdims=True)


def apply_temperature(Y_pred, T=MODEL_TEMPERATURE):
    """Temperature scaling: raises probabilities to power 1/T and renormalizes.

    T > 1 softens predictions (reduces overconfidence).
    T < 1 sharpens predictions.
    T = 1 is identity.
    """
    if abs(T - 1.0) < 1e-6:
        return Y_pred
    log_p = np.log(np.maximum(Y_pred, 1e-12))
    scaled = log_p / T
    scaled -= scaled.max(axis=-1, keepdims=True)  # numerical stability
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum(axis=-1, keepdims=True)


def score_predictions(Y_true, Y_pred, weighted=False, floor=0.001):
    """Compute leaderboard score and KL, optionally entropy-weighted."""
    Y_pred = normalize_predictions(Y_pred, floor=floor)
    safe_true = np.clip(Y_true, 1e-12, 1.0)
    kl_per_cell = np.where(
        Y_true > 0,
        Y_true * (np.log(safe_true) - np.log(Y_pred)),
        0.0,
    ).sum(axis=1)

    if weighted:
        weights = compute_cell_entropy(Y_true)
        total_weight = max(weights.sum(), 1e-12)
        kl_avg = float((weights * kl_per_cell).sum() / total_weight)
    else:
        kl_avg = float(np.mean(kl_per_cell))

    score = float(100 * np.exp(-kl_avg))
    return score, kl_avg


def save_model(model, path=MODEL_PATH, temperature=MODEL_TEMPERATURE):
    """Persist model with metadata so strategy changes invalidate old pickles."""
    payload = {
        "version": MODEL_VERSION,
        "model_family": MODEL_FAMILY,
        "sample_weight_power": MODEL_SAMPLE_WEIGHT_POWER,
        "temperature": temperature,
        "model": model,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_saved_model(path=MODEL_PATH):
    """Load model plus metadata, handling legacy plain-model pickles."""
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload

    return payload, {"version": 0}


def train_model(X, Y, sample_weight_power=MODEL_SAMPLE_WEIGHT_POWER):
    """Train an XGBoost/LightGBM/CatBoost ensemble to predict 6-class distributions.

    Training weights follow the entropy-weighted leaderboard metric so
    high-uncertainty boundary cells matter more during fitting.
    """
    print(f"Training on {X.shape[0]} examples, {X.shape[1]} features, {Y.shape[1]} outputs")
    sample_weight = get_training_sample_weights(Y, power=sample_weight_power)

    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    xgb_model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            n_jobs=32,
            random_state=42,
            verbosity=0,
        )
    )

    lgbm_model = MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=32,
            random_state=42,
            verbose=-1,
        )
    )

    cat_model = MultiOutputRegressor(
        CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            thread_count=32,
            random_seed=42,
            verbose=0,
            allow_const_label=True,
            bootstrap_type='No',
        )
    )

    model = EnsembleModel([xgb_model, lgbm_model, cat_model])
    model.fit(X, Y, sample_weight=sample_weight)
    return model


def predict_round(model, details, seed_idx, global_features=None, temperature=MODEL_TEMPERATURE):
    """Generate predictions for a single seed."""
    initial_state = details["initial_states"][seed_idx]
    grid = np.array(initial_state["grid"])
    settlements = initial_state["settlements"]
    h, w = grid.shape

    # Extract features for all cells
    X, coords = extract_all_features(grid, settlements, global_features)

    # Predict
    Y_pred = model.predict(X)

    # Reshape to grid
    prediction = np.zeros((h, w, NUM_CLASSES))
    for i, (y, x) in enumerate(coords):
        if grid[y, x] == 10:
            # Ocean → class 0
            prediction[y, x, 0] = 1.0
        elif grid[y, x] == 5:
            # Mountain → class 5
            prediction[y, x, 5] = 1.0
        else:
            prediction[y, x] = Y_pred[i]

    # Ensure valid probabilities: clip, floor, normalize
    prediction = np.clip(prediction, 0, 1)
    prediction = np.maximum(prediction, 0.001)
    prediction = prediction / prediction.sum(axis=-1, keepdims=True)

    # Apply temperature scaling to dynamic cells
    if abs(temperature - 1.0) > 1e-6:
        dynamic_mask = (grid != 10) & (grid != 5)
        h, w = grid.shape
        flat_pred = prediction[dynamic_mask]  # (n_dynamic, 6)
        flat_pred = apply_temperature(flat_pred, temperature)
        prediction[dynamic_mask] = flat_pred

    return prediction


def evaluate_model(model, X, Y, weighted=False):
    """Evaluate model using KL divergence."""
    Y_pred = model.predict(X)
    return score_predictions(Y, Y_pred, weighted=weighted)


def bayesian_update(prior_dist, observed_class, n_observations=1, prior_weight=10):
    """Update a prior distribution with observed samples.

    prior_dist: (6,) probability distribution from model
    observed_class: int, the class we observed
    n_observations: how many times we observed this class
    prior_weight: how many "virtual observations" the prior is worth
    """
    # Convert prior to pseudo-counts
    counts = prior_dist * prior_weight

    # Add observation
    counts[observed_class] += n_observations

    # Normalize back to distribution
    updated = counts / counts.sum()
    return updated


def solve_with_model():
    """Main solving function using ML model."""
    # Load or train model
    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        model, meta = load_saved_model(MODEL_PATH)
        if meta.get("version", 0) < MODEL_VERSION:
            print(f"Saved model is outdated (v{meta.get('version', 0)}), retraining...")
            X, Y = load_training_data()
            model = train_model(X, Y)
            save_model(model, MODEL_PATH)
    else:
        print("Training model from analysis data...")
        X, Y = load_training_data()
        model = train_model(X, Y)

        # Evaluate
        score_u, kl_u = evaluate_model(model, X, Y, weighted=False)
        score_w, kl_w = evaluate_model(model, X, Y, weighted=True)
        print(f"Training score: unweighted={score_u:.2f} (KL={kl_u:.4f}), weighted={score_w:.2f} (KL={kl_w:.4f})")

        # Save model
        save_model(model, MODEL_PATH)
        print(f"Saved model to {MODEL_PATH}")

    return model


def api_get(path):
    r = requests.get(f"{BASE}{path}", headers=HEADERS)
    r.raise_for_status()
    return r.json()


def api_post(path, payload):
    r = requests.post(f"{BASE}{path}", headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Just train and evaluate
        print("Loading training data...")
        X, Y = load_training_data()

        # Cross-validate: train on 3 rounds, test on 1
        print("\nCross-validation by round:")
        round_dirs = sorted([d for d in os.listdir(DATA_DIR)
                             if d.startswith("round") and os.path.isdir(os.path.join(DATA_DIR, d))])

        # We need to split by round for proper CV
        # For now, just train on all and report training score
        model = train_model(X, Y)
        score_u, kl_u = evaluate_model(model, X, Y, weighted=False)
        score_w, kl_w = evaluate_model(model, X, Y, weighted=True)
        print(f"\nTraining score: unweighted={score_u:.2f} (KL={kl_u:.4f}), weighted={score_w:.2f} (KL={kl_w:.4f})")
        print(f"(For reference: observation-only scored 0.08-2.15, heuristic scored 49.56)")

        # Save
        save_model(model, MODEL_PATH)
        print(f"Saved model to {MODEL_PATH}")

    elif len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        # Leave-one-round-out cross-validation
        print("Loading training data by round...")
        round_dirs = sorted([d for d in os.listdir(DATA_DIR)
                             if d.startswith("round") and os.path.isdir(os.path.join(DATA_DIR, d))])

        all_data = {}
        for round_dir_name in round_dirs:
            round_dir = os.path.join(DATA_DIR, round_dir_name)
            details_path = os.path.join(round_dir, "details.json")
            if not os.path.exists(details_path):
                continue

            with open(details_path) as f:
                details = json.load(f)

            round_X = []
            round_Y = []
            for seed_idx in range(5):
                analysis_path = os.path.join(round_dir, f"analysis_seed{seed_idx}.json")
                if not os.path.exists(analysis_path):
                    continue

                with open(analysis_path) as f:
                    analysis = json.load(f)

                grid = np.array(details["initial_states"][seed_idx]["grid"])
                settlements = details["initial_states"][seed_idx]["settlements"]
                gt = np.array(analysis["ground_truth"])

                gf = compute_global_features_from_gt(gt, grid)
                for y in range(grid.shape[0]):
                    for x in range(grid.shape[1]):
                        if grid[y, x] in (10, 5):
                            continue
                        features = extract_features(grid, settlements, y, x, gf)
                        round_X.append(features)
                        round_Y.append(gt[y, x])

            if round_X:
                all_data[round_dir_name] = (np.array(round_X), np.array(round_Y))
                print(f"  {round_dir_name}: {len(round_X)} examples")

        # Leave-one-out CV
        print("\nLeave-one-round-out cross-validation:")
        for test_round in all_data:
            train_X = []
            train_Y = []
            for r in all_data:
                if r != test_round:
                    train_X.append(all_data[r][0])
                    train_Y.append(all_data[r][1])

            if not train_X:
                continue

            train_X = np.concatenate(train_X)
            train_Y = np.concatenate(train_Y)
            test_X, test_Y = all_data[test_round]

            model = train_model(train_X, train_Y)
            score_u, kl_u = evaluate_model(model, test_X, test_Y, weighted=False)
            score_w, kl_w = evaluate_model(model, test_X, test_Y, weighted=True)
            print(
                f"  Test on {test_round}: "
                f"unweighted={score_u:.2f} (KL={kl_u:.4f}), "
                f"weighted={score_w:.2f} (KL={kl_w:.4f})"
            )

    else:
        print("Usage:")
        print("  uv run python ml_solver.py train     # Train model on analysis data")
        print("  uv run python ml_solver.py evaluate   # Cross-validate by round")
