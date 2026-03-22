"""
ML model improvement experiments.
Tests: hyperparameter tuning, LightGBM, target transforms, deeper models.
Evaluates with both 200-sim and simulated 8-sim scoring.
"""

import json, os, time, numpy as np, pickle
from ml_solver import (
    extract_features, compute_global_features_from_gt,
    evaluate_model, NUM_CLASSES, DATA_DIR, TERRAIN_TYPES
)
from sklearn.multioutput import MultiOutputRegressor

np.random.seed(42)


def load_cv_data():
    """Load all round data for CV, organized by round."""
    round_dirs = sorted([d for d in os.listdir(DATA_DIR)
                         if d.startswith("round") and os.path.isdir(os.path.join(DATA_DIR, d))])
    all_data = {}
    for rd in round_dirs:
        rdir = os.path.join(DATA_DIR, rd)
        dpath = os.path.join(rdir, "details.json")
        if not os.path.exists(dpath):
            continue
        with open(dpath) as f:
            details = json.load(f)
        rX, rY = [], []
        for seed in range(5):
            apath = os.path.join(rdir, f"analysis_seed{seed}.json")
            if not os.path.exists(apath):
                continue
            with open(apath) as f:
                gt = np.array(json.load(f)["ground_truth"])
            grid = np.array(details["initial_states"][seed]["grid"])
            settlements = details["initial_states"][seed]["settlements"]
            gf = compute_global_features_from_gt(gt, grid)
            for y in range(grid.shape[0]):
                for x in range(grid.shape[1]):
                    if grid[y, x] in (10, 5):
                        continue
                    rX.append(extract_features(grid, settlements, y, x, gf))
                    rY.append(gt[y, x])
        if rX:
            all_data[rd] = (np.array(rX), np.array(rY))
    return all_data


def score_predictions(Y_true, Y_pred):
    """Compute KL-based score (200-sim metric)."""
    Y_pred = np.clip(Y_pred, 0, 1)
    Y_pred = np.maximum(Y_pred, 0.001)
    Y_pred = Y_pred / Y_pred.sum(axis=-1, keepdims=True)
    kl = 0
    for i in range(len(Y_true)):
        for c in range(NUM_CLASSES):
            if Y_true[i, c] > 0:
                kl += Y_true[i, c] * np.log(Y_true[i, c] / Y_pred[i, c])
    kl /= len(Y_true)
    return 100 * np.exp(-kl)


def loo_cv(all_data, train_fn, label=""):
    """Leave-one-round-out CV. Returns per-round scores and average."""
    scores = []
    for test_round in sorted(all_data.keys()):
        train_X = np.concatenate([all_data[r][0] for r in all_data if r != test_round])
        train_Y = np.concatenate([all_data[r][1] for r in all_data if r != test_round])
        test_X, test_Y = all_data[test_round]

        model = train_fn(train_X, train_Y)
        Y_pred = model.predict(test_X)
        score = score_predictions(test_Y, Y_pred)
        scores.append(score)

    avg = np.mean(scores)
    return scores, avg


def make_xgb(n_estimators=500, max_depth=6, learning_rate=0.05,
             min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
             reg_alpha=0, reg_lambda=1):
    """Create XGBoost training function with given params."""
    from xgboost import XGBRegressor
    def train_fn(X, Y):
        model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, min_child_weight=min_child_weight,
                subsample=subsample, colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                n_jobs=32, random_state=42, verbosity=0,
            )
        )
        model.fit(X, Y)
        return model
    return train_fn


def make_lgbm(n_estimators=500, max_depth=6, learning_rate=0.05,
              num_leaves=31, min_child_samples=20, subsample=1.0,
              colsample_bytree=1.0, reg_alpha=0, reg_lambda=0):
    """Create LightGBM training function."""
    from lightgbm import LGBMRegressor
    def train_fn(X, Y):
        model = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                subsample=subsample, colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                n_jobs=32, random_state=42, verbose=-1,
            )
        )
        model.fit(X, Y)
        return model
    return train_fn


def make_rf(n_estimators=500, max_depth=15):
    """Create RandomForest training function."""
    from sklearn.ensemble import RandomForestRegressor
    def train_fn(X, Y):
        model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                n_jobs=32, random_state=42,
            )
        )
        model.fit(X, Y)
        return model
    return train_fn


def make_ensemble(*train_fns):
    """Create ensemble that averages predictions from multiple models."""
    class EnsemblePredictor:
        def __init__(self, models):
            self.models = models
        def predict(self, X):
            preds = [m.predict(X) for m in self.models]
            return np.mean(preds, axis=0)

    def train_fn(X, Y):
        models = [fn(X, Y) for fn in train_fns]
        return EnsemblePredictor(models)
    return train_fn


if __name__ == "__main__":
    print("Loading CV data...")
    all_data = load_cv_data()
    print(f"Loaded {len(all_data)} rounds\n")

    experiments = {}

    # Baseline: current XGBoost
    print("=" * 60)
    print("1. Baseline XGBoost (500 trees, depth 6, lr 0.05)")
    print("=" * 60)
    t0 = time.time()
    _, avg = loo_cv(all_data, make_xgb())
    print(f"   AVG: {avg:.2f} ({time.time()-t0:.0f}s)")
    experiments["XGB_baseline"] = avg

    # XGBoost: more trees
    print("\n2. XGBoost 1000 trees")
    t0 = time.time()
    _, avg = loo_cv(all_data, make_xgb(n_estimators=1000))
    print(f"   AVG: {avg:.2f} ({time.time()-t0:.0f}s)")
    experiments["XGB_1000trees"] = avg

    # XGBoost: deeper
    print("\n3. XGBoost depth 8")
    t0 = time.time()
    _, avg = loo_cv(all_data, make_xgb(max_depth=8))
    print(f"   AVG: {avg:.2f} ({time.time()-t0:.0f}s)")
    experiments["XGB_depth8"] = avg

    # XGBoost: lower learning rate + more trees
    print("\n4. XGBoost lr=0.02, 1000 trees")
    t0 = time.time()
    _, avg = loo_cv(all_data, make_xgb(n_estimators=1000, learning_rate=0.02))
    print(f"   AVG: {avg:.2f} ({time.time()-t0:.0f}s)")
    experiments["XGB_lr02_1000"] = avg

    # XGBoost: regularized
    print("\n5. XGBoost with regularization (alpha=0.1, lambda=5)")
    t0 = time.time()
    _, avg = loo_cv(all_data, make_xgb(reg_alpha=0.1, reg_lambda=5))
    print(f"   AVG: {avg:.2f} ({time.time()-t0:.0f}s)")
    experiments["XGB_regularized"] = avg

    # XGBoost: subsampling
    print("\n6. XGBoost with subsampling (0.8 rows, 0.8 cols)")
    t0 = time.time()
    _, avg = loo_cv(all_data, make_xgb(subsample=0.8, colsample_bytree=0.8))
    print(f"   AVG: {avg:.2f} ({time.time()-t0:.0f}s)")
    experiments["XGB_subsample"] = avg

    # XGBoost: deeper + more trees + lower lr
    print("\n7. XGBoost depth 8, 1500 trees, lr=0.03")
    t0 = time.time()
    _, avg = loo_cv(all_data, make_xgb(max_depth=8, n_estimators=1500, learning_rate=0.03))
    print(f"   AVG: {avg:.2f} ({time.time()-t0:.0f}s)")
    experiments["XGB_deep_slow"] = avg

    # LightGBM: default-ish
    print("\n8. LightGBM (500 trees, 31 leaves)")
    t0 = time.time()
    _, avg = loo_cv(all_data, make_lgbm())
    print(f"   AVG: {avg:.2f} ({time.time()-t0:.0f}s)")
    experiments["LGBM_default"] = avg

    # LightGBM: more leaves
    print("\n9. LightGBM (500 trees, 63 leaves)")
    t0 = time.time()
    _, avg = loo_cv(all_data, make_lgbm(num_leaves=63))
    print(f"   AVG: {avg:.2f} ({time.time()-t0:.0f}s)")
    experiments["LGBM_63leaves"] = avg

    # LightGBM: tuned
    print("\n10. LightGBM (1000 trees, 63 leaves, lr=0.03)")
    t0 = time.time()
    _, avg = loo_cv(all_data, make_lgbm(n_estimators=1000, num_leaves=63, learning_rate=0.03))
    print(f"   AVG: {avg:.2f} ({time.time()-t0:.0f}s)")
    experiments["LGBM_tuned"] = avg

    # Ensemble: XGB + LGBM
    print("\n11. Ensemble: XGB + LGBM")
    t0 = time.time()
    _, avg = loo_cv(all_data, make_ensemble(make_xgb(), make_lgbm()))
    print(f"   AVG: {avg:.2f} ({time.time()-t0:.0f}s)")
    experiments["Ensemble_XGB_LGBM"] = avg

    # Ensemble: XGB + LGBM + RF
    print("\n12. Ensemble: XGB + LGBM + RF")
    t0 = time.time()
    _, avg = loo_cv(all_data, make_ensemble(make_xgb(), make_lgbm(), make_rf()))
    print(f"   AVG: {avg:.2f} ({time.time()-t0:.0f}s)")
    experiments["Ensemble_XGB_LGBM_RF"] = avg

    # XGBoost: min_child_weight tuning
    print("\n13. XGBoost min_child_weight=5")
    t0 = time.time()
    _, avg = loo_cv(all_data, make_xgb(min_child_weight=5))
    print(f"   AVG: {avg:.2f} ({time.time()-t0:.0f}s)")
    experiments["XGB_mcw5"] = avg

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (sorted by score)")
    print("=" * 60)
    for name, score in sorted(experiments.items(), key=lambda x: -x[1]):
        delta = score - experiments["XGB_baseline"]
        marker = " <-- BEST" if score == max(experiments.values()) else ""
        print(f"  {name:30s}: {score:.2f} ({delta:+.2f}){marker}")
