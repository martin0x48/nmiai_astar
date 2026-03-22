"""
Auto-solver: single-run script that checks for new rounds and solves them.
Uses ML model trained on ground truth analysis data from past rounds.
Observations are used for Bayesian correction of ML predictions.

Run: uv run python auto_solve.py
"""

import json
import time
import os
import sys
import pickle
import requests
import numpy as np
from datetime import datetime, timezone

TOKEN = "YOUR_JWT_TOKEN_HERE"
BASE = "https://api.ainm.no/astar-island"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

CELL_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
NUM_CLASSES = 6
DATA_DIR = "/home/penguin/astar/data"
MODEL_PATH = "/home/penguin/astar/model.pkl"
PROB_FLOOR = 0.001
ULTRA_SPARSE_SETTLEMENT_MAX = 0.035
ULTRA_SPARSE_EMPTY_MIN = 0.72
ULTRA_SPARSE_FOREST_MAX = 0.255


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def api_get(path):
    r = requests.get(f"{BASE}{path}", headers=HEADERS)
    r.raise_for_status()
    return r.json()


def api_post(path, payload):
    r = requests.post(f"{BASE}{path}", headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()


def simulate_query(round_id, seed_index, x, y, w=15, h=15):
    """Query the API for a viewport of the final state."""
    payload = {
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport_x": x, "viewport_y": y,
        "viewport_width": w, "viewport_height": h,
    }
    r = requests.post(f"{BASE}/simulate", headers=HEADERS, json=payload)
    return r.json()


def compute_viewports(map_w, map_h):
    """Compute viewport positions for full coverage of map."""
    import math
    nx = math.ceil(map_w / 15)
    ny = math.ceil(map_h / 15)

    if nx * ny <= 10:
        # Full coverage possible
        step_x = (map_w - 15) / max(1, nx - 1) if nx > 1 else 0
        step_y = (map_h - 15) / max(1, ny - 1) if ny > 1 else 0
        viewports = []
        for gy in range(ny):
            for gx in range(nx):
                x = max(0, min(int(round(gx * step_x)), map_w - 15))
                y = max(0, min(int(round(gy * step_y)), map_h - 15))
                viewports.append((x, y))
        return list(dict.fromkeys(viewports))
    else:
        # Can't fully cover - tile without overlap
        viewports = []
        for gy in range(math.ceil(map_h / 15)):
            for gx in range(math.ceil(map_w / 15)):
                x = min(gx * 15, max(0, map_w - 15))
                y = min(gy * 15, max(0, map_h - 15))
                viewports.append((x, y))
        return list(dict.fromkeys(viewports))


def get_heuristic_prior(init_val):
    """Probability distribution for unobserved cells."""
    probs = np.zeros(NUM_CLASSES)
    if init_val == 4:      # Forest
        probs[:] = [0.03, 0.26, 0.08, 0.01, 0.61, 0.01]
    elif init_val == 11:   # Plains
        probs[:] = [0.72, 0.22, 0.01, 0.01, 0.03, 0.01]
    elif init_val == 1:    # Settlement
        probs[:] = [0.05, 0.55, 0.10, 0.05, 0.20, 0.05]
    elif init_val == 2:    # Port
        probs[:] = [0.05, 0.15, 0.55, 0.05, 0.15, 0.05]
    else:
        probs[:] = [0.60, 0.20, 0.05, 0.02, 0.10, 0.03]
    return probs


def summarize_global_features(all_global_features):
    """Average observed round-level features across observed seeds."""
    if not all_global_features:
        return None
    gfs = list(all_global_features.values())
    return {
        "settlement_ratio": float(np.mean([gf["settlement_ratio"] for gf in gfs])),
        "empty_ratio": float(np.mean([gf["empty_ratio"] for gf in gfs])),
        "forest_ratio": float(np.mean([gf["forest_ratio"] for gf in gfs])),
    }


def should_ignore_globals(round_feature_summary, observed_seed_count):
    """Detect the round-22-like ultra-sparse regime where globals hurt badly.

    This guardrail is intentionally narrow:
    - only activates when we observed all 5 seeds
    - requires very low settlement, very high empty ratio, and limited forest
    """
    if observed_seed_count < 5 or round_feature_summary is None:
        return False
    return (
        round_feature_summary["settlement_ratio"] < ULTRA_SPARSE_SETTLEMENT_MAX
        and round_feature_summary["empty_ratio"] > ULTRA_SPARSE_EMPTY_MIN
        and round_feature_summary["forest_ratio"] < ULTRA_SPARSE_FOREST_MAX
    )


def retrain_model_if_new_data():
    """Retrain model if we have new analysis data since last training."""
    from ml_solver import MODEL_VERSION, load_saved_model, load_training_data, save_model, train_model

    analysis_files = []
    for d in os.listdir(DATA_DIR):
        rdir = os.path.join(DATA_DIR, d)
        if os.path.isdir(rdir):
            for f in os.listdir(rdir):
                if f.startswith("analysis_seed"):
                    analysis_files.append(os.path.join(rdir, f))

    if not analysis_files:
        return

    needs_retrain = not os.path.exists(MODEL_PATH)
    model_version = 0

    if os.path.exists(MODEL_PATH):
        try:
            _, meta = load_saved_model(MODEL_PATH)
            model_version = meta.get("version", 0)
        except Exception:
            model_version = 0
            needs_retrain = True

    if model_version < MODEL_VERSION:
        needs_retrain = True

    if not needs_retrain:
        model_mtime = os.path.getmtime(MODEL_PATH)
        newest_analysis = max(os.path.getmtime(f) for f in analysis_files)
        if model_mtime >= newest_analysis:
            return  # model is up to date

    log(
        f"Retraining model ({len(analysis_files)} analysis files, "
        f"saved_version={model_version}, target_version={MODEL_VERSION})..."
    )
    X, Y = load_training_data()
    if len(X) > 0:
        model = train_model(X, Y)
        save_model(model, MODEL_PATH)
        log(f"Retrained model on {len(X)} examples")


def load_ml_model():
    """Load the trained ML model."""
    from ml_solver import MODEL_VERSION, load_saved_model, load_training_data, save_model, train_model

    if not os.path.exists(MODEL_PATH):
        log("No ML model found, training from analysis data...")
        X, Y = load_training_data()
        if len(X) == 0:
            log("No training data available, falling back to heuristics")
            return None
        model = train_model(X, Y)
        save_model(model, MODEL_PATH)
        log(f"Trained and saved model ({len(X)} examples)")
        return model

    try:
        model, meta = load_saved_model(MODEL_PATH)
    except Exception as e:
        log(f"Failed to load saved model ({e}), retraining...")
        X, Y = load_training_data()
        if len(X) == 0:
            log("No training data available, falling back to heuristics")
            return None
        model = train_model(X, Y)
        save_model(model, MODEL_PATH)
        return model

    if meta.get("version", 0) < MODEL_VERSION:
        log(
            f"Saved model version {meta.get('version', 0)} is outdated; "
            f"retraining to version {MODEL_VERSION}"
        )
        X, Y = load_training_data()
        if len(X) == 0:
            log("No training data available, falling back to heuristics")
            return None
        model = train_model(X, Y)
        save_model(model, MODEL_PATH)
        return model

    log(f"Loaded ML model (version {meta.get('version', 0)})")
    return model


def ml_predict_seed(model, details, seed_idx, global_features=None):
    """Generate ML predictions for a single seed."""
    from ml_solver import predict_round
    return predict_round(model, details, seed_idx, global_features)


def bayesian_update(prediction, observed, map_h, map_w, prior_weight=10):
    """Update ML predictions with observation samples using Bayesian updating."""
    # Count observations per cell per class
    obs_counts = np.zeros((map_h, map_w, NUM_CLASSES))
    obs_total = np.zeros((map_h, map_w))

    for obs_grid in observed:
        for y in range(map_h):
            for x in range(map_w):
                if obs_grid[y, x] >= 0:
                    cls = CELL_TO_CLASS.get(obs_grid[y, x], 0)
                    obs_counts[y, x, cls] += 1
                    obs_total[y, x] += 1

    # Bayesian update: prior (ML) + observations
    updated = prediction.copy()
    for y in range(map_h):
        for x in range(map_w):
            if obs_total[y, x] > 0:
                # Convert prior to pseudo-counts
                prior_counts = prediction[y, x] * prior_weight
                # Add observation counts
                posterior_counts = prior_counts + obs_counts[y, x]
                # Normalize
                updated[y, x] = posterior_counts / posterior_counts.sum()

    return updated


def get_settlement_stats_from_replay(round_id, seed_idx):
    """Fetch frame 0 settlement stats from replay API (has pop, food, etc.).
    Only works for completed rounds — returns None for active rounds."""
    try:
        r = requests.post(f"{BASE}/replay",
            headers={"Content-Type": "application/json"},
            cookies={"access_token": TOKEN},
            json={"round_id": round_id, "seed_index": seed_idx})
        time.sleep(2)  # rate limit
        if r.status_code == 200:
            data = r.json()
            return data['frames'][0]['settlements']
        elif r.status_code == 403:
            return None  # active round, expected
    except Exception as e:
        log(f"  Replay API failed: {e}")
    return None


def solve_round_simulator_OLD(round_info, details, round_dir):
    """OLD VERSION — kept for reference. Do not use.
    Solve using v2 simulator + observation calibration.

    Strategy:
    1. Run 500 Monte Carlo sims per seed → baseline distributions
    2. Use 10 queries per seed to observe actual final state samples
    3. Compute per-class calibration ratios from observations vs simulator
    4. Apply calibration to improve predictions
    """
    from sim_search.fast_sim_v2 import run_simulation, CELL_TO_CLASS
    import time as _time

    round_id = round_info["id"]
    map_w = round_info["map_width"]
    map_h = round_info["map_height"]
    N_SIMS = 2000  # smooth distributions — 2000 sims × 26ms = 52s per seed on 1 core

    # Load best params
    params_path = os.path.join(os.path.dirname(__file__), "sim_search", "best_params_v2.npy")
    if not os.path.exists(params_path):
        log("ERROR: No v2 simulator params found")
        return False

    params = np.load(params_path)
    log(f"Loaded v2 simulator params")

    # Budget check
    budget = api_get("/budget")
    queries_left = budget["queries_max"] - budget["queries_used"]
    queries_per_seed = queries_left // 5
    log(f"Budget: {queries_left} queries, {queries_per_seed} per seed for calibration")

    # Compute viewports for observation
    all_viewports = compute_viewports(map_w, map_h)

    # === Phase 0: Observe seed 0 first to detect expansion level ===
    log("Phase 0: Observing seed 0 to detect expansion level...")
    state0 = details["initial_states"][0]
    grid0 = np.array(state0["grid"], dtype=np.int32)
    h0, w0 = grid0.shape

    obs_grid_s0 = np.full((h0, w0), -1, dtype=int)
    viewports_s0 = all_viewports[:min(9, queries_per_seed)]
    for vp_idx, (vx, vy) in enumerate(viewports_s0):
        for attempt in range(3):
            result = simulate_query(round_id, 0, vx, vy, 15, 15)
            if "error" not in result:
                break
            _time.sleep(10)
        else:
            continue
        vp = result["viewport"]
        for row_i, row in enumerate(result["grid"]):
            for col_i, val in enumerate(row):
                gy = vp["y"] + row_i
                gx = vp["x"] + col_i
                if 0 <= gy < h0 and 0 <= gx < w0:
                    obs_grid_s0[gy, gx] = val
        _time.sleep(4.5)

    # Compute observed settlement density
    mask0 = (grid0 != 10) & (grid0 != 5) & (obs_grid_s0 >= 0)
    if mask0.sum() > 0:
        obs_sett = sum(1 for y in range(h0) for x in range(w0)
                      if mask0[y,x] and obs_grid_s0[y,x] in (1, 2)) / mask0.sum()
    else:
        obs_sett = 0.12
    log(f"  Observed settlement density: {obs_sett:.4f}")

    # === Phase 0b: Find best growth rate by trying multiple in parallel ===
    # Run simulator at different growth rates, pick best match to observations
    growth_rates_to_try = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.163, 0.20]

    # Only adapt growth rate for EXTREME cases
    # Backtest showed: 3D search overfits to noise and HURTS on normal rounds
    # Default simulator scores 95+ on normal rounds — don't mess with it
    if obs_sett < 0.01:
        params[0] = 0.02
        log(f"  COLLAPSE round detected (sett={obs_sett:.4f}) → growth_rate=0.02")
    elif obs_sett > 0.22:
        params[0] = 0.20
        log(f"  HIGH EXPANSION round detected (sett={obs_sett:.4f}) → growth_rate=0.20")
    else:
        log(f"  Normal round (sett={obs_sett:.4f}) → using default growth_rate={params[0]:.3f}")

    for seed_idx in range(5):
        log(f"--- Seed {seed_idx} (simulator + calibration, gr={selected_growth:.3f}) ---")

        # Get initial state
        state = details["initial_states"][seed_idx]
        grid = np.array(state["grid"], dtype=np.int32)
        h, w = grid.shape

        # Try to get rich settlement data from replay API
        settlements = get_settlement_stats_from_replay(round_id, seed_idx)
        if settlements is None:
            settlements = state["settlements"]
            for i, s in enumerate(settlements):
                s.setdefault('population', 1.0)
                s.setdefault('food', 0.5)
                s.setdefault('wealth', 0.3)
                s.setdefault('defense', 0.5)
                s.setdefault('owner_id', i)  # CRITICAL: each settlement is its own faction
            log(f"  Using default settlement stats ({len(settlements)} settlements, unique owners)")
        else:
            log(f"  Rich settlement stats from replay ({len(settlements)} settlements)")

        # Save settlement stats
        sett_path = os.path.join(round_dir, "replays", f"seed{seed_idx}_frame0_settlements.json")
        os.makedirs(os.path.dirname(sett_path), exist_ok=True)
        with open(sett_path, 'w') as f:
            json.dump(settlements, f)

        # Convert to arrays
        sett_x = np.array([s['x'] for s in settlements], dtype=np.int32)
        sett_y = np.array([s['y'] for s in settlements], dtype=np.int32)
        sett_pop = np.array([s['population'] for s in settlements], dtype=np.float64)
        sett_food = np.array([s['food'] for s in settlements], dtype=np.float64)
        sett_wealth = np.array([s['wealth'] for s in settlements], dtype=np.float64)
        sett_defense = np.array([s['defense'] for s in settlements], dtype=np.float64)
        sett_owner = np.array([s['owner_id'] for s in settlements], dtype=np.int32)
        sett_has_port = np.array([1 if s.get('has_port', False) else 0 for s in settlements], dtype=np.int32)
        sett_alive = np.array([1 if s.get('alive', True) else 0 for s in settlements], dtype=np.int32)

        # Step 1: Run Monte Carlo simulations
        t0 = _time.time()
        counts = np.zeros((h, w, 6), dtype=np.int32)
        for sim_idx in range(N_SIMS):
            final_grid = run_simulation(
                grid, sett_x, sett_y, sett_pop, sett_food,
                sett_wealth, sett_defense, sett_owner, sett_has_port, sett_alive,
                sim_idx * 7919 + 42, params,
            )
            for y in range(h):
                for x in range(w):
                    cls = CELL_TO_CLASS.get(int(final_grid[y, x]), 0)
                    counts[y, x, cls] += 1
        t1 = _time.time()
        log(f"  {N_SIMS} sims in {t1-t0:.1f}s")

        sim_pred = counts.astype(np.float64) / N_SIMS

        # Step 2: Observe actual final state samples for calibration
        obs_counts = np.zeros((h, w, 6))
        obs_total = np.zeros((h, w))

        # Seed 0: reuse phase 0 observations (already queried)
        if seed_idx == 0:
            for y in range(h):
                for x in range(w):
                    if obs_grid_s0[y, x] >= 0:
                        cls = CELL_TO_CLASS.get(obs_grid_s0[y, x], 0)
                        obs_counts[y, x, cls] += 1
                        obs_total[y, x] += 1
            log(f"  Reusing {int(obs_total.sum())} phase 0 observations")

        # Seeds 1-4: query fresh observations
        remaining_queries = queries_per_seed if seed_idx > 0 else 0
        viewports_to_query = all_viewports[:remaining_queries]
        for vp_idx, (vx, vy) in enumerate(viewports_to_query):
            log(f"  Calibration query {vp_idx+1}/{len(viewports_to_query)}: ({vx},{vy})")
            for attempt in range(3):
                result = simulate_query(round_id, seed_idx, vx, vy, 15, 15)
                if "error" in result:
                    log(f"  Error: {result['error']}, retrying...")
                    _time.sleep(10)
                else:
                    break
            else:
                continue

            vp = result["viewport"]
            for row_i, row in enumerate(result["grid"]):
                for col_i, val in enumerate(row):
                    gy = vp["y"] + row_i
                    gx = vp["x"] + col_i
                    if 0 <= gy < h and 0 <= gx < w:
                        cls = CELL_TO_CLASS.get(val, 0)
                        obs_counts[gy, gx, cls] += 1
                        obs_total[gy, gx] += 1

            n_queries_used += 1
            log(f"  Budget: {result.get('queries_used','?')}/{result.get('queries_max','?')}")
            _time.sleep(4.5)

        # Step 3: Compute calibration ratios from observed vs simulated
        # Compare per-class averages on observed dynamic cells
        mask = (grid != 10) & (grid != 5) & (obs_total > 0)
        obs_dist = np.zeros_like(obs_counts)
        for y in range(h):
            for x in range(w):
                if obs_total[y, x] > 0:
                    obs_dist[y, x] = obs_counts[y, x] / obs_total[y, x]

        calibration_ratios = np.ones(6)
        if mask.sum() > 0:
            for c in range(6):
                sim_avg = sim_pred[mask, c].mean()
                obs_avg = obs_dist[mask, c].mean()
                if sim_avg > 0.005:
                    calibration_ratios[c] = obs_avg / sim_avg
            log(f"  Calibration ratios: E={calibration_ratios[0]:.2f} S={calibration_ratios[1]:.2f} "
                f"P={calibration_ratios[2]:.2f} R={calibration_ratios[3]:.2f} "
                f"F={calibration_ratios[4]:.2f} M={calibration_ratios[5]:.2f}")

        # Step 4: Apply calibration + Bayesian blend
        prediction = sim_pred.copy()

        # Global calibration
        for y in range(h):
            for x in range(w):
                if grid[y, x] in (10, 5):
                    continue
                prediction[y, x] = sim_pred[y, x] * calibration_ratios

        # Bayesian blend with observations where available (pw=20)
        prior_weight = 20
        for y in range(h):
            for x in range(w):
                if obs_total[y, x] > 0 and grid[y, x] not in (10, 5):
                    prior_counts = prediction[y, x] * prior_weight
                    posterior = (prior_counts + obs_counts[y, x]) / (prior_weight + obs_total[y, x])
                    prediction[y, x] = posterior

        # Normalize
        prediction = np.maximum(prediction, PROB_FLOOR)
        prediction = prediction / prediction.sum(axis=-1, keepdims=True)

        # Static cells
        for y in range(h):
            for x in range(w):
                if grid[y, x] == 10:
                    prediction[y, x] = [0.99999, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002]
                elif grid[y, x] == 5:
                    prediction[y, x] = [0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.99999]

        # Save observations
        obs_path = os.path.join(round_dir, f"obs_seed{seed_idx}.json")
        with open(obs_path, "w") as f:
            json.dump({"obs_counts": obs_counts.tolist(), "obs_total": obs_total.tolist()}, f)

        # Submit
        result = api_post("/submit", {
            "round_id": round_info["id"],
            "seed_index": seed_idx,
            "prediction": prediction.tolist(),
        })
        log(f"  Submitted: {result}")

    return True


def solve_round_simulator(round_info, details, round_dir):
    """Solve using v2 simulator — simple, backtested approach.

    Backtest results (round 9):
    - Simulator alone: 95.24
    - Simulator + calibration: 95.24 (no improvement)
    - Simulator + 3D search + calibration: 66.27 (WORSE!)
    - ML model: 75.63

    Strategy: Keep it simple.
    1. Observe seed 0 (9 queries) → detect collapse/expansion
    2. Adjust growth rate only for extremes (<1% or >22% settlements)
    3. Run 2000 Monte Carlo sims per seed
    4. Submit raw distributions (no calibration — it doesn't help)
    """
    from sim_search.fast_sim_v2 import run_simulation, CELL_TO_CLASS
    import time as _time

    round_id = round_info["id"]
    map_w = round_info["map_width"]
    map_h = round_info["map_height"]
    N_SIMS = 2000

    params_path = os.path.join(os.path.dirname(__file__), "sim_search", "best_params_v2.npy")
    if not os.path.exists(params_path):
        log("ERROR: No v2 simulator params found")
        return False

    params = np.load(params_path)
    log(f"Loaded v2 simulator params (default growth={params[0]:.3f})")

    # === Observe seed 0 to detect expansion level ===
    budget = api_get("/budget")
    queries_left = budget["queries_max"] - budget["queries_used"]
    log(f"Budget: {queries_left} queries")

    all_viewports = compute_viewports(map_w, map_h)
    state0 = details["initial_states"][0]
    grid0 = np.array(state0["grid"], dtype=np.int32)
    h0, w0 = grid0.shape

    obs_grid_s0 = np.full((h0, w0), -1, dtype=int)
    n_obs_queries = min(9, queries_left)
    for vp_idx, (vx, vy) in enumerate(all_viewports[:n_obs_queries]):
        log(f"  Observe query {vp_idx+1}/{n_obs_queries}: ({vx},{vy})")
        for attempt in range(3):
            result = simulate_query(round_id, 0, vx, vy, 15, 15)
            if "error" not in result:
                break
            _time.sleep(10)
        else:
            continue
        vp = result["viewport"]
        for row_i, row in enumerate(result["grid"]):
            for col_i, val in enumerate(row):
                gy = vp["y"] + row_i
                gx = vp["x"] + col_i
                if 0 <= gy < h0 and 0 <= gx < w0:
                    obs_grid_s0[gy, gx] = val
        log(f"  Budget: {result.get('queries_used','?')}/{result.get('queries_max','?')}")
        _time.sleep(4.5)

    # Settlement density from observation
    mask0 = (grid0 != 10) & (grid0 != 5) & (obs_grid_s0 >= 0)
    if mask0.sum() > 0:
        obs_sett = sum(1 for y in range(h0) for x in range(w0)
                      if mask0[y,x] and obs_grid_s0[y,x] in (1, 2)) / mask0.sum()
    else:
        obs_sett = 0.12
    log(f"  Observed settlement density: {obs_sett:.4f}")

    # Save observation
    obs_path = os.path.join(round_dir, "obs_seed0_phase1.json")
    with open(obs_path, "w") as f:
        json.dump({"observed_grid": obs_grid_s0.tolist(),
                   "settlement_density": obs_sett}, f)

    # Adjust growth rate ONLY for extreme rounds
    if obs_sett < 0.01:
        params[0] = 0.02
        log(f"  COLLAPSE detected → growth_rate=0.02")
    elif obs_sett > 0.22:
        params[0] = 0.20
        log(f"  HIGH EXPANSION detected → growth_rate=0.20")
    else:
        log(f"  Normal round → keeping default growth_rate={params[0]:.3f}")

    # === Run simulator for all 5 seeds ===
    for seed_idx in range(5):
        log(f"--- Seed {seed_idx} ({N_SIMS} sims, gr={params[0]:.3f}) ---")

        state = details["initial_states"][seed_idx]
        grid = np.array(state["grid"], dtype=np.int32)
        h, w = grid.shape

        # Try replay API for rich settlement data
        settlements = get_settlement_stats_from_replay(round_id, seed_idx)
        if settlements is None:
            settlements = state["settlements"]
            for i, s in enumerate(settlements):
                s.setdefault('population', 1.0)
                s.setdefault('food', 0.5)
                s.setdefault('wealth', 0.3)
                s.setdefault('defense', 0.5)
                s.setdefault('owner_id', i)  # each settlement is its own faction

        # Save settlement stats
        sett_path = os.path.join(round_dir, "replays",
                                 f"seed{seed_idx}_frame0_settlements.json")
        os.makedirs(os.path.dirname(sett_path), exist_ok=True)
        with open(sett_path, 'w') as f:
            json.dump(settlements, f)

        sett_x = np.array([s['x'] for s in settlements], dtype=np.int32)
        sett_y = np.array([s['y'] for s in settlements], dtype=np.int32)
        sett_pop = np.array([s['population'] for s in settlements], dtype=np.float64)
        sett_food = np.array([s['food'] for s in settlements], dtype=np.float64)
        sett_wealth = np.array([s['wealth'] for s in settlements], dtype=np.float64)
        sett_defense = np.array([s['defense'] for s in settlements], dtype=np.float64)
        sett_owner = np.array([s['owner_id'] for s in settlements], dtype=np.int32)
        sett_has_port = np.array([1 if s.get('has_port', False) else 0
                                  for s in settlements], dtype=np.int32)
        sett_alive = np.array([1 if s.get('alive', True) else 0
                               for s in settlements], dtype=np.int32)

        # Run Monte Carlo
        t0 = _time.time()
        counts = np.zeros((h, w, 6), dtype=np.int32)
        for sim_idx in range(N_SIMS):
            final_grid = run_simulation(
                grid, sett_x, sett_y, sett_pop, sett_food,
                sett_wealth, sett_defense, sett_owner, sett_has_port, sett_alive,
                sim_idx * 7919 + 42, params,
            )
            for y in range(h):
                for x in range(w):
                    cls = CELL_TO_CLASS.get(int(final_grid[y, x]), 0)
                    counts[y, x, cls] += 1
        t1 = _time.time()
        log(f"  {N_SIMS} sims in {t1-t0:.1f}s")

        # Build prediction — raw distributions, no calibration
        prediction = counts.astype(np.float64) / N_SIMS
        prediction = np.maximum(prediction, PROB_FLOOR)
        prediction = prediction / prediction.sum(axis=-1, keepdims=True)

        # Static cells
        for y in range(h):
            for x in range(w):
                if grid[y, x] == 10:
                    prediction[y, x] = [0.99999, 0.000002, 0.000002,
                                        0.000002, 0.000002, 0.000002]
                elif grid[y, x] == 5:
                    prediction[y, x] = [0.000002, 0.000002, 0.000002,
                                        0.000002, 0.000002, 0.99999]

        # Submit
        result = api_post("/submit", {
            "round_id": round_info["id"],
            "seed_index": seed_idx,
            "prediction": prediction.tolist(),
        })
        log(f"  Submitted: {result}")
        _time.sleep(2)

    return True


def solve_round(round_info):
    """Solve a round — uses v2 simulator if available, falls back to ML model."""
    round_id = round_info["id"]
    round_num = round_info["round_number"]
    map_w = round_info["map_width"]
    map_h = round_info["map_height"]

    log(f"Solving round {round_num} ({round_id})")
    log(f"Map: {map_w}x{map_h}, closes: {round_info['closes_at']}")

    # Save round data
    round_dir = os.path.join(DATA_DIR, f"round{round_num}")
    os.makedirs(round_dir, exist_ok=True)

    # Get round details (initial states)
    details = api_get(f"/rounds/{round_id}")
    with open(os.path.join(round_dir, "details.json"), "w") as f:
        json.dump(details, f)
    log(f"Saved round details to {round_dir}/details.json")

    # Use ML model (reliable 75-86 scores)
    # Simulator approach moved to ~/astar-sim for separate development
    model = load_ml_model()
    if model is None:
        log("ERROR: No model available, cannot solve")
        return

    # Budget check
    budget = api_get("/budget")
    queries_left = budget["queries_max"] - budget["queries_used"]
    log(f"Budget: {queries_left} queries remaining")

    # Query strategy (avg5):
    # Observe all 5 seeds (9 queries each = 45 total), compute global features
    # from each observation, predict 5 times, average. Benchmark showed +0.22
    # proxy points vs seed-0-only.

    from ml_solver import compute_global_features_from_observation

    all_viewports = compute_viewports(map_w, map_h)
    queries_per_seed = min(len(all_viewports), 9)
    total_queries_needed = queries_per_seed * 5

    if queries_left < total_queries_needed:
        # Fallback: seed-0-only if budget is tight
        log(f"Budget too low for avg5 ({queries_left} < {total_queries_needed}), using seed-0 only")
        seeds_to_observe = [0]
        queries_per_seed = min(queries_per_seed, queries_left)
    else:
        seeds_to_observe = list(range(5))

    log(f"Observing seeds {seeds_to_observe} with {queries_per_seed} queries each")

    # Observe each seed and collect global features
    all_obs_grids = {}
    all_global_features = {}

    for obs_seed in seeds_to_observe:
        initial_grid_obs = np.array(details["initial_states"][obs_seed]["grid"])
        obs_grid = np.full((map_h, map_w), -1, dtype=int)

        for vp_idx, (vx, vy) in enumerate(all_viewports[:queries_per_seed]):
            log(f"  Seed {obs_seed} query {vp_idx+1}/{queries_per_seed}: viewport ({vx},{vy})")
            for attempt in range(3):
                result = simulate_query(round_id, obs_seed, vx, vy, 15, 15)
                if "error" in result:
                    log(f"  Error: {result['error']}, retrying in 10s...")
                    time.sleep(10)
                else:
                    break
            else:
                continue

            vp = result["viewport"]
            for row_i, row in enumerate(result["grid"]):
                for col_i, val in enumerate(row):
                    gy = vp["y"] + row_i
                    gx = vp["x"] + col_i
                    if 0 <= gy < map_h and 0 <= gx < map_w:
                        obs_grid[gy, gx] = val

            log(f"  Budget: {result.get('queries_used','?')}/{result.get('queries_max','?')}")
            time.sleep(4.5)

        gf = compute_global_features_from_observation(obs_grid, initial_grid_obs)
        all_obs_grids[obs_seed] = obs_grid
        all_global_features[obs_seed] = gf
        log(f"  Seed {obs_seed} globals: sett={gf['settlement_ratio']:.3f}, "
            f"empty={gf['empty_ratio']:.3f}, forest={gf['forest_ratio']:.3f}")

    # Save all observations
    round_feature_summary = summarize_global_features(all_global_features)
    ignore_globals = should_ignore_globals(round_feature_summary, len(seeds_to_observe))
    if round_feature_summary is not None:
        log(
            "Observed-round summary: "
            f"sett={round_feature_summary['settlement_ratio']:.3f}, "
            f"empty={round_feature_summary['empty_ratio']:.3f}, "
            f"forest={round_feature_summary['forest_ratio']:.3f}"
        )
    if ignore_globals:
        log(
            "Ultra-sparse guard triggered: observed regime matches the round-22-style "
            "failure mode, so global-feature conditioning will be skipped"
        )

    obs_path = os.path.join(round_dir, "obs_all_seeds.json")
    with open(obs_path, "w") as f:
        json.dump({
            "obs_grids": {str(k): v.tolist() for k, v in all_obs_grids.items()},
            "global_features": {str(k): v for k, v in all_global_features.items()},
            "round_feature_summary": round_feature_summary,
            "ignore_globals": ignore_globals,
        }, f)

    # For each target seed: predict once per observation, then average
    for seed_idx in range(5):
        log(f"--- Seed {seed_idx} ---")

        if ignore_globals:
            prediction = ml_predict_seed(model, details, seed_idx, None)
            log("  Used no-global prediction due to ultra-sparse guard")
        else:
            predictions = []
            for obs_seed in seeds_to_observe:
                pred = ml_predict_seed(model, details, seed_idx, all_global_features[obs_seed])
                predictions.append(pred)

            # Average predictions across observations
            prediction = np.mean(predictions, axis=0)
            log(f"  Averaged {len(predictions)} predictions (temperature-scaled)")

        # Ensure valid probabilities
        initial_grid = np.array(details["initial_states"][seed_idx]["grid"])
        prediction = np.clip(prediction, 0, 1)
        prediction = np.maximum(prediction, PROB_FLOOR)
        prediction = prediction / prediction.sum(axis=-1, keepdims=True)

        # Static cells: near-perfect confidence (tiny floor for API safety)
        for y in range(map_h):
            for x in range(map_w):
                if initial_grid[y, x] == 10:  # Ocean
                    prediction[y, x] = [0.99999, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002]
                elif initial_grid[y, x] == 5:  # Mountain
                    prediction[y, x] = [0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.99999]

        # Submit
        result = api_post("/submit", {
            "round_id": round_id,
            "seed_index": seed_idx,
            "prediction": prediction.tolist(),
        })
        log(f"  Submitted: {result}")

    # Check scores
    log("=== Scores ===")
    my_rounds = api_get("/my-rounds")
    for r in my_rounds:
        log(f"Round {r['round_number']}: score={r.get('round_score')}, rank={r.get('rank')}, submitted={r['seeds_submitted']}/5")

    # Try analysis for any completed rounds
    rounds = api_get("/rounds")
    for r in rounds:
        if r["status"] != "active":
            log(f"Checking analysis for round {r['round_number']}...")
            for seed in range(5):
                try:
                    resp = requests.get(f"{BASE}/analysis/{r['id']}/{seed}", headers=HEADERS)
                    if resp.status_code == 200:
                        analysis = resp.json()
                        analysis_path = os.path.join(DATA_DIR, f"round{r['round_number']}", f"analysis_seed{seed}.json")
                        os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
                        with open(analysis_path, "w") as f:
                            json.dump(analysis, f)
                        log(f"  Seed {seed}: saved analysis")
                    else:
                        log(f"  Seed {seed}: {resp.status_code}")
                    time.sleep(2)  # rate limit
                except Exception as e:
                    log(f"  Seed {seed}: error - {e}")


def main():
    """Run once: check for active rounds, solve if new, grab analysis for completed."""
    log("Checking rounds...")

    # Retrain model if new analysis data is available
    try:
        retrain_model_if_new_data()
    except Exception as e:
        log(f"Model retrain check failed: {e}")

    # Find which rounds we've already submitted for
    submitted_ids = set()
    try:
        my_rounds = api_get("/my-rounds")
        for r in my_rounds:
            if r["seeds_submitted"] >= 5:
                submitted_ids.add(r["id"])
            if r.get("round_score") is not None:
                log(f"Score - Round {r['round_number']}: score={r['round_score']}, rank={r.get('rank')}")
    except Exception as e:
        log(f"Error checking my-rounds: {e}")

    rounds = api_get("/rounds")

    # Solve any active rounds we haven't submitted for
    for r in rounds:
        if r["status"] == "active" and r["id"] not in submitted_ids:
            log(f"New active round: {r['round_number']}")
            solve_round(r)
        elif r["status"] == "active":
            log(f"Round {r['round_number']}: already submitted, skipping")

    # Grab analysis for completed rounds
    for r in rounds:
        if r["status"] != "active":
            round_dir = os.path.join(DATA_DIR, f"round{r['round_number']}")
            analysis_path = os.path.join(round_dir, "analysis_seed0.json")
            if os.path.exists(analysis_path):
                continue  # already have it
            log(f"Round {r['round_number']} completed, grabbing analysis...")
            os.makedirs(round_dir, exist_ok=True)
            for seed in range(5):
                try:
                    resp = requests.get(f"{BASE}/analysis/{r['id']}/{seed}", headers=HEADERS)
                    if resp.status_code == 200:
                        path = os.path.join(round_dir, f"analysis_seed{seed}.json")
                        with open(path, "w") as f:
                            json.dump(resp.json(), f)
                        log(f"  Saved analysis seed {seed}")
                    else:
                        log(f"  Seed {seed}: {resp.status_code}")
                    time.sleep(2)  # rate limit
                except Exception as e:
                    log(f"  Error: {e}")

    log("Done.")


if __name__ == "__main__":
    main()
