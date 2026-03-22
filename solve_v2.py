"""
Astar Island solver v2 - improved observation-based approach.
Key improvements over v1:
- Saves observation data to disk for resubmission
- Uses lower probability floor (0.001) for higher score on observed cells
- Can resubmit without re-querying
- Rate limiting with proper retry
"""

import json
import time
import os
import requests
import numpy as np
from collections import Counter

TOKEN = "YOUR_JWT_TOKEN_HERE"
BASE = "https://api.ainm.no/astar-island"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

CELL_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
NUM_CLASSES = 6
OBS_DIR = "/home/penguin/astar/observations"


def api_get(path):
    r = requests.get(f"{BASE}{path}", headers=HEADERS)
    r.raise_for_status()
    return r.json()


def api_post(path, payload):
    r = requests.post(f"{BASE}{path}", headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()


def simulate(round_id, seed_index, x, y, w=15, h=15):
    payload = {
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport_x": x, "viewport_y": y,
        "viewport_width": w, "viewport_height": h,
    }
    r = requests.post(f"{BASE}/simulate", headers=HEADERS, json=payload)
    return r.json()


def compute_viewports(map_w, map_h, max_viewports=9):
    """Compute viewport positions to cover as much of the map as possible.
    Uses 15x15 viewports with overlap to ensure full coverage."""
    import math

    # Calculate grid dimensions needed
    nx = math.ceil(map_w / 15)  # minimum viewports in x
    ny = math.ceil(map_h / 15)  # minimum viewports in y

    # If we have enough viewports for full coverage, use evenly spaced positions
    if nx * ny <= max_viewports:
        # Calculate step to spread viewports evenly
        step_x = (map_w - 15) / max(1, nx - 1) if nx > 1 else 0
        step_y = (map_h - 15) / max(1, ny - 1) if ny > 1 else 0

        viewports = []
        for gy in range(ny):
            for gx in range(nx):
                x = int(round(gx * step_x))
                y = int(round(gy * step_y))
                x = max(0, min(x, map_w - 15))
                y = max(0, min(y, map_h - 15))
                viewports.append((x, y))
        return list(dict.fromkeys(viewports))[:max_viewports]
    else:
        # Not enough viewports for full coverage - use greedy placement
        # Start with corners and center
        viewports = []
        positions = set()
        step = 15  # no overlap to maximize coverage

        for gy in range(math.ceil(map_h / step)):
            for gx in range(math.ceil(map_w / step)):
                x = min(gx * step, max(0, map_w - 15))
                y = min(gy * step, max(0, map_h - 15))
                pos = (x, y)
                if pos not in positions:
                    positions.add(pos)
                    viewports.append(pos)

        return viewports[:max_viewports]


def save_observations(round_id, seed_idx, observed_grid, map_h, map_w):
    """Save observation data for later resubmission."""
    os.makedirs(OBS_DIR, exist_ok=True)
    path = os.path.join(OBS_DIR, f"r{round_id[:8]}_s{seed_idx}.json")
    data = {
        "round_id": round_id,
        "seed_index": seed_idx,
        "observed_grid": observed_grid.tolist(),
        "map_height": map_h,
        "map_width": map_w,
    }
    with open(path, 'w') as f:
        json.dump(data, f)
    return path


def load_observations(round_id, seed_idx):
    """Load previously saved observations."""
    path = os.path.join(OBS_DIR, f"r{round_id[:8]}_s{seed_idx}.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return np.array(data["observed_grid"])
    return None


def build_prediction(observed, initial_grid, map_h, map_w, floor=0.001):
    """Build prediction from observations and initial state."""
    prediction = np.zeros((map_h, map_w, NUM_CLASSES), dtype=float)

    for y in range(map_h):
        for x in range(map_w):
            init_val = initial_grid[y, x]

            if observed[y, x] >= 0:
                # Observed cell - very high confidence
                obs_class = CELL_TO_CLASS.get(observed[y, x], 0)
                prediction[y, x, obs_class] = 1.0
            elif init_val in (10, 5):
                # Static terrain (ocean/mountain)
                cls = CELL_TO_CLASS.get(init_val, 0)
                prediction[y, x, cls] = 1.0
            else:
                # Unobserved dynamic cell - heuristic prior
                prediction[y, x] = get_heuristic_prior(init_val)

    # Apply floor and normalize
    prediction = np.maximum(prediction, floor)
    sums = prediction.sum(axis=-1, keepdims=True)
    prediction = prediction / sums
    return prediction


def get_heuristic_prior(init_val):
    """Get probability distribution for unobserved cells based on initial terrain.

    Calibrated from round 1 observation data:
    - Plains: 75% stays plains, 22% settlement, 3% forest, 0% ruin
    - Forest: 63% stays forest, 27% settlement, 8% port, 2% plains
    - Settlement: 50% stays settlement, 50% forest (reclaimed) - small sample
    - Port: similar to settlement but more likely to stay port
    """
    probs = np.zeros(NUM_CLASSES)
    if init_val == 4:  # Forest
        #             empty  sett  port  ruin  forest mountain
        probs[:] = [0.03, 0.26, 0.08, 0.01, 0.61, 0.01]
    elif init_val == 11:  # Plains
        probs[:] = [0.72, 0.22, 0.01, 0.01, 0.03, 0.01]
    elif init_val == 1:  # Settlement
        probs[:] = [0.05, 0.55, 0.10, 0.05, 0.20, 0.05]
    elif init_val == 2:  # Port
        probs[:] = [0.05, 0.15, 0.55, 0.05, 0.15, 0.05]
    else:
        probs[:] = [0.60, 0.20, 0.05, 0.02, 0.10, 0.03]
    return probs


def observe_seed(round_id, seed_idx, map_w, map_h, max_queries=9):
    """Observe a single seed using viewport queries. Returns observed grid."""
    viewports = compute_viewports(map_w, map_h)[:max_queries]
    observed = np.full((map_h, map_w), -1, dtype=int)

    for vp_idx, (vx, vy) in enumerate(viewports):
        print(f"  Query {vp_idx+1}/{len(viewports)}: viewport ({vx},{vy}) 15x15")
        result = simulate(round_id, seed_idx, vx, vy, 15, 15)

        if "error" in result:
            print(f"  Error: {result['error']}")
            break

        vp = result["viewport"]
        grid = result["grid"]
        for row_i, row in enumerate(grid):
            for col_i, val in enumerate(row):
                gy = vp["y"] + row_i
                gx = vp["x"] + col_i
                if 0 <= gy < map_h and 0 <= gx < map_w:
                    observed[gy, gx] = val

        print(f"  Budget: {result.get('queries_used', '?')}/{result.get('queries_max', '?')}")
        time.sleep(4.5)  # Rate limit: 15 queries/min = 4s between

    return observed


def observe_and_submit(round_id, details, budget, mode="full"):
    """
    Observe and submit predictions.
    mode: "full" = use queries to observe, "resubmit" = use saved data only
    """
    map_w = details.get("map_width", 40)
    map_h = details.get("map_height", 40)
    queries_left = budget["queries_max"] - budget["queries_used"]

    if mode == "full":
        queries_per_seed = min(queries_left // 5, 9)
        print(f"Using {queries_per_seed} queries per seed ({queries_left} total left)")
    elif mode == "resubmit":
        queries_per_seed = 0
        print("Resubmitting with saved observations only")

    for seed_idx in range(5):
        print(f"\n--- Seed {seed_idx} ---")
        initial_grid = np.array(details["initial_states"][seed_idx]["grid"])

        # Try loading saved observations
        observed = load_observations(round_id, seed_idx)
        if observed is not None:
            obs_count = (observed >= 0).sum()
            print(f"  Loaded saved observations: {obs_count}/{map_h*map_w} cells")
            if mode == "resubmit" or obs_count >= map_h * map_w:
                # Use saved data
                pass
            else:
                # Merge with new observations
                print("  Querying for additional coverage...")
                new_obs = observe_seed(round_id, seed_idx, map_w, map_h, queries_per_seed)
                # Merge: prefer new observations
                mask = new_obs >= 0
                observed[mask] = new_obs[mask]
                save_observations(round_id, seed_idx, observed, map_h, map_w)
        elif mode == "full" and queries_per_seed > 0:
            observed = observe_seed(round_id, seed_idx, map_w, map_h, queries_per_seed)
            save_observations(round_id, seed_idx, observed, map_h, map_w)
        else:
            print("  No saved observations and no queries available!")
            observed = np.full((map_h, map_w), -1, dtype=int)

        # Build and submit prediction
        obs_count = (observed >= 0).sum()
        total_cells = map_h * map_w
        print(f"  Coverage: {obs_count}/{total_cells} ({obs_count/total_cells*100:.0f}%)")

        # Use lower floor when we have high coverage
        floor = 0.001 if obs_count > total_cells * 0.9 else 0.01
        prediction = build_prediction(observed, initial_grid, map_h, map_w, floor=floor)

        # Verify
        sums = prediction.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=0.01)
        assert (prediction >= 0).all()

        result = api_post("/submit", {
            "round_id": round_id,
            "seed_index": seed_idx,
            "prediction": prediction.tolist(),
        })
        print(f"  Submit: {result}")


def check_analysis(round_id):
    """Check post-round analysis for ground truth comparison."""
    print("\n=== Post-round Analysis ===")
    for seed_idx in range(5):
        try:
            r = requests.get(f"{BASE}/analysis/{round_id}/{seed_idx}", headers=HEADERS)
            if r.status_code == 200:
                data = r.json()
                print(f"\nSeed {seed_idx}: {json.dumps(data, indent=2)[:500]}")
            else:
                print(f"Seed {seed_idx}: {r.status_code} - {r.text[:200]}")
        except Exception as e:
            print(f"Seed {seed_idx}: Error - {e}")


def main():
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    # Get active round
    rounds = api_get("/rounds")
    active = [r for r in rounds if r["status"] == "active"]

    if mode == "analysis":
        # Check analysis for all rounds
        for r in rounds:
            check_analysis(r["id"])
        return

    if mode == "scores":
        result = api_get("/my-rounds")
        print(json.dumps(result, indent=2))
        return

    if not active:
        print("No active rounds!")
        # Check analysis for completed rounds
        completed = [r for r in rounds if r["status"] != "active"]
        for r in completed:
            check_analysis(r["id"])
        return

    round_info = active[0]
    round_id = round_info["id"]
    print(f"Round: {round_info['round_number']} ({round_id})")
    print(f"Map: {round_info['map_width']}x{round_info['map_height']}")
    print(f"Closes: {round_info['closes_at']}")

    details = api_get(f"/rounds/{round_id}")
    budget = api_get("/budget")
    print(f"Budget: {budget['queries_used']}/{budget['queries_max']} used")

    if mode == "resubmit":
        print("\n=== Resubmitting with saved observations ===")
        observe_and_submit(round_id, details, budget, mode="resubmit")
    elif mode == "full":
        observe_and_submit(round_id, details, budget, mode="full")
    else:
        print(f"Unknown mode: {mode}")

    # Check scores
    print("\n=== Scores ===")
    result = api_get("/my-rounds")
    for r in result:
        print(f"Round {r['round_number']}: score={r.get('round_score')}, rank={r.get('rank')}")
        if r.get('seed_scores'):
            for i, s in enumerate(r['seed_scores']):
                print(f"  Seed {i}: {s}")


if __name__ == "__main__":
    main()
