"""
Astar Island solver - simple observation-based approach.
Strategy: Use 10 queries per seed (50 total) to cover the map,
then predict based on observations + initial terrain for unobserved cells.
"""

import json
import time
import requests
import numpy as np
from collections import Counter

TOKEN = "YOUR_JWT_TOKEN_HERE"
BASE = "https://api.ainm.no/astar-island"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# Prediction classes: 0=Empty/Ocean/Plains, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain
# Grid cell values: 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain, 10=Ocean, 11=Plains
CELL_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
NUM_CLASSES = 6
PROB_FLOOR = 0.01


def get_round():
    r = requests.get(f"{BASE}/rounds", headers=HEADERS)
    rounds = r.json()
    active = [r for r in rounds if r["status"] == "active"]
    if not active:
        print("No active rounds!")
        return None
    return active[0]


def get_round_details(round_id):
    r = requests.get(f"{BASE}/rounds/{round_id}", headers=HEADERS)
    return r.json()


def get_budget():
    r = requests.get(f"{BASE}/budget", headers=HEADERS)
    return r.json()


def simulate(round_id, seed_index, x, y, w=15, h=15):
    payload = {
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport_x": x,
        "viewport_y": y,
        "viewport_width": w,
        "viewport_height": h,
    }
    r = requests.post(f"{BASE}/simulate", headers=HEADERS, json=payload)
    return r.json()


def submit(round_id, seed_index, prediction):
    payload = {
        "round_id": round_id,
        "seed_index": seed_index,
        "prediction": prediction,
    }
    r = requests.post(f"{BASE}/submit", headers=HEADERS, json=payload)
    return r.json()


def compute_viewports(map_w, map_h, num_queries=10):
    """Compute viewport positions to cover as much of the map as possible.
    With 10 queries of 15x15 on a 40x40 map, we can cover most of it.
    Strategy: tile with some overlap to ensure full coverage.
    """
    # 40x40 map, 15x15 viewport. 3 viewports across = 45 cells (covers 40).
    # 3x3 = 9 viewports covers the whole map. Use 9 queries + 1 spare.
    viewports = []
    # 3x3 tiling
    positions = []
    step = 13  # 15 - 2 overlap
    for gy in range(3):
        for gx in range(3):
            x = min(gx * step, map_w - 15)
            y = min(gy * step, map_h - 15)
            positions.append((x, y))

    # Take first num_queries
    viewports = positions[:num_queries]
    return viewports


def apply_floor_and_normalize(probs):
    """Apply probability floor and renormalize."""
    probs = np.maximum(probs, PROB_FLOOR)
    sums = probs.sum(axis=-1, keepdims=True)
    return probs / sums


def initial_grid_to_class(grid_value):
    """Map initial grid values to prediction class."""
    return CELL_TO_CLASS.get(grid_value, 0)


def is_static_terrain(grid_value):
    """Check if terrain is static (won't change during simulation)."""
    return grid_value in (10, 5)  # Ocean, Mountain


def main():
    # Get active round
    round_info = get_round()
    if not round_info:
        return
    round_id = round_info["id"]
    map_w = round_info["map_width"]
    map_h = round_info["map_height"]
    print(f"Round: {round_info['round_number']} ({round_id})")
    print(f"Map: {map_w}x{map_h}, closes at {round_info['closes_at']}")

    # Get initial states
    details = get_round_details(round_id)
    budget = get_budget()
    queries_left = budget["queries_max"] - budget["queries_used"]
    print(f"Budget: {queries_left} queries remaining")

    # Plan: 10 queries per seed (9 for full coverage + 1 spare)
    queries_per_seed = queries_left // 5
    print(f"Using {queries_per_seed} queries per seed")

    viewports = compute_viewports(map_w, map_h, queries_per_seed)
    print(f"Viewport positions: {viewports}")

    for seed_idx in range(5):
        print(f"\n--- Seed {seed_idx} ---")
        initial_state = details["initial_states"][seed_idx]
        initial_grid = np.array(initial_state["grid"])  # shape: (H, W)

        # Observed final grid - start empty
        observed = np.full((map_h, map_w), -1, dtype=int)
        observation_count = np.zeros((map_h, map_w, NUM_CLASSES), dtype=int)

        # Query the simulator
        for vp_idx, (vx, vy) in enumerate(viewports):
            print(f"  Query {vp_idx+1}: viewport ({vx},{vy}) 15x15")
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
                        cls = CELL_TO_CLASS.get(val, 0)
                        observation_count[gy, gx, cls] += 1

            print(f"  Queries used: {result.get('queries_used', '?')}/{result.get('queries_max', '?')}")
            time.sleep(0.25)  # Rate limit safety

        # Build prediction
        prediction = np.zeros((map_h, map_w, NUM_CLASSES), dtype=float)

        for y in range(map_h):
            for x in range(map_w):
                init_val = initial_grid[y, x]

                if observed[y, x] >= 0:
                    # We observed this cell - use observation
                    obs_class = CELL_TO_CLASS.get(observed[y, x], 0)
                    prediction[y, x, obs_class] = 1.0
                elif is_static_terrain(init_val):
                    # Static terrain - predict same as initial
                    cls = initial_grid_to_class(init_val)
                    prediction[y, x, cls] = 1.0
                else:
                    # Unobserved dynamic cell - use heuristic prior
                    # Based on initial terrain type
                    if init_val == 4:  # Forest
                        # Forest is mostly static but can be cleared for settlement
                        prediction[y, x, 4] = 0.80  # stays forest
                        prediction[y, x, 0] = 0.08  # becomes plains
                        prediction[y, x, 1] = 0.06  # becomes settlement
                        prediction[y, x, 2] = 0.02
                        prediction[y, x, 3] = 0.02
                        prediction[y, x, 5] = 0.02
                    elif init_val == 11:  # Plains
                        prediction[y, x, 0] = 0.50  # stays plains/empty
                        prediction[y, x, 1] = 0.20  # becomes settlement
                        prediction[y, x, 3] = 0.10  # becomes ruin
                        prediction[y, x, 4] = 0.10  # becomes forest
                        prediction[y, x, 2] = 0.05  # becomes port
                        prediction[y, x, 5] = 0.05
                    elif init_val == 1:  # Settlement
                        prediction[y, x, 1] = 0.40  # stays settlement
                        prediction[y, x, 3] = 0.20  # becomes ruin
                        prediction[y, x, 0] = 0.15  # becomes empty/plains
                        prediction[y, x, 2] = 0.10  # becomes port
                        prediction[y, x, 4] = 0.10  # forest reclaims
                        prediction[y, x, 5] = 0.05
                    elif init_val == 2:  # Port
                        prediction[y, x, 2] = 0.40
                        prediction[y, x, 1] = 0.20
                        prediction[y, x, 3] = 0.15
                        prediction[y, x, 0] = 0.15
                        prediction[y, x, 4] = 0.05
                        prediction[y, x, 5] = 0.05
                    else:
                        # Default uniform-ish
                        prediction[y, x, 0] = 0.50
                        prediction[y, x, 1] = 0.15
                        prediction[y, x, 2] = 0.05
                        prediction[y, x, 3] = 0.10
                        prediction[y, x, 4] = 0.15
                        prediction[y, x, 5] = 0.05

        # Apply floor and normalize
        prediction = apply_floor_and_normalize(prediction)

        # Verify
        sums = prediction.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=0.01), f"Sums not 1.0: min={sums.min()}, max={sums.max()}"
        assert (prediction >= 0).all(), "Negative probabilities!"

        # Submit
        pred_list = prediction.tolist()
        result = submit(round_id, seed_idx, pred_list)
        print(f"  Submit result: {result}")

    # Check scores
    print("\n--- Checking results ---")
    r = requests.get(f"{BASE}/my-rounds", headers=HEADERS)
    print(json.dumps(r.json(), indent=2))


if __name__ == "__main__":
    main()
