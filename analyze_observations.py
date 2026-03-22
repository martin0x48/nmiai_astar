"""Analyze observation data to understand simulation dynamics."""
import json
import numpy as np
from collections import Counter

TOKEN = "YOUR_JWT_TOKEN_HERE"

with open('/home/penguin/astar/round1_data.json') as f:
    round_data = json.load(f)

# Load simulation results
import glob, os
sim_files = sorted(glob.glob('/home/penguin/astar/sim_*.json'))
print(f"Simulation files: {sim_files}")

# We only have 1 saved file from the test query. Let's analyze what we can from the round data.
# Focus on comparing initial states across seeds

CELL_NAMES = {0: 'Empty', 1: 'Settlement', 2: 'Port', 3: 'Ruin', 4: 'Forest', 5: 'Mountain', 10: 'Ocean', 11: 'Plains'}

for seed_idx in range(5):
    state = round_data['initial_states'][seed_idx]
    grid = np.array(state['grid'])
    settlements = state['settlements']
    
    counts = Counter(grid.flatten())
    print(f"\nSeed {seed_idx}:")
    print(f"  Terrain: " + ", ".join(f"{CELL_NAMES.get(k,k)}={v}" for k,v in sorted(counts.items())))
    print(f"  Settlements: {len(settlements)}, Ports: {sum(1 for s in settlements if s['has_port'])}")
    
    # Where are settlements relative to terrain?
    sett_positions = [(s['x'], s['y']) for s in settlements]
    grid_at_sett = [grid[y,x] for x,y in sett_positions]
    print(f"  Grid values at settlement positions: {Counter(grid_at_sett)}")

# Compare grids across seeds to find what's shared
print("\n--- Cross-seed terrain comparison ---")
grids = [np.array(round_data['initial_states'][i]['grid']) for i in range(5)]

# Which cells are the same across all seeds?
all_same = np.ones((40,40), dtype=bool)
for i in range(1, 5):
    all_same &= (grids[0] == grids[i])

print(f"Cells identical across all seeds: {all_same.sum()} / 1600")

# What changes?
diff_mask = ~all_same
diff_positions = list(zip(*np.where(diff_mask)))
print(f"Cells that differ: {len(diff_positions)}")

# What are the values in cells that differ?
print("\nValue distribution in differing cells:")
for seed_idx in range(5):
    vals = grids[seed_idx][diff_mask]
    print(f"  Seed {seed_idx}: {Counter(sorted(vals))}")

# What are the values in cells that are the same?
same_vals = grids[0][all_same]
print(f"\nStatic cell values: {Counter(sorted(same_vals))}")

# Analyze the test simulation result
print("\n--- Simulation result (seed 0, viewport 0,0) ---")
with open('/home/penguin/astar/sim_s0_0_0.json') as f:
    sim = json.load(f)

sim_grid = np.array(sim['grid'])
init_grid = grids[0][0:15, 0:15]

print(f"Initial vs Final terrain changes in viewport (0,0)-(15,15):")
for y in range(15):
    for x in range(15):
        iv = init_grid[y, x]
        fv = sim_grid[y, x]
        if iv != fv:
            print(f"  ({x},{y}): {CELL_NAMES.get(iv,iv)} -> {CELL_NAMES.get(fv,fv)}")

print(f"\nFinal terrain counts: {Counter(sorted(sim_grid.flatten()))}")
print(f"Initial terrain counts: {Counter(sorted(init_grid.flatten()))}")

# Settlement analysis
print(f"\nSettlements in final viewport: {len(sim['settlements'])}")
owners = Counter(s['owner_id'] for s in sim['settlements'])
print(f"Owners: {owners}")
ports = [s for s in sim['settlements'] if s['has_port']]
print(f"Ports: {len(ports)}")
print(f"Population range: {min(s['population'] for s in sim['settlements']):.2f} - {max(s['population'] for s in sim['settlements']):.2f}")
print(f"Food range: {min(s['food'] for s in sim['settlements']):.2f} - {max(s['food'] for s in sim['settlements']):.2f}")
