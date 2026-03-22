"""Quick status checker - run periodically to monitor rounds and scores."""
import requests
import json

TOKEN = "YOUR_JWT_TOKEN_HERE"
BASE = "https://api.ainm.no/astar-island"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# Rounds
r = requests.get(f"{BASE}/rounds", headers=HEADERS)
rounds = r.json()
for rd in rounds:
    print(f"Round {rd['round_number']}: status={rd['status']}, closes={rd.get('closes_at', 'N/A')}")

# Budget
r = requests.get(f"{BASE}/budget", headers=HEADERS)
b = r.json()
print(f"\nBudget: {b['queries_used']}/{b['queries_max']} used")

# My rounds / scores
r = requests.get(f"{BASE}/my-rounds", headers=HEADERS)
my_rounds = r.json()
print(f"\nScores:")
for rd in my_rounds:
    print(f"  Round {rd['round_number']}: score={rd.get('round_score')}, rank={rd.get('rank')}, submitted={rd['seeds_submitted']}/5")
    if rd.get('seed_scores'):
        for i, s in enumerate(rd['seed_scores']):
            print(f"    Seed {i}: {s}")

# Try analysis for completed rounds
for rd in rounds:
    if rd['status'] != 'active':
        print(f"\nAnalysis for round {rd['round_number']}:")
        for seed in range(5):
            r = requests.get(f"{BASE}/analysis/{rd['id']}/{seed}", headers=HEADERS)
            if r.status_code == 200:
                data = r.json()
                # Print just summary
                print(f"  Seed {seed}: {json.dumps(data)[:300]}")
            else:
                print(f"  Seed {seed}: {r.status_code}")
            break  # just check first seed
