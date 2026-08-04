from itertools import combinations_with_replacement

import numpy as np

# ==========================================
# 1. EDIT THESE VALUES BEFORE EACH ROUND
# ==========================================
CLASS_SIZE = 290
TARGET_RANK = 100
SIMULATIONS = 25000
TOTAL_TOKENS = 3000
TOKEN_INCREMENT = 300

# Herd Assumptions: How do you think the class will bet?
# Ensure these add up to 1.0 (100% of the class)
HERD_MAX_EV_PCT = 0.7  # 70% dump everything on the single highest EV horse
HERD_SPREAD_PCT = 1 - HERD_MAX_EV_PCT  # 30% spread bets across top 2 or 3 horses

# The 8 Horses: Update odds and TRUE probabilities (from hints) here.
# NOTE: true_probs must sum to exactly 1.0
odds = np.array([
    4.0,
    4.0,
    6.67,
    6.67,
    16.67,
    16.67,
    25.0,
    25.0
])
true_probs = np.array([
    0.3,
    0.2,
    0.132,
    0.168,
    0.0552,
    0.0648,
    0.0416,
    0.0384
])

# ==========================================
# 2. INTERNAL CALCULATIONS & SIMULATION (RANDOMIZED & VECTORIZED)
# ==========================================
num_horses = len(odds)
evs = odds * true_probs

# Define how many players fall into each bucket
num_max_ev_players = int(CLASS_SIZE * HERD_MAX_EV_PCT)
num_spread_players = CLASS_SIZE - num_max_ev_players

# 1. The "Herd" (Max EV Bettors) -> all do the exact same thing
max_ev_horse_idx = np.argmax(evs)
herd_max_ev_bets = np.zeros((num_max_ev_players, num_horses))
herd_max_ev_bets[:, max_ev_horse_idx] = TOTAL_TOKENS

# 2. The "Spreaders" -> each player gets a unique, random split across Positive EV horses
# We define "Positive EV" as an expected return greater than 1.0
positive_ev_indices = np.where(evs > 1.0)[0]

# Fallback: If the board is terrible and fewer than 2 horses have +EV, use Top 3 EV horses
if len(positive_ev_indices) < 2:
    positive_ev_indices = np.argsort(evs)[-3:]

# Use a Dirichlet distribution to generate random fractions that perfectly sum to 1.0
# Each row represents a unique player's random split strategy
spread_fractions = np.random.dirichlet(np.ones(len(positive_ev_indices)), size=num_spread_players)

herd_spread_bets = np.zeros((num_spread_players, num_horses))
herd_spread_bets[:, positive_ev_indices] = spread_fractions * TOTAL_TOKENS

# Combine both groups into a single matrix representing the whole class [290 players, 8 horses]
class_bets = np.vstack((herd_max_ev_bets, herd_spread_bets))

# Run the Monte Carlo Simulation (10,000 realities)
race_outcomes = np.random.multinomial(10, true_probs, size=SIMULATIONS)
race_outcomes_fraction = race_outcomes / 10  # Shape: [10000, 8]

# 3. Calculate Scores for the ENTIRE class at once using matrix multiplication
# (class_bets * odds) gives payout multipliers -> Shape: [290, 8]
# Dot product with race_outcomes_fraction gets final scores -> Shape: [10000, 290] 
class_scores = np.dot(race_outcomes_fraction, (class_bets * odds).T)

# 4. Find the Cutoff Score for rank 100
# Sort the scores for each simulation descending (axis=1, then [:, ::-1] reverses it)
sorted_scores = np.sort(class_scores, axis=1)[:, ::-1]

# Grab the score at the 100th position (index TARGET_RANK - 1) for every simulation
cutoff_scores = sorted_scores[:, TARGET_RANK - 1]

# ==========================================
# 3. STRATEGY OPTIMIZATION (GRID SEARCH)
# ==========================================
# To keep computation fast, we test distributions in blocks of tokens
blocks = TOTAL_TOKENS // TOKEN_INCREMENT

print("Generating combinations and multiplying matrices (fast!)...")

# 1. Pre-build all 1,716 possible token combinations into a single matrix
combos = list(combinations_with_replacement(range(num_horses), blocks))
bets_matrix = np.zeros((len(combos), num_horses))

for i, combo in enumerate(combos):
    for horse_idx in combo:
        bets_matrix[i, horse_idx] += TOKEN_INCREMENT

# 2. Vectorize the payouts (bets_matrix * odds) -> Shape: [1716, 8]
bets_with_odds = bets_matrix * odds 

# 3. Vectorize the win fractions (race_outcomes / 10) -> Shape: [10000, 8]
race_outcomes_fraction = race_outcomes / 10

# 4. Matrix Multiplication to get all scores instantly -> Shape: [1716, 10000]
# This calculates how EVERY strategy did in EVERY simulation in one calculation
all_scores = np.dot(bets_with_odds, race_outcomes_fraction.T)

# 5. Broadcast comparison against the cutoff scores -> Shape: [1716, 10000] (Booleans)
successes = all_scores > cutoff_scores

# 6. Sum the successes across the 10000 simulations for each of the 1716 strategies
win_rates = np.sum(successes, axis=1) / SIMULATIONS

# 7. Find the index of the strategy with the highest win rate
best_idx = np.argmax(win_rates)
best_win_rate = win_rates[best_idx]
best_strategy = bets_matrix[best_idx]

assert best_strategy is not None, "No strategy found!"
# ==========================================
# 4. RESULTS
# ==========================================
print("\n--- OPTIMAL TOURNAMENT STRATEGY ---")
print(f"Targeting Top {TARGET_RANK} against {CLASS_SIZE} players.")
print(f"Herd Behavior: {HERD_MAX_EV_PCT*100}% playing Max EV. {HERD_MAX_EV_PCT * CLASS_SIZE} people \n")

print("Your Optimal Token Split:")
for i in range(num_horses):
    if best_strategy[i] > 0:
        print(f"Horse {i+1} (Odds {odds[i]}, True Prob {true_probs[i]*100}%): {int(best_strategy[i])} tokens")

print(f"\nEstimated probability of achieving Top {TARGET_RANK}: {best_win_rate * 100:.2f}%")