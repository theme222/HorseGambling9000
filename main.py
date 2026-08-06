import argparse
import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

# Default parameters
DEFAULT_RACE_DATA_FILE = Path("races/race1.json")
DEFAULT_CLASS_SIZE = 290
DEFAULT_TARGET_RANK = 100
DEFAULT_SIMULATIONS = 10000
DEFAULT_TOTAL_TOKENS = 3000
DEFAULT_TOKEN_INCREMENT = 300
DEFAULT_HERD_MAX_EV_PCT = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation tool to optimize token allocation strategies in horse gambling."
    )
    parser.add_argument(
        "--race",
        "--race-data-file",
        dest="race_data_file",
        type=Path,
        default=DEFAULT_RACE_DATA_FILE,
        help=f"Path to JSON file containing race odds and probabilities (default: {DEFAULT_RACE_DATA_FILE})",
    )
    parser.add_argument(
        "--size",
        "--class-size",
        dest="class_size",
        type=int,
        default=DEFAULT_CLASS_SIZE,
        help=f"Total participating players in tournament (default: {DEFAULT_CLASS_SIZE})",
    )
    parser.add_argument(
        "--rank",
        "--target-rank",
        dest="target_rank",
        type=int,
        default=DEFAULT_TARGET_RANK,
        help=f"Rank threshold to achieve (default: {DEFAULT_TARGET_RANK})",
    )
    parser.add_argument(
        "--simulations",
        "--sims",
        dest="simulations",
        type=int,
        default=DEFAULT_SIMULATIONS,
        help=f"Number of Monte Carlo race iterations (default: {DEFAULT_SIMULATIONS})",
    )
    parser.add_argument(
        "--tokens",
        "--total-tokens",
        dest="total_tokens",
        type=int,
        default=DEFAULT_TOTAL_TOKENS,
        help=f"Total betting tokens available per player (default: {DEFAULT_TOTAL_TOKENS})",
    )
    parser.add_argument(
        "--increment",
        "--token-increment",
        dest="token_increment",
        type=int,
        default=DEFAULT_TOKEN_INCREMENT,
        help=f"Resolution step for token distribution grid search (default: {DEFAULT_TOKEN_INCREMENT})",
    )
    parser.add_argument(
        "--herd",
        "--herd-max-ev-pct",
        dest="herd_max_ev_pct",
        type=float,
        default=DEFAULT_HERD_MAX_EV_PCT,
        help=f"Estimated fraction of competitors playing Max EV (default: {DEFAULT_HERD_MAX_EV_PCT})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    race_data_file = args.race_data_file
    class_size = args.class_size
    target_rank = args.target_rank
    simulations = args.simulations
    total_tokens = args.total_tokens
    token_increment = args.token_increment
    herd_max_ev_pct = args.herd_max_ev_pct

    # Load race data (odds and TRUE probabilities from JSON)
    with open(race_data_file, "r") as f:
        race_data = json.load(f)

    odds = np.array(race_data["odds"])
    true_probs = np.array(race_data["true_probs"])

    # ==========================================
    # INTERNAL CALCULATIONS & SIMULATION (RANDOMIZED & VECTORIZED)
    # ==========================================
    num_horses = len(odds)
    evs = odds * true_probs

    # Define how many players fall into each bucket
    num_max_ev_players = int(class_size * herd_max_ev_pct)
    num_spread_players = class_size - num_max_ev_players

    # 1. The "Herd" (Max EV Bettors) -> all do the exact same thing
    max_ev_horse_idx = np.argmax(evs)
    herd_max_ev_bets = np.zeros((num_max_ev_players, num_horses))
    herd_max_ev_bets[:, max_ev_horse_idx] = total_tokens

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
    herd_spread_bets[:, positive_ev_indices] = spread_fractions * total_tokens

    # Combine both groups into a single matrix representing the whole class
    class_bets = np.vstack((herd_max_ev_bets, herd_spread_bets))

    # Run the Monte Carlo Simulation
    race_outcomes = np.random.multinomial(10, true_probs, size=simulations)
    race_outcomes_fraction = race_outcomes / 10

    # 3. Calculate Scores for the ENTIRE class at once using matrix multiplication
    class_scores = np.dot(race_outcomes_fraction, (class_bets * odds).T)

    # 4. Find the Cutoff Score for rank
    sorted_scores = np.sort(class_scores, axis=1)[:, ::-1]

    # Grab the score at the target position (index target_rank - 1) for every simulation
    cutoff_scores = sorted_scores[:, target_rank - 1]

    # ==========================================
    # STRATEGY OPTIMIZATION (GRID SEARCH)
    # ==========================================
    # To keep computation fast, we test distributions in blocks of tokens
    blocks = total_tokens // token_increment

    # 1. Pre-build all possible token combinations into a single matrix
    combos = list(combinations_with_replacement(range(num_horses), blocks))
    bets_matrix = np.zeros((len(combos), num_horses))

    for i, combo in enumerate(combos):
        for horse_idx in combo:
            bets_matrix[i, horse_idx] += token_increment

    # 2. Vectorize the payouts (bets_matrix * odds)
    bets_with_odds = bets_matrix * odds

    # 3. Vectorize the win fractions
    race_outcomes_fraction = race_outcomes / 10

    # 4. Matrix Multiplication to get all scores instantly
    all_scores = np.dot(bets_with_odds, race_outcomes_fraction.T)

    # 5. Broadcast comparison against the cutoff scores
    successes = all_scores > cutoff_scores

    # 6. Sum the successes across simulations for each strategy
    win_rates = np.sum(successes, axis=1) / simulations

    # 7. Find the index of the strategy with the highest win rate
    best_idx = np.argmax(win_rates)
    best_win_rate = win_rates[best_idx]
    best_strategy = bets_matrix[best_idx]

    assert best_strategy is not None, "No strategy found!"

    # ==========================================
    # RESULTS
    # ==========================================
    print("\n--- OPTIMAL TOURNAMENT STRATEGY ---")
    print(f"Targeting Top {target_rank} against {class_size} players.")
    print(f"Herd Behavior: {herd_max_ev_pct * 100:.1f}% playing Max EV ({num_max_ev_players} people).\n")

    print("Your Optimal Token Split:")
    for i in range(num_horses):
        if best_strategy[i] > 0:
            print(f"Horse {i+1} (Odds {odds[i]}, True Prob {true_probs[i]*100:.1f}%): {int(best_strategy[i])} tokens")

    print(f"\nEstimated probability of achieving Top {target_rank}: {best_win_rate * 100:.2f}%")


if __name__ == "__main__":
    main()