import argparse
import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

# Default parameters
DEFAULT_RACE_DATA_FILE = Path("races/race1.json")
DEFAULT_CLASS_SIZE = 290
DEFAULT_TARGET_RANK = 100
DEFAULT_SIMULATIONS = 90000
DEFAULT_TOTAL_TOKENS = 3000
DEFAULT_TOKEN_INCREMENT = 300

# Default group distribution percentages (must sum to 1.0)
DEFAULT_MAX_EV_PCT = 0.30
DEFAULT_SAFE_EV_PCT = 0.20
DEFAULT_HIGH_VAR_PCT = 0.15
DEFAULT_POS_EV_PCT = 0.15
DEFAULT_RANDOM_ALL_PCT = 0.10
DEFAULT_DO_NOTHING_PCT = 0.10


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
        "--max-ev-pct",
        "--pct-max-ev",
        dest="max_ev_pct",
        type=float,
        default=None,
        help=f"Fraction of competitors placing all tokens on Max EV horse (default: {DEFAULT_MAX_EV_PCT})",
    )
    parser.add_argument(
        "--safe-ev-pct",
        "--safest-ev-pct",
        "--pct-safe-ev",
        "--pct-safest-ev",
        dest="safe_ev_pct",
        type=float,
        default=None,
        help=f"Fraction of competitors placing all tokens on safest +EV horse (default: {DEFAULT_SAFE_EV_PCT})",
    )
    parser.add_argument(
        "--high-var-pct",
        "--highest-variance-pct",
        "--high-variance-pct",
        "--pct-high-var",
        "--pct-highest-variance",
        dest="high_var_pct",
        type=float,
        default=None,
        help=f"Fraction of competitors spreading across top 2 highest odds +EV horses (default: {DEFAULT_HIGH_VAR_PCT})",
    )
    parser.add_argument(
        "--pos-ev-pct",
        "--positive-ev-pct",
        "--pct-pos-ev",
        "--pct-positive-ev",
        dest="pos_ev_pct",
        type=float,
        default=None,
        help=f"Fraction of competitors randomly spreading across positive EV horses (default: {DEFAULT_POS_EV_PCT})",
    )
    parser.add_argument(
        "--random-all-pct",
        "--all-choices-pct",
        "--pct-random-all",
        "--pct-all-choices",
        dest="random_all_pct",
        type=float,
        default=None,
        help=f"Fraction of competitors randomly spreading across all choices (default: {DEFAULT_RANDOM_ALL_PCT})",
    )
    parser.add_argument(
        "--do-nothing-pct",
        "--keep-tokens-pct",
        "--pct-do-nothing",
        "--pct-keep-tokens",
        dest="do_nothing_pct",
        type=float,
        default=None,
        help=f"Fraction of competitors keeping all tokens and placing no bets (default: {DEFAULT_DO_NOTHING_PCT})",
    )
    return parser.parse_args()


def resolve_group_percentages(
    max_ev: float | None,
    safe_ev: float | None,
    high_var: float | None,
    pos_ev: float | None,
    random_all: float | None,
    do_nothing: float | None,
) -> tuple[float, float, float, float, float, float]:
    """Resolve, normalize, and assert competitor group proportions across all 6 interest groups."""
    base_defaults = [
        DEFAULT_MAX_EV_PCT,
        DEFAULT_SAFE_EV_PCT,
        DEFAULT_HIGH_VAR_PCT,
        DEFAULT_POS_EV_PCT,
        DEFAULT_RANDOM_ALL_PCT,
        DEFAULT_DO_NOTHING_PCT,
    ]
    inputs = [max_ev, safe_ev, high_var, pos_ev, random_all, do_nothing]

    # If no flags were provided, return default distribution
    if all(x is None for x in inputs):
        resolved = tuple(base_defaults)
        assert np.isclose(sum(resolved), 1.0), (
            f"Sum of all groups of interest must equal 1.0, got {sum(resolved)}"
        )
        return resolved

    # If all flags were explicitly provided, assert they sum to 1.0
    if all(x is not None for x in inputs):
        total = sum(inputs)
        assert np.isclose(total, 1.0), (
            f"Sum of all groups of interest must equal 1.0, got {total}"
        )
        return tuple(inputs)

    # If partially specified:
    specified_sum = sum(x for x in inputs if x is not None)
    if specified_sum > 1.0 and not np.isclose(specified_sum, 1.0):
        raise AssertionError(
            f"Sum of specified group percentages ({specified_sum}) exceeds 1.0"
        )
    elif np.isclose(specified_sum, 1.0):
        # Specified values take 100%, un-specified get 0
        resolved = tuple((x if x is not None else 0.0) for x in inputs)
        assert np.isclose(sum(resolved), 1.0), (
            f"Sum of all groups of interest must equal 1.0, got {sum(resolved)}"
        )
        return resolved

    remaining = 1.0 - specified_sum
    unspecified_default_sum = sum(
        base_defaults[i] for i in range(6) if inputs[i] is None
    )

    results = []
    for i in range(6):
        if inputs[i] is not None:
            results.append(inputs[i])
        else:
            if unspecified_default_sum > 0:
                results.append(
                    remaining * (base_defaults[i] / unspecified_default_sum)
                )
            else:
                results.append(0.0)

    resolved = tuple(results)
    assert np.isclose(sum(resolved), 1.0), (
        f"Sum of all groups of interest must equal 1.0, got {sum(resolved)}"
    )
    return resolved


def calculate_group_sizes(class_size: int, pcts: tuple[float, ...]) -> list[int]:
    """Apportion class_size into integer counts using the largest remainder method."""
    exact = [p * class_size for p in pcts]
    counts = [int(x) for x in exact]
    remainder = class_size - sum(counts)

    fractional_parts = [exact[i] - counts[i] for i in range(len(pcts))]
    sorted_indices = sorted(
        range(len(pcts)), key=lambda i: fractional_parts[i], reverse=True
    )

    for i in range(remainder):
        counts[sorted_indices[i % len(pcts)]] += 1

    return counts


def main():
    args = parse_args()

    race_data_file = args.race_data_file
    class_size = args.class_size
    target_rank = args.target_rank
    simulations = args.simulations
    total_tokens = args.total_tokens
    token_increment = args.token_increment

    pct_max_ev, pct_safe_ev, pct_high_var, pct_pos_ev, pct_random_all, pct_do_nothing = (
        resolve_group_percentages(
            args.max_ev_pct,
            args.safe_ev_pct,
            args.high_var_pct,
            args.pos_ev_pct,
            args.random_all_pct,
            args.do_nothing_pct,
        )
    )

    # Assert sum of all groups of interest is equal to 1.0
    group_sum = (
        pct_max_ev
        + pct_safe_ev
        + pct_high_var
        + pct_pos_ev
        + pct_random_all
        + pct_do_nothing
    )
    assert np.isclose(group_sum, 1.0), (
        f"Sum of all groups of interest must equal 1.0, got {group_sum}"
    )

    # Load race data (odds and TRUE probabilities from JSON)
    with open(race_data_file, "r") as f:
        race_data = json.load(f)

    odds = np.array(race_data["odds"])
    true_probs = np.array(race_data["true_probs"])

    # Assert sum of all horse true probabilities is equal to 1.0
    prob_sum = np.sum(true_probs)
    assert np.isclose(prob_sum, 1.0), (
        f"Sum of horse true probabilities must equal 1.0, got {prob_sum}"
    )

    # ==========================================
    # INTERNAL CALCULATIONS & SIMULATION (RANDOMIZED & VECTORIZED)
    # ==========================================
    num_horses = len(odds)
    evs = odds * true_probs

    # Apportion player counts across all 6 groups of interest
    (
        num_max_ev,
        num_safe_ev,
        num_high_var,
        num_pos_ev,
        num_random_all,
        num_do_nothing,
    ) = calculate_group_sizes(
        class_size,
        (
            pct_max_ev,
            pct_safe_ev,
            pct_high_var,
            pct_pos_ev,
            pct_random_all,
            pct_do_nothing,
        ),
    )

    # Positive EV horses definition (EV > 1.0)
    positive_ev_indices = np.where(evs > 1.0)[0]

    # 1. MAX EV Bettors -> put all tokens on the horse with highest expected value
    max_ev_horse_idx = int(np.argmax(evs))
    max_ev_bets = np.zeros((num_max_ev, num_horses))
    if num_max_ev > 0:
        max_ev_bets[:, max_ev_horse_idx] = total_tokens

    # 2. Safest EV Bettors -> put all tokens on the +EV horse with the highest true win probability
    if len(positive_ev_indices) > 0:
        safe_ev_horse_idx = int(
            positive_ev_indices[np.argmax(true_probs[positive_ev_indices])]
        )
    else:
        top_ev_indices = np.argsort(evs)[-min(3, num_horses):]
        safe_ev_horse_idx = int(
            top_ev_indices[np.argmax(true_probs[top_ev_indices])]
        )

    safe_ev_bets = np.zeros((num_safe_ev, num_horses))
    if num_safe_ev > 0:
        safe_ev_bets[:, safe_ev_horse_idx] = total_tokens

    # 3. Highest Variance Bettors -> random Dirichlet distribution of top 2 highest odds +EV horses
    if len(positive_ev_indices) >= 2:
        sorted_pos_by_odds = positive_ev_indices[
            np.argsort(odds[positive_ev_indices])[::-1]
        ]
        high_var_indices = sorted_pos_by_odds[:2]
    elif len(positive_ev_indices) == 1:
        top_ev_indices = np.argsort(evs)[-min(3, num_horses):]
        sorted_top_by_odds = top_ev_indices[
            np.argsort(odds[top_ev_indices])[::-1]
        ]
        high_var_indices = sorted_top_by_odds[:min(2, len(sorted_top_by_odds))]
    else:
        top_ev_indices = np.argsort(evs)[-min(3, num_horses):]
        sorted_top_by_odds = top_ev_indices[
            np.argsort(odds[top_ev_indices])[::-1]
        ]
        high_var_indices = sorted_top_by_odds[:min(2, len(sorted_top_by_odds))]

    high_var_bets = np.zeros((num_high_var, num_horses))
    if num_high_var > 0:
        high_var_fractions = np.random.dirichlet(
            np.ones(len(high_var_indices)), size=num_high_var
        )
        high_var_bets[:, high_var_indices] = high_var_fractions * total_tokens

    # 4. Positive EV Spreaders -> random Dirichlet distribution across +EV horses (EV > 1.0)
    pos_ev_spread_indices = positive_ev_indices
    if len(pos_ev_spread_indices) < 2:
        pos_ev_spread_indices = np.argsort(evs)[-min(3, num_horses):]

    pos_ev_bets = np.zeros((num_pos_ev, num_horses))
    if num_pos_ev > 0:
        pos_ev_fractions = np.random.dirichlet(
            np.ones(len(pos_ev_spread_indices)), size=num_pos_ev
        )
        pos_ev_bets[:, pos_ev_spread_indices] = pos_ev_fractions * total_tokens

    # 5. All Choices Spreaders -> random Dirichlet distribution across all possible choices
    random_all_bets = np.zeros((num_random_all, num_horses))
    if num_random_all > 0:
        random_all_fractions = np.random.dirichlet(
            np.ones(num_horses), size=num_random_all
        )
        random_all_bets = random_all_fractions * total_tokens

    # 6. Keeping Tokens (Do Nothing) -> bet 0 tokens, keeping all starting tokens
    do_nothing_bets = np.zeros((num_do_nothing, num_horses))

    # Combine all 6 groups into a single matrix representing the whole class
    class_bets = np.vstack((
        max_ev_bets,
        safe_ev_bets,
        high_var_bets,
        pos_ev_bets,
        random_all_bets,
        do_nothing_bets,
    ))

    # Track unspent tokens per player for final score calculation
    player_spent_tokens = np.sum(class_bets, axis=1)
    player_unspent_tokens = total_tokens - player_spent_tokens

    # Run the Monte Carlo Simulation
    race_outcomes = np.random.multinomial(10, true_probs, size=simulations)
    race_outcomes_fraction = (race_outcomes / 10).astype(np.float64)

    # Calculate Scores for the ENTIRE class at once: winnings + unspent tokens
    class_payouts = np.dot(race_outcomes_fraction, (class_bets * odds).T)
    class_scores = class_payouts + player_unspent_tokens

    # Find the Cutoff Score for rank
    sorted_scores = np.sort(class_scores, axis=1)[:, ::-1]

    # Grab the score at the target position (index target_rank - 1) for every simulation
    cutoff_scores = sorted_scores[:, target_rank - 1]

    # ==========================================
    # STRATEGY OPTIMIZATION (GRID SEARCH)
    # ==========================================
    # Test distributions in blocks of tokens (including option to keep cash)
    blocks = total_tokens // token_increment

    # 1. Pre-build all possible token combinations into a single matrix
    # Index num_horses corresponds to keeping tokens in cash (doing nothing with those tokens)
    combos = list(combinations_with_replacement(range(num_horses + 1), blocks))
    bets_matrix = np.zeros((len(combos), num_horses), dtype=np.float64)

    for i, combo in enumerate(combos):
        for choice in combo:
            if choice < num_horses:
                bets_matrix[i, choice] += token_increment

    strategy_unspent = (total_tokens - np.sum(bets_matrix, axis=1)).astype(np.float64)
    bets_with_odds = (bets_matrix * odds).astype(np.float64)

    # 2. Vectorized Batched Evaluation against cutoff scores
    win_rates = np.zeros(len(combos), dtype=np.float64)
    batch_size = 5000
    for start in range(0, len(combos), batch_size):
        end = min(start + batch_size, len(combos))
        batch_scores = (
            np.dot(bets_with_odds[start:end], race_outcomes_fraction.T)
            + strategy_unspent[start:end, None]
        )
        win_rates[start:end] = np.mean(batch_scores > cutoff_scores, axis=1)

    # 3. Find the index of the strategy with the highest win rate
    best_idx = int(np.argmax(win_rates))
    best_win_rate = win_rates[best_idx]
    best_strategy = bets_matrix[best_idx]
    best_unspent = int(strategy_unspent[best_idx])

    assert best_strategy is not None, "No strategy found!"

    # ==========================================
    # RESULTS
    # ==========================================
    print("\n--- OPTIMAL TOURNAMENT STRATEGY ---")
    print(f"Targeting Top {target_rank} against {class_size} players.")
    print("Competitor Breakdown:")
    print(f"  1. Max EV Bettors:              {pct_max_ev * 100:5.1f}% ({num_max_ev} people)")
    print(f"  2. Safest EV Bettors:           {pct_safe_ev * 100:5.1f}% ({num_safe_ev} people)")
    print(f"  3. Highest Variance Bettors:    {pct_high_var * 100:5.1f}% ({num_high_var} people)")
    print(f"  4. Positive EV Spreaders:       {pct_pos_ev * 100:5.1f}% ({num_pos_ev} people)")
    print(f"  5. All Choices Spreaders:       {pct_random_all * 100:5.1f}% ({num_random_all} people)")
    print(f"  6. Keeping Tokens (No Bet):     {pct_do_nothing * 100:5.1f}% ({num_do_nothing} people)\n")

    print("Your Optimal Token Split:")
    for i in range(num_horses):
        if best_strategy[i] > 0:
            print(
                f"Horse {i+1} (Odds {odds[i]}, True Prob {true_probs[i]*100:.1f}%): {int(best_strategy[i])} tokens"
            )
    if best_unspent > 0:
        print(f"Kept Tokens (Cash): {best_unspent} tokens")

    print(f"\nEstimated probability of achieving Top {target_rank}: {best_win_rate * 100:.2f}%")


if __name__ == "__main__":
    main()