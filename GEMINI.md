# HorseGambling9000 - Context & Overview

## Project Summary
**HorseGambling9000** is a Monte Carlo simulation tool designed to optimize token allocation strategies in a competitive horse gambling environment (e.g., a classroom tournament setting).

Rather than simply maximizing expected value (EV)—which leads to heavy overlap with the mass of players betting on the single highest EV horse—this tool seeks strategies with enough variance to achieve a target rank threshold (e.g., placing in the Top 100 out of 290 competitors).

---

## Project Structure & Key Components

- **[`main.py`](file:///home/sirat/Code/HorseGambling9000/main.py)**: The primary entry point and simulation engine.
  - **Data Loading**: Loads race odds and true win probabilities from JSON files in the `races/` directory using a `pathlib.Path` object specified by `RACE_DATA_FILE`.
  - **Competitor Group Modeling**: Models opponent behavior across 6 groups of interest:
    1. **MAX EV Players**: Put all tokens on the single highest expected value horse.
    2. **Safest EV Players**: Put all tokens on the +EV horse with the highest true win probability.
    3. **Highest Variance Bettors**: Randomly distribute tokens across the top 2 highest odds +EV horses using a Dirichlet distribution.
    4. **Positive EV Spreaders**: Randomly distribute tokens across all +EV horses ($EV > 1.0$) using a Dirichlet distribution.
    5. **All Choices Spreaders**: Randomly distribute tokens across all possible choices/horses using a Dirichlet distribution.
    6. **Do Nothing / Keep Tokens**: Place 0 bets, retaining all starting tokens as their final score.
  - **Vectorized Monte Carlo Engine**: Uses NumPy matrix operations to simulate race outcomes and quickly compute cutoff scores across all competing players.
  - **Grid Search Strategy Optimizer**: Tests token split combinations (in steps set by `TOKEN_INCREMENT`, including retaining cash) to find the strategy maximizing the probability of beating the target rank cutoff score.

- **[`thing.sh`](file:///home/sirat/Code/HorseGambling9000/thing.sh)**: Example evaluation shell script executing `main.py` with custom group proportions.

- **[`races/`](file:///home/sirat/Code/HorseGambling9000/races/)**: Subdirectory containing race specification files in JSON format (e.g., `race0.json`, `race1.json`, `race2.json`).

- **[`results/`](file:///home/sirat/Code/HorseGambling9000/results/)**: Subdirectory containing generated evaluation output logs.

- **[`README.md`](file:///home/sirat/Code/HorseGambling9000/README.md)**: Original notes describing the vibe-coded nature and core intention of the strategy simulator.

---

## Configuration & Usage

The simulator accepts CLI options (falling back to standard defaults if omitted):

| Parameter / CLI Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--race`, `--race-data-file` | `Path` | `races/race1.json` | Path to JSON file containing race odds and probabilities |
| `--size`, `--class-size` | `int` | `290` | Total number of participating players in the tournament |
| `--rank`, `--target-rank` | `int` | `100` | Rank threshold to achieve (e.g. 100 for Top 100) |
| `--simulations`, `--sims` | `int` | `90000` | Number of Monte Carlo race iterations |
| `--tokens`, `--total-tokens` | `int` | `3000` | Total betting tokens available per player |
| `--increment`, `--token-increment` | `int` | `300` | Resolution step for token distribution grid search |
| `--max-ev-pct` | `float` | `0.30` | Fraction of competitors betting all tokens on the Max EV horse |
| `--safe-ev-pct`, `--safest-ev-pct` | `float` | `0.20` | Fraction of competitors betting all tokens on the safest +EV horse |
| `--high-var-pct`, `--highest-variance-pct` | `float` | `0.15` | Fraction of competitors spreading across top 2 highest odds +EV horses |
| `--pos-ev-pct` | `float` | `0.15` | Fraction of competitors randomly spreading across positive EV horses |
| `--random-all-pct`, `--all-choices-pct` | `float` | `0.10` | Fraction of competitors randomly spreading across all choices |
| `--do-nothing-pct`, `--keep-tokens-pct` | `float` | `0.10` | Fraction of competitors keeping all tokens and placing no bets |

### Running the Simulator
This project uses `uv`.

```bash
# Run with default settings
uv run main.py

# Run with custom parameters
uv run main.py --size 250 --rank 50 --max-ev-pct 0.3 --safe-ev-pct 0.2 --high-var-pct 0.15

# Run via script
./thing.sh
```
