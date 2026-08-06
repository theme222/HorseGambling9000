# HorseGambling9000 - Context & Overview

## Project Summary
**HorseGambling9000** is a Monte Carlo simulation tool designed to optimize token allocation strategies in a competitive horse gambling environment (e.g., a classroom tournament setting).

Rather than simply maximizing expected value (EV)—which leads to heavy overlap with the "herd" of players betting on the single highest EV horse—this tool seeks strategies with enough variance to achieve a target rank threshold (e.g., placing in the Top 100 out of 290 competitors).

---

## Project Structure & Key Components

- **[`main.py`](file:///home/sirat/Code/HorseGambling9000/main.py)**: The primary entry point and simulation engine.
  - **Data Loading**: Loads race odds and true win probabilities from JSON files in the `races/` directory using a `pathlib.Path` object specified by `RACE_DATA_FILE`.
  - **Herd Modeling**: Models opponent behavior split between "Max EV Bettors" (who put all tokens on the single highest EV horse) and "Spreaders" (who randomly distribute bets across +EV horses using a Dirichlet distribution).
  - **Vectorized Monte Carlo Engine**: Uses NumPy matrix operations to simulate 10,000 race outcomes and quickly compute cutoff scores across all competing players.
  - **Grid Search Strategy Optimizer**: Tests token split combinations (in steps set by `TOKEN_INCREMENT`) to find the strategy maximizing the probability of beating the target rank cutoff score.

- **[`evaluate.sh`](file:///home/sirat/Code/HorseGambling9000/evaluate.sh)**: Batch evaluation shell script that sweeps herd percentages from 0.0 to 1.0 (in 0.1 increments) for a given race number and saves standard output to `results/race<NUM>.out`.

- **[`races/`](file:///home/sirat/Code/HorseGambling9000/races/)**: Subdirectory containing race specification files in JSON format (e.g., `race0.json`, `race1.json`).

- **[`results/`](file:///home/sirat/Code/HorseGambling9000/results/)**: Subdirectory containing generated evaluation output logs (e.g., `race0.out`, `race1.out`).

- **[`README.md`](file:///home/sirat/Code/HorseGambling9000/README.md)**: Original notes describing the vibe-coded nature and core intention of the strategy simulator.

---

## Configuration & Usage

The simulator accepts CLI options (falling back to standard defaults if omitted):

| Parameter / CLI Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--race`, `--race-data-file` | `Path` | `races/race1.json` | Path to JSON file containing race odds and probabilities |
| `--size`, `--class-size` | `int` | `290` | Total number of participating players in the tournament |
| `--rank`, `--target-rank` | `int` | `100` | Rank threshold to achieve (e.g. 100 for Top 100) |
| `--simulations`, `--sims` | `int` | `10000` | Number of Monte Carlo race iterations |
| `--tokens`, `--total-tokens` | `int` | `3000` | Total betting tokens available per player |
| `--increment`, `--token-increment` | `int` | `300` | Resolution step for token distribution grid search |
| `--herd`, `--herd-max-ev-pct` | `float` | `0.5` | Estimated fraction of competitors who dump all tokens on top EV horse |

### Running the Simulator
This project uses `uv`.

```bash
# Run with default settings
uv run main.py

# Run with custom parameters
uv run main.py --size 250 --rank 50 --herd 0.3

# Batch evaluate a race across herd percentages 0.0 to 1.0 (e.g. race 1 -> results/race1.out)
./evaluate.sh 1
```
