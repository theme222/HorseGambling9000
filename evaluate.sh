#!/usr/bin/env bash

RACE_NUM="${1:-1}"
RACE_FILE="races/race${RACE_NUM}.json"
OUT_FILE="results/race${RACE_NUM}.out"

if [ ! -f "$RACE_FILE" ]; then
    echo "Error: Race file $RACE_FILE does not exist."
    exit 1
fi

mkdir -p results
for h in $(seq 0 0.1 1); do
    echo -e "\n--- HERD: $h ---"
    uv run main.py --race "$RACE_FILE" --herd "$h"
done > "$OUT_FILE"

echo "Evaluation complete for Race ${RACE_NUM}. Saved to ${OUT_FILE}"
