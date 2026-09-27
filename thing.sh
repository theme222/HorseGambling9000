#!/bin/bash

uv run main.py \
    --race races/race4.json \
    --size 287 \
    --increment 250 \
    --rank 30 \
    --max-ev-pct 0.08 \
    --safe-ev-pct 0.15 \
    --high-var-pct 0.16 \
    --pos-ev-pct 0.48 \
    --random-all-pct 0.13 \
    --do-nothing-pct 0.00 > results/race4.out
    
# uv run main.py \
#     --race races/race2.json \
#     --size 287 \
#     --rank 80 \
#     --max-ev-pct 0.16 \
#     --safe-ev-pct 0.13 \
#     --high-var-pct 0.13 \
#     --pos-ev-pct 0.51 \
#     --random-all-pct 0.07 \
#     --do-nothing-pct 0.00 > results/race2.out