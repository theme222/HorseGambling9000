#!/bin/bash

uv run main.py \
    --race races/race3.json \
    --size 287 \
    --increment 250 \
    --rank 90 \
    --max-ev-pct 0.12 \
    --safe-ev-pct 0.14 \
    --high-var-pct 0.10 \
    --pos-ev-pct 0.50 \
    --random-all-pct 0.12 \
    --do-nothing-pct 0.02 > results/race3.out
    
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