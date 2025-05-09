#!/bin/bash

python3 inverse_folding/inverse_folding/inference_surface.py \
    --model=ckpt/progen2-small-surface-p80k-2r-finetuned/e2 \
    --device=cuda \
    --batch_size=8 \
    --max_length=1024 \
    --t=0.01 \
    --k=10 \
    --test_data_dir=data/processed_surface_data/test
