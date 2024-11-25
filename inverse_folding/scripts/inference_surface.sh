#!/bin/bash

python3 inverse_folding/inverse_folding/inference_surface.py \
    --model=ckpt/progen2-small-surface-p80k-2r-finetuned/e2 \
    --device=cuda \
    --batch_size=8 \
    --max_length=1024 \
    --t=0.01 \
    --k=10 \
    --test_data_dir=data/processed_surface_data/test


# for i in {1..4}; do
#     echo "Running inference for model e${i}..."

#     python3 inverse_folding/inverse_folding/inference_surface.py \
#         --model="ckpt/progen2-small-surface-p4f-finetuned/e${i}" \
#         --device=cuda \
#         --batch_size=8 \
#         --max_length=1024 \
#         --t=0.01 \
#         --k=10 \
#         --test_data_dir="data/processed_surface_data/test"

#     echo "Finished inference for model e${i}"
#     echo "----------------------------------------"
# done
