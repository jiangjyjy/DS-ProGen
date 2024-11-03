python3 inverse_folding/inverse_folding/inference_surface.py \
    --model=/home/v-yantingli/mmp/ckpt/progen2-small-surface_1103/e2 \
    --device=cuda \
    --batch_size=8 \
    --max_length=1024 \
    --t=0.01 \
    --k=10 \
    --test_data_dir=/home/v-yantingli/mmp/data/processed_surface_data/test