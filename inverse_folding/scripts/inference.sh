python3 inverse_folding/inverse_folding/inference.py \
    --model=checkpoints/progen2-base \
    --device=cuda \
    --batch_size=8 \
    --max_length=1024 \
    --t=0.1 \
    --k=10 \
    --test_data='/home/v-yantingli/mmp/data/processed_data/test_h.pkl'