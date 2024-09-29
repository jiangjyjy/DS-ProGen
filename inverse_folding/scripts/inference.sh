python3 inverse_folding/inverse_folding/inference.py \
    --model=ckpt/progen2-small-finetuned/e5 \
    --device=cuda \
    --batch_size=8 \
    --max_length=1024 \
    --t=0.1 \
    --k=10 \
    --test_data='/home/v-yantingli/mmp/data/processed_data_new/test_h.pkl'