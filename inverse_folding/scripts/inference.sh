python3 inverse_folding/inverse_folding/inference.py \
    --model=/home/v-yantingli/mmp/ckpt/progen2-small-p4m-finetuned1029/e2 \
    --device=cuda \
    --batch_size=8 \
    --max_length=1024 \
    --t=0.01 \
    --k=10 \
    --test_data_dir='/home/v-yantingli/mmp/data/processed_data_new'

