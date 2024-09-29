python3 inverse_folding/inverse_folding/finetune.py \
    --model=checkpoints/progen2-small \
    --train_file=/home/v-yantingli/mmp/data/processed_data_new/train.pkl \
    --test_file=/home/v-yantingli/mmp/data/processed_data_new/valid.pkl \
    --device=cuda \
    --epochs=20 \
    --batch_size=16 \
    --accumulation_steps=2 \
    --lr=1e-4 \
    --decay=cosine \
    --warmup_steps=200 \
    --eval_steps=1 \
    --eval_before_train \
    # --model_parallel