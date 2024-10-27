python3 inverse_folding/inverse_folding/finetune.py \
    --model=/home/v-yantingli/mmp/ckpt/progen2-small-p4m/e1 \
    --train_file=/home/v-yantingli/mmp/data/processed_data_new/train_all.pkl \
    --test_file=/home/v-yantingli/mmp/data/processed_data_new/test_hme.pkl \
    --device=cuda \
    --epochs=10 \
    --batch_size=8 \
    --accumulation_steps=2 \
    --lr=5e-5 \
    --decay=cosine \
    --warmup_steps=1500 \
    --eval_steps=1 \
    --checkpoint_steps=1 \
    --eval_before_train \
    --save_path=/home/v-yantingli/mmp/ckpt/progen2-small-p4m-finetuned2 \
    # --sec_struc \
    # --model_parallel \
