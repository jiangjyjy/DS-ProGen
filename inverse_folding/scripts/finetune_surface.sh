python3 inverse_folding/inverse_folding/finetune_surface.py \
    --model=/home/v-yantingli/mmp/ckpt/progen2-small-surface_afdb/e3 \
    --train_file=/home/v-yantingli/mmp/data/processed_surface_data/train \
    --test_file=/home/v-yantingli/mmp/data/processed_surface_data/test \
    --device=cuda \
    --epochs=16 \
    --batch_size=8 \
    --accumulation_steps=2 \
    --lr=5e-5 \
    --decay=cosine \
    --warmup_steps=200 \
    --eval_steps=1 \
    --checkpoint_steps=1 \
    --eval_before_train \
    --save_path=/home/v-yantingli/mmp/ckpt/progen2-small-surface_1103 \
    # --sec_struc \
    # --model_parallel \