python3 inverse_folding/inverse_folding/finetune_surface.py \
    --model=checkpoints/progen2-small \
    --train_file=data/processed_surface_data/train \
    --test_file=data/processed_surface_data/test \
    --device=cuda \
    --epochs=5 \
    --batch_size=8 \
    --accumulation_steps=2 \
    --lr=1e-4 \
    --decay=cosine \
    --warmup_steps=200 \
    --eval_steps=1 \
    --checkpoint_steps=1 \
    --eval_before_train \
    --save_path=ckpt/progen2-small-surface-p80k-2r \
    # --sec_struc \
    # --model_parallel \