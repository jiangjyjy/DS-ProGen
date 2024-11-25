python3 inverse_folding/inverse_folding/finetune.py \
    --model=ckpt/progen2-small-p80k/e2 \
    --train_file=data/processed_8w_data/train.pkl \
    --test_file=data/processed_8w_data/test.pkl \
    --device=cuda \
    --epochs=5 \
    --batch_size=8 \
    --accumulation_steps=2 \
    --lr=5e-5 \
    --decay=cosine \
    --warmup_steps=1500 \
    --eval_steps=1 \
    --checkpoint_steps=1 \
    --eval_before_train \
    --save_path=ckpt/progen2-small-p80k-finetuned \
    # --sec_struc \
    # --model_parallel \
