python3 ../scripts/sample.py \
    --model=checkpoints/progen2-small-finetuned/e5 \
    --device=cuda \
    --batch_size=8 \
    --iters=1 \
    --max_length=512 \
    --t=1.0 \
    --k=10 \
    --prompt="<|pf03668|>1MEVVIVTGMSGAGK"