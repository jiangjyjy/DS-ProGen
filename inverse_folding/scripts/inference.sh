for i in {1..5}; do
    MODEL="ckpt/progen2-small-finetuned/e$i"
    echo "Running inference with model $MODEL"
    
    python3 inverse_folding/inverse_folding/inference.py \
        --model=$MODEL \
        --device=cuda \
        --batch_size=8 \
        --max_length=1024 \
        --t=0.1 \
        --k=10 \
        --test_data_dir='/home/v-yantingli/mmp/data/processed_data_new'
done
