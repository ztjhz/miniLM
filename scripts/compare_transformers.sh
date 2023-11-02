WANDB_API_KEY=your_wandb_api_key \
    deepspeed --include localhost:0,1,2,3 \
    pipeline/compare_transformers.py \
    --dataset yelp \
    --model t5
