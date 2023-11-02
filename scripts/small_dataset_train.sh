WANDB_API_KEY=your_wandb_api_key \
    deepspeed --include localhost:0,1,2,3 \
    pipeline/small_dataset.py \
    --init train \
    --model roberta \
    --run_name RoBERTa-TrainFromScratch-Imdb
