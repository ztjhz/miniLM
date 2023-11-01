WANDB_API_KEY=2f2372d1a623873ef37a1451ce0ec60b6451ff3c \
    deepspeed --include localhost:0,1,2,3 \
    pipeline/llama.py
    