# SC4001 Project

- [Training](#training)
  - [Cuda Out of Memory](#cuda-out-of-memory)

## Training

This project uses deep speed to optimise training for Llama2.

In command line: `deepspeed <program>.py --deepspeed ds_config.json`

Deployment in notebook (https://huggingface.co/docs/transformers/main_classes/deepspeed#deployment-in-notebooks):

```python
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
```

### Cuda Out of Memory

If you encouter CUDA out of memory while running the deep speed, reduce `allgather_bucket_size` and `reduce_bucket_size` in the `ds_config.json`
