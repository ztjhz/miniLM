import deepspeed
import wandb

from datasets import load_dataset

from models.SlicedLlama import SlicedLlama, compute_loss
from utils.preprocessing import tokenize, train_val_test_split

NUM_EPOCH = 3

def main():
    # wandb.init()
    dataset = load_dataset("imdb")
    tokenized_datasets = tokenize(dataset, "meta-llama/Llama-2-7b-hf")
    train_dataset, val_dataset, test_dataset = train_val_test_split(tokenized_datasets)

    model = SlicedLlama(num_labels=2)
    model_engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                        model_parameters=model.parameters(),
                                                        config="ds_config.json")


    for epoch in range(NUM_EPOCH):
        for step, (data, labels) in enumerate(train_dataset):
            # forward
            all_output_logits = model_engine(data)

            # compute loss
            summed_loss, all_layer_loss = compute_loss(all_layer_logits=all_output_logits, labels=labels, 
                                num_labels=model.num_labels, num_layers=model.num_layers)

            # backward propagation
            model_engine.backward(summed_loss)

            # weight update
            # deepspeed handles the optimization steps which includes the optimizer zero grad
            model_engine.step()

            #save checkpoint
            # if step % args.save_interval:
            #     client_sd['step'] = step
            #     ckpt_id = loss.item()
            #     model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd = client_sd)

