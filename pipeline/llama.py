import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parents[1]))

import argparse

import json

import deepspeed
import wandb
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from datasets import load_dataset
from transformers import EvalPrediction

from models.SlicedLlama import SlicedLlama, compute_loss
from utils.preprocessing import tokenize, train_val_test_split, subset_dataset
from utils.trainer import compute_metrics

NUM_EPOCH = 3

def main():
    parser = argparse.ArgumentParser(description='Sliced model experiment')
    parser.add_argument("--dataset", choices=['imdb', 'yelp'], default='imdb', help="Dataset to use")
    
    # for deepspeed
    parser.add_argument("--local_rank")

    args = parser.parse_args()

    print(args)

    # set up dataset
    if args.dataset == 'imdb':
        dataset = load_dataset("imdb")
        num_labels = 2
    elif args.dataset == 'yelp':
        dataset = load_dataset("yelp_review_full")
        num_labels = 5
    else:
        raise NotImplementedError

    # set up model
    with open("ds_config_llama.json", "r") as f:
        df_config = json.load(f)
    model = SlicedLlama(num_labels=num_labels)
    model_engine, _, _, _ = deepspeed.initialize(model=model,
                                                model_parameters=model.parameters(),
                                                config=df_config)
    
    # set up dataset
    tokenized_datasets = tokenize(dataset, "meta-llama/Llama-2-7b-hf")
    train_dataset, val_dataset, test_dataset = train_val_test_split(tokenized_datasets)

    if args.dataset == 'yelp':
        # yelp dataset has 650k train which would take very long to train
        # so we subset it
        train_dataset = subset_dataset(train_dataset, size=25_000, seed=42)
        val_dataset = subset_dataset(val_dataset, size=25_000, seed=42)
        test_dataset = subset_dataset(test_dataset, size=25_000, seed=42)

    # set up dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=df_config["train_batch_size"])
    val_dataloader = DataLoader(val_dataset, batch_size=df_config["train_batch_size"])
    test_dataloader = DataLoader(test_dataset, batch_size=df_config["train_batch_size"])

    # start wandb tracking
    run_name = f"Sliced Llama - {args.dataset}"
    if model_engine.global_rank == 0:
        wandb.init(
            project="text-sentiment-analysis",
            entity="sc4001",
            name=run_name,
            config=df_config
            )

    for epoch in range(NUM_EPOCH):
        if model_engine.global_rank == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCH}")

        for step, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch['input_ids'].to(torch.cuda.current_device())
            attention_mask = batch['attention_mask'].to(torch.cuda.current_device())
            labels = batch['label'].to(torch.cuda.current_device())

            # set to train mode
            model_engine.train()

            # forward
            all_output_logits = model_engine(input_ids=input_ids, 
                                             attention_mask=attention_mask)

            # compute loss
            summed_loss, all_layer_loss = compute_loss(all_layer_logits=all_output_logits, labels=labels, 
                                                       num_labels=model.num_labels, num_layers=model.num_layers)

            # backward propagation
            model_engine.backward(summed_loss)

            # weight update
            # deepspeed handles the optimization steps which includes the optimizer zero grad
            model_engine.step()

            # prepare wandb logs
            wandb_log = {}
            for i in range(model.num_layers):
                wandb_log[f"train_loss/layer_{i+1}"] = all_layer_loss[i]

            ########### validation ###########
            if step % 1000 == 0:
                # set to eval mode
                model_engine.eval()

                all_labels = torch.tensor([]).to(torch.cuda.current_device())
                all_layer_logits = torch.tensor([]).to(torch.cuda.current_device())
                with torch.no_grad():
                    for batch in val_dataloader:
                        input_ids = batch['input_ids'].to(torch.cuda.current_device())
                        attention_mask = batch['attention_mask'].to(torch.cuda.current_device())
                        labels = batch['label'].to(torch.cuda.current_device()) # labels: (batch_size)

                        # forward
                        all_output_logits = model_engine(input_ids=input_ids,
                                                         attention_mask=attention_mask) # (num_layers, batch_size, num_labels)

                        # accumulate all labels
                        all_labels = torch.concat([all_labels, labels])

                        # accumulate all the layer logits
                        all_layer_logits = torch.concat([all_layer_logits, all_output_logits], dim=1)

                # compute loss
                _, all_layer_loss = compute_loss(all_layer_logits=all_layer_logits, labels=all_labels, 
                                                            num_labels=model.num_labels, num_layers=model.num_layers) # (num_layers)
                # prepare wandb logs
                for i in range(model.num_layers):
                    wandb_log[f"eval_loss/layer_{i+1}_loss"] = all_layer_loss[i]

                # compute metrics
                for i in range(model.num_layers):
                    pred = EvalPrediction(predictions=all_layer_logits[i], label_ids=all_labels.long())
                    metrics = compute_metrics(pred=pred)

                    # prepare wandb logs
                    for key, value in metrics.items():
                        wandb_log[f"eval_{key}/layer_{i+1}"] = value

            ########### save checkpoint ###########
            if step % 1000 == 0:
                model_engine.save_checkpoint(save_dir="checkpoints")

            # log to wandb
            if model_engine.global_rank == 0:
                wandb.log(wandb_log)
    
    # end wandb tracking
    if model_engine.global_rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()