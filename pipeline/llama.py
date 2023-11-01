import deepspeed
import wandb
import torch

from datasets import load_dataset
from transformers import EvalPrediction

from models.SlicedLlama import SlicedLlama, compute_loss
from utils.preprocessing import tokenize, train_val_test_split
from utils.trainer import compute_metrics

NUM_EPOCH = 3

def main():
    wandb.init(
        project="text-sentiment-analysis",
        entity="sc4001",
        name="Sliced Llama"
        )
    dataset = load_dataset("imdb")
    tokenized_datasets = tokenize(dataset, "meta-llama/Llama-2-7b-hf")
    train_dataset, val_dataset, test_dataset = train_val_test_split(tokenized_datasets)

    model = SlicedLlama(num_labels=2)
    model_engine, _, _, _ = deepspeed.initialize(model=model,
                                                        model_parameters=model.parameters(),
                                                        config="ds_config.json")


    for epoch in range(NUM_EPOCH):
        for step, (data, labels) in enumerate(train_dataset):
            # set to train mode
            model_engine.train()

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

            # prepare wandb logs
            wandb_log = {"train": {}}
            for i in range(model.num_layers):
                wandb_log["train"][f"layer_{i+1}_loss"] = all_layer_loss[i]

            ########### validation ###########
            if step % 50 == 0:
                wandb_log["eval"] = {}

                # set to eval mode
                model_engine.eval()

                all_labels = torch.tensor([])
                all_layer_logits = torch.tensor([])
                with torch.no_grad():
                    for batch in val_dataset:
                        inputs, labels = batch # labels: (batch_size)

                        # forward
                        all_output_logits = model_engine(inputs) # (num_layers, batch_size, num_labels)

                        # accumulate all labels
                        all_labels = torch.concat([all_labels, labels])

                        # accumulate all the layer logits
                        all_layer_logits = torch.concat([all_layer_logits, all_output_logits], dim=1)

                # compute loss
                _, all_layer_loss = compute_loss(all_layer_logits=all_layer_logits, labels=all_labels, 
                                                            num_labels=model.num_labels, num_layers=model.num_layers) # (num_layers)
                # prepare wandb logs
                for i in range(model.num_layers):
                    wandb_log["eval"][f"layer_{i+1}_loss"] = all_layer_loss[i]

                # compute metrics
                for i in range(model.num_layers):
                    pred = EvalPrediction(predictions=all_layer_logits[i], label_ids=all_labels)
                    metrics = compute_metrics(pred=pred)

                    # prepare wandb logs
                    for key, value in metrics.items():
                        wandb_log["eval"][f"layer_{i+1}_{key}"] = value

            wandb.log(wandb_log)

if __name__ == "__main__":
    main()