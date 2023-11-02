import os

import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC

from transformers import Trainer, TrainingArguments, EvalPrediction

import wandb

os.environ["WANDB_ENTITY"] = "sc4001" # name of W&B team 
os.environ["WANDB_PROJECT"] = "text-sentiment-analysis" # name of W&B project 

wandb.login()

# default optimizer: AdamW
training_args = TrainingArguments(
    output_dir='./results', # output directory of results
    num_train_epochs=3, # number of train epochs
    report_to='wandb', # enable logging to W&B
    evaluation_strategy='steps', # check evaluation metrics at each epoch
    logging_steps = 10, # we will log every 10 steps
    eval_steps = 200, # we will perform evaluation every 200 steps
    save_steps = 200, # we will save the model every 200 steps
    save_total_limit = 5, # we only save the last 5 checkpoints (including the best one)
    load_best_model_at_end = True, # we will load the best model at the end of training
    metric_for_best_model = 'accuracy', # metric to see which model is better
    deepspeed='ds_config.json', # deep speed integration
    
    #### effective batch_size = per_device_train_batch_size x gradient_accumulation_steps ####
    #### We set effective batch_size to 32 (8 x 4) ####
    per_device_train_batch_size=int(8 / torch.cuda.device_count()), # batch size per device
    per_device_eval_batch_size=int(8 / torch.cuda.device_count()), # eval batch size per device
    gradient_accumulation_steps=4, # gradient accumulation
)


def compute_metrics(pred: EvalPrediction):
    """
    Compute metrics using torchmetrics for a given set of predictions and labels.

    Args:
    pred (EvalPrediction): An object containing model predictions and labels.

    Returns:
    dict: A dictionary containing metric results.
    """
    # Extract labels and predictions
    labels = pred.label_ids
    preds = pred.predictions

    # for t5 model, the predictions is in the form of a tuple with the logits as the only element in the tuple
    if isinstance(preds, tuple):
        preds = preds[0]

    num_classes = preds.shape[1]

    # Convert to torch tensors
    labels = torch.tensor(labels)
    preds = torch.tensor(preds)

    # Initialize metrics
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(torch.cuda.current_device())
    precision = Precision(task="multiclass", num_classes=num_classes).to(torch.cuda.current_device())
    recall = Recall(task="multiclass", num_classes=num_classes).to(torch.cuda.current_device())
    f1 = F1Score(task="multiclass", num_classes=num_classes).to(torch.cuda.current_device())
    auroc = AUROC(task="multiclass", num_classes=num_classes).to(torch.cuda.current_device())

    # Calculate metrics (automatically does argmax)
    accuracy_score = accuracy(preds, labels)
    precision_score = precision(preds, labels)
    recall_score = recall(preds, labels)
    f1_score = f1(preds, labels)
    auroc_score = auroc(preds, labels)


    # Convert to CPU for serialization
    return {
        "accuracy": accuracy_score.cpu().item(),
        "precision": precision_score.cpu().item(),
        "recall": recall_score.cpu().item(),
        "f1": f1_score.cpu().item(),
        "auroc": auroc_score.cpu().item(),
    }

class CustomTrainer(Trainer):
    def __init__(self, *args, run_name: str = None, trainer_args: TrainingArguments = None, **kwargs):
        if not trainer_args:
            # set default training arguments if not supplied
            trainer_args = training_args
        if run_name:
            trainer_args.run_name = run_name # specify the run name for wandb logging
        super().__init__(*args, compute_metrics=compute_metrics, args=trainer_args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override the default compute_loss. 
        Use Cross Entropy Loss for multiclass classification (>= 2).
        """
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute cross entropy loss
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss