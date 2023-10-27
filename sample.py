from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

from utils.trainer import CustomTrainer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
dataset = load_dataset("imdb")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]


model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

trainer = CustomTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()