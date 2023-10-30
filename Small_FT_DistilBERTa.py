from transformers import AutoTokenizer, AutoModelForSequenceClassification

from datasets import load_dataset

from utils.trainer import CustomTrainer

# Load model directly
# https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

dataset = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]


trainer = CustomTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()