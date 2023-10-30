from transformers import AutoTokenizer, AutoModelForSequenceClassification

from datasets import load_dataset

from utils.trainer import CustomTrainer

# Load model directly
# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", num_labels=3)

dataset = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]


trainer = CustomTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()