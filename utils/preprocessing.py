from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

def train_val_test_split(dataset: Dataset | DatasetDict, seed: int = 42):
    """
    Splits the dataset into training, validation, and test sets.

    :param dataset: The dataset to split.
    :param seed: Random seed for reproducibility.
    :return: A tuple containing the train, validation, and test splits.
    """
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    if "validation" in dataset:
        # dataset already contains the validation split
        val_dataset = dataset["validation"]
    else:
        # dataset does not contain the validation split
        train_dataset, val_dataset = train_dataset.train_test_split(test_size=0.3, seed=seed).values()
    
    return (train_dataset, val_dataset, test_dataset)

def tokenize(dataset: Dataset | DatasetDict, tokenizer_name: str):
    def _tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized_datasets = dataset.map(_tokenize, batched=True)
    return tokenized_datasets
