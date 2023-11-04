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

def tokenize(dataset: Dataset | DatasetDict, tokenizer_name: str, input_col_name: str = "text"):
    def _tokenize(examples):
        return tokenizer(examples[input_col_name], padding='max_length', truncation=True, max_length=512)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_datasets = dataset.map(_tokenize, batched=True).select_columns(["input_ids", "attention_mask", "label"]).with_format("torch")
    return tokenized_datasets

def subset_dataset(dataset: Dataset | DatasetDict, 
                   size: int,
                   seed: int = 42):
    shuffled_dataset = dataset.shuffle(seed=seed)
    new_dataset = shuffled_dataset.select(range(size))
    return new_dataset
