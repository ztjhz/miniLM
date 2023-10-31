import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parents[1]))

import argparse

from datasets import load_dataset

from models.SlicedLlama2 import get_sliced_llama2
from utils.trainer import CustomTrainer
from utils.preprocessing import train_val_test_split, tokenize

def main():
    parser = argparse.ArgumentParser()

    # Number of layers for the sliced Llama2
    parser.add_argument('--num_layers', type=int, choices=range(1, 13), default=1,
                        help='Number of layers (from 1 to 12). Default is 1.')
    args = parser.parse_args()

    # log the number of layers
    print(f"Number of layers: {args.num_layers}")

    model = get_sliced_llama2(num_labels=2, num_layers=args.num_layers)

    dataset = load_dataset("imdb")
    tokenized_datasets = tokenize(dataset, "meta-llama/Llama-2-7b-hf")
    train_dataset, val_dataset, test_dataset = train_val_test_split(tokenized_datasets)

    trainer = CustomTrainer(
        run_name="Bert-CompareTransformers-Imdb",
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    test_metrics = trainer.predict(test_dataset=test_dataset)


if __name__ == "__main__":
    main()
