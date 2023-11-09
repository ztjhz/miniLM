import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parents[1]))

import argparse

import wandb

from transformers import AutoModelForSequenceClassification
from datasets import load_dataset

from utils.trainer import CustomTrainer
from utils.preprocessing import tokenize, train_val_test_split, subset_dataset

def main():
    parser = argparse.ArgumentParser(description='Small dataset experiments')

    parser.add_argument("--dataset", choices=['imdb', 'yelp', 'sst2'], default='imdb', help="Dataset to use")
    parser.add_argument("--model", choices=['roberta', 'gpt2', 't5'], default='roberta', help='Model to use')
    # we provide an option to subset the dataset to reduce training time
    parser.add_argument("--subset_yelp", type=bool, default=False, help='Whether to subset the dataset')

    # for deepspeed
    parser.add_argument("--local_rank")

    args = parser.parse_args()

    print(args)

    run_name = f"{args.model}-CompareTransformers-{args.dataset}"

    if args.subset_yelp:
        run_name += "_subset"

    # set up dataset
    if args.dataset == 'imdb':
        dataset = load_dataset("imdb")
        num_labels = 2
        input_col_name = "text"
    elif args.dataset =='yelp':
        dataset = load_dataset("yelp_review_full")
        num_labels = 5
        input_col_name = "text" 
    elif args.dataset == 'sst2':
        dataset = load_dataset("sst2")
        num_labels = 2
        input_col_name = "sentence"
    else:
        raise NotImplementedError

    # set up model
    if args.model == 'roberta':
        model_name = "roberta-base"
    elif args.model == 'gpt2':
        model_name = "gpt2"
    elif args.model == 't5':
        model_name = "t5-base"
    else:
        raise NotImplementedError
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if model.config.pad_token_id == None:
        model.config.pad_token_id = model.config.eos_token_id


    # prepare dataset
    tokenized_datasets = tokenize(dataset, model_name, input_col_name=input_col_name)
    train_dataset, val_dataset, test_dataset = train_val_test_split(tokenized_datasets)

    if args.dataset =='yelp' and args.subset_yelp == True:
        # yelp dataset has 650k train which would take very long to train
        # we provide an option to subset the dataset to reduce training time
        train_dataset = subset_dataset(train_dataset, size=25_000, seed=42)
        val_dataset = subset_dataset(val_dataset, size=25_000, seed=42)
        test_dataset = subset_dataset(test_dataset, size=25_000, seed=42)

    # create trainer
    trainer = CustomTrainer(
        run_name=run_name,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # train
    trainer.train()



if __name__ == "__main__":
    main()