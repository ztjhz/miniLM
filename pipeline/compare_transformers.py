import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parents[1]))

import argparse

import wandb

from transformers import AutoModelForSequenceClassification
from datasets import load_dataset

from utils.trainer import CustomTrainer, training_args
from utils.preprocessing import tokenize, train_val_test_split

def main():
    parser = argparse.ArgumentParser(description='Small dataset experiments')

    parser.add_argument("--dataset", choices=['imdb', 'yelp', 'sst2'], default='imdb', help="Dataset to use")
    parser.add_argument("--model", choices=['roberta', 'gpt2', 't5'], default='roberta', help='Model to use')

    # for deepspeed
    parser.add_argument("--local_rank")

    args = parser.parse_args()

    print(args)

    run_name = f"{args.model}-CompareTransformers-{args.dataset}"

    # set up dataset
    if args.dataset == 'imdb':
        dataset = load_dataset("imdb")
        num_labels = 2
        input_col_name = "text"
    elif args.dataset =='yelp':
        dataset = load_dataset("yelp_review_full")
        num_labels = 5
        input_col_name = "text"
        
        # reduce epochs to 1 because train set is 650k
        training_args.num_train_epochs = 1

        # reduce frequency of eval and eval because got many steps
        training_args.save_steps = 1000
        training_args.eval_steps = 1000
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

    # prepare dataset
    tokenized_datasets = tokenize(dataset, model_name, input_col_name=input_col_name)
    train_dataset, val_dataset, test_dataset = train_val_test_split(tokenized_datasets)

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