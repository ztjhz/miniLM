import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parents[1]))

import argparse

import wandb

from transformers import AutoModelForSequenceClassification, RobertaConfig
from datasets import load_dataset

from utils.trainer import CustomTrainer, training_args
from utils.preprocessing import tokenize, train_val_test_split

def main():
    parser = argparse.ArgumentParser(description='Small dataset experiments')

    parser.add_argument("--init", choices=['train', 'finetune'], default='train', help="Whether to train from scratch or finetune")
    parser.add_argument("--model", choices=['roberta'], default='roberta', help='What model to use')
    parser.add_argument("--run_name", type=str, help="Run name of the wandb experiment")

    # for deepspeed
    parser.add_argument("--local_rank")

    args = parser.parse_args()

    print(args)
    
    if not args.run_name:
        print("Warning! WandB run name not supplied!")

    # set up model
    if args.init == 'train':
        model_name = "roberta-base"
        model = AutoModelForSequenceClassification.from_config(RobertaConfig.from_pretrained(model_name))
        training_args.num_train_epochs = 20
    elif args.init == 'finetune':
        model_name = "siebert/sentiment-roberta-large-english"
        model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
    else:
        raise NotImplementedError


    # set up dataset
    dataset = load_dataset("imdb")
    tokenized_datasets = tokenize(dataset, model_name)
    train_dataset, val_dataset, test_dataset = train_val_test_split(tokenized_datasets)

    # create trainer
    trainer = CustomTrainer(
        trainer_args=training_args,
        run_name=args.run_name,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # train
    trainer.train()



if __name__ == "__main__":
    main()