import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split

import optuna

import torch
from torch.utils.tensorboard import SummaryWriter

from hpo_workshop.tokenizer import BytesTokenizer
from hpo_workshop.dataset import get_dataloader
from hpo_workshop.training_loop import train
from hpo_workshop.rnn import RNNPredictor

def main():
    parser = argparse.ArgumentParser(description='Basic Hyper Parameter optimization example')
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--random-seed', help="The random seed to use", type=int, default=1729)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    labels = df['p_np'].tolist()
    smiles_list = df['smiles'].tolist()

    tokenizer = BytesTokenizer()
    
    device = torch.device('cuda')

    max_epochs=5
    n_folds = 10
    batch_size = 256
    num_workers = 4
    n_hpo_trials = 20

    torch.random.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.random_seed)
    for visible_index, heldout_indices in skf.split(smiles_list, labels):
        tb_writer = SummaryWriter('optuna_runs')
        
        visible_labels = [labels[i] for i in visible_index]
        train_indices, dev_indices = train_test_split(visible_index, stratify=visible_labels, shuffle=True, test_size=0.2, random_state=args.random_seed)
        
        train_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=train_indices, tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        dev_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=dev_indices, tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers)
        heldout_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=heldout_indices, tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers)
        
        model_kwargs = dict(tokenizer=tokenizer, device=device)

        def optimization_function(trial: optuna.Trial):
            model_hparams = dict(embedding_dim=trial.suggest_categorical('embedding_dim', [8, 16, 32, 64, 128]),
                                 d_model=trial.suggest_int(
                                     'd_model', [8, 16, 32, 64, 128, 256, 512, 1024]),
                                 num_layers=trial.suggest_int(
                                     'num_layers', 1, 5),
                                 bidirectional=trial.suggest_categorical(
                                     'bidirectional', [True, False]),
                                 dropout=trial.suggest_float('dropout', 0, 1),
                                 learning_rate=trial.suggest_float(
                                     'learning_rate', 1e-5, 1e-2, log=True),
                                 weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True))

            best_roc_auc = train(train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, test_dataloader=heldout_dataloader, writer=tb_writer,
                                 max_epochs=max_epochs, model_class=RNNPredictor, model_args=tuple(), model_kwargs=model_kwargs, model_hparams=model_hparams)
            return best_roc_auc

        study = optuna.create_study(direction='maximize')
        study.optimize(optimization_function, n_trials=n_hpo_trials)
        best_model_params = study.best_params
        final_roc_auc = train(train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, test_dataloader=heldout_dataloader, writer=tb_writer,
                                 max_epochs=max_epochs, model_class=RNNPredictor, model_args=tuple(), model_kwargs=model_kwargs, model_hparams=best_model_params)
            
        tb_writer.close()
            
    args = parser.parse_args()


if __name__ == '__main__':
    main()