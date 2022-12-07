Cross validation example code
=============================

Here will go through an example where we change some code to use optuna. 
You can find the code :download:`here <_static/code_archive.zip>`.

The example implements a simple Long Short-Term Memory (LSTM) network on a 
binary sequence prediction dataset. The sequences are molecules encoded in 
the SMILES format and the prediction target is whether they risk penetrating 
the Blood Brain Barrier or not.

The network is defined in the file :code:`hpo_workshop/rnn.py` and implemented pytorch, in this example we're mainly interested in the initialization of the network:

.. code-block:: python
    :emphasize-lines: 2-21

    class RNNPredictor(nn.Module):
        def __init__(self, *, tokenizer, device, embedding_dim, d_model, num_layers, bidirectional, dropout, learning_rate, weight_decay) -> None:
            super().__init__()
            self.tokenizer = tokenizer
            self.device = device
            self.embedding = nn.Embedding(tokenizer.get_num_embeddings(), 
                                        embedding_dim=embedding_dim, 
                                        padding_idx=0)
            self.recurrent_layers = nn.LSTM(input_size=embedding_dim, 
                                            hidden_size=d_model, 
                                            num_layers=num_layers,
                                            bidirectional=bidirectional,
                                            dropout=dropout,
                                            )
            self.num_directions = 1
            if bidirectional:
                self.num_directions = 2
            self.output_layers = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.num_directions*d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
            self.to(self.device)
        
        def forward(self, sequence_batch, lengths):
            embedded_sequences = self.embedding(sequence_batch)
            packed_sequence = pack_padded_sequence(embedded_sequences, lengths, enforce_sorted=False)
            output, (h_n, c_n) = self.recurrent_layers(packed_sequence)
            final_state = h_n[-1]
            if self.num_directions == 2:
                final_state = torch.cat((final_state, h_n[-1]), dim=-1)
            logits = self.output_layers(final_state)
            return logits

        def loss_on_batch(self, batch):
            sequence_batch, lengths, labels = batch
            logit_prediction = self(sequence_batch.to(self.device), lengths)
            loss = self.loss_fn(logit_prediction.squeeze(), labels.to(self.device))
            return loss

        def train_batch(self, batch):
            self.train()
            self.optimizer.zero_grad()
            loss = self.loss_on_batch(batch)
            loss.backward()
            self.optimizer.step()
            return loss.item()

        def eval_batch(self, batch):
            self.eval()
            with torch.no_grad():
                loss = self.loss_on_batch(batch)
                return loss.item()

        def eval_and_predict_batch(self, batch):
            self.eval()
            with torch.no_grad():
                sequence_batch, lengths, labels = batch
                logit_prediction = self(sequence_batch.to(self.device), lengths)
                loss = self.loss_fn(logit_prediction.squeeze(), labels.to(self.device))
                prob_predictions = torch.sigmoid(logit_prediction)
                return loss.item(), labels.cpu().numpy(), prob_predictions.cpu().numpy()


As you can see, we're taking the hyper parameters of the network as keyword arguments to 
the :code:`__init__` method. Our goal is to find good settings for these hyper parameters.

Manual hyper parameter Optimization
-----------------------------------

First we start with the basic "Grad Student Descent" example in :code:`scripts/basic_neural_network.py`. The important part is given in the training loop:

.. code-block:: python

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.random_seed)
    for visible_index, heldout_indices in skf.split(smiles_list, labels):
        tb_writer = SummaryWriter('basic_runs')
        
        visible_labels = [labels[i] for i in visible_index]
        train_indices, dev_indices = train_test_split(visible_index, stratify=visible_labels, shuffle=True, test_size=0.2, random_state=args.random_seed)
        
        train_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=train_indices,
                                          tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        dev_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=dev_indices,
                                        tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers)
        heldout_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=heldout_indices,
                                            tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers)

        model_kwargs = dict(tokenizer=tokenizer, device=device)

        model_hparams = dict(embedding_dim=128,
                             d_model=128,
                             num_layers=3,
                             bidirectional=True,
                             dropout=0.2,
                             learning_rate=0.001,
                             weight_decay=0.0001)
        
        heldout_roc_auc = train(train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, test_dataloader=heldout_dataloader, writer=tb_writer,
                                max_epochs=max_epochs, model_class=RNNPredictor, model_args=tuple(), model_kwargs=model_kwargs, model_hparams=model_hparams)

        tb_writer.close()

Here we are manually setting hyper parameters and then training our models with these. Using tensordboard we can 
essentially track how good they are. These runs will be stored in :code:`basic_runs` so you need to run tensorboard like:

.. code-block:: bash

    $ tensorboard --logdir=basic_runs

Try running the experiment a couple of times while changing hyper parameters and see if you can get better results.


Hyper Parameter Optimization with Optuna
----------------------------------------

We'll now take a look at how we can easily extend the above example using Optuna. We will replace the work we did with 
manually setting hyper parameters with a loop which automatically searches the hyper parameter space. We need to import optuna 
and create a new study object for our hyper parameter optimization. We will perform a separate study for each fold in our cross validation.

.. code-block:: python
    :emphasize-lines: 1,30

    import optuna 

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.random_seed)
    for visible_index, heldout_indices in skf.split(smiles_list, labels):
        tb_writer = SummaryWriter('basic_runs')
        
        visible_labels = [labels[i] for i in visible_index]
        train_indices, dev_indices = train_test_split(visible_index, stratify=visible_labels, shuffle=True, test_size=0.2, random_state=args.random_seed)
        
        train_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=train_indices,
                                          tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        dev_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=dev_indices,
                                        tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers)
        heldout_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=heldout_indices,
                                            tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers)

        model_kwargs = dict(tokenizer=tokenizer, device=device)

        model_hparams = dict(embedding_dim=128,
                             d_model=128,
                             num_layers=3,
                             bidirectional=True,
                             dropout=0.2,
                             learning_rate=0.001,
                             weight_decay=0.0001)
        
        heldout_roc_auc = train(train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, test_dataloader=heldout_dataloader, writer=tb_writer,
                                max_epochs=max_epochs, model_class=RNNPredictor, model_args=tuple(), model_kwargs=model_kwargs, model_hparams=model_hparams)

        study = optuna.create_study(direction='maximize')
        
        tb_writer.close()

We've now created a new study with the objective of maximizing an objective function. There are two interfaces for running 
optunas optimization: the :code:`study.ask()` / :code:`study.tell()` interface and the :code:`study.optimize()` interface. 
We looked at how to use :code:`study.ask()` / :code:`study.tell()` in the noteboook before and will now use :code:`study.optimize()` 
instead.

:code:`study.optimize()` takes a function to optimize as an input, and we'll implement it inlined in our optimization loop so it can refer the datasets we've set up.

.. code-block:: python
    :emphasize-lines: 19-29,32

    import optuna 

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.random_seed)
    for visible_index, heldout_indices in skf.split(smiles_list, labels):
        tb_writer = SummaryWriter('basic_runs')
        
        visible_labels = [labels[i] for i in visible_index]
        train_indices, dev_indices = train_test_split(visible_index, stratify=visible_labels, shuffle=True, test_size=0.2, random_state=args.random_seed)
        
        train_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=train_indices,
                                          tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        dev_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=dev_indices,
                                        tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers)
        heldout_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=heldout_indices,
                                            tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers)

        model_kwargs = dict(tokenizer=tokenizer, device=device)
        
        def optimization_function(trial: optuna.Trial):
            model_hparams = dict(embedding_dim=128,
                                d_model=128,
                                num_layers=3,
                                bidirectional=True,
                                dropout=0.2,
                                learning_rate=0.001,
                                weight_decay=0.0001)
            heldout_roc_auc = train(train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, test_dataloader=heldout_dataloader, writer=tb_writer,
                                max_epochs=max_epochs, model_class=RNNPredictor, model_args=tuple(), model_kwargs=model_kwargs, model_hparams=model_hparams)
            return heldout_roc_auc

        study = optuna.create_study(direction='maximize')
        study.optimize(optimization_function, n_trials=20)

        tb_writer.close()


We've now set up the study infrastructure, but we're still not actually 
performing any search. The :code:`optimization_function` we defined takes 
a :code:`optuna.Trial` object as its argument, and this is our interface 
to the actual hyper parameter search procedure.

We extend our :code:`optimization_function` so that the values for the hyper 
parameters are give by the trial:

.. code-block:: python
    :emphasize-lines: 20-26

    import optuna 

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.random_seed)
    for visible_index, heldout_indices in skf.split(smiles_list, labels):
        tb_writer = SummaryWriter('basic_runs')
        
        visible_labels = [labels[i] for i in visible_index]
        train_indices, dev_indices = train_test_split(visible_index, stratify=visible_labels, shuffle=True, test_size=0.2, random_state=args.random_seed)
        
        train_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=train_indices,
                                          tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        dev_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=dev_indices,
                                        tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers)
        heldout_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=heldout_indices,
                                            tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers)

        model_kwargs = dict(tokenizer=tokenizer, device=device)
        
        def optimization_function(trial: optuna.Trial):
            model_hparams = dict(embedding_dim=trial.suggest_categorical('embedding_dim', [8, 16, 32, 64, 128]),
                                 d_model=trial.suggest_categorical('d_model', [8, 16, 32, 64, 128, 256, 512, 1024]),
                                 num_layers=trial.suggest_int('num_layers', 1, 5),
                                 bidirectional=trial.suggest_categorical('bidirectional', [True, False]),
                                 dropout=trial.suggest_float('dropout', 0, 1),
                                 learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                                 weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True))

            heldout_roc_auc = train(train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, test_dataloader=heldout_dataloader, writer=tb_writer,
                                max_epochs=max_epochs, model_class=RNNPredictor, model_args=tuple(), model_kwargs=model_kwargs, model_hparams=model_hparams)
            return heldout_roc_auc

        study = optuna.create_study(direction='maximize')
        study.optimize(optimization_function, n_trials=20)

        tb_writer.close()

Here we are using most of optunas variable types. We're using the :code:`suggest_categorical` 
method to sample from a set of arbitrary python objects. We could have used :code:`suggest_int`
for the :code:`embedding_dim` and :code:`d_model` hyper parameters, but by supplying a logits
we're able to focus specific orders of magnitude instead.

For the :code:`learning_rate` and :code:`weight_decay` parameters, we want to 
explore the values geometrically, so we set the attribute :code:`log=True`. This samples the values 
from a log-transformed space instead, so that we for example are roughly as likely to sample values in the range
:math:`[10^{-4},10^{-3}]` as in the range :math:`[10^{-3},10^{-2}]`. If we don't do this, our sampling 
will be skewed towards larger values.

We have changed our basic version of the training to automatically search for hyper paramters using Optuna.