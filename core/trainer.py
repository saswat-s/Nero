from ontolearn import KnowledgeBase
from typing import List, Tuple, Set
from owlapy.model import OWLEquivalentClassesAxiom, OWLClass, IRI, OWLObjectIntersectionOf, OWLObjectUnionOf, \
    OWLObjectSomeValuesFrom, OWLObjectInverseOf, OWLObjectProperty, OWLThing
from argparse import ArgumentParser

from owlapy.render import DLSyntaxObjectRenderer

import random
from collections import deque
from .model import NERO

import torch
from torch import nn
import numpy as np
from .static_funcs import *
from .util_classes import *
from .neural_arch import *
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Trainer:
    """ Trainer of Neural Class Expression Learner"""

    def __init__(self, learning_problems, args, logger):
        # self.knowledge_base = knowledge_base
        self.learning_problems = learning_problems
        # List of URIs representing instances / individuals
        self.instances = None
        # Input arguments
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.storage_path = self.args['storage_path']
        self.logger.info(self)

        if self.args['quality_function_training'] == 'accuracy':
            self.quality_function = accuracy
        elif self.args['quality_function_training'] == 'fmeasure':
            self.quality_function = f_measure
        else:
            raise KeyError

    def __str__(self):
        return f'Trainer: |C|={self.args["num_named_classes"]},' \
               f'|I|={self.args["num_instances"]},' \
               f'|D|={len(self.learning_problems)},' \
               f'|T|={len(self.learning_problems.target_class_expressions)},' \
               f'd:{self.args["num_embedding_dim"]},' \
               f'Quality func for training:{self.args["quality_function_training"]},' \
               f'NumEpoch={self.args["num_epochs"]},' \
               f'LR={self.args["learning_rate"]},' \
               f'BatchSize={self.args["batch_size"]},' \
               f'Device:{self.device}'

    def describe_configuration(self, model, loss_func):
        """
        Describe the selected model and hyperparameters
        :param loss_func:
        :param model:
        :return:
        """
        self.logger.info(f'{model},d:{self.args["num_embedding_dim"]}, '
                         f'|Theta|={sum([p.numel() for p in model.parameters()])}, '
                         f'Loss={loss_func}')

    def neural_architecture_selection(self):
        param = {'num_embedding_dim': self.args['num_embedding_dim'],
                 'num_instances': self.args['num_instances'],
                 'num_outputs': len(self.learning_problems.target_class_expressions)}

        arc = self.args['neural_architecture']
        if arc == 'DeepSet':
            model = DeepSet(param)
        elif arc == 'DeepSetBase':
            model = DeepSetBase(param)
        elif arc == 'ST':
            model = ST(param)
        else:
            raise NotImplementedError(f'There is no {arc} model implemented')

        return NERO(model=model,
                    quality_func=self.quality_function,
                    target_class_expressions=self.learning_problems.target_class_expressions,
                    instance_idx_mapping=self.learning_problems.str_individuals_to_idx)

    def select_loss_and_optim(self, model_params)->Tuple:
        # During our training, using MSE results in better results, less num of concepts explored
        if self.args['loss_func'] == 'MSELoss':
            loss_func = torch.nn.MSELoss()
        elif self.args['loss_func'] == 'CrossEntropyLoss':
            loss_func = torch.nn.CrossEntropyLoss()
        elif self.args['loss_func'] == 'HuberLoss':
            loss_func = torch.nn.HuberLoss()
        else:
            raise KeyError

        optimizer = torch.optim.Adam(model_params, lr=self.args['learning_rate'])
        return loss_func, optimizer

    def training_loop(self):
        """
         (1) Initialize the model
         (2) Select the loss function and the optimizer
         (3)

        Parameter: None
        ---------

        Returns: NERO
        ---------

        """
        # (1) Initialize the model.
        model = self.neural_architecture_selection()
        # (2) Select the loss function and Optimizer.
        loss_func, optimizer = self.select_loss_and_optim(model_params=model.parameters())
        # (1) Set model in training mode.
        model.train()
        # (2) Send model to selected device.
        model.to(self.device)
        # (3) Store average loss per epoch
        losses = []
        # (4) Start training loop
        printout_constant = (self.args['num_epochs'] // 10) + 1
        if self.args['num_epochs'] > 0:
            self.logger.info('Data being labelled')
            # (4.1) Initialize the mini-batch loader
            data_loader = torch.utils.data.DataLoader(
                Dataset(self.learning_problems, num_workers_for_labelling=self.args['num_workers']),
                batch_size=self.args['batch_size'],
                num_workers=self.args['num_workers'], shuffle=True)

            # (4.2) Describe the training setting.
            self.logger.info('Training starts.')
            self.describe_configuration(model, loss_func)
            # Validation on randomly sampled 1 percent of the data
            num_val_lp = 1 + len(self.learning_problems) // 100
            start_time = time.time()
            model.train()

            # (5) Iterate training data
            for it in range(1, self.args['num_epochs'] + 1):
                epoch_loss = 0
                # (6) Mini-batch.
                for xpos, xneg, y in data_loader:
                    # (6.1) Send the batch into device.
                    xpos, xneg, y = xpos.to(self.device), xneg.to(self.device), y.to(self.device)
                    # (6.2) Zero the parameter gradients.
                    optimizer.zero_grad()
                    # (6.3) Forward.
                    predictions = model.forward(xpos=xpos, xneg=xneg)
                    # (6.4) Compute Loss.
                    batch_loss = loss_func(input=predictions, target=y)
                    epoch_loss += batch_loss.item()
                    # (6.5) Backward loss.
                    batch_loss.backward()
                    # (6.6) Update parameters according.
                    optimizer.step()
                # (7) Store epoch loss
                losses.append(epoch_loss)
                # (8) Print-out
                if it % printout_constant == 0:
                    self.logger.info(f'{it}.th epoch loss: {epoch_loss}')
                if it % self.args['val_at_every_epochs'] == 0:
                    # At each time sample 10% of the training data.
                    # or lp=random.choices(self.learning_problems, k=num_val_lp)
                    self.validate(model, [self.learning_problems[i] for i in range(num_val_lp)], args={'topK': 3},
                                  info='Validation on Training Data Starts')
                    model.train()

            training_time = time.time() - start_time
            # Save
            self.logger.info(f'TrainingRunTime {training_time / 60:.3f} minutes')
            self.logger.info('Save the loss epoch trajectory')
            np.savetxt(fname=self.storage_path + "/loss_per_epoch.csv", X=np.array(losses), delimiter=",")
            self.logger.info('Save Weights')
            save_weights(model, self.storage_path)

        model.eval()
        self.logger.info('Training Loop ends')
        return model

    def validate(self, ncel, lp, args, info):
        self.logger.info(f'{info}')
        ncel.eval()
        ncel_results = dict()

        for _, (p, n) in enumerate(lp):
            with torch.no_grad():
                # use_search=args['use_search'],kb_path=args['path_knowledge_base']
                ncel_report = ncel.fit(str_pos=p, str_neg=n, topK=args['topK'])
            ncel_report.update({'P': p, 'N': n, 'F-measure': f_measure(instances=ncel_report['Instances'],
                                                                       positive_examples=set(p),
                                                                       negative_examples=set(n)),
                                })

            ncel_results[_] = ncel_report
        avg_f1_ncel = np.array([i['F-measure'] for i in ncel_results.values()]).mean()
        avg_runtime_ncel = np.array([i['Runtime'] for i in ncel_results.values()]).mean()
        avg_expression_ncel = np.array([i['NumClassTested'] for i in ncel_results.values()]).mean()
        self.logger.info(
            f'Avg. F-measure NERO:{avg_f1_ncel}\t Avg. Runtime:{avg_runtime_ncel}\t Avg. Expression Tested:{avg_expression_ncel} in {len(lp)} LPs ')
        self.logger.info('Validation Ends')

    def serialize_ncel(self, model):
        # (2) Serialize model and weights
        embeddings = model.embeddings_to_numpy()
        df = pd.DataFrame(embeddings, index=self.instances)
        df.to_csv(self.storage_path + '/instance_embeddings.csv')

        if self.args['plot_embeddings'] > 0:
            low_emb = PCA(n_components=2).fit_transform(embeddings)
            plt.scatter(low_emb[:, 0], low_emb[:, 1])
            plt.show()

    def start(self) -> NERO:
        """
         Train and Store

        Parameter: None
        ---------

        Returns: NERO
        ---------

        """
        self.logger.info('Start the training phase')
        # (1) Training loop
        model = self.training_loop()
        self.serialize_ncel(model)
        return model
