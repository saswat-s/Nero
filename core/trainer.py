from ontolearn import KnowledgeBase
from typing import List, Tuple, Set
from owlapy.model import OWLEquivalentClassesAxiom, OWLClass, IRI, OWLObjectIntersectionOf, OWLObjectUnionOf, \
    OWLObjectSomeValuesFrom, OWLObjectInverseOf, OWLObjectProperty, OWLThing
from argparse import ArgumentParser

from owlapy.render import DLSyntaxObjectRenderer

import random
from collections import deque
from .model import NCEL

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
        # self.renderer = DLSyntaxObjectRenderer()
        self.logger = logger
        self.storage_path = self.args['storage_path']

    def describe_configuration(self, model):
        """
        Describe the selected model and hyperparameters
        :param model:
        :return:
        """
        self.logger.info(f'Training data size:{len(self.learning_problems)}\t'
                         f'Number of Labels:{len(self.learning_problems.target_class_expressions)}')

        num_param = sum([p.numel() for p in model.parameters()])
        self.logger.info("Number of free parameters: {0}".format(num_param))
        self.logger.info(model)

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

        return NCEL(model=model,
                    quality_func=f_measure,
                    target_class_expressions=self.learning_problems.target_class_expressions,
                    instance_idx_mapping=self.learning_problems.instance_idx_mapping)

    def training_loop(self):

        # (1) Initialize the model.
        model = self.neural_architecture_selection()
        # (2) Describe the training setting.
        self.describe_configuration(model)
        # (3) Initialize the training. MSE seemed to yield better results, less num of concepts explored
        loss_func = torch.nn.MSELoss()
        # loss_func = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.args['learning_rate'])
        self.logger.info('Data being labelled')
        # (4) Initialize the mini-batch loader
        """
        # DatasetWithOnFlyLabelling uses less memory but takes too much time although all cpus are used
        data_loader = torch.utils.data.DataLoader(DatasetWithOnFlyLabelling(self.learning_problems),
                                                  batch_size=self.args['batch_size'],
                                                  num_workers=self.args['num_workers'], shuffle=True)
        
        """
        self.logger.info('Training starts')
        # (1) Set model in training mode.
        model.train()
        # (2) Send model to selected device.
        model.to(self.device)
        # (3) Store average loss per epoch
        losses = []
        # (4) Start training loop
        printout_constant = (self.args['num_epochs'] // 10) + 1
        if self.args['num_epochs'] > 0:
            data_loader = torch.utils.data.DataLoader(Dataset(self.learning_problems),
                                                      batch_size=self.args['batch_size'],
                                                      num_workers=self.args['num_workers'], shuffle=True)

            start_time = time.time()
            # For every some epochs, we should change the size of input
            for it in range(1, self.args['num_epochs'] + 1):
                epoch_loss = 0
                # (5) Mini-batch.
                for xpos, xneg, y in data_loader:
                    # (5.1) Send the batch into device.
                    xpos, xneg, y = xpos.to(self.device), xneg.to(self.device), y.to(self.device)
                    # (5.2) Zero the parameter gradients.
                    optimizer.zero_grad()
                    # (5.3) Forward.
                    predictions = model.forward(xpos=xpos, xneg=xneg)
                    # (5.4) Compute Loss.
                    batch_loss = loss_func(y, predictions)
                    epoch_loss += batch_loss.item()
                    # (5.5) Backward loss.
                    batch_loss.backward()
                    # (5.6) Update parameters according.
                    optimizer.step()
                # (6) Store epoch loss
                losses.append(epoch_loss)
                # (7) Print-out
                if it % printout_constant == 0:
                    self.logger.info(f'{it}.th epoch loss: {epoch_loss}')

                if it % self.args['val_at_every_epochs'] == 0:
                    self.validate(model, lp=self.learning_problems, args={'topK': 100})
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

    def validate(self, ncel, lp, args):
        self.logger.info('Validation Starts')
        ncel.eval()
        ncel_results = dict()

        for _, (p, n) in enumerate(lp):
            ncel_report = ncel.fit(pos=p, neg=n, topK=args['topK'], local_search=False)
            ncel_report.update({'P': p, 'N': n, 'F-measure': f_measure(instances=ncel_report['Instances'],
                                                                       positive_examples=set(p),
                                                                       negative_examples=set(n)),
                                })

            ncel_results[_] = ncel_report
        avg_f1_ncel = np.array([i['F-measure'] for i in ncel_results.values()]).mean()
        avg_runtime_ncel = np.array([i['Runtime'] for i in ncel_results.values()]).mean()
        avg_expression_ncel = np.array([i['NumClassTested'] for i in ncel_results.values()]).mean()
        self.logger.info(
            f'Avg. F-measure NCEL:{avg_f1_ncel}\t Avg. Runtime:{avg_runtime_ncel}\t Avg. Expression Tested:{avg_expression_ncel} in {len(lp)} LPs ')
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

    def start(self) -> NCEL:
        self.logger.info('Start the training phase')
        # (1) Training loop
        model = self.training_loop()
        self.serialize_ncel(model)

        return model
