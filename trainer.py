from ontolearn import KnowledgeBase
from typing import List, Tuple, Set
from owlapy.model import OWLEquivalentClassesAxiom, OWLClass, IRI, OWLObjectIntersectionOf, OWLObjectUnionOf, \
    OWLObjectSomeValuesFrom, OWLObjectInverseOf, OWLObjectProperty, OWLThing
from argparse import ArgumentParser

from ontolearn.learning_problem_generator import LearningProblemGenerator
from owlapy.render import DLSyntaxObjectRenderer

import random
from collections import deque
from model import DT

import torch
from torch import nn
import numpy as np
from static_funcs import *
from util_classes import *
import json
import pandas as pd
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, knowledge_base: KnowledgeBase, learning_problem_generator: LearningProblemGenerator, args):
        self.knowledge_base = knowledge_base
        self.learning_problem_generator = learning_problem_generator
        # List of URIs representing instances / individuals
        self.instances = None
        # Input arguments
        self.args = args
        # Create an experiment folder
        self.storage_path, _ = create_experiment_folder(folder_name='Experiments')
        self.logger = create_logger(name='Trainer', p=self.storage_path)
        # cuda device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info('Device:{0}'.format(self.device))
        if torch.cuda.is_available():
            self.logger.info('Name of selected Divice:{0}'.format(torch.cuda.get_device_name(self.device)))

    def save_as_json(self, obj,name=None):
        with open(self.storage_path + f'/{name}.json', 'w') as file_descriptor:
            json.dump(obj, file_descriptor, indent=3)

    def generate_class_expressions(self) -> List[Set[str]]:
        """
        Generate Target Expressions
        :return: Targets and Instances
        """
        self.logger.info('generate_class_expressions')
        renderer = DLSyntaxObjectRenderer()
        # (3.1) Store generated class expressions
        target_expressions = set()
        for valid_rl_state in self.learning_problem_generator.generate_examples(num_problems=1000, max_length=100,
                                                                                min_length=1, num_diff_runs=10,
                                                                                min_num_instances=10):
            target_expressions.add(valid_rl_state)
        # Target class expressions, ALC formulas
        target_expressions: List[RL_State] = sorted(list(target_expressions), key=lambda x: x.length,
                                                    reverse=False)
        # All instances belonging to targets
        target_individuals: List[Set[str]] = [{i.get_iri().as_str() for i in s.instances} for s in
                                              target_expressions]
        # Save after convert it to list
        self.save_as_json([[renderer.render(target_expressions[idx].concept), list(examples)] for idx, examples in
                           enumerate(target_individuals)],name='target_expressions')
        # @TODO Generate label/target hierarchy and stored it as json file This should containing bottom to up hierarchy
        return target_individuals

    def generate_training_data(self, target_instances):
        """

        :param target_instances:
        :return:
        """
        self.logger.info('generate_training_data')
        # (2) Convert instance into list of URIs
        # Could do it more efficiently
        self.instances = [i.get_iri().as_str() for i in self.knowledge_base.individuals()]
        instance_to_index = {indv: i for i, indv in enumerate(self.instances)}

        examples = []
        size_of_positive_example_set = self.args.input_set_size
        # (4) Generate RANDOM TRAINING DATA
        for i in range(self.args.num_of_data_points):
            # https://docs.python.org/3/library/random.html#random.choices
            examples.append(random.choices(self.instances, k=size_of_positive_example_set))


        X, Y = [], []
        for e in examples:
            # Positive examples to integer index
            X.append([instance_to_index[_] for _ in e])
            # Create labels/score for concepts
            Y.append([target_scores(target_instances=target_instances, positive_examples=set(e)) for target_instances in
                      target_instances])
        # (5) Send (4) into Dataloader to iterate on it in mini-batch fashion
        data_loader = torch.utils.data.DataLoader(Dataset(inputs=X, outputs=Y), batch_size=self.args.batch_size,
                                                  num_workers=4, shuffle=True)
        # Free some memory
        del X, Y

        return data_loader

    def training_loop(self, target_individuals):
        # (1) Generate Training Data Points
        data_loader = self.generate_training_data(target_individuals)

        # (2) Initialize DT
        model = DT(param={'num_embedding_dim': self.args.num_embedding_dim,
                          'num_instances': len(self.instances),
                          'num_outputs': len(target_individuals)})

        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)

        self.logger.info('Training Loop starts')
        # (1) Set model in training mode.
        model.train()
        # (2) Send model to selected device.
        model.to(self.device)
        # (3) Store average loss per epoch
        losses = []
        # (4) Start training loop
        printout_constant = (self.args.num_epochs // 20) + 1
        start_time = time.time()
        for it in range(1, self.args.num_epochs + 1):
            epoch_loss = 0
            # (5) Mini-batch.
            for x, y in data_loader:
                # (5.1) Send the batch into device.
                x, y = x.to(self.device), y.to(self.device)

                # (5.2) Zero the parameter gradients.
                optimizer.zero_grad()
                # (5.3) Forward.
                predictions = model(x)
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
        training_time = time.time() - start_time
        # Save
        self.logger.info(f'TrainingRunTime {training_time / 60:.3f} minutes')
        self.logger.info('Save the loss epoch trajectory')
        np.savetxt(fname=self.storage_path + "/loss_per_epoch.csv", X=np.array(losses), delimiter=",")
        self.logger.info('Save Weights')
        save_weights(model, self.storage_path)

        model.eval()
        return model

    def start(self):
        """
        Training starts
        :return:
        """
        self.logger.info('Start')
        # (1) .
        target_individuals = self.generate_class_expressions()
        # (4) L
        model = self.training_loop(target_individuals)

        embeddings=model.embeddings.weight.data.detach().numpy()
        df = pd.DataFrame(embeddings, index=self.instances)

        df.to_csv(self.storage_path + '/instance_embeddings.csv')

        if self.args.plot_embeddings>0:
            #import umap
            #reducer = umap.UMAP()
            #low_emb = reducer.fit_transform(embeddings)
            from sklearn.decomposition import PCA
            low_emb = PCA(n_components=2).fit_transform(embeddings)
            plt.scatter(low_emb[:, 0],low_emb[:, 1])
            plt.show()
