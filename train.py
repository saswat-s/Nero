"""
====================================================================
Deep Tunnelling
====================================================================
Drill with training.
Authors: Caglar Demir

(1) Parse input knowledge base
(2) Generate 100_000 Class Expressions that are "informative"
    # (3) Generate learning problems, where a learning problem E is a set of examples/instance
    # considered as positive examples. E has variable Size
    # (4) Extend (3) by adding randomly sampled examples
    # (5) Let D denotes the set generated in (3) and (4)
    # (6) For each learning problem X_i, compute Y_i that is a vector of F1-scores
    # (7) Summary: D_i = { ({e_x, e_y, ...e_z }_i ,Y_i) }_i=0 ^N has variable size, Y_i has 10^5 size
    # (8) For each D_i, let x_i denote input set and Y_i label
    # (9) Let \mathbf{x_i} \in R^{3,D} represent the mean, median, and sum of X_i; permutation invariance baby :)
    # (10) Train sigmoid(net) to minimize binary cross entropy; multi-label classification problem

    # We can use (10) for
    #                 class expression learning
    #                 ?Explainable Clustering ?
    #                 ?link prediction? (h,r,x) look the range of  top10K(Y_i)
https://github.com/RDFLib/sparqlwrapper/blob/master/scripts/example-dbpedia.py
"""
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

random.seed(0)


class Trainer:
    def __init__(self, knowledge_base: KnowledgeBase, learning_problem_generator: LearningProblemGenerator, args):
        self.knowledge_base = knowledge_base
        self.learning_problem_generator = learning_problem_generator
        # List of URIs representing instances / individuals
        self.instances = None
        self.args = args
        # Create an experiment folder
        self.storage_path, _ = create_experiment_folder(folder_name='Experiments')
        self.logger = create_logger(name='Trainer', p=self.storage_path)
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

    def training_loop(self, model, loss_func, optimizer, data_loader, num_epochs):
        self.logger.info('Training Loop starts')
        # (1) Set model in training mode.
        model.train()
        # (2) Send model to selected device.
        model.to(self.device)
        # (3) Store average loss per epoch
        losses = []
        # (4) Start training loop
        printout_constant = (num_epochs // 20) + 1
        start_time = time.time()
        for it in range(1, num_epochs + 1):
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
        target_individuals = self.generate_class_expressions()
        data_loader = self.generate_training_data(target_individuals)

        # Model definition
        model = DT(param={'num_embedding_dim': self.args.num_embedding_dim,
                          'num_instances': len(self.instances),
                          'num_outputs': len(target_individuals)})

        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        model = self.training_loop(model, loss_func, optimizer, data_loader, self.args.num_epochs)

        df = pd.DataFrame(model.embeddings.weight.data.detach().numpy(), index=self.instances)

        df.to_csv(self.storage_path + '/instance_embeddings.csv')

def main(args):
    """

    :param args:
    :return:
    """
    # (1) Parse input KB
    kb = KnowledgeBase(path=args.path_knowledge_base,
                       reasoner_factory=ClosedWorld_ReasonerFactory)
    # (3) Generate Class Expressions semi-randomly
    lpg = LearningProblemGenerator(knowledge_base=kb,
                                   min_length=args.min_length,
                                   max_length=args.max_length,
                                   min_num_instances=args.min_num_instances_ratio_per_concept * kb.individuals_count(),
                                   max_num_instances=args.max_num_instances_ratio_per_concept * kb.individuals_count())

    trainer = Trainer(knowledge_base=kb, learning_problem_generator=lpg, args=args)
    trainer.start()


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str,
                        default='KGs/Family/family-benchmark_rich_background.owl'
                        )
    # Concept Generation Related
    parser.add_argument("--min_length", type=int, default=0, help='Min length of concepts to be used')
    parser.add_argument("--max_length", type=int, default=5, help='Max length of concepts to be used')
    parser.add_argument("--min_num_instances_ratio_per_concept", type=float, default=.01)
    parser.add_argument("--max_num_instances_ratio_per_concept", type=float, default=.90)

    # Neural related
    parser.add_argument("--input_set_size", type=int, default=10, help='Input set size for expression learning.')
    parser.add_argument("--num_of_data_points", type=int, default=1000, help='Total number of randomly sampled training data points')
    parser.add_argument("--num_embedding_dim", type=int, default=25, help='Number of embedding dimensions.')
    parser.add_argument("--learning_rate", type=int, default=.01, help='Learning Rate')
    parser.add_argument("--num_epochs", type=int, default=50, help='Number of iterations over the entire dataset.')
    parser.add_argument("--batch_size", type=int, default=1024)
    main(parser.parse_args())
