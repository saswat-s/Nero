from logging import Logger

from ontolearn import KnowledgeBase

from typing import List, Tuple, Set, Dict
from owlapy.model import OWLEquivalentClassesAxiom, OWLClass, IRI, OWLObjectIntersectionOf, OWLObjectUnionOf, \
    OWLObjectSomeValuesFrom, OWLObjectInverseOf, OWLObjectProperty, OWLThing

from ontolearn.learning_problem_generator import LearningProblemGenerator
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.binders import DLLearnerBinder

from .static_funcs import *
from .util_classes import *
from .trainer import Trainer

import numpy as np
import pandas as pd
from collections import deque
import json
import os
from random import randint


class Experiment:
    def __init__(self, args):
        self.storage_path, _ = create_experiment_folder(folder_name='Experiments')
        self.logger = create_logger(name='Experimenter', p=self.storage_path)

        self.args = args
        self.logger.info('Knowledge Base being Initialized')
        # Initialize instances to conduct the experiment
        self.kb = KnowledgeBase(path=self.args['path_knowledge_base'],
                                reasoner_factory=ClosedWorld_ReasonerFactory)
        assert self.kb.individuals_count() > 0
        self.logger.info('Learning Problems being generated')
        # @TODO: Generate and label examples on the fly
        self.lp = LP(**generate_training_data(self.kb, self.args, logger=self.logger))
        self.args['storage_path'] = self.storage_path
        self.logger.info('Trainer initialized')
        self.trainer = Trainer(knowledge_base=self.kb, learning_problems=self.lp, args=self.args, logger=self.logger)
        self.instance_str = list(self.lp.instance_idx_mapping.keys())

    def start(self):
        self.logger.info('Experiment starts')

        # (1) Train NCEL

        ncel = self.trainer.start()
        """
        lp = [(random.choices(self.instance_str, k=self.args['num_individual_per_example']),
               random.choices(self.instance_str, k=self.args['num_individual_per_example'])) for _ in
              range(self.args['num_of_learning_problems_testing'])]
        """
        with open(self.args['path_lp']) as json_file:
            settings = json.load(json_file)
        lp = [(list(v['positive_examples']), list(v['negative_examples'])) for k, v in
              settings['problems'].items()]

        # (2) Evaluate NCEL
        self.evaluate(ncel, lp, self.args)

    def evaluate(self, ncel, lp, args):
        self.logger.info('Evaluation Starts')

        ncel_results = dict()
        celoe_results = dict()

        for _, (p, n) in enumerate(lp):
            ncel_report = ncel.fit(pos=p, neg=n, topK=args['topK'])
            ncel_report.update({'P': p, 'N': n, 'F-measure': f_measure(instances=ncel_report['Instances'],
                                                                       positive_examples=set(p),
                                                                       negative_examples=set(n)),
                                })

            ncel_results[_] = ncel_report
            if args['eval_dl_learner']:
                celoe = DLLearnerBinder(binary_path=args['dl_learner_binary_path'], kb_path=args['path_knowledge_base'],
                                        model='celoe')
                best_pred_celoe = celoe.fit(pos=p, neg=n, max_runtime=3).best_hypothesis()
                celoe_results[_] = {'P': p, 'N': n,
                                    'Prediction': best_pred_celoe['Prediction'],
                                    'F-measure': best_pred_celoe['F-measure'],
                                    'NumClassTested': best_pred_celoe['NumClassTested'],
                                    'Runtime': best_pred_celoe['Runtime'],
                                    }
        avg_f1_ncel = np.array([i['F-measure'] for i in ncel_results.values()]).mean()
        avg_runtime_ncel = np.array([i['Runtime'] for i in ncel_results.values()]).mean()
        avg_expression_ncel = np.array([i['NumClassTested'] for i in ncel_results.values()]).mean()
        self.logger.info(
            f'Average F-measure NCEL:{avg_f1_ncel}\t Avg. Runtime:{avg_runtime_ncel}\t Avg. Expression Tested:{avg_expression_ncel} in {len(lp)} ')
        if args['eval_dl_learner']:
            avg_f1_celoe = np.array([i['F-measure'] for i in celoe_results.values()]).mean()
            avg_runtime_celoe = np.array([i['Runtime'] for i in celoe_results.values()]).mean()
            avg_expression_celoe = np.array([i['NumClassTested'] for i in celoe_results.values()]).mean()

            self.logger.info(
                f'Average F-measure CELOE:{avg_f1_celoe}\t Avg. Runtime:{avg_runtime_celoe}\t Avg. Expression Tested:{avg_expression_celoe}')
