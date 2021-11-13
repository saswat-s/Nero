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
        self.lp = LP(**generate_training_data(self.kb, self.args,logger=self.logger))
        self.args['storage_path'] = self.storage_path
        self.logger.info('Trainer initialized')
        self.trainer = Trainer(knowledge_base=self.kb, learning_problems=self.lp, args=self.args, logger=self.logger)

    def start(self):
        self.logger.info('Experiment starts')

        # (1) Train NCEL
        ncel = self.trainer.start()
        # (2) Evaluate NCEL
        self.evaluate(ncel, self.lp, self.args)

    def evaluate(self, ncel, lp, args):
        self.logger.info('Evaluation Starts')
        str_all_targets = [i for i in ncel.get_target_class_expressions()]

        instance_str = list(lp.instance_idx_mapping.keys())
        # (1) Enter the absolute path of the input knowledge base
        # (3) Initialize CELOE, OCEL, and ELTL
        celoe = DLLearnerBinder(binary_path=args['dl_learner_binary_path'], kb_path=args['path_knowledge_base'],
                                model='celoe')
        # (4) Fit (4) on the learning problems and show the best concept.
        ncel_results = dict()
        celoe_results = dict()

        for _ in range(args['num_of_learning_problems_testing']):
            # Variable
            #p = random.choices(instance_str, k=randint(1, args['max_num_individual_per_example']))
            #n = random.choices(instance_str, k=randint(1, args['max_num_individual_per_example']))
            p = random.choices(instance_str, k=args['num_individual_per_example'])
            n = random.choices(instance_str, k=args['num_individual_per_example'])

            ncel_report = ncel.fit(pos=p, neg=n, topK=args['topK'])
            ncel_report.update({'P': p, 'N': n, 'F-measure': f_measure(instances=ncel_report['Instances'],
                                                                       positive_examples=set(p),
                                                                       negative_examples=set(n)),
                                })

            ncel_results[_] = ncel_report
            best_pred_celoe = celoe.fit(pos=p, neg=n, max_runtime=1).best_hypothesis()
            if best_pred_celoe['Prediction'] in str_all_targets:
                pass
            else:
                self.logger.info(f'{best_pred_celoe["Prediction"]} not found in labels')
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
            f'Average F-measure NCEL:{avg_f1_ncel}\t Avg. Runtime:{avg_runtime_ncel}\t Avg. Expression Tested:{avg_expression_ncel} in {args["num_of_learning_problems_testing"]} randomly generated learning problems')
        if len(celoe_results) > 0:
            avg_f1_celoe = np.array([i['F-measure'] for i in celoe_results.values()]).mean()
            avg_runtime_celoe = np.array([i['Runtime'] for i in celoe_results.values()]).mean()
            avg_expression_celoe = np.array([i['NumClassTested'] for i in celoe_results.values()]).mean()

            self.logger.info(
                f'Average F-measure CELOE:{avg_f1_celoe}\t Avg. Runtime:{avg_runtime_celoe}\t Avg. Expression Tested:{avg_expression_celoe}')
