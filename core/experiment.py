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
import os
from random import randint


class Experiment:
    def __init__(self, args):
        self.args = args

        # (1) Create Logging & Experiment folder for serialization
        self.storage_path, _ = create_experiment_folder(folder_name='Experiments')
        self.logger = create_logger(name='Experimenter', p=self.storage_path)
        self.args['storage_path'] = self.storage_path

        # (2) Initialize KB
        self.logger.info('Knowledge Base being Initialized')
        kb = KnowledgeBase(path=self.args['path_knowledge_base'],
                           reasoner_factory=ClosedWorld_ReasonerFactory)
        self.args['num_instances'] = kb.individuals_count()
        self.args['num_named_classes'] = len([i for i in kb.ontology().classes_in_signature()])

        # (3) Initialize Learning problems
        self.logger.info('Learning Problems being generated')
        self.lp = LP(**generate_training_data(kb, self.args, logger=self.logger))
        del kb

        # (4) Init Trainer
        self.trainer = Trainer(learning_problems=self.lp, args=self.args, logger=self.logger)

        self.instance_str = list(self.lp.instance_idx_mapping.keys())
        self.__describe_and_store()

    def __describe_and_store(self):
        assert self.args['num_instances'] > 0
        # Sanity checking
        renderer = DLSyntaxObjectRenderer()
        # cuda device
        self.logger.info('Device:{0}'.format(self.trainer.device))
        if torch.cuda.is_available():
            self.logger.info('Name of selected Device:{0}'.format(torch.cuda.get_device_name(self.trainer.device)))

        save_as_json(storage_path=self.storage_path,
                     obj={i: {'Pos': e_pos, 'Neg': e_neg} for i, (e_pos, e_neg) in
                          enumerate(zip(self.lp.e_pos, self.lp.e_neg))},
                     name='training_learning_problems')

        save_as_json(storage_path=self.storage_path,
                     obj=self.lp.instance_idx_mapping, name='instance_idx_mapping')
        save_as_json(storage_path=self.storage_path, obj={i: {'DL-Syntax': renderer.render(cl.concept),
                                                              'ExpressionChain': [renderer.render(_.concept) for _ in
                                                                                  retrieve_concept_chain(cl)]}
                                                          for i, cl in
                                                          enumerate(self.lp.target_class_expressions)},
                     name='target_class_expressions')
        self.logger.info('Learning Problem object serialized')
        save_as_json(storage_path=self.storage_path, obj=self.args, name='settings')

        self.logger.info('Describe the experiment')
        self.logger.info(
            f'Number of named classes: {self.args["num_named_classes"]}\t'
            f'Number of individuals: {self.args["num_instances"]}'
        )

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
            ncel_report = ncel.fit(pos=p, neg=n, topK=args['topK'], local_search=False)
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
