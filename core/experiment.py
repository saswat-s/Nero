import random
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
from .dl_expression import TargetClassExpression
import numpy as np
import pandas as pd
from collections import deque
import os
from random import randint
import time
import gc


class Experiment:
    """ Main class for conducting experiments """

    def __init__(self, args):
        self.args = args
        # (1) Create Logging & Experiment folder for serialization
        self.storage_path, _ = create_experiment_folder(folder_name='Experiments')
        self.logger = create_logger(name='Experimenter', p=self.storage_path)
        self.args['storage_path'] = self.storage_path

        # (2) Initialize KB.
        kb = self.initialize_knowledge_base()
        # (2) Initialize Training Data (D: {(E^+,E^-)})_i ^N .
        self.lp = self.construct_targets_and_problems(kb)
        # (4) Init Trainer.
        self.trainer = Trainer(learning_problems=self.lp, args=self.args, logger=self.logger)

        self.instance_str = list(self.lp.instance_idx_mapping.keys())
        self.describe_and_store()
        del kb
        gc.collect()

    def initialize_knowledge_base(self):
        self.logger.info(f"Knowledge Base being Initialized {self.args['path_knowledge_base']}")
        kb = KnowledgeBase(path=self.args['path_knowledge_base'],
                           reasoner_factory=ClosedWorld_ReasonerFactory)
        # (2.1) Store some info about KB
        self.args['num_instances'] = kb.individuals_count()
        self.args['num_named_classes'] = len([i for i in kb.ontology().classes_in_signature()])
        self.args['num_properties'] = len([i for i in itertools.chain(kb.ontology().data_properties_in_signature(),
                                                                      kb.ontology().object_properties_in_signature())])
        # (2.2) Log some info about data
        self.logger.info(f'Number of individuals: {self.args["num_instances"]}')
        self.logger.info(f'Number of named classes / expressions: {self.args["num_named_classes"]}')
        self.logger.info(f'Number of properties / roles : {self.args["num_properties"]}')
        try:
            assert self.args['num_instances'] > 0
        except AssertionError:
            print(f'Number of entities can not be 0, *** {self.args["num_instances"]}')
            print('Background knowledge should be OWL 2.')
            exit(1)
        return kb

    def construct_targets_and_problems(self, kb: KnowledgeBase) -> LP:
        # (3) Initialize Learning problems
        target_class_expressions, instance_idx_mapping = select_target_expressions(kb, self.args, logger=self.logger)
        # e_pos, e_neg = generate_random_learning_problems(instance_idx_mapping, self.args)
        e_pos, e_neg = generate_learning_problems_from_targets(target_class_expressions, instance_idx_mapping,
                                                               self.args, logger=self.logger)
        lp = LP(e_pos=e_pos, e_neg=e_neg, instance_idx_mapping=instance_idx_mapping,
                target_class_expressions=target_class_expressions)
        self.logger.info(lp)
        return lp

    def describe_and_store(self) -> None:
        assert self.args['num_instances'] > 0
        self.logger.info('Experimental Setting is being serialized.')
        if torch.cuda.is_available():
            self.logger.info('Name of selected Device:{0}'.format(torch.cuda.get_device_name(self.trainer.device)))
        # (1) Store Learning Problems
        self.logger.info('Serialize Learning Problems.')
        save_as_json(storage_path=self.storage_path,
                     obj={i: {'Pos': e_pos, 'Neg': e_neg} for i, (e_pos, e_neg) in
                          enumerate(zip(self.lp.e_pos, self.lp.e_neg))},
                     name='training_learning_problems')
        self.logger.info('Serialize Index of Instances.')
        # (2) Store Integer mapping of instance: index of individuals
        save_as_json(storage_path=self.storage_path,
                     obj=self.lp.instance_idx_mapping, name='instance_idx_mapping')
        # (3) Store Target Class Expressions with respective expression chain from T -> ... -> TargetExp
        # Instead of storing as list of objects, we can store targets as pandas dataframe
        self.logger.info('Serialize Pandas Dataframe containing target expressions')
        df = pd.DataFrame([t.__dict__ for t in self.lp.target_class_expressions])
        df.to_csv(path_or_buf=self.storage_path + '/target_class_expressions.csv')
        # print(total_size(self.lp.target_class_expressions))
        # print(df.memory_usage(deep=True).sum())
        # Pandas require more memory than self.lp.target_class_expressions or our memory calculating of a list of
        # items in correct
        del df
        gc.collect()
        self.logger.info('Serialize Targets as json.')
        save_as_json(storage_path=self.storage_path, obj={target_cl.label_id: {'label_id': target_cl.label_id,
                                                                               'name': target_cl.name,
                                                                               'expression_chain': target_cl.expression_chain,
                                                                               'idx_individuals': list(
                                                                                   target_cl.idx_individuals),
                                                                               }
                                                          for target_cl in self.lp.target_class_expressions},
                     name='target_class_expressions')

        self.args['num_outputs'] = len(self.lp.target_class_expressions)
        # (4) Store input settings
        save_as_json(storage_path=self.storage_path, obj=self.args, name='settings')
        # (5) Log details about input KB.

    def start(self):
        # (1) Train model & Validate
        self.logger.info('Experiment starts')
        start_time = time.time()
        ncel = self.trainer.start()

        if self.args['path_lp'] is not None:
            # (2) Load learning problems
            with open(self.args['path_lp']) as json_file:
                settings = json.load(json_file)
            lp = [(k, list(v['positive_examples']), list(v['negative_examples'])) for k, v in
                  settings['problems'].items()]

            # (2) Evaluate model
            self.evaluate(ncel, lp, self.args)

        self.logger.info(f'Total Runtime of the experiment:{time.time() - start_time}')

    def evaluate(self, ncel, lp, args)->None:
        self.logger.info('Evaluation Starts')

        ncel_results = dict()
        celoe_results = dict()
        # (1) Iterate over input learning problems.
        for _, (goal_exp, p, n) in enumerate(lp):
            ncel_report = ncel.fit(pos=p, neg=n, topK=args['topK'], local_search=False)
            ncel_report.update({'P': p, 'N': n, 'F-measure': f_measure(instances=ncel_report['Instances'],
                                                                       positive_examples=set(p),
                                                                       negative_examples=set(n)),
                                })
            ncel_report.update({'Target': goal_exp})
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
        if len(ncel_results) < 50:
            for k, v in ncel_results.items():
                self.logger.info(
                    f'{k}.th test LP:\tTarget:{v["Target"]}\tPred:{v["Prediction"]}\tF-measure:{v["F-measure"]:.3f}\tNumClassTested:{v["NumClassTested"]}')

        # Overall Preport
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
        self.logger.info('Evaluation Ends')
