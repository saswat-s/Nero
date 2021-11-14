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
    """ Main class for conducting experiments """

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
        # cuda device
        self.logger.info('Device:{0}'.format(self.trainer.device))
        if torch.cuda.is_available():
            self.logger.info('Name of selected Device:{0}'.format(torch.cuda.get_device_name(self.trainer.device)))
        # (1) Store Learning Problems
        save_as_json(storage_path=self.storage_path,
                     obj={i: {'Pos': e_pos, 'Neg': e_neg} for i, (e_pos, e_neg) in
                          enumerate(zip(self.lp.e_pos, self.lp.e_neg))},
                     name='training_learning_problems')
        # (2) Store Integer mapping of instance: index of individuals
        save_as_json(storage_path=self.storage_path,
                     obj=self.lp.instance_idx_mapping, name='instance_idx_mapping')
        # (3) Store Target Class Expressions with respective expression chain from T -> ... -> TargetExp
        save_as_json(storage_path=self.storage_path, obj={i: {'DL-Syntax': target_cl.name,
                                                              'ExpressionChain': self.lp.expressions_chain[
                                                                  target_cl.name]}
                                                          for i, target_cl in
                                                          enumerate(self.lp.target_class_expressions)},
                     name='target_class_expressions')
        # (4) Store input settings
        save_as_json(storage_path=self.storage_path, obj=self.args, name='settings')
        # (5) Log details about input KB.
        self.logger.info('Describe the experiment')
        self.logger.info(
            f'Number of named classes: {self.args["num_named_classes"]}\t'
            f'Number of individuals: {self.args["num_instances"]}'
        )

    def start(self):
        self.logger.info('Experiment starts')
        # (1) Train NCEL
        ncel = self.trainer.start()

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
        self.logger.info('Evaluation Ends')


def generate_training_data(kb, args, logger):
    """

    :param logger:
    :param kb:
    :param args:
    :return:
    """
    # (1) Individual to integer mapping
    instance_idx_mapping = {individual.get_iri().as_str(): i for i, individual in enumerate(kb.individuals())}
    number_of_target_expressions = args['number_of_target_expressions']
    # (2) Select labels
    # TODO Add union and intersection
    target_class_expressions, expressions_chain = select_target_expressions(kb, number_of_target_expressions,
                                                                            instance_idx_mapping, logger)
    (e_pos, e_neg) = generate_random_learning_problems(instance_idx_mapping, args)

    return {'e_pos': e_pos, 'e_neg': e_neg, 'instance_idx_mapping': instance_idx_mapping,
            'expressions_chain': expressions_chain,
            'target_class_expressions': target_class_expressions}


def select_target_expressions(kb, number_of_target_expressions, instance_idx_mapping, logger) -> Tuple[
    List[TargetClassExpression], Dict]:
    """
    Select Target Expression
    :return:
    """
    # (1) Preparation
    renderer = DLSyntaxObjectRenderer()
    target_class_expressions = set()
    rl_state = RL_State(kb.thing, parent_node=None, is_root=True)
    rl_state.length = kb.cl(kb.thing)
    rl_state.instances = set(kb.individuals(rl_state.concept))
    target_class_expressions.add(rl_state)
    quantifiers = set()

    rho = LengthBasedRefinement(knowledge_base=kb)
    # (2) Refine Top concept
    for i in apply_rho_on_rl_state(rl_state, rho, kb):
        # (3) Store a class expression has indv.
        if len(i.instances) > 0:
            target_class_expressions.add(i)
            # (4) Store for later refinement if concept is \forall or \exists
            if isinstance(i.concept, OWLObjectAllValuesFrom) or isinstance(i.concept, OWLObjectSomeValuesFrom):
                quantifiers.add(i)
            if len(target_class_expressions) == number_of_target_expressions:
                logger.info(f'{number_of_target_expressions} target expressions generated')
                break
    # (5) Refine
    if len(target_class_expressions) < number_of_target_expressions:
        for selected_states in quantifiers:
            if len(target_class_expressions) == number_of_target_expressions:
                break
            for ref_selected_states in apply_rho_on_rl_state(selected_states, rho, kb):
                if len(ref_selected_states.instances) > 0:
                    if len(target_class_expressions) == number_of_target_expressions:
                        break
                    target_class_expressions.add(ref_selected_states)
    # Sanity checking:target_class_expressions must contain sane number of unique expressions
    assert len({renderer.render(i.concept) for i in target_class_expressions}) == len(target_class_expressions)

    # Sort targets w.r.t. their lenghts
    # Store all target instances
    # These computation can be avoided via Priorty Queue above
    target_class_expressions: List[RL_State] = sorted(list(target_class_expressions), key=lambda x: x.length,
                                                      reverse=False)
    labels = []
    expressions_chain = dict()
    for i in target_class_expressions:
        target = TargetClassExpression(name=renderer.render(i.concept),
                                       individuals={_.get_iri().as_str() for _ in i.instances},
                                       idx_individuals={instance_idx_mapping[_.get_iri().as_str()] for _ in i.instances}
                                       )
        expressions_chain[target.name] = [renderer.render(x.concept) for x in retrieve_concept_chain(i)]
        labels.append(target)
    return labels, expressions_chain


def generate_random_learning_problems(instance_idx_mapping: Dict,
                                      args: Dict) -> Tuple[List[int], List[int]]:
    """
    Generate Learning problems
    :param instance_idx_mapping:
    :param target_idx_individuals:
    :param args: hyperparameters
    :return: a list of ordered learning problems. Each inner list contains same amount of positive and negative
     examples
    """
    instances_idx_list = list(instance_idx_mapping.values())

    pos_examples = []
    neg_examples = []
    num_individual_per_example = args['num_individual_per_example']
    for i in range(args['num_of_learning_problems_training']):
        # Varianable length
        # pos_examples.append(random.choices(instances_idx_list, k=randint(1, max_num_individual_per_example)))
        # neg_examples.append(random.choices(instances_idx_list, k=randint(1, max_num_individual_per_example)))

        pos_examples.append(random.choices(instances_idx_list, k=num_individual_per_example))
        neg_examples.append(random.choices(instances_idx_list, k=num_individual_per_example))

    return pos_examples, neg_examples
