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

        # (2) Initialize KB
        kb = self.initialize_knowledge_base()
        # (2) Initialize Training Data (D: {(E^+,E^-)})_i ^N
        self.lp = self.construct_targets_and_problems(kb)
        self.logger.info(self.lp)
        # (4) Init Trainer
        self.trainer = Trainer(learning_problems=self.lp, args=self.args, logger=self.logger)
        self.logger.info(self.trainer)

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

    def construct_targets_and_problems(self, kb):
        # (3) Initialize Learning problems
        target_class_expressions, instance_idx_mapping = select_target_expressions(kb, self.args, logger=self.logger)
        # e_pos, e_neg = generate_random_learning_problems(instance_idx_mapping, self.args)
        e_pos, e_neg = generate_learning_problems_from_targets(target_class_expressions, instance_idx_mapping,
                                                               self.args)
        return LP(e_pos=e_pos, e_neg=e_neg, instance_idx_mapping=instance_idx_mapping,
                  target_class_expressions=target_class_expressions)

    def describe_and_store(self):
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
        # Instead of storing as list of objects, we can store targets as pandas dataframe

        df = pd.DataFrame([t.__dict__ for t in self.lp.target_class_expressions])
        df.to_csv(path_or_buf=self.storage_path + '/target_class_expressions.csv')
        # print(total_size(self.lp.target_class_expressions))
        # print(df.memory_usage(deep=True).sum())
        # Pandas require more memory than self.lp.target_class_expressions or our memory calculating of a list of
        # items in correct

        del df
        gc.collect()
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
        # (1) Train model
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

    def evaluate(self, ncel, lp, args):
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


def select_target_expressions(kb, args, logger) -> Tuple[List[TargetClassExpression], Dict]:
    """
    Select target expressions
    :param kb:
    :param args:
    :param logger:
    :return: a list of target expressions and a dictionary of URI to integer index
    """
    logger.info('Learning Problems being generated')
    # (1) Individual to integer mapping
    instance_idx_mapping = {individual.get_iri().as_str(): i for i, individual in enumerate(kb.individuals())}
    number_of_target_expressions = args['number_of_target_expressions']
    # (2) Target Expression selection
    if args['target_expression_selection'] == 'diverse_target_expression_selection':
        target_class_expressions = diverse_target_expression_selection(kb,
                                                                       args['tolerance_for_search_unique_target_exp'],
                                                                       number_of_target_expressions,
                                                                       instance_idx_mapping,
                                                                       logger)
    elif args['target_expression_selection'] == 'random_target_expression_selection':
        target_class_expressions = random_target_expression_selection(kb,
                                                                      number_of_target_expressions,
                                                                      instance_idx_mapping,
                                                                      logger)
    else:
        raise KeyError(f'target_expression_selection:{args["target_expression_selection"]}')
    return target_class_expressions, instance_idx_mapping


def target_expressions_via_refining_top(rho, kb, number_of_target_expressions, num_of_all_individuals,
                                        instance_idx_mapping):
    rl_state = RL_State(kb.thing, parent_node=None, is_root=True)
    rl_state.length = kb.concept_len(kb.thing)
    rl_state.instances = set(kb.individuals(rl_state.concept))
    renderer = DLSyntaxObjectRenderer()
    target_class_expressions = set()
    target_idx_instance_set = set()
    quantifiers = set()

    # (1) Refine Top concept
    for i in apply_rho_on_rl_state(rl_state, rho, kb):
        # (3) Continue only concept is not empty.
        if num_of_all_individuals > len(i.instances) > 0:
            # (3.1) Add OWL class expression if, its instances is not already seen
            poss_target_idx_individuals = frozenset(instance_idx_mapping[_.get_iri().as_str()] for _ in i.instances)
            if poss_target_idx_individuals not in target_idx_instance_set:
                # (3.1.) Add instances
                target_idx_instance_set.add(poss_target_idx_individuals)
                # ( 3.2.) Create an instance
                target = TargetClassExpression(
                    label_id=len(target_idx_instance_set),
                    name=renderer.render(i.concept),
                    idx_individuals=poss_target_idx_individuals,
                    expression_chain=[renderer.render(x.concept) for x in
                                      retrieve_concept_chain(i)]
                )
                # Add the created instance
                target_class_expressions.add(target)

            # (4) Store for later refinement if concept is \forall or \exists
            if isinstance(i.concept, OWLObjectAllValuesFrom) or isinstance(i.concept, OWLObjectSomeValuesFrom):
                quantifiers.add(i)
            if len(target_class_expressions) == number_of_target_expressions:
                break
    gc.collect()
    return target_class_expressions, target_idx_instance_set, quantifiers


def refine_selected_expressions(rho, kb, quantifiers, target_class_expressions, target_idx_instance_set,
                                tolerance_for_search_unique_target_exp, instance_idx_mapping,
                                number_of_target_expressions, num_of_all_individuals) -> None:
    renderer = DLSyntaxObjectRenderer()
    if len(target_class_expressions) < number_of_target_expressions:
        for selected_states in quantifiers:
            if len(target_class_expressions) >= number_of_target_expressions:
                break
            not_added = 0
            for ref_selected_states in apply_rho_on_rl_state(selected_states, rho, kb):
                if not_added == tolerance_for_search_unique_target_exp:
                    break
                if num_of_all_individuals > len(ref_selected_states.instances) > 0:
                    # () Check whether we have enough target class expressions
                    if len(target_class_expressions) >= number_of_target_expressions:
                        break
                    # (3.1) Add OWL class expresssion if, its instances is not already seen
                    # poss_target_individuals = frozenset(_.get_iri().as_str() for _ in ref_selected_states.instances)
                    poss_target_idx_individuals = frozenset(
                        instance_idx_mapping[_.get_iri().as_str()] for _ in ref_selected_states.instances)
                    if poss_target_idx_individuals not in target_idx_instance_set:
                        # (3.1.) Add instances
                        target_idx_instance_set.add(poss_target_idx_individuals)
                        # ( 3.2.) Create an instance
                        target = TargetClassExpression(
                            label_id=len(target_idx_instance_set),
                            name=renderer.render(ref_selected_states.concept),
                            idx_individuals=poss_target_idx_individuals,
                            expression_chain=[renderer.render(x.concept) for x in
                                              retrieve_concept_chain(ref_selected_states)]
                        )
                        # Add the created instance
                        target_class_expressions.add(target)
                    else:
                        not_added += 1
                else:
                    not_added += 1
                if len(target_class_expressions) >= number_of_target_expressions:
                    break
            if len(target_class_expressions) >= number_of_target_expressions:
                break

    gc.collect()


def intersect_and_union_expressions_from_iterable(target_class_expressions, target_idx_instance_set,
                                                  number_of_target_expressions):
    while len(target_idx_instance_set) < number_of_target_expressions:

        res = set()
        for i in target_class_expressions:
            for j in target_class_expressions:

                if i == j:
                    continue

                i_and_j = i * j
                if i_and_j.size > 0 and (i_and_j.idx_individuals not in target_idx_instance_set):
                    res.add(i_and_j)
                    target_idx_instance_set.add(i_and_j.idx_individuals)
                    i_and_j.label_id = len(target_idx_instance_set)
                else:
                    del i_and_j

                if len(target_idx_instance_set) >= number_of_target_expressions:
                    break

                i_or_j = i + j
                if i_or_j.size > 0 and (i_or_j.idx_individuals not in target_idx_instance_set):
                    res.add(i_or_j)
                    target_idx_instance_set.add(i_or_j.idx_individuals)
                    i_or_j.label_id = len(target_idx_instance_set)
                else:
                    del i_or_j

                if len(target_idx_instance_set) >= number_of_target_expressions:
                    break
        target_class_expressions.update(res)


def diverse_target_expression_selection(kb, tolerance_for_search_unique_target_exp, number_of_target_expressions,
                                        instance_idx_mapping, logger) -> Tuple[
    List[TargetClassExpression], Dict]:
    """
    (1) Refine Top expression and obtain all possible ALC expressions up to length 3
    (1.1) Consider only those expression as labels whose set of individuals has not been seen before
    (1.2.) E.g. {{....}, {.}, {...}}. Only  consider those expressions as labels that do not cover all individuals
    (2)
    Select Target Expression
    :return:
    """
    # Preparation
    rho = LengthBasedRefinement(knowledge_base=kb)
    num_of_all_individuals = kb.individuals_count()
    target_class_expressions, target_idx_instance_set, quantifiers = target_expressions_via_refining_top(rho=rho,
                                                                                                         kb=kb,
                                                                                                         number_of_target_expressions=number_of_target_expressions,
                                                                                                         num_of_all_individuals=num_of_all_individuals,
                                                                                                         instance_idx_mapping=instance_idx_mapping)
    logger.info(
        f'{len(target_class_expressions)} number of target expressions are obtained from the most general expression.')
    assert len(target_idx_instance_set) == len(target_class_expressions)

    refine_selected_expressions(rho, kb, quantifiers, target_class_expressions, target_idx_instance_set,
                                tolerance_for_search_unique_target_exp, instance_idx_mapping,
                                number_of_target_expressions, num_of_all_individuals)
    logger.info(
        f'{len(target_class_expressions)} number of target expressions are obtained from the most general expression and quantifiers')
    assert len(target_idx_instance_set) == len(target_class_expressions)
    intersect_and_union_expressions_from_iterable(target_class_expressions, target_idx_instance_set,
                                                  number_of_target_expressions)
    logger.info(
        f'{len(target_class_expressions)} number of target expressions are obtained from the most general expression, quantifiers, and intersect/union all previous expressions')
    assert len(target_idx_instance_set) == len(target_class_expressions)

    result = []
    for ith, tce in enumerate(target_class_expressions):
        tce.label_id = ith
        result.append(tce)
    gc.collect()
    return result


def random_target_expression_selection(kb, number_of_target_expressions, instance_idx_mapping, logger) -> Tuple[
    List[TargetClassExpression], Dict]:
    """
    Select Target Expression
    :return:
    """
    # @TODO followed same method of not using RL_State as done in entropy_based_target_expression_selection
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
    for id_t, i in enumerate(target_class_expressions):
        target = TargetClassExpression(
            label_id=id_t,
            name=renderer.render(i.concept),
            idx_individuals=frozenset(instance_idx_mapping[_.get_iri().as_str()] for _ in i.instances),
            expression_chain=[renderer.render(x.concept) for x in retrieve_concept_chain(i)]
        )
        labels.append(target)
    return labels


def generate_learning_problems_from_targets(target_class_expressions: List[TargetClassExpression],
                                            instance_idx_mapping: Dict,
                                            args: Dict) -> Tuple[List[int], List[int]]:
    """
    Sample pos from targets

    :param target_class_expressions:
    :param instance_idx_mapping:
    :param args:
    :return:
    """
    random.seed(0)
    instances_idx_list = list(instance_idx_mapping.values())

    pos_examples = []
    neg_examples = []
    num_individual_per_example = args['num_individual_per_example']
    for i in range(args['num_of_learning_problems_training']):
        for tce in target_class_expressions:
            pos_examples.append(random.choices(list(tce.idx_individuals), k=num_individual_per_example))
            neg_examples.append(random.choices(instances_idx_list, k=num_individual_per_example))
    assert len(pos_examples) == len(neg_examples)
    return pos_examples, neg_examples


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
    assert len(pos_examples) == len(neg_examples)
    return pos_examples, neg_examples
