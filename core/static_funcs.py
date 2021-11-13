from typing import List, Tuple, Set, Dict
from ontolearn.search import RL_State
import datetime
import logging
import os
import time
import torch
from collections import deque
from ontolearn.learning_problem_generator import LearningProblemGenerator
from ontolearn.refinement_operators import LengthBasedRefinement
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.model import OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom
import random

from random import randint


def generate_random_learning_problems(instance_idx_mapping: Dict, target_idx_individuals: List[List[int]],
                                      args: Dict) -> Tuple[List[int], List[int]]:
    """
    Generate Learning problems
    :param instance_idx_mapping:
    :param target_idx_individuals:
    :param args: hyperparameters
    :return: a list of ordered learning problems. Each inner list contains same amount of positive and negative
     examples
    """
    assert isinstance(target_idx_individuals, list)
    assert isinstance(target_idx_individuals[0], list)
    assert isinstance(target_idx_individuals[0][0], int)
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


def apply_rho_on_rl_state(rl_state, rho, kb):
    for i in rho.refine(rl_state.concept):
        next_rl_state = RL_State(i, parent_node=rl_state)
        next_rl_state.length = kb.cl(next_rl_state.concept)
        next_rl_state.instances = set(kb.individuals(next_rl_state.concept))
        yield next_rl_state


def generate_training_data(kb, args, logger):
    """

    :param kb:
    :param args:
    :return:
    """
    # (3) Obtain labels
    instance_idx_mapping = {individual.get_iri().as_str(): i for i, individual in enumerate(kb.individuals())}
    renderer = DLSyntaxObjectRenderer()
    number_of_target_expressions = args['number_of_target_expressions']
    # Generate target_class_expressions
    target_class_expressions = set()
    rl_state = RL_State(kb.thing, parent_node=None, is_root=True)
    rl_state.length = kb.cl(kb.thing)
    rl_state.instances = set(kb.individuals(rl_state.concept))
    target_class_expressions.add(rl_state)
    quantifiers = set()

    rho = LengthBasedRefinement(knowledge_base=kb)
    for i in apply_rho_on_rl_state(rl_state, rho, kb):
        if len(i.instances) > 0:
            target_class_expressions.add(i)
            if isinstance(i.concept, OWLObjectAllValuesFrom) or isinstance(i.concept, OWLObjectSomeValuesFrom):
                quantifiers.add(i)
            if len(target_class_expressions) == number_of_target_expressions:
                logger.info(f'{number_of_target_expressions} target expressions generated')
                break
    if len(target_class_expressions) < number_of_target_expressions:
        for selected_states in quantifiers:
            for ref_selected_states in apply_rho_on_rl_state(selected_states, rho, kb):
                if len(ref_selected_states.instances) > 0:
                    target_class_expressions.add(ref_selected_states)
                    if len(target_class_expressions) == number_of_target_expressions:
                        break

    # Sanity checking:target_class_expressions must contain sane number of unique expressions
    assert len({renderer.render(i.concept) for i in target_class_expressions}) == len(target_class_expressions)

    # Sort it for the convenience. Not a must ALC formulas
    target_class_expressions: List[RL_State] = sorted(list(target_class_expressions), key=lambda x: x.length,
                                                      reverse=False)
    # All instances belonging to targets
    target_individuals: List[Set[str]] = [{i.get_iri().as_str() for i in s.instances} for s in
                                          target_class_expressions]

    target_idx_individuals = [[instance_idx_mapping[x] for x in i] for i in target_individuals]

    (e_pos, e_neg) = generate_random_learning_problems(instance_idx_mapping, target_idx_individuals, args)

    return {'e_pos': e_pos, 'e_neg': e_neg, 'instance_idx_mapping': instance_idx_mapping,
            'target_class_expressions': target_class_expressions, 'target_idx_individuals': target_idx_individuals}


def generate_target_class_expressions(lpg, kb, args):
    renderer = DLSyntaxObjectRenderer()

    number_of_target_expressions = args['number_of_target_expressions']
    # Generate target_class_expressions
    target_class_expressions = set()
    rl_state = RL_State(kb.thing, parent_node=None, is_root=True)
    rl_state.length = kb.cl(kb.thing)
    rl_state.instances = set(kb.individuals(rl_state.concept))
    target_class_expressions.add(rl_state)
    quantifiers = set()
    for i in lpg.apply_rho_on_rl_state(rl_state):
        if len(i.instances) > 0:
            target_class_expressions.add(i)
            if isinstance(i.concept, OWLObjectAllValuesFrom) or isinstance(i.concept, OWLObjectSomeValuesFrom):
                quantifiers.add(i)
            if len(target_class_expressions) == number_of_target_expressions:
                break
    for selected_states in quantifiers:
        for ref_selected_states in lpg.apply_rho_on_rl_state(selected_states):
            if len(ref_selected_states.instances) > 0:
                target_class_expressions.add(ref_selected_states)
        if len(target_class_expressions) == number_of_target_expressions:
            break

    # Sanity checking:target_class_expressions must contain sane number of unique expressions
    assert len({renderer.render(i.concept) for i in target_class_expressions}) == len(target_class_expressions)

    # Sort it for the convenience. Not a must ALC formulas
    target_class_expressions: List[RL_State] = sorted(list(target_class_expressions), key=lambda x: x.length,
                                                      reverse=False)
    # All instances belonging to targets
    target_individuals: List[Set[str]] = [{i.get_iri().as_str() for i in s.instances} for s in
                                          target_class_expressions]
    return target_class_expressions, target_individuals


def f_measure(*, instances: Set, positive_examples: Set, negative_examples: Set):
    tp = len(positive_examples.intersection(instances))
    # tn = len(learning_problem.kb_neg.difference(instances))

    fp = len(negative_examples.intersection(instances))
    fn = len(positive_examples.difference(instances))

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        return 0.0

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        return 0.0

    if precision == 0 or recall == 0:
        return 0.0

    f_1 = 2 * ((precision * recall) / (precision + recall))
    return round(f_1, 5)


def retrieve_concept_chain(rl_state: RL_State) -> List[RL_State]:
    hierarchy = deque()
    if rl_state.parent_node:
        hierarchy.appendleft(rl_state.parent_node)
        while hierarchy[-1].parent_node is not None:
            hierarchy.append(hierarchy[-1].parent_node)
        hierarchy.appendleft(rl_state)
    return list(hierarchy)


def create_experiment_folder(folder_name='Experiments'):
    directory = os.getcwd() + '/' + folder_name + '/'
    folder_name = str(datetime.datetime.now())
    path_of_folder = directory + folder_name
    os.makedirs(path_of_folder)
    return path_of_folder, path_of_folder[:path_of_folder.rfind('/')]


def create_logger(*, name, p):
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(p + '/info.log')
    fh.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def save_weights(model, storage_path):
    model.to('cpu')
    torch.save(model.state_dict(), storage_path + f'/final_model.pt')
