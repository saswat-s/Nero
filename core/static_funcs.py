from typing import List, Tuple, Set, Dict
from ontolearn.search import RL_State
import datetime
import logging
import os
import time
import torch
import json
from collections import deque
from ontolearn.learning_problem_generator import LearningProblemGenerator
from ontolearn.refinement_operators import LengthBasedRefinement
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.model import OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom
import random
from random import randint

from sys import getsizeof, stderr
from itertools import chain
from collections import deque

try:
    from reprlib import repr
except ImportError:
    pass

import torch
from owlapy.model import OWLOntology, OWLReasoner
from owlapy.owlready2 import OWLOntology_Owlready2
from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
from .dl_expression import TargetClassExpression # Needs to be added to static_classes ?
import gc
def select_target_expressions(kb, args, logger) -> Tuple[List[TargetClassExpression], Dict]:
    """
    Select target expressions
    :param kb:
    :param args:
    :param logger:
    :return: a list of target expressions and a dictionary of URI to integer index
    """
    logger.info('Target Expressions being selected.')
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
    rl_state.instances = set(kb.str_individuals(rl_state.concept))
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
                                            args: Dict, logger) -> Tuple[List[int], List[int]]:
    """
    Sample pos from targets

    :param target_class_expressions:
    :param instance_idx_mapping:
    :param args:
    :return:
    """
    logger.info('Learning Problems are being sampled from Targets')
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


def ClosedWorld_ReasonerFactory(onto: OWLOntology) -> OWLReasoner:
    assert isinstance(onto, OWLOntology_Owlready2)
    base_reasoner = OWLReasoner_Owlready2_TempClasses(ontology=onto)
    reasoner = OWLReasoner_FastInstanceChecker(ontology=onto,
                                               base_reasoner=base_reasoner,
                                               negation_default=True)
    return reasoner

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    CD: obtained from one and only  Raymond Hettinger :)
    source : https://code.activestate.com/recipes/577504/

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


import resource


def using(point=""):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return '''%s: usertime=%s systime=%s mem=%s mb ''' % (point, usage[0], usage[1], usage[2] / 1024.0)



def save_as_json(*, storage_path=None, obj=None, name=None):
    with open(storage_path + f'/{name}.json', 'w') as file_descriptor:
        json.dump(obj, file_descriptor, indent=3)


def apply_rho_on_rl_state(rl_state, rho, kb):
    for i in rho.refine(rl_state.concept):
        next_rl_state = RL_State(i, parent_node=rl_state)
        next_rl_state.length = kb.concept_len(next_rl_state.concept)
        next_rl_state.instances = set(kb.individuals(next_rl_state.concept))
        yield next_rl_state


def generate_target_class_expressions(lpg, kb, args):
    renderer = DLSyntaxObjectRenderer()

    number_of_target_expressions = args['number_of_target_expressions']
    # Generate target_class_expressions
    target_class_expressions = set()
    rl_state = RL_State(kb.thing, parent_node=None, is_root=True)
    rl_state.length = kb.concept_len(kb.thing)
    rl_state.instances = set(kb.str_individuals(rl_state.concept))
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
    target_class_expressions: List[str] = [renderer.render(i.concept) for i in target_class_expressions]

    return target_class_expressions, target_individuals

def compute_f1_target(target_class_expressions, pos, neg):
    pos = set(pos)
    neg = set(neg)
    return [f_measure(instances=t.idx_individuals, positive_examples=pos, negative_examples=neg) for t in
            target_class_expressions]

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
    return f_1


def accuracy(*, instances: Set, positive_examples: Set, negative_examples: Set):
        tp = len(positive_examples.intersection(instances))
        tn = len(negative_examples.difference(instances))

        fp = len(negative_examples.intersection(instances))
        fn = len(positive_examples.difference(instances))

        return (tp + tn) / (tp+tn+fp+fn)


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
