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


class TargetClassExpression_old:
    def __init__(self, *, label_id: int, name: str, individuals: Set, idx_individuals: Set, expression_chain: List):
        assert isinstance(label_id, int)
        assert isinstance(name, str)
        assert isinstance(individuals, frozenset)
        assert isinstance(idx_individuals, frozenset)
        assert isinstance(expression_chain, list)

        self.label_id = label_id
        self.name = name
        self.individuals = individuals
        self.idx_individuals = idx_individuals
        self.expression_chain = expression_chain
        assert len(self.individuals) == len(self.idx_individuals)
        self.num_individuals = len(self.individuals)

    def __str__(self):
        return f'{self.name}\tIndv:{self.num_individuals}'

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        return TargetClassExpression(
            label_id=-1,
            name=f'({self.name}) ⊓ ({other.name})',
            individuals=self.individuals.intersection(other.individuals),
            idx_individuals=self.idx_individuals.intersection(other.idx_individuals),
            expression_chain=self.expression_chain + [other.name])

    def __add__(self, other):
        return TargetClassExpression(
            label_id=-2,
            name=f'({self.name}) ⊔ ({other.name})',
            individuals=self.individuals.union(other.individuals),
            idx_individuals=self.idx_individuals.union(other.idx_individuals),
            expression_chain=self.expression_chain + [other.name])


class TargetClassExpression:
    def __init__(self, *, label_id: int, name: str, idx_individuals: Set, expression_chain: List):
        assert isinstance(label_id, int)
        assert isinstance(name, str)
        assert isinstance(idx_individuals, frozenset)
        assert isinstance(expression_chain, list)

        self.label_id = label_id
        self.name = name
        self.idx_individuals = idx_individuals
        self.expression_chain = expression_chain
        self.num_individuals = len(self.idx_individuals)
    @property
    def size(self):
        return self.num_individuals

    def __str__(self):
        return f'{self.name}\tIndv:{self.num_individuals}'

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        return TargetClassExpression(
            label_id=-1,
            name=f'({self.name}) ⊓ ({other.name})',
            idx_individuals=self.idx_individuals.intersection(other.idx_individuals),
            expression_chain=self.expression_chain + [other.name])

    def __add__(self, other):
        return TargetClassExpression(
            label_id=-2,
            name=f'({self.name}) ⊔ ({other.name})',
            idx_individuals=self.idx_individuals.union(other.idx_individuals),
            expression_chain=self.expression_chain + [other.name])


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
    target_class_expressions: List[str] = [renderer.render(i.concept) for i in target_class_expressions]

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
    return f_1


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
