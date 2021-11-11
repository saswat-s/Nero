from typing import List, Tuple, Set
from ontolearn.search import RL_State
import datetime
import logging
import os
import time
import torch
from collections import deque


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
