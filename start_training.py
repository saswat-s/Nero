"""
====================================================================
Deep Tunnelling
====================================================================
Drill with training.
Authors: Caglar Demir

(1) Parse input knowledge base
(2) Generate 100_000 Class Expressions that are "informative"
    # (3) Generate learning problems, where a learning problem E is a set of examples/instance
    # considered as positive examples. E has variable Size
    # (4) Extend (3) by adding randomly sampled examples
    # (5) Let D denotes the set generated in (3) and (4)
    # (6) For each learning problem X_i, compute Y_i that is a vector of F1-scores
    # (7) Summary: D_i = { ({e_x, e_y, ...e_z }_i ,Y_i) }_i=0 ^N has variable size, Y_i has 10^5 size
    # (8) For each D_i, let x_i denote input set and Y_i label
    # (9) Let \mathbf{x_i} \in R^{3,D} represent the mean, median, and sum of X_i; permutation invariance baby :)
    # (10) Train sigmoid(net) to minimize binary cross entropy; multi-label classification problem

    # We can use (10) for
    #                 class expression learning
    #                 ?Explainable Clustering ?
    #                 ?link prediction? (h,r,x) look the range of  top10K(Y_i)
https://github.com/RDFLib/sparqlwrapper/blob/master/scripts/example-dbpedia.py
"""
from ontolearn import KnowledgeBase
from typing import List, Tuple, Set
from owlapy.model import OWLEquivalentClassesAxiom, OWLClass, IRI, OWLObjectIntersectionOf, OWLObjectUnionOf, \
    OWLObjectSomeValuesFrom, OWLObjectInverseOf, OWLObjectProperty, OWLThing
from argparse import ArgumentParser

from ontolearn.learning_problem_generator import LearningProblemGenerator
from owlapy.render import DLSyntaxObjectRenderer

import random
from collections import deque
from model import DT

import torch
from torch import nn
import numpy as np
from static_funcs import *
from util_classes import *
import json
import pandas as pd
from trainer import Trainer

random.seed(0)


def generate_target_class_expressions(lpg, kb):
    renderer = DLSyntaxObjectRenderer()

    # Generate target_class_expressions
    target_class_expressions = []
    rl_state = RL_State(kb.thing, parent_node=None, is_root=True)
    rl_state.length = kb.cl(kb.thing)
    rl_state.instances = set(kb.individuals(rl_state.concept))
    target_class_expressions.append(rl_state)
    for i in lpg.apply_rho_on_rl_state(rl_state):
        if len(i.instances) > 0:
            target_class_expressions.append(i)
    # Sanity checking:target_class_expressions must contain sane number of unique expressions
    assert len({renderer.render(i.concept) for i in target_class_expressions}) == len(target_class_expressions)

    # Sort it for the convenience. Not a must ALC formulas
    target_class_expressions: List[RL_State] = sorted(list(target_class_expressions), key=lambda x: x.length,
                                                      reverse=False)

    # All instances belonging to targets
    target_individuals: List[Set[str]] = [{i.get_iri().as_str() for i in s.instances} for s in
                                          target_class_expressions]

    return target_class_expressions, target_individuals


def generate_random_learning_problems(instance_idx_mapping, args):
    # (2) Convert instance into list of URIs
    # Could do it more efficiently
    instances = list(instance_idx_mapping.keys())  # [i.get_iri().as_str() for i in knowledge_base.individuals()]

    X = []
    size_of_positive_example_set = args['input_set_size']
    # (4) Generate RANDOM TRAINING DATA
    for i in range(args['num_of_data_points']):
        # https://docs.python.org/3/library/random.html#random.choices
        e = random.choices(instances, k=size_of_positive_example_set)
        X.append([instance_idx_mapping[_] for _ in e])
    return X


def main(args):
    """

    :param args:
    :return:
    """
    # (1) Parse input KB
    kb = KnowledgeBase(path=args['path_knowledge_base'],
                       reasoner_factory=ClosedWorld_ReasonerFactory)
    # (2) Generate Class Expressions semi-randomly
    lpg = LearningProblemGenerator(knowledge_base=kb,
                                   min_length=args['min_length'],
                                   max_length=args['max_length'],
                                   min_num_instances=args[
                                                         'min_num_instances_ratio_per_concept'] * kb.individuals_count(),
                                   max_num_instances=args[
                                                         'max_num_instances_ratio_per_concept'] * kb.individuals_count())
    # Labels
    target_class_expressions, target_individuals = generate_target_class_expressions(lpg, kb)
    instance_idx_mapping = {indv.get_iri().as_str(): i for i, indv in enumerate(kb.individuals())}

    learning_problems = generate_random_learning_problems(instance_idx_mapping, args)

    lp = LP(learning_problems, list(instance_idx_mapping.keys()), instance_idx_mapping, target_class_expressions,
            target_individuals)
    # conda run - n envname python
    # (3) Train DT
    trainer = Trainer(knowledge_base=kb, learning_problems=lp, args=args)
    trainer.start()

    # (4) Load trained model


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str,
                        default='KGs/Family/family-benchmark_rich_background.owl'
                        )
    # Concept Generation Related
    parser.add_argument("--min_length", type=int, default=0, help='Min length of concepts to be used')
    parser.add_argument("--max_length", type=int, default=5, help='Max length of concepts to be used')
    parser.add_argument("--min_num_instances_ratio_per_concept", type=float, default=.01)
    parser.add_argument("--max_num_instances_ratio_per_concept", type=float, default=.90)

    # Neural related
    parser.add_argument("--input_set_size", type=int, default=2, help='Input set size for expression learning.')

    parser.add_argument("--num_of_data_points", type=int, default=1000,
                        help='Total number of randomly sampled training data points')
    parser.add_argument("--num_embedding_dim", type=int, default=25, help='Number of embedding dimensions.')
    parser.add_argument("--learning_rate", type=int, default=.01, help='Learning Rate')
    parser.add_argument("--num_epochs", type=int, default=1000, help='Number of iterations over the entire dataset.')
    parser.add_argument("--batch_size", type=int, default=1024)

    # Analysis Related
    parser.add_argument("--plot_embeddings", type=int, default=1, help='1 => Yes, 0 => No')

    main(vars(parser.parse_args()))
