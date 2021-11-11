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
from typing import List, Tuple, Set, Dict
from owlapy.model import OWLEquivalentClassesAxiom, OWLClass, IRI, OWLObjectIntersectionOf, OWLObjectUnionOf, \
    OWLObjectSomeValuesFrom, OWLObjectInverseOf, OWLObjectProperty, OWLThing
from argparse import ArgumentParser

from ontolearn.learning_problem_generator import LearningProblemGenerator
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.model import OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom
import random
from collections import deque
import torch
from torch import nn
import numpy as np
from static_funcs import *
from util_classes import *
import json
import pandas as pd
from trainer import Trainer
from ontolearn.binders import DLLearnerBinder
import json

random.seed(0)


def generate_target_class_expressions(lpg, kb):
    renderer = DLSyntaxObjectRenderer()

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
    for selected_states in quantifiers:
        for ref_selected_states in lpg.apply_rho_on_rl_state(selected_states):
            if len(ref_selected_states.instances) > 0:
                target_class_expressions.add(ref_selected_states)
    # Sanity checking:target_class_expressions must contain sane number of unique expressions
    assert len({renderer.render(i.concept) for i in target_class_expressions}) == len(target_class_expressions)

    # Sort it for the convenience. Not a must ALC formulas
    target_class_expressions: List[RL_State] = sorted(list(target_class_expressions), key=lambda x: x.length,
                                                      reverse=False)
    # All instances belonging to targets
    target_individuals: List[Set[str]] = [{i.get_iri().as_str() for i in s.instances} for s in
                                          target_class_expressions]
    return target_class_expressions, target_individuals


def generate_random_learning_problems(instance_idx_mapping: Dict, target_idx_individuals: List[List[int]],
                                      args: Dict) -> List[Tuple[List[int], List[int]]]:
    """
    Generate Learning problems
    :param instance_idx_mapping:
    :param target_idx_individuals:
    :param args: hyperparameters
    :return: a list of ordered learning problems. Each inner list contains same amounth of positive and negative
     examples
    """
    assert isinstance(target_idx_individuals, list)
    assert isinstance(target_idx_individuals[0], list)
    assert isinstance(target_idx_individuals[0][0], int)
    # (1) Obtain set of instance idx to sample negative examples via set difference
    instances_idx_set = set(instance_idx_mapping.values())
    instances_idx_list = list(instance_idx_mapping.values())

    X = []
    size_of_positive_example_set = args['input_set_size']
    """
    # (4) Generate RANDOM TRAINING DATA
    for i in range(args['num_of_data_points']):
        # https://docs.python.org/3/library/random.html#random.choices
        e = random.choices(instances, k=size_of_positive_example_set)
        X.append([instance_idx_mapping[_] for _ in e])
    """
    for i in range(args['num_of_data_points']):
        pos = random.choices(instances_idx_list, k=size_of_positive_example_set)
        neg = random.choices(instances_idx_list, k=size_of_positive_example_set)
        X.append((pos, neg))
    """
    for i in target_idx_individuals:
        pos = random.choices(i, k=size_of_positive_example_set)
        all_poss_neg_samples = list(instances_idx_set.difference(set(i)))
        if len(all_poss_neg_samples) == 0:
            continue
        neg = random.choices(all_poss_neg_samples, k=size_of_positive_example_set)
        X.append((pos, neg))
    """

    return X


def generate_trainin_data(kb, args):
    # (2) Generate Class Expressions semi-randomly
    lpg = LearningProblemGenerator(knowledge_base=kb,
                                   min_length=args['min_length'],
                                   max_length=args['max_length'],
                                   min_num_instances=args[
                                                         'min_num_instances_ratio_per_concept'] * kb.individuals_count(),
                                   max_num_instances=args[
                                                         'max_num_instances_ratio_per_concept'] * kb.individuals_count())
    # (3) Obtain labels
    instance_idx_mapping = {individual.get_iri().as_str(): i for i, individual in enumerate(kb.individuals())}
    target_class_expressions, target_individuals = generate_target_class_expressions(lpg, kb)
    target_idx_individuals = [[instance_idx_mapping[x] for x in i] for i in target_individuals]

    learning_problems: List[Tuple[List[int], List[int]]] = generate_random_learning_problems(instance_idx_mapping,
                                                                                             target_idx_individuals,
                                                                                             # consider only top 10
                                                                                             args)

    lp = LP(learning_problems, instance_idx_mapping, target_class_expressions, target_idx_individuals)
    return lp


def evaluate(ncel, lp, args):
    print('Evaluation Starts')
    str_all_targets = [i for i in ncel.get_target_class_expressions()]

    instance_str = list(lp.instance_idx_mapping.keys())
    # (1) Enter the absolute path of the input knowledge base
    kb_path = '/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/KGs/Family/family-benchmark_rich_background.owl'
    # (2) To download DL-learner,  https://github.com/SmartDataAnalytics/DL-Learner/releases.
    dl_learner_binary_path = '/home/demir/Desktop/DL/dllearner-1.4.0/'
    # (3) Initialize CELOE, OCEL, and ELTL
    celoe = DLLearnerBinder(binary_path=dl_learner_binary_path, kb_path=kb_path, model='celoe')
    # (4) Fit (4) on the learning problems and show the best concept.
    ncel_results = dict()
    celoe_results = dict()
    for _ in range(10):
        p = random.choices(instance_str, k=args['input_set_size'])
        n = random.choices(instance_str, k=args['input_set_size'])

        ncel_report = ncel.fit(pos=p, neg=n)
        ncel_report.update({'P': p, 'N': n, 'F-measure': f_measure(instances=ncel_report['Instances'],
                                                                   positive_examples=set(p),
                                                                   negative_examples=set(n)),
                            })
        best_pred_celoe = celoe.fit(pos=p, neg=n, max_runtime=1).best_hypothesis()

        if best_pred_celoe['Prediction'] in str_all_targets:
            pass
        else:
            print(f'{best_pred_celoe["Prediction"]} not found in labels')

        ncel_results[_] = ncel_report

        celoe_results[_] = {'P': p, 'N': n,
                            'Prediction': best_pred_celoe['Prediction'],
                            'F-measure': best_pred_celoe['F-measure'],
                            'NumClassTested': best_pred_celoe['NumClassTested'],
                            'Runtime': best_pred_celoe['Runtime'],
                            }

    avg_f1_ncel = np.array([i['F-measure'] for i in ncel_results.values()]).mean()
    avg_runtime_ncel = np.array([i['Runtime'] for i in ncel_results.values()]).mean()
    avg_expression_ncel = np.array([i['NumClassTested'] for i in ncel_results.values()]).mean()

    print(f'Average F-measure NCEL:{avg_f1_ncel}\t Avg. Runtime:{avg_runtime_ncel}\t Avg. Expression Tested:{avg_expression_ncel}')

    avg_f1_celoe = np.array([i['F-measure'] for i in celoe_results.values()]).mean()
    avg_runtime_celoe = np.array([i['Runtime'] for i in celoe_results.values()]).mean()
    avg_expression_celoe = np.array([i['NumClassTested'] for i in celoe_results.values()]).mean()

    print(f'Average F-measure CELOE:{avg_f1_celoe}\t Avg. Runtime:{avg_runtime_celoe}\t Avg. Expression Tested:{avg_expression_celoe}')


def main(args):
    """

    :param args:
    :return:
    """
    # (1) Parse input KB
    kb = KnowledgeBase(path=args['path_knowledge_base'],
                       reasoner_factory=ClosedWorld_ReasonerFactory)
    lp = generate_trainin_data(kb, args)
    # conda run - n envname python
    # (3) Train DT
    ncel = Trainer(knowledge_base=kb, learning_problems=lp, args=args).start()

    evaluate(ncel, lp, args)


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str,
                        default='KGs/Family/family-benchmark_rich_background.owl'
                        )
    # Concept Generation Related
    parser.add_argument("--min_length", type=int, default=0, help='Min length of concepts to be used')
    parser.add_argument("--max_length", type=int, default=7, help='Max length of concepts to be used')
    parser.add_argument("--min_num_instances_ratio_per_concept", type=float, default=.0001)
    parser.add_argument("--max_num_instances_ratio_per_concept", type=float, default=.999)

    # Neural related
    parser.add_argument("--input_set_size", type=int, default=20, help='Input set size for expression learning.')

    parser.add_argument("--num_of_data_points", type=int, default=100,
                        help='Total number of randomly sampled training data points')
    parser.add_argument("--num_embedding_dim", type=int, default=25, help='Number of embedding dimensions.')
    parser.add_argument("--learning_rate", type=int, default=.01, help='Learning Rate')
    parser.add_argument("--num_epochs", type=int, default=1, help='Number of iterations over the entire dataset.')
    parser.add_argument("--batch_size", type=int, default=1024)

    # Analysis Related
    parser.add_argument("--plot_embeddings", type=int, default=0, help='1 => Yes, 0 => No')

    main(vars(parser.parse_args()))
