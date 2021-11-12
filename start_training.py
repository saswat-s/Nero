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
from argparse import ArgumentParser

from core.experiment import Experiment
import torch
from torch import nn
import random

random.seed(0)


def main(args):
    """

    :param args:
    :return:
    """
    Experiment(args).start()


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str,
                        default='/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/KGs/Family/family-benchmark_rich_background.owl',
                        help='The absolute path of a knowledge base required.')
    parser.add_argument("--dl_learner_binary_path", type=str, default='dllearner-1.4.0/')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of cpus used during batching')

    # Class Expression Learning
    parser.add_argument("--num_individual_per_example", type=int, default=20,
                        help='Input set size for expression learning.')
    parser.add_argument("--num_of_learning_problems_training", type=int, default=20,
                        help='Total number of randomly generated learning problems for training.')
    parser.add_argument("--num_of_learning_problems_testing", type=int, default=10,
                        help='Total number of randomly generated learning problems for testing.')
    # Neural related
    parser.add_argument("--neural_architecture", type=str, default='ST',
                        help='[ST,PIL]')

    parser.add_argument("--number_of_target_expressions", type=int, default=500,
                        help='Randomly select target class expressions as labels.')

    # Hyperparameters of Neural Class Expression
    parser.add_argument("--num_embedding_dim", type=int, default=25, help='Number of embedding dimensions.')
    # Training Related
    parser.add_argument("--learning_rate", type=int, default=.001, help='Learning Rate')
    parser.add_argument("--num_epochs", type=int, default=100, help='Number of iterations over the entire dataset.')
    parser.add_argument("--batch_size", type=int, default=1024)
    # Inference Related
    parser.add_argument("--topK", type=int, default=250,
                        help='Test the highest topK target expressions')

    # Analysis Related
    parser.add_argument("--plot_embeddings", type=int, default=0, help='1 => Yes, 0 => No')

    main(vars(parser.parse_args()))
