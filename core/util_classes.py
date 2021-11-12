import torch
from owlapy.model import OWLOntology, OWLReasoner
from owlapy.owlready2 import OWLOntology_Owlready2
from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
from .static_funcs import f_measure
from multiprocessing import Pool
from typing import List
import numpy as np
import itertools


class LP:
    def __init__(self, learning_problems: List[List[int]], instance_idx_mapping, target_class_expressions,
                 target_idx_individuals):
        """

        :param learning_problems: a list of ordered learning problems. Each inner list contains same amounth of positive and negative
        :param instance_idx_mapping:
        :param target_class_expressions:
        :param target_idx_individuals:
        """
        self.data_points = learning_problems
        self.instance_idx_mapping = instance_idx_mapping
        self.target_class_expressions = target_class_expressions
        self.target_idx_individuals = target_idx_individuals

    def __str__(self):
        return f'<LP object at {hex(id(self))}>\tdata_points: {self.data_points.shape}\t|target_class_expressions|:{len(self.target_class_expressions)}'

    def __len__(self):
        return len(self.data_points)


def ClosedWorld_ReasonerFactory(onto: OWLOntology) -> OWLReasoner:
    assert isinstance(onto, OWLOntology_Owlready2)
    base_reasoner = OWLReasoner_Owlready2_TempClasses(ontology=onto)
    reasoner = OWLReasoner_FastInstanceChecker(ontology=onto,
                                               base_reasoner=base_reasoner,
                                               negation_default=True)
    return reasoner


def dummy(all_targets, pos, neg):
    res = []
    pos = set(pos)
    neg = set(neg)
    for target_instances in all_targets:
        target_instances: target_instances[int]  # containing ordered positive and negative examples
        res.append(f_measure(instances=set(target_instances), positive_examples=pos, negative_examples=neg))
    return res

class Dataset(torch.utils.data.Dataset):
    def __init__(self, lp: LP):
        self.lp = lp
        self.num_data_points = len(self.lp)
        self.Y = []

        with Pool(processes=4) as pool:
            self.Y = list(
                pool.starmap(dummy, ((self.lp.target_idx_individuals, pos, neg) for (pos, neg) in self.lp.data_points)))

        """
        for pos, neg in self.lp.data_points:
            res = []
            pos = set(pos)
            neg = set(neg)
            for target_instances in self.lp.target_idx_individuals:
                target_instances: target_instances[int]  # containing ordered positive and negative examples
                res.append(f1_score(instances=set(target_instances), positive_examples=pos, negative_examples=neg))
            self.Y.append(res)
        """
        self.X = torch.LongTensor(self.lp.data_points)
        self.Y = torch.FloatTensor(self.Y)
        n, two, size_examples = self.X.shape

        self.X = torch.reshape(self.X, (n, two * size_examples))
        # Expensive Sanity checking
        # Flatten data points into a single list
        # Flatten data points stored in pytorchTensor
        # Through utilizing the order of datapoints, check whether they are equal
        assert list(itertools.chain.from_iterable(itertools.chain.from_iterable(self.lp.data_points))) == list(
            itertools.chain.from_iterable(self.X.tolist()))
        # To free some memory
        self.lp.data_points = None

        self.Xpos, self.Xneg = torch.hsplit(self.X, 2)

        del self.X

    def __len__(self):
        return self.num_data_points

    def __getitem__(self, idx):
        return self.Xpos[idx], self.Xneg[idx], self.Y[idx]
