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
    def __init__(self, *, e_pos: List[List[int]], e_neg: List[List[int]], instance_idx_mapping,
                 target_class_expressions,
                 target_idx_individuals):
        """

        :param learning_problems: a list of ordered learning problems. Each inner list contains same amounth of positive and negative
        :param instance_idx_mapping:
        :param target_class_expressions:
        :param target_idx_individuals:
        """
        assert len(e_pos) == len(e_neg)
        self.e_pos = e_pos
        self.e_neg = e_neg
        self.num_learning_problems=len(self.e_pos)
        self.instance_idx_mapping = instance_idx_mapping
        self.target_class_expressions = target_class_expressions
        self.target_idx_individuals = target_idx_individuals

    def __str__(self):
        return f'<LP object at {hex(id(self))}>\tdata_points: {self.data_points.shape}\t|target_class_expressions|:{len(self.target_class_expressions)}'

    def __len__(self):
        return self.num_learning_problems


def ClosedWorld_ReasonerFactory(onto: OWLOntology) -> OWLReasoner:
    assert isinstance(onto, OWLOntology_Owlready2)
    base_reasoner = OWLReasoner_Owlready2_TempClasses(ontology=onto)
    reasoner = OWLReasoner_FastInstanceChecker(ontology=onto,
                                               base_reasoner=base_reasoner,
                                               negation_default=True)
    return reasoner


def compute_f1_target(all_targets, pos, neg):
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
                pool.starmap(compute_f1_target, ((self.lp.target_idx_individuals, pos, neg) for (pos, neg) in zip(self.lp.e_pos, self.lp.e_neg))))

        self.Xpos = torch.LongTensor(self.lp.e_pos)
        self.Xneg = torch.LongTensor(self.lp.e_neg)
        self.Y = torch.FloatTensor(self.Y)

    def __len__(self):
        return self.num_data_points

    def __getitem__(self, idx):
        return self.Xpos[idx], self.Xneg[idx], self.Y[idx]
