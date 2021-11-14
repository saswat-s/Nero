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
from owlapy.render import DLSyntaxObjectRenderer


class LP:
    def __init__(self, *, e_pos: List[List[int]], e_neg: List[List[int]], instance_idx_mapping,
                 expressions_chain,
                 target_class_expressions):
        assert len(e_pos) == len(e_neg)
        self.e_pos = e_pos
        self.e_neg = e_neg
        self.num_learning_problems = len(self.e_pos)
        self.instance_idx_mapping = instance_idx_mapping
        self.idx_instance_mapping = dict(zip(instance_idx_mapping.values(),instance_idx_mapping.keys()))
        self.expressions_chain=expressions_chain
        self.target_class_expressions = target_class_expressions

    def __str__(self):
        return f'<LP object at {hex(id(self))}>\tdata_points: {self.num_learning_problems}\t|target_class_expressions|:{len(self.target_class_expressions)}'

    def __len__(self):
        return self.num_learning_problems

    def __iter__(self):
        for pos, neg in zip(self.e_pos, self.e_neg):
            yield [self.idx_instance_mapping[i] for i in pos], [self.idx_instance_mapping[i] for i in neg]


def ClosedWorld_ReasonerFactory(onto: OWLOntology) -> OWLReasoner:
    assert isinstance(onto, OWLOntology_Owlready2)
    base_reasoner = OWLReasoner_Owlready2_TempClasses(ontology=onto)
    reasoner = OWLReasoner_FastInstanceChecker(ontology=onto,
                                               base_reasoner=base_reasoner,
                                               negation_default=True)
    return reasoner


def compute_f1_target(target_class_expressions, pos, neg):
    res = []
    pos = set(pos)
    neg = set(neg)
    for t in target_class_expressions:
        res.append(f_measure(instances=t.idx_individuals, positive_examples=pos, negative_examples=neg))
    return res


class Dataset(torch.utils.data.Dataset):
    def __init__(self, lp: LP):
        self.lp = lp
        self.num_data_points = len(self.lp)
        self.Y = []

        with Pool(processes=4) as pool:
            self.Y = list(
                pool.starmap(compute_f1_target, ((self.lp.target_class_expressions, pos, neg) for (pos, neg) in
                                                 zip(self.lp.e_pos, self.lp.e_neg))))

        self.Xpos = torch.LongTensor(self.lp.e_pos)
        self.Xneg = torch.LongTensor(self.lp.e_neg)
        self.Y = torch.FloatTensor(self.Y)

    def __len__(self):
        return self.num_data_points

    def __getitem__(self, idx):
        return self.Xpos[idx], self.Xneg[idx], self.Y[idx]
