import torch
from owlapy.model import OWLOntology, OWLReasoner
from owlapy.owlready2 import OWLOntology_Owlready2
from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
from .static_funcs import compute_f1_target
from .expression import TargetClassExpression
from multiprocessing import Pool
from typing import List, Iterable, Dict
import numpy as np
import itertools
from owlapy.render import DLSyntaxObjectRenderer


class LP:
    """ Learning Problem holder"""

    def __init__(self, *, e_pos: List[List[int]], e_neg: List[List[int]], instance_idx_mapping: Dict,
                 target_class_expressions: List[TargetClassExpression]):
        # (1) Sanity checking: we have at least 1 learning problem
        assert len(e_pos) == len(e_neg) and len(e_neg) > 0
        # (2) Sanity checking: first learning problem is stored in a list
        assert isinstance(e_pos[0], list) and isinstance(e_neg[-1], list)
        # (3) Sanity checking: The last item of the first learning problem is integer
        assert isinstance(e_pos[0][-1], int) and isinstance(e_neg[-1][0], int)
        assert isinstance(instance_idx_mapping, dict) and len(instance_idx_mapping) > 0
        assert isinstance(target_class_expressions, list) and len(target_class_expressions) > 0

        # (1) Individual to integer mapping
        self.e_pos = e_pos
        self.e_neg = e_neg
        self.num_learning_problems = len(self.e_pos)
        # {str_individuals:integer} ,e.g. {individual.get_iri().as_str(): i for i, individual in enumerate(kb.individuals())}
        self.str_individuals_to_idx = instance_idx_mapping
        self.idx_to_str_individuals = dict(zip(instance_idx_mapping.values(), instance_idx_mapping.keys()))
        self.target_class_expressions = target_class_expressions

    def __str__(self):
        return f'<LP object at {hex(id(self))}>\t|Training Data (D)|: {self.num_learning_problems}\t|Tasks|:{len(self.target_class_expressions)}'

    def __len__(self):
        return self.num_learning_problems

    def __iter__(self):
        for pos, neg in zip(self.e_pos, self.e_neg):
            yield [self.idx_to_str_individuals[i] for i in pos], [self.idx_to_str_individuals[i] for i in neg]

    def __getitem__(self, i):
        if isinstance(i, Iterable):
            res = []
            for ith in i:
                res.append(([self.idx_to_str_individuals[_] for _ in self.e_pos[ith]],
                            [self.idx_to_str_individuals[_] for _ in self.e_neg[ith]]))
            return res
        else:
            assert isinstance(i, int)
            return [self.idx_to_str_individuals[_] for _ in self.e_pos[i]], [self.idx_to_str_individuals[i] for i in
                                                                             self.e_neg[i]]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, lp: LP, num_workers_for_labelling: int):
        if num_workers_for_labelling == 0:
            num_workers_for_labelling = 1
        self.lp = lp
        self.num_workers_for_labelling = num_workers_for_labelling
        self.num_data_points = len(self.lp)
        self.Y = []
        # This is quite memory expensive.;perfectfmeasure binarieze
        with Pool(processes=self.num_workers_for_labelling) as pool:
            self.Y = list(
                pool.starmap(compute_f1_target, ((self.lp.target_class_expressions, pos, neg) for (pos, neg) in
                                                 zip(self.lp.e_pos, self.lp.e_neg))))

        self.Xpos = torch.LongTensor(self.lp.e_pos) # torch.ShortTensor
        self.Xneg = torch.LongTensor(self.lp.e_neg) # torch.ShortTensor
        self.Y = torch.FloatTensor(self.Y) # torch.HalfTensor

    def __len__(self):
        return self.num_data_points

    def __getitem__(self, idx):
        return self.Xpos[idx], self.Xneg[idx], self.Y[idx]


class DatasetWithOnFlyLabelling(torch.utils.data.Dataset):
    def __init__(self, lp: LP):
        self.lp = lp
        self.num_data_points = len(self.lp)

    def __len__(self):
        return self.num_data_points

    def __getitem__(self, idx):
        # We save memory but not runtime ?!
        pos_idx, neg_idx = self.lp.e_pos[idx], self.lp.e_neg[idx]
        f1 = compute_f1_target(self.lp.target_class_expressions, pos_idx, neg_idx)
        return torch.LongTensor(pos_idx), torch.LongTensor(neg_idx), torch.FloatTensor(f1)
