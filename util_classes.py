import torch
from owlapy.model import OWLOntology, OWLReasoner
from owlapy.owlready2 import OWLOntology_Owlready2
from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
from static_funcs import target_scores
# @TODO Note quite sure
# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool


class LP:
    def __init__(self, learning_problems, instances, instance_idx_mapping, target_class_expressions,
                 target_individuals):
        self.data_points = learning_problems
        self.instances = instances
        self.instance_idx_mapping = instance_idx_mapping
        self.target_class_expressions = target_class_expressions
        self.target_idx_individuals = [[self.instance_idx_mapping[x] for x in i] for i in target_individuals]

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


def temp(x):
    return len(x)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, lp: LP):
        self.lp = lp
        self.num_data_points = len(self.lp)
        self.Y = []
        for dp in self.lp.data_points:
            res = []
            for i_label in self.lp.target_idx_individuals:
                res.append(target_scores(i_label, dp))
            self.Y.append(res)

        # with Pool(processes=4) as pool:
        #    Y = list(pool.starmap(target_scores, ([i_label, dp] for i_label in self.lp.target_idx_individuals
        #                                          for dp in self.lp.data_points), chunksize=100))

        self.X = torch.LongTensor(self.lp.data_points)
        self.Y = torch.FloatTensor(self.Y)

        print(self.Y.max())

        # To free some memory
        self.lp.data_points = None

    def __len__(self):
        return self.num_data_points

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
