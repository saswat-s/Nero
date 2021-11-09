import torch
from owlapy.model import OWLOntology, OWLReasoner
from owlapy.owlready2 import OWLOntology_Owlready2
from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker


def ClosedWorld_ReasonerFactory(onto: OWLOntology) -> OWLReasoner:
    assert isinstance(onto, OWLOntology_Owlready2)
    base_reasoner = OWLReasoner_Owlready2_TempClasses(ontology=onto)
    reasoner = OWLReasoner_FastInstanceChecker(ontology=onto,
                                               base_reasoner=base_reasoner,
                                               negation_default=True)
    return reasoner


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        self.X = torch.LongTensor(inputs)
        self.Y = torch.FloatTensor(outputs)
        assert len(self.X) == len(self.Y)
        self.num_data_points = len(self.X)

    def __len__(self):
        return self.num_data_points

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

