import torch
from torch import nn
from typing import Dict, List
from ontolearn.search import RL_State
from owlapy.render import DLSyntaxObjectRenderer
import time

class NCEL:
    def __init__(self, model: torch.nn.Module, quality_func, target_class_expressions: List[RL_State], instance_idx_mapping: Dict):
        self.model = model
        self.quality_func = quality_func
        self.target_class_expressions = target_class_expressions
        self.instance_idx_mapping = instance_idx_mapping
        self.renderer = DLSyntaxObjectRenderer()

    def forward(self, xpos, xneg):
        return self.model(xpos, xneg)

    def fit(self, pos, neg, topK: int):
        start_time = time.time()

        xpos = torch.LongTensor([[self.instance_idx_mapping[i] for i in pos]])
        xneg = torch.LongTensor([[self.instance_idx_mapping[i] for i in neg]])
        pred = self.forward(xpos, xneg)

        sort_values, sort_idxs = torch.sort(pred, dim=1, descending=True)
        sort_idxs = sort_idxs.cpu().numpy()[0]

        results = []
        # We could apply multi_processing here
        # Explore only top K class expressions that have received highest K scores
        for i in sort_idxs[:topK]:
            s = self.quality_func(instances={i.get_iri().as_str() for i in self.target_class_expressions[i].instances},
                                  positive_examples=set(pos), negative_examples=set(neg))
            results.append((s, self.target_class_expressions[i]))

        num_expression_tested=len(results)
        results: List[RL_State] = sorted(results, key=lambda x: x[0], reverse=False)

        f1, top_pred = results.pop()

        report = {'Prediction': self.renderer.render(top_pred.concept),
                  'Instances': {i.get_iri().as_str() for i in top_pred.instances},
                  'NumClassTested': num_expression_tested,
                  'Runtime': time.time() - start_time,
                  }

        return report

    def __str__(self):
        return f'NCEL with {self.model}'

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)

    def state_dict(self):
        return self.model.state_dict()

    def parameters(self):
        return self.model.parameters()

    def embeddings_to_numpy(self):
        return self.model.embeddings.weight.data.detach().numpy()

    def get_target_class_expressions(self):
        return (self.renderer.render(cl.concept) for cl in self.target_class_expressions)
