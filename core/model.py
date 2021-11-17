import torch
from torch import nn
from typing import Dict, List, Iterable
from ontolearn.search import RL_State
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.model import OWLClass, OWLObjectComplementOf, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, \
    OWLObjectUnionOf, OWLObjectIntersectionOf, OWLClassExpression, OWLNothing, OWLThing, OWLNaryBooleanClassExpression
from ontolearn.refinement_operators import LengthBasedRefinement
from .static_funcs import apply_rho_on_rl_state
import time


class NCEL:
    def __init__(self, model: torch.nn.Module,
                 quality_func,
                 target_class_expressions,
                 instance_idx_mapping: Dict):
        self.model = model
        self.quality_func = quality_func
        self.target_class_expressions = target_class_expressions
        self.instance_idx_mapping = instance_idx_mapping
        self.renderer = DLSyntaxObjectRenderer()

    def forward(self, *, xpos, xneg):
        return self.model(xpos, xneg)

    def positive_embeddings_from_iterable_of_individuals(self, pos: Iterable[str]):
        pred = self.forward(xpos=torch.LongTensor([[self.instance_idx_mapping[i] for i in pos]]),
                            xneg=torch.LongTensor([[self.instance_idx_mapping[i] for i in neg]]))
        return self.model(xpos, xneg)

    def negative_embeddings(self, xpos):
        return self.model(xpos, xneg)

    def __intersection_topK(self, results, set_pos, set_neg):
        """
        Intersect top K class expressions

        This often deteriorates the performance. This may indicate that topK concepts explain the different aspect of
        the goal expression
        :param results:
        :param set_pos:
        :param set_neg:
        :return:
        """
        # apply some operation
        for (_, exp_i) in results:
            for (__, exp_j) in results:
                if exp_i == exp_j:
                    continue

                next_rl_state = RL_State(OWLObjectIntersectionOf((exp_i.concept, exp_j.concept)), parent_node=exp_i)
                next_rl_state.length = self.kb.cl(next_rl_state.concept)
                next_rl_state.instances = set(self.kb.individuals(next_rl_state.concept))
                quality = self.quality_func(
                    instances={i.get_iri().as_str() for i in next_rl_state.instances},
                    positive_examples=set_pos, negative_examples=set_neg)
                # Do not assing quality for target states
                print(exp_i)
                print(exp_j)
                print(quality)

    def __union_topK(self, results, set_pos, set_neg):
        """
        Union topK expressions
        :param results:
        :param set_pos:
        :param set_neg:
        :return:
        """
        # apply some operation
        for (_, exp_i) in results:
            for (__, exp_j) in results:
                if exp_i == exp_j:
                    continue

                next_rl_state = RL_State(OWLObjectUnionOf((exp_i.concept, exp_j.concept)), parent_node=exp_i)
                next_rl_state.length = self.kb.cl(next_rl_state.concept)
                next_rl_state.instances = set(self.kb.individuals(next_rl_state.concept))
                quality = self.quality_func(
                    instances={i.get_iri().as_str() for i in next_rl_state.instances},
                    positive_examples=set_pos, negative_examples=set_neg)

    def __refine_topK(self, results, set_pos, set_neg, stop_at):
        extended_results = []
        for ith, (_, topK_target_expression) in enumerate(results):
            for refinement_topK in apply_rho_on_rl_state(topK_target_expression, self.rho, self.kb):
                s: float = self.quality_func(
                    instances={i.get_iri().as_str() for i in refinement_topK.instances},
                    positive_examples=set_pos, negative_examples=set_neg)
                if s > _:
                    # print(f'Refinement ({s}) is better than its parent ({_})')
                    extended_results.append((s, refinement_topK))
                    if s == 1.0:
                        print('Goal found in the local search')
                        break
                if ith == stop_at:
                    break
        return extended_results

    def fit(self, pos: [str], neg: [str], topK: int, local_search=False) -> Dict:
        try:
            assert topK > 0
        except AssertionError:
            print(f'topK must be greater than 0. Currently:{topK}')
        start_time = time.time()
        goal_found = False
        try:
            idx_pos = [self.instance_idx_mapping[i] for i in pos]
        except KeyError:
            print('Ensure that Positive examples can be found in the input KG')
            print(pos)
            exit(1)
        try:
            idx_neg = [self.instance_idx_mapping[i] for i in neg]
        except KeyError:
            print('Ensure that Positive examples can be found in the input KG')
            print(neg)
            exit(1)
        pred = self.forward(xpos=torch.LongTensor([idx_pos]),
                            xneg=torch.LongTensor([idx_neg]))

        sort_values, sort_idxs = torch.sort(pred, dim=1, descending=True)
        sort_idxs = sort_idxs.cpu().numpy()[0]

        results = []
        # We could apply multi_processing here
        # Explore only top K class expressions that have received highest K scores
        set_pos = set(pos)
        set_neg = set(neg)
        for i in sort_idxs[:topK]:
            s: float = self.quality_func(
                instances=self.target_class_expressions[i].individuals,
                positive_examples=set_pos, negative_examples=set_neg)
            results.append((s, self.target_class_expressions[i]))
            if s == 1.0:
                # print('Goal Found in the tunnelling')
                goal_found = True
                break
        # self.__intersection_topK(results, set_pos, set_neg)
        # self.__union_topK(results, set_pos, set_neg)
        if goal_found is False and local_search:
            extended_results = self.__refine_topK(results, set_pos, set_neg, stop_at=topK)
            results.extend(extended_results)

        num_expression_tested = len(results)
        results = sorted(results, key=lambda x: x[0], reverse=True)
        f1, top_pred = results[0]

        report = {'Prediction': top_pred.name,
                  'Instances': top_pred.individuals,
                  'F1-Score': f1,
                  'NumClassTested': num_expression_tested,
                  'Runtime': time.time() - start_time,
                  }

        return report

    def predict(self, pos: [str], neg: [str], topK: int, local_search=False) -> List:
        try:
            assert topK > 0
        except AssertionError:
            print(f'topK must be greater than 0. Currently:{topK}')
        goal_found = False
        try:
            idx_pos = [self.instance_idx_mapping[i] for i in pos]
        except KeyError:
            print('Ensure that Positive examples can be found in the input KG')
            print(pos)
            exit(1)
        try:
            idx_neg = [self.instance_idx_mapping[i] for i in neg]
        except KeyError:
            print('Ensure that Positive examples can be found in the input KG')
            print(neg)
            exit(1)
        pred = self.forward(xpos=torch.LongTensor([idx_pos]),
                            xneg=torch.LongTensor([idx_neg]))

        sort_values, sort_idxs = torch.sort(pred, dim=1, descending=True)
        sort_idxs = sort_idxs.cpu().numpy()[0]

        results = []
        # We could apply multi_processing here
        # Explore only top K class expressions that have received highest K scores
        set_pos = set(pos)
        set_neg = set(neg)
        for i in sort_idxs[:topK]:
            s: float = self.quality_func(
                instances=self.target_class_expressions[i].individuals,
                positive_examples=set_pos, negative_examples=set_neg)
            results.append((s, self.target_class_expressions[i]))
            if s == 1.0:
                break

        return sorted(results, key=lambda x: x[0], reverse=True)

    def __str__(self):
        return f'NCEL with {self.model.name}'

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
