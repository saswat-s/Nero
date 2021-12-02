import torch
from torch import nn
from typing import Dict, List, Iterable, Set
from .dl_expression import TargetClassExpression, ClassExpression

"""
from ontolearn.search import RL_State
from owlapy.model import OWLClass, OWLObjectComplementOf, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, \
    OWLObjectUnionOf, OWLObjectIntersectionOf, OWLClassExpression, OWLNothing, OWLThing, OWLNaryBooleanClassExpression
from ontolearn.refinement_operators import LengthBasedRefinement
"""
from owlapy.render import DLSyntaxObjectRenderer
from .static_funcs import apply_rho_on_rl_state
import time
from .data_struct import ExpressionQueue


class NERO:
    def __init__(self, model: torch.nn.Module,
                 quality_func,
                 target_class_expressions,
                 instance_idx_mapping: Dict):
        self.model = model
        self.quality_func = quality_func
        # expression ordered by id.
        self.target_class_expressions = target_class_expressions
        self.str_target_class_expression_to_label_id = {i.name: i.label_id for i in self.target_class_expressions}

        self.instance_idx_mapping = instance_idx_mapping
        self.inverse_instance_idx_mapping = dict(
            zip(self.instance_idx_mapping.values(), self.instance_idx_mapping.keys()))
        self.renderer = DLSyntaxObjectRenderer()
        self.max_top_k = len(self.target_class_expressions)

        self.set_str_all_instances = set(list(self.instance_idx_mapping.keys()))

    def get_target_exp_found_in_chain(self, expression_chain):
        for i in expression_chain:
            if i in self.str_target_class_expression_to_label_id:
                yield self.target_class_expressions[self.str_target_class_expression_to_label_id[i]]

    def call_quality_function(self, *, set_str_individual: Set[str], set_str_pos: Set[str],
                              set_str_neg: Set[str]) -> float:
        return self.quality_func(instances=set_str_individual, positive_examples=set_str_pos,
                                 negative_examples=set_str_neg)

    def forward(self, *, xpos, xneg):
        return self.model(xpos, xneg)

    def positive_expression_embeddings(self, individuals: Iterable[str]):
        return self.model.positive_expression_embeddings(
            torch.LongTensor([[self.instance_idx_mapping[i] for i in individuals]]))

    def negative_expression_embeddings(self, individuals: Iterable[str]):
        return self.model.negative_expression_embeddings(
            torch.LongTensor([[self.instance_idx_mapping[i] for i in individuals]]))

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
                quality = self.call_quality_function(
                    set_str_individual={i.get_iri().as_str() for i in next_rl_state.instances},
                    set_str_pos=set_pos,
                    set_str_neg=set_neg)
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
        if topK is None:
            topK = self.max_top_k
        try:
            assert topK > 0
        except AssertionError:
            print(f'topK must be greater than 0. Currently:{topK}')
            topK = self.max_top_k
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
        set_str_pos = set(pos)
        set_str_neg = set(neg)

        for i in sort_idxs[:topK]:
            str_instances = {self.inverse_instance_idx_mapping[i] for i in
                             self.target_class_expressions[i].idx_individuals}
            s: float = self.quality_func(
                instances=str_instances,
                positive_examples=set_str_pos, negative_examples=set_str_neg)
            results.append((s, self.target_class_expressions[i], str_instances))
            if s == 1.0:
                # print('Goal Found in the tunnelling')
                goal_found = True
                break
        if goal_found is False and local_search:
            extended_results = self.__refine_topK(results, set_pos, set_neg, stop_at=topK)
            results.extend(extended_results)

        num_expression_tested = len(results)
        results = sorted(results, key=lambda x: x[0], reverse=True)
        f1, top_pred, top_str_instances = results[0]

        report = {'Prediction': top_pred.name,
                  'Instances': top_str_instances,
                  'F1-Score': f1,
                  'NumClassTested': num_expression_tested,
                  'Runtime': time.time() - start_time,
                  }

        return report

    def __predict_sanity_checking(self, pos: [str], neg: [str], topK: int = None, local_search=False):
        if topK is None:
            topK = self.max_top_k
        elif isinstance(topK, int) or isinstance(topK, flat):
            try:
                assert topK > 0
                topK = int(round(topK))
            except AssertionError:
                print(f'topK must be greater than 0. Currently:{topK}')
                topK = self.max_top_k

        assert len(pos) > 0
        assert len(neg) > 0

    def single_dl_exp_learning(self, set_str_pos, set_str_neg, n):

        _, sort_idxs = torch.sort(
            self.forward(xpos=torch.LongTensor([[self.instance_idx_mapping[i] for i in set_str_pos]]),
                         xneg=torch.LongTensor([[self.instance_idx_mapping[i] for i in set_str_neg]])), dim=1,
            descending=True)
        sort_idxs = sort_idxs.cpu().numpy()[0]
        results = ExpressionQueue()
        for idx_target in sort_idxs[:100]:
            expression = self.target_class_expressions[idx_target]
            str_instances = {self.inverse_instance_idx_mapping[_] for _ in expression.idx_individuals}

            # True positive and True Negative should be as large as possible
            tp = len(set_str_pos.intersection(str_instances))
            tn = len(set_str_neg.difference(str_instances))
            # False Negative and False Positive should be as less as possible
            fn = len(set_str_pos.difference(str_instances))
            fp = len(set_str_neg.intersection(str_instances))

            # fully coverage score and as small as possible
            s = (tp + tn) / (tp + tn + fp + fn)

            # Sort expression with their negative coverage of intersection of E^- and Target exp
            results.put(s, expression, str_instances)
        return results.get_top(n)

    def apply_local_search(self, top_pred_exp_queue: ExpressionQueue, set_str_pos: Set[str], set_str_neg: Set[str]):
        first_preds = ExpressionQueue()
        refinements_first_preds = ExpressionQueue()

        # Iterate over top K expressions.
        for (f1, tcl, tcl_set_str_individuals) in top_pred_exp_queue:
            first_preds.put(quality=f1, tce=tcl, str_individuals=tcl_set_str_individuals)
            tcl_set_str_individuals: set
            tcl: TargetClassExpression
            # (1) Calculate Total Positive Coverage, i.e. \forall e \in E^+ check G \models h(e).
            total_positive_coverage_rate = len(tcl_set_str_individuals & set_str_pos) / len(set_str_neg)
            if total_positive_coverage_rate == 1.0:
                # (1.1) We need to remove some individuals from tcl that are found in E^-
                negs_to_be_removed = tcl_set_str_individuals & set_str_neg
                print('|I AND E^-|/ |E^-| :', len(tcl_set_str_individuals & set_str_neg) / len(set_str_neg))
                print('|I AND E^-|/ |I| :', len(tcl_set_str_individuals & set_str_neg) / len(tcl_set_str_individuals))

                print(f'Domain:{tcl}\tQuality{f1:.2f}')
                # Find a target expression that does not cover
                for expresssions in self.target_class_expressions:
                    str_instances = {self.inverse_instance_idx_mapping[_] for _ in expresssions.idx_individuals}

                    score = (1 + len(str_instances.intersection(set_str_pos))) / (
                                1 + len(str_instances.intersection(set_str_neg)))

                    refinements_first_preds.put(score, expresssions, str_instances)

                for s, exp, str_indiviudals in refinements_first_preds.get_top(10000):
                    """
                    best_ref_tcl = ClassExpression(name=f'({tcl.name}) ⊔  ({exp.name})',
                                                   individuals=tcl_set_str_individuals.union(str_indiviudals),
                                                   expression_chain=tcl.expression_chain + [exp.name])
                    """

                    s: float = self.quality_func(instances=str_indiviudals,
                                                 positive_examples=set_str_pos,
                                                 negative_examples=set_str_neg)
                    if s>.82:
                        print(s)

                exit(1)
                ref_tcl = ClassExpression(name=f'({tcl.name}) ⊓  ({expresssions.name})',
                                          individuals=tcl_set_str_individuals.intersection(str_instances),
                                          expression_chain=tcl.expression_chain + [expresssions.name])
                assert len(ref_tcl.individuals.intersection(set_str_neg)) == 0

                print(best_ref_tcl)
                s: float = self.quality_func(instances=best_ref_tcl.individuals, positive_examples=set_str_pos,
                                             negative_examples=set_str_neg)
                print(s)
                exit(1)

                for (score, exp, str_indiviudals) in self.single_dl_exp_learning(set_str_pos=negs_to_be_removed,
                                                                                 set_str_neg=set_str_neg, n=3):
                    print(len(str_indiviudals & set_str_pos))
                    print(len(str_indiviudals & set_str_neg))

                    continue

                    s: float = self.quality_func(instances=ref_tcl.individuals, positive_examples=set_str_pos,
                                                 negative_examples=set_str_pos)
                    print(s)
                exit(1)

                for i in self.get_target_exp_found_in_chain(tcl.expression_chain):
                    s: float = self.quality_func(
                        instances={self.inverse_instance_idx_mapping[_] for _ in i.idx_individuals},
                        positive_examples=set_str_pos,
                        negative_examples=set_str_pos)
                    print(i, s)
                exit(1)

                exit(1)

                for (f1_exp, exp, str_indiviudals) in self.single_dl_exp_learning(negs_to_be_removed,
                                                                                  self.top_neg_instances(
                                                                                      negs_to_be_removed), n=10):
                    str_indiviudals: set
                    print(f1_exp)
                    print(exp)

                    exit(1)

                    s: float = self.quality_func(
                        instances=tcl_set_str_individuals.union(self.top_neg_instances(str_indiviudals)),
                        positive_examples=set_str_pos,
                        negative_examples=set_str_pos)
                    # print(s)

                    exit(1)
                    s: float = self.quality_func(instances=str_indiviudals,
                                                 positive_examples=set_str_pos,
                                                 negative_examples=set_str_pos)

                    ref_tcl = ClassExpression(name=f'({tcl.name}) ⊔ ( {exp.name})',
                                              individuals=str_indiviudals.union(tcl_set_str_individuals),
                                              expression_chain=tcl.expression_chain + [exp.name])

                    s: float = self.quality_func(instances=ref_tcl.individuals,
                                                 positive_examples=set_str_pos,
                                                 negative_examples=set_str_pos)

                    ref_tcl = ClassExpression(name=f'({tcl.name}) ⊓  ( {exp.name})',
                                              individuals=str_indiviudals.intersection(tcl_set_str_individuals),
                                              expression_chain=tcl.expression_chain + [exp.name])

                    s2: float = self.quality_func(instances=ref_tcl.individuals,
                                                  positive_examples=set_str_pos,
                                                  negative_examples=set_str_pos)
                    print('Refinement Intersect and Union Domain Quality:', s, s2)

                exit(1)
            else:
                """ Dont do anything"""
                continue

            # tcl_set_str_individuals.issuperset(pos)
            # print(f1, tcl.name, len(tcl_set_str_individuals & pos), len(tcl_set_str_individuals & neg))

        exit(1)

    def predict(self, str_pos: [str], str_neg: [str], topK: int = None, local_search=True) -> List:
        start_time = time.time()
        self.__predict_sanity_checking(pos=str_pos, neg=str_neg, topK=topK, local_search=local_search)
        set_pos, set_neg = set(str_pos), set(str_neg)
        idx_pos, idx_neg = [], []
        results = ExpressionQueue()

        try:
            idx_pos = [self.instance_idx_mapping[i] for i in str_pos]
            idx_neg = [self.instance_idx_mapping[i] for i in str_neg]
        except KeyError:
            print('Ensure that URIs are valid and can be found in the input KG')
            print(str_pos)
            print(str_neg)
            exit(1)

        _, sort_idxs = torch.sort(self.forward(xpos=torch.LongTensor([idx_pos]),
                                               xneg=torch.LongTensor([idx_neg])), dim=1, descending=True)
        sort_idxs = sort_idxs.cpu().numpy()[0]

        goal_found = False
        for the_exploration, idx_target in enumerate(sort_idxs[:topK]):
            str_instances = {self.inverse_instance_idx_mapping[_] for _ in
                             self.target_class_expressions[idx_target].idx_individuals}

            s = self.call_quality_function(set_str_individual=str_instances,
                                           set_str_pos=set_pos,
                                           set_str_neg=set_neg)
            results.put(s,
                        self.target_class_expressions[idx_target],
                        str_instances)
            if s == 1.0:
                goal_found = True
                break

        # If Goal is not found in top K predictions
        if goal_found is False and local_search:
            self.apply_local_search(top_pred_exp_queue=results, set_str_pos=set_pos, set_str_neg=set_neg)
        else:
            return results, len(results), time.time() - start_time

    def __str__(self):
        return f'NERO with {self.model.name}'

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
