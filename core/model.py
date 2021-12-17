import torch
from torch import nn
from typing import Dict, List, Iterable, Set
from .expression import ClassExpression,TargetClassExpression
from owlapy.render import DLSyntaxObjectRenderer
from .static_funcs import apply_rho_on_rl_state
import time
from .data_struct import ExpressionQueue, State, SearchTree
from ontolearn import KnowledgeBase
from .refinement_operator import SimpleRefinement


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

        self.retrieve_counter = 0

    def get_target_exp_found_in_chain(self, expression_chain):
        for i in expression_chain:
            if i in self.str_target_class_expression_to_label_id:
                yield self.target_class_expressions[self.str_target_class_expression_to_label_id[i]]

    def call_quality_function(self, *, set_str_individual: Set[str], set_str_pos: Set[str],
                              set_str_neg: Set[str]) -> float:
        self.retrieve_counter += 1
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

    def predict_sanity_checking(self, pos: [str], neg: [str], topK: int = None):
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

    def str_to_index_mapping(self, individuals: List[str]) -> List[int]:
        try:
            return [self.instance_idx_mapping[i] for i in individuals]
        except KeyError:
            print('Ensure that URIs are valid and can be found in the input KG')
            print(individuals)
            exit(1)

    def retrieval_of_individuals(self, target_class_expression):
        return {self.inverse_instance_idx_mapping[_] for _ in target_class_expression.idx_individuals}

    def predict(self, str_pos: [str], str_neg: [str], topK: int = None, local_search=True) -> ExpressionQueue:
        """

        :param str_pos: string rep of E^+
        :param str_neg: string rep of E^+
        :param topK: number of top ranked expressions to be tested
        :param local_search: Return a queue of top ranked expressions sorted as descending order of qualities
        :return:
        """
        start_time = time.time()
        self.predict_sanity_checking(pos=str_pos, neg=str_neg, topK=topK)
        set_pos, set_neg = set(str_pos), set(str_neg)
        idx_pos = self.str_to_index_mapping(str_pos)
        idx_neg = self.str_to_index_mapping(str_neg)
        goal_found = False

        # (2) Initialize a priority queue for top K Target Expressions.
        results = ExpressionQueue()
        # (3) Predict scores and sort index target expressions in descending order of assigned scores.
        _, sort_idxs = torch.sort(self.forward(xpos=torch.LongTensor([idx_pos]),
                                               xneg=torch.LongTensor([idx_neg])), dim=1, descending=True)
        sort_idxs = sort_idxs.cpu().numpy()[0]
        # (4) Iterate over the sorted index of target expressions.
        for the_exploration, idx_target in enumerate(sort_idxs[:topK]):
            # (4.1.) Retrieval of instance.
            str_instances = self.retrieval_of_individuals(self.target_class_expressions[idx_target])

            quality_score = self.call_quality_function(set_str_individual=str_instances,
                                                       set_str_pos=set_pos,
                                                       set_str_neg=set_neg)
            results.put(quality_score, self.target_class_expressions[idx_target], str_instances)
            if quality_score == 1.0:
                goal_found = True
                break

        # If Goal is not found in top K predictions
        # Later call CELOE if goal not found
        return results, len(results), time.time() - start_time

    def fit(self, str_pos: [str], str_neg: [str], topK: int = None, use_search=None, kb_path=None) -> Dict:
        """

        :param kb_path:
        :param use_search:
        :param kb:
        :param str_pos: string rep of E^+
        :param str_neg: string rep of E^+
        :param topK: number of top ranked expressions to be tested
        :param local_search: Return a queue of top ranked expressions sorted as descending order of qualities
        :return:
        """
        # (1) Initialize Learning Problem.
        start_time = time.time()
        self.predict_sanity_checking(pos=str_pos, neg=str_neg, topK=topK)
        self.retrieve_counter = 0
        set_pos, set_neg = set(str_pos), set(str_neg)
        idx_pos = self.str_to_index_mapping(str_pos)
        idx_neg = self.str_to_index_mapping(str_neg)
        goal_found = False

        # (2) Initialize a priority queue for top K Target Expressions.
        top_prediction_queue = SearchTree()

        # (3) Predict scores and sort index target expressions in descending order of assigned scores.
        _, sort_idxs = torch.sort(self.forward(xpos=torch.LongTensor([idx_pos]),
                                               xneg=torch.LongTensor([idx_neg])), dim=1, descending=True)
        sort_idxs = sort_idxs.cpu().numpy()[0]

        # (4) Iterate over the sorted index of target expressions.
        for idx_target in sort_idxs[:topK]:
            # (5) Retrieval of instance.
            str_instances = self.retrieval_of_individuals(self.target_class_expressions[idx_target])
            # (6) Compute Quality.
            quality_score = self.call_quality_function(set_str_individual=str_instances,
                                                       set_str_pos=set_pos,
                                                       set_str_neg=set_neg)
            # (7) Put CE into the priority queue.
            top_prediction_queue.put(ClassExpression(name=self.target_class_expressions[idx_target].name,
                                                     str_individuals=str_instances,
                                                     expression_chain=self.target_class_expressions[
                                                         idx_target].expression_chain,
                                                     quality=quality_score))

            # (8) If goal is found, we do not need to compute scores.
            if quality_score == 1.0:
                goal_found = True
                break

        # (9) IF goal is not found, we do search
        if goal_found is False:
            if use_search == 'Continues':
                best_pred = top_prediction_queue.get()
                top_prediction_queue.put(best_pred)
                self.apply_continues_search(top_prediction_queue, set_pos, set_neg)
                best_constructed_expression = top_prediction_queue.get()
                if best_constructed_expression > best_pred:
                    best_pred = best_constructed_expression
            elif use_search == 'IntersectNegatives':
                # Let t \in Top, m \in Least
                # Assumption
                # (1) \for all i in E^+ t(i)=1 and \exist i in E^- t(i)=1
                # (2) \for all i in E^- m(i)=1 and \exist i in E^+ m(i)=1
                # (3) Intersect m and take AND
                st_to_intersect = SearchTree()

                topK_lowest_predictions = sort_idxs[-10:]
                for idx_target in topK_lowest_predictions:
                    # (5) Retrieval of instance. and negate it
                    str_instances = self.set_str_all_instances - self.retrieval_of_individuals(
                        self.target_class_expressions[idx_target])
                    # (6) Compute Quality.
                    quality_score = self.call_quality_function(set_str_individual=str_instances,
                                                               set_str_pos=set_pos,
                                                               set_str_neg=set_neg)
                    # (7) Put CE into the priority queue.
                    st_to_intersect.put(
                        ClassExpression(name='Neg(' + self.target_class_expressions[idx_target].name + ')',
                                        str_individuals=str_instances,
                                        expression_chain=self.target_class_expressions[idx_target].expression_chain + [
                                            'NEG'],
                                        quality=quality_score))
                results = self.apply_continues_search_with_negatives(top_prediction_queue, st_to_intersect, set_pos,
                                                                     set_neg)
                best_pred = results.get()
            else:
                best_pred = top_prediction_queue.get()
        else:
            best_pred = top_prediction_queue.get()

        f1, name, str_instances = best_pred.quality, best_pred.name, best_pred.str_individuals
        report = {'Prediction': best_pred.name,
                  'Instances': str_instances,
                  'F-measure': f1,
                  'NumClassTested': self.retrieve_counter,
                  'Runtime': time.time() - start_time,
                  }
        return report

    def generate_expression(self, *, i, kb, expression_chain):
        expression = ClassExpression(name=self.renderer.render(i),
                                     str_individuals=set([_.get_iri().as_str() for _ in kb.individuals(i)]),
                                     expression_chain=expression_chain)
        return expression

    def apply_continues_search(self, top_states, set_str_pos: Set[str], set_str_neg: Set[str]):
        """

        :param kb:
        :param top_states:
        :param set_str_pos:
        :param set_str_neg:
        :return:
        """
        num_explore_concepts = len(top_states) // 10
        max_size = len(top_states) + num_explore_concepts
        # (1) Get embeddings of E^+.
        pos_emb = self.positive_expression_embeddings(set_str_pos)
        # (2) Get embeddings of E^-.
        neg_emb = self.negative_expression_embeddings(set_str_neg)
        last_sim = None
        st = []
        small = []
        # (3) Iterate over most promising states and compute similarity scores between E^+ and E^-
        for i in top_states:
            st.append(i)
            # (4) Embedding of most promising state
            target_emb = self.positive_expression_embeddings(i.str_individuals)
            # (5) Compute MSE Loss between | ((4) -neg_emb) - pos_emb|
            sim = torch.nn.functional.mse_loss(input=target_emb - neg_emb, target=pos_emb)
            if last_sim is None:
                last_sim = sim

            if sim < last_sim:
                small.append(i)
                last_sim = sim

        for i in small:
            for j in st:
                if i == j:
                    continue
                i_or_j = i + j
                i_or_j.quality = self.call_quality_function(set_str_individual=i_or_j.str_individuals,
                                                            set_str_pos=set_str_pos,
                                                            set_str_neg=set_str_neg)
                if i_or_j.quality > i.quality:
                    top_states.put(i_or_j)
                    if len(top_states) == max_size:
                        break

                i_and_j = i * j
                i_and_j.quality = self.call_quality_function(set_str_individual=i_and_j.str_individuals,
                                                             set_str_pos=set_str_pos,
                                                             set_str_neg=set_str_neg)
                if i_and_j.quality > i.quality:
                    top_states.put(i_and_j)
                    if len(top_states) == max_size:
                        break
            if len(top_states) == max_size:
                break
        assert len(top_states) <= max_size

    def apply_continues_search_with_negatives(self, top_states, st_to_intersect, set_str_pos: Set[str],
                                              set_str_neg: Set[str]):
        """

        :param kb:
        :param top_states:
        :param set_str_pos:
        :param set_str_neg:
        :return:
        """
        st = SearchTree()
        result = SearchTree()
        for i in top_states:
            for j in st_to_intersect:
                i_and_j = i * j
                i_and_j.quality = self.call_quality_function(set_str_individual=i_and_j.str_individuals,
                                                             set_str_pos=set_str_pos,
                                                             set_str_neg=set_str_neg)
                st.put(i_and_j)
                if len(st) == top_states:
                    break
            if len(st) == top_states:
                break
        result.extend_queue(st)
        result.extend_queue(top_states)

        return result

    def apply_combinatorial_local_search(self, kb: KnowledgeBase, most_promissing_states, set_str_pos: Set[str],
                                         set_str_neg: Set[str]):
        rho = SimpleRefinement(knowledge_base=kb)

        st = SearchTree()
        top_refinement_class_expressions = []
        for i in rho.top_refinements:
            if self.renderer.render(i) == '⊥':
                continue
            expression = self.generate_expression(i=i, kb=kb, expression_chain=['TOP'])
            expression.quality = self.call_quality_function(set_str_individual=expression.str_individuals,
                                                            set_str_pos=set_str_pos,
                                                            set_str_neg=set_str_neg)
            # Either Full coverage of Positives
            if expression.str_individuals.issuperset(set_str_pos) > 0:
                st.put(expression)
                top_refinement_class_expressions.append(expression)

            # Non coverage of negatives
            elif len(expression.str_individuals & set_str_neg) == 0:
                st.put(expression)
                top_refinement_class_expressions.append(expression)
            else:
                """ Ignore concepts that do not capture all positives """
        st2 = SearchTree()
        # Iterate over refinements in descending order of quality
        for i in st:
            st2.put(i)
            for j in top_refinement_class_expressions:
                if i == j:
                    continue

                quality_intersection = self.call_quality_function(
                    set_str_individual=i.str_individuals & j.str_individuals,
                    set_str_pos=set_str_pos,
                    set_str_neg=set_str_neg)
                if quality_intersection > i.quality:
                    i_and_j = i * j
                    i_and_j.quality = quality_intersection
                    st2.put(i_and_j)

                quality_union = self.call_quality_function(set_str_individual=i.str_individuals | j.str_individuals,
                                                           set_str_pos=set_str_pos,
                                                           set_str_neg=set_str_neg)

                if quality_union > i.quality:
                    i_or_j = i + j
                    i_or_j.quality = quality_union
                    st2.put(i_or_j)

        final_set = SearchTree()
        for i in st2:
            final_set.put(i)
            for mps in most_promissing_states:
                final_set.put(mps)
                i_or_mps = i + mps
                i_or_mps.quality = self.call_quality_function(set_str_individual=i_or_mps.str_individuals,
                                                              set_str_pos=set_str_pos,
                                                              set_str_neg=set_str_neg)
                if mps.quality < i_or_mps.quality:
                    final_set.put(i_or_mps)

                i_and_mps = i * mps
                i_and_mps.quality = self.call_quality_function(set_str_individual=i_and_mps.str_individuals,
                                                               set_str_pos=set_str_pos,
                                                               set_str_neg=set_str_neg)
                if mps.quality < i_and_mps.quality:
                    final_set.put(i_and_mps)

        return final_set

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
