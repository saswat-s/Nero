import torch
from torch import nn
from typing import Dict, List, Iterable, Set
from .expression import ClassExpression, TargetClassExpression
from owlapy.render import DLSyntaxObjectRenderer
from .static_funcs import apply_rho_on_rl_state, ClosedWorld_ReasonerFactory
import time
from .data_struct import ExpressionQueue, State, SearchTree
from ontolearn import KnowledgeBase
from .refinement_operator import SimpleRefinement


class NERO:
    def __init__(self, model: torch.nn.Module,
                 quality_func,
                 target_class_expressions,
                 instance_idx_mapping: Dict):
        assert len(target_class_expressions) > 2
        self.model = model
        self.quality_func = quality_func
        self.instance_idx_mapping = instance_idx_mapping
        self.inverse_instance_idx_mapping = dict(
            zip(self.instance_idx_mapping.values(), self.instance_idx_mapping.keys()))
        # expression ordered by id.
        self.target_class_expressions = target_class_expressions

        self.str_target_class_expression_to_label_id = {i.name: i.label_id for i in self.target_class_expressions}

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

    """
    def predict(self, str_pos: [str], str_neg: [str], topK: int = None, local_search=True) -> ExpressionQueue:
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
    """

    def fit(self, str_pos: [str], str_neg: [str], topK: int = None, use_search=None, kb_path=None) -> Dict:
        """
        Given set of positive and negative indviduals, fit returns
        {'Prediction': best_pred.name,
        'Instances': ...,
        'F-measure': ...,
        'NumClassTested': ...,
        'Runtime': ...}
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
        pred_vec = self.forward(xpos=torch.LongTensor([idx_pos]),
                                xneg=torch.LongTensor([idx_neg]))

        sort_val, sort_idxs = torch.sort(pred_vec, dim=1, descending=True)
        sort_idxs = sort_idxs.cpu().numpy()[0]

        # (4) Iterate over the sorted index of target expressions.
        for idx_target in sort_idxs[:topK]:
            # (5) Retrieval of instance.
            str_individuals = self.retrieval_of_individuals(self.target_class_expressions[idx_target])

            # (6) Compute Quality.
            quality_score = self.call_quality_function(set_str_individual=str_individuals,
                                                       set_str_pos=set_pos,
                                                       set_str_neg=set_neg)
            self.target_class_expressions[idx_target].quality = quality_score
            self.target_class_expressions[idx_target].str_individuals = str_individuals
            # (7) Put CE into the priority queue.
            top_prediction_queue.put(self.target_class_expressions[idx_target], key=-quality_score)
            # (8) If goal is found, we do not need to compute scores.
            if quality_score == 1.0:
                goal_found = True
                break
        assert len(top_prediction_queue) > 0
        # (9) IF goal is not found, we do search
        if goal_found is False:
            if use_search == 'Continues':
                best_pred = top_prediction_queue.get()
                top_prediction_queue.put(best_pred, key=-best_pred.quality)
                self.apply_continues_search(top_prediction_queue, set_pos, set_neg)
                best_constructed_expression = top_prediction_queue.get()
                if best_constructed_expression > best_pred:
                    best_pred = best_constructed_expression
            elif use_search == 'SmartInit':
                best_pred = top_prediction_queue.get()
                top_prediction_queue.put(best_pred, key=-best_pred.quality)
                top_prediction_queue = self.search_with_init(kb_path, top_prediction_queue, set_pos, set_neg)
                best_constructed_expression = top_prediction_queue.get()
                if best_constructed_expression > best_pred:
                    best_pred = best_constructed_expression
            elif use_search == 'None' or use_search is None:
                assert len(top_prediction_queue) > 0
                best_pred = top_prediction_queue.get()
            else:
                print(use_search)
                raise KeyError
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

    def search_with_init(self, kb_path, top_prediction_queue, set_pos, set_neg):
        """
        Standard search with smart initialization

        :param top_prediction_queue:
        :param rho:
        :param set_pos:
        :param set_neg:
        :return:
        """
        kb = KnowledgeBase(path=kb_path,
                           reasoner_factory=ClosedWorld_ReasonerFactory)
        rho = SimpleRefinement(knowledge_base=kb)

        heuristic_st = SearchTree()
        goal_found = False
        # (2) Iterate over advantages states
        while len(top_prediction_queue) > 0:
            # (2.1) Get top ranked Description Logic Expressions: C
            nero_mode_class_expression = top_prediction_queue.get()
            if nero_mode_class_expression.type in ['union_expression', 'intersection_expression']:
                nero_mode_class_expression.concepts = rho.construct_two_exp_from_chain(nero_mode_class_expression)

            # (2.2) Compute heuristic val: C
            heuristic_st.put(nero_mode_class_expression,
                             key=-nero_mode_class_expression.quality)
            # (2.3) Add a path from T ->... ->C
            for class_expression_c in rho.chain_gen(nero_mode_class_expression.expression_chain):
                # (2.3.) Obtain Path from T-> C
                if class_expression_c not in heuristic_st and class_expression_c not in top_prediction_queue:
                    class_expression_c.quality = self.call_quality_function(
                        set_str_individual=class_expression_c.str_individuals,
                        set_str_pos=set_pos,
                        set_str_neg=set_neg)
                    # (2.4.) Add states into tree
                    heuristic_st.put(class_expression_c,
                                     key=-class_expression_c.quality)
                    if class_expression_c.quality == 1:
                        goal_found = True
                        break
                else:
                    """ Ignore class_expression_c"""
            if goal_found:
                break

        explored_states = SearchTree()
        exploited_states = SearchTree()
        explored_states.extend_queue(heuristic_st)

        for i in range(3):
            s = explored_states.get()
            if s in exploited_states:
                continue
            exploited_states.put(s, key=-s.quality)

            for x in rho.refine(s):
                if len(x.str_individuals) and (x not in explored_states or x not in exploited_states):
                    x.quality = self.call_quality_function(
                        set_str_individual=x.str_individuals,
                        set_str_pos=set_pos,
                        set_str_neg=set_neg)
                    explored_states.put(x, key=-x.quality)
                if x.quality == 1.0:
                    break

        exploited_states.extend_queue(explored_states)
        return exploited_states

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

        sim = torch.nn.functional.mse_loss(input=neg_emb, target=pos_emb)

        print(sim)
        exit(1)
        st = []
        # (3) Iterate over most promising states and compute similarity scores between E^+ and E^-
        while len(top_states) > 0:
            i = top_states.get()

            full_coverage = len(i.str_individuals.intersection(set_str_pos)) / len(set_str_pos)

            if full_coverage == 1.0:
                fp = len(i.str_individuals.intersection(set_str_neg))
                st.append((fp, i))
            continue
            target_emb = self.positive_expression_embeddings(i.str_individuals)
            e_pos_idst = torch.cdist(target_emb, pos_emb, p=2).numpy()[0][0]
            e_neg_idst = torch.cdist(target_emb, neg_emb, p=2).numpy()[0][0]
            # print(f'{i} | dist. Pos: {e_pos_idst:.3f} |dist. Neg:{e_neg_idst:.3f}')
            # print(f'{i} | Semantic Dist: {e_neg_idst-e_pos_idst:.3f}')
            semantic_dist = e_pos_idst - e_neg_idst
            st.append((semantic_dist, i))
            """
            # (4) Embedding of most promising state
            target_emb = self.positive_expression_embeddings(i.str_individuals)
            # (5) Compute MSE Loss between | ((4) -neg_emb) - pos_emb|
            sim = torch.nn.functional.mse_loss(input=target_emb - neg_emb, target=pos_emb)
            #small.append((sim,i))

            if last_sim is None:
                last_sim = sim

            if sim < last_sim:
                small.append(i)
                last_sim = sim
            """
        st = sorted(st, key=lambda x: x[0], reverse=False)  # aschending order

        for _, i in st:
            print(i, _)

        exit(1)
        st = sorted(st, key=lambda x: x[0], reverse=False)  # aschending order
        for _, i in st[:10]:
            print(_, i)
        print('###')
        st = sorted(st, key=lambda x: x[1].quality, reverse=True)
        for _, i in st[:10]:
            print(_, i)

        exit(1)

        # small=sorted(small,key=lambda x:x[1],reverse=True)
        for _, i in st:
            for _, j in st:
                if i == j:
                    continue
                if i.str_individuals == j.str_individuals:
                    continue

                tp_i_str_pos = i.str_individuals.intersection(set_str_pos)
                tp_j_str_pos = j.str_individuals.intersection(set_str_pos)

                fp_i_str_pos = i.str_individuals.intersection(set_str_neg)
                fp_j_str_pos = j.str_individuals.intersection(set_str_neg)

                tp_coefficient = len(tp_i_str_pos.intersection(tp_j_str_pos)) / (len(tp_j_str_pos) + len(tp_i_str_pos))
                fp_coefficient = len(fp_i_str_pos.intersection(fp_j_str_pos)) / (len(fp_i_str_pos) + len(fp_j_str_pos))

                print(tp_coefficient)
                print(fp_coefficient)

                print(i)
                print(j)
                i_or_j = i + j
                i_or_j.quality = self.call_quality_function(set_str_individual=i_or_j.str_individuals,
                                                            set_str_pos=set_str_pos,
                                                            set_str_neg=set_str_neg)

                print(i_or_j)
                if i_or_j.quality > i.quality:
                    top_states.put(i_or_j)
                    if len(top_states) == max_size:
                        break

                i_and_j = i * j
                if len(i_and_j.str_individuals) == 0:
                    continue

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

    # def get_target_class_expressions(self):
    #    return (self.renderer.render(cl.concept) for cl in self.target_class_expressions)

    # def generate_expression(self, *, i, kb, expression_chain):
    #    expression = ClassExpression(name=self.renderer.render(i),
    #                                 str_individuals=set([_.get_iri().as_str() for _ in kb.individuals(i)]),
    #                                 expression_chain=expression_chain)
    #    return expression
