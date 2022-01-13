import pandas as pd
import torch
from torch import nn
from typing import Dict, List, Iterable, Set, Tuple
from .expression import *
from owlapy.render import DLSyntaxObjectRenderer
from .static_funcs import ClosedWorld_ReasonerFactory
import time
from .data_struct import SearchTree
from ontolearn import KnowledgeBase
from .refinement_operator import SimpleRefinement


class NERO:
    def __init__(self, model: torch.nn.Module,
                 quality_func,
                 target_class_expressions,
                 instance_idx_mapping: Dict, target_retrieval_data_frame=None, kb_path=None):
        assert len(target_class_expressions) > 2
        self.model = model
        self.quality_func = quality_func
        self.instance_idx_mapping = instance_idx_mapping
        self.str_all_individuals = set(self.instance_idx_mapping.keys())
        # self.inverse_instance_idx_mapping = dict(
        #    zip(self.instance_idx_mapping.values(), self.instance_idx_mapping.keys()))

        # expression ordered by id.
        self.target_class_expressions = target_class_expressions
        self.max_top_k = len(self.target_class_expressions)

        self.retrieve_counter = 0
        self.target_retrieval_data_frame = target_retrieval_data_frame
        self.dummy=kb_path
        if kb_path:
            # (1) Read atomic expressions
            self.str_to_atomic_exp = self.extract_nc(
                self.target_class_expressions[self.target_class_expressions['length'] == 1])
            # (2) Read negated atomic expressions
            self.neg_nc = self.negate_nc(self.str_to_atomic_exp.values(), self.str_all_individuals)
            # (3) Read universal quantifiers
            self.str_to_universal_quantifiers = self.extract_quantifiers(
                self.target_class_expressions[
                    self.target_class_expressions['type'] == 'universal_quantifier_expression'],
                UniversalQuantifierExpression, self.str_to_atomic_exp)
            # (3) Read existential quantifiers
            self.str_to_existential_quantifiers = self.extract_quantifiers(
                self.target_class_expressions[
                    self.target_class_expressions['type'] == 'existantial_quantifier_expression'],
                ExistentialQuantifierExpression, self.str_to_atomic_exp)
            # (4) Easy access
            self.expression = {**self.str_to_atomic_exp, **self.neg_nc, **self.str_to_existential_quantifiers,
                               **self.str_to_universal_quantifiers}
            # (5) Create hierarchies
            self.str_to_sh_down, self.str_to_sh_up = self.construct_subsumption_hierarchy()

            # (6) Create leaf atomic concepts
            self.set_of_least_general_atomic_expressions = {self.expression[k] for k, v in self.str_to_sh_down.items()
                                                            if len(v) == 0}
            # (7) Refinement of rho(⊤)
            self.top_refinements = set()
            self.top_refinements.update({i for i in self.str_to_sh_down['⊤']})
            # rho(⊤) contains all negated leaf nodes
            self.top_refinements.update(
                {self.expression['¬' + k] for k, v in self.str_to_sh_down.items() if len(v) == 0})
            self.set_of_most_general_universal_quantifiers = set()
            self.set_of_most_general_existential_quantifiers = set()
            for _, exp1 in self.str_to_universal_quantifiers.items():
                filler_exp1 = exp1.filler
                # If a filler is bottom, ignore it
                if filler_exp1 == '⊥':
                    continue
                # If filler is a T, add it.
                if filler_exp1 == '⊤':
                    self.set_of_most_general_universal_quantifiers.add(exp1)
                    continue
                # if filler is a leaf exp, ignore it
                if filler_exp1 in self.set_of_least_general_atomic_expressions:
                    continue

                if isinstance(filler_exp1, str):
                    if filler_exp1 in self.str_to_sh_up:
                        for x in self.str_to_sh_up[filler_exp1]:
                            n = exp1.name.split('.')[0]
                            new_name = n + '.' + x.name
                            if new_name in self.expression:
                                self.set_of_most_general_universal_quantifiers.add(self.expression[new_name])
                else:
                    if filler_exp1.name in self.str_to_sh_up:
                        for x in self.str_to_sh_up[filler_exp1.name]:
                            n = exp1.name.split('.')[0]
                            new_name = n + '.' + x.name
                            if new_name in self.expression:
                                self.set_of_most_general_universal_quantifiers.add(self.expression[new_name])

            for _, exp1 in self.str_to_existential_quantifiers.items():
                filler_exp1 = exp1.filler
                # If a filler is bottom, ignore it
                if filler_exp1 == '⊥':
                    continue
                # If filler is a T, add it.
                if filler_exp1 == '⊤':
                    self.set_of_most_general_universal_quantifiers.add(exp1)
                    continue
                # if filler is a leaf exp, ignore it
                if filler_exp1 in self.set_of_least_general_atomic_expressions:
                    continue
                if isinstance(filler_exp1, str):
                    if filler_exp1 in self.str_to_sh_up:
                        for x in self.str_to_sh_up[filler_exp1]:
                            n = exp1.name.split('.')[0]
                            new_name = n + '.' + x.name
                            if new_name in self.expression:
                                self.set_of_most_general_existential_quantifiers.add(self.expression[new_name])
                else:
                    if filler_exp1.name in self.str_to_sh_up:
                        for x in self.str_to_sh_up[filler_exp1.name]:
                            n = exp1.name.split('.')[0]
                            new_name = n + '.' + x.name
                            if new_name in self.expression:
                                self.set_of_most_general_existential_quantifiers.add(self.expression[new_name])

            self.top_refinements.update(self.set_of_most_general_universal_quantifiers)
            self.top_refinements.update(self.set_of_most_general_existential_quantifiers)
        else:
            self.expression = dict()

    def construct_subsumption_hierarchy(self) -> Tuple[Dict, Dict]:
        """ Construct Direct subsumption up and down hierarchy """
        # Direct subsumption down hierarchy
        str_to_sh_down = dict()
        # Direct subsumption up hierarchy
        str_to_sh_up = dict()
        for _, exp in self.str_to_atomic_exp.items():
            str_to_sh_down.setdefault(_, set())
            for i in exp.expression_chain:
                str_to_sh_down.setdefault(i, set()).add(exp)
        for k, v in str_to_sh_down.items():
            # print(k, [i.name for i in v]) #to plot direct submsumpion hierahry
            for i in v:
                str_to_sh_up.setdefault(i.name, set())
                if k == '⊤':
                    str_to_sh_up.setdefault(i.name, set()).add('T')
                else:
                    if k in self.expression:
                        str_to_sh_up.setdefault(i.name, set()).add(self.expression[k])

        return str_to_sh_down, str_to_sh_up

    @staticmethod
    def leaf_atomic_expressions(x: Dict, y: Dict) -> Set:
        """ Construct leaf expressions from a hierarhy """
        res = set()
        for k, v in y.items():
            if k not in x:
                res.add(v)
        return res

    @staticmethod
    def extract_nc(df: pd.DataFrame):
        """ Deserialize atomic expressions from a pandas DataFrame"""
        res = dict()
        for i, d in df.iterrows():
            new_obj = AtomicExpression(name=d['name'],
                                       label_id=d['label_id'],
                                       str_individuals=eval(d['str_individuals']),
                                       expression_chain=eval(d['expression_chain']),
                                       idx_individuals=eval(d['idx_individuals']))
            res[new_obj.name] = new_obj
        return res

    @staticmethod
    def negate_nc(x: Set, N_I: Set):
        return {'¬' + i.name: ComplementOfAtomicExpression(
            name='¬' + i.name,
            atomic_expression=i,
            str_individuals=N_I.difference(i.str_individuals),
            expression_chain=i.expression_chain) for i in x}

    @staticmethod
    def extract_quantifiers(df: pd.DataFrame, cls, atomic_to_exp):
        res = dict()
        for i, d in df.iterrows():
            str_role, str_filler = d['name'].split()[1].split('.')
            filler = atomic_to_exp[str_filler] if str_filler in atomic_to_exp else str_filler
            new_obj = cls(name=d['name'],
                          role=Role(name=str_role),
                          filler=filler,
                          str_individuals=eval(d['str_individuals']),
                          expression_chain=eval(d['expression_chain']),
                          idx_individuals=eval(d['idx_individuals']))
            res[new_obj.name] = new_obj
        return res

    def str_to_expression(self, x: str) -> ClassExpression:
        """ An ALC expressions in string format to an instance of ClassExpression """
        if x in self.expression:
            return self.expression[x]
        # Check in the pandas frame
        row = self.target_class_expressions[self.target_class_expressions['name'] == x]
        return self.pandas_series_to_exp(row)

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

    def str_to_index_mapping(self, individuals: List[str]) -> List[int]:
        try:
            return [self.instance_idx_mapping[i] for i in individuals]
        except KeyError:
            print('Ensure that URIs are valid and can be found in the input KG')
            print(individuals)
            exit(1)

    def retrieval_of_individuals(self, target_class_expression):
        if target_class_expression.str_individuals:
            return target_class_expression.str_individuals
        elif target_class_expression.idx_individuals is None:
            str_row = \
                self.target_retrieval_data_frame[
                    self.target_retrieval_data_frame['name'] == target_class_expression.name][
                    'str_individuals'].values[0]
            return eval(str_row)
        else:
            return {self.inverse_instance_idx_mapping[_] for _ in target_class_expression.idx_individuals}

    def generate_exp_from_pandas_series(self, d):
        """ Create an instance of expression from pandas series """
        type_ = d['type'].item()
        name = d['name'].item()

        if type_ == 'atomic_expression':
            new_obj = AtomicExpression(name=name,
                                       label_id=d['label_id'].item(),
                                       str_individuals=eval(d['str_individuals'].item()),
                                       expression_chain=eval(d['expression_chain'].item()),
                                       idx_individuals=eval(d['idx_individuals'].item()))
        elif type_ == 'negated_expression':
            new_obj = ComplementOfAtomicExpression(name=d['name'].item(),
                                                   atomic_expression=self.str_to_expression(d['name'].item()[1:]),
                                                   # '¬X => X'
                                                   str_individuals=eval(d['str_individuals'].item()),
                                                   expression_chain=eval(d['expression_chain'].item()),
                                                   idx_individuals=eval(d['idx_individuals'].item()))
        elif type_ == 'existantial_quantifier_expression':
            # '∃ married.⊤' -> married, ⊤
            str_role, str_filler = d['name'].item().split()[1].split('.')
            # filler must be atomic
            new_obj = ExistentialQuantifierExpression(name=d['name'].item(),
                                                      role=Role(name=str_role),
                                                      filler=self.str_to_expression(str_filler),
                                                      str_individuals=eval(d['str_individuals'].item()),
                                                      expression_chain=eval(d['expression_chain'].item()),
                                                      idx_individuals=eval(d['idx_individuals'].item()))
        elif type_ == 'universal_quantifier_expression':
            # '∃ married.⊤' -> married, ⊤
            str_role, str_filler = d['name'].item().split()[1].split('.')
            # filler must be atomic
            new_obj = UniversalQuantifierExpression(name=d['name'].item(),
                                                    role=Role(name=str_role),
                                                    filler=self.str_to_expression(str_filler),
                                                    str_individuals=eval(d['str_individuals'].item()),
                                                    expression_chain=eval(d['expression_chain'].item()),
                                                    idx_individuals=eval(d['idx_individuals'].item()))
        elif type_ == 'union_expression':
            #            print(d)
            # '(atomic_expression at 0x7f63ab372130 | XXXX | Indv:104 | Quality:-1.000, atomic_expression at 0x7f63ab349340 | YYYY | Indv:35 | Quality:-1.000)'
            str_concepts = d['concepts'].item()[1:-1].split(',')
            str_a, str_b = str_concepts[0].split(' | ')[1], str_concepts[1].split(' | ')[1]
            new_obj = UnionClassExpression(name=d['name'].item(),
                                           length=int(d['length'].item()),
                                           concepts=(self.str_to_expression(str_a), self.str_to_expression(str_b)),
                                           # (tuple(A,B)
                                           str_individuals=eval(d['str_individuals'].item()),
                                           expression_chain=eval(d['expression_chain'].item()),
                                           idx_individuals=eval(d['idx_individuals'].item()))
        elif type_ == 'intersection_expression':
            #            print(d)
            # '(atomic_expression at 0x7f63ab372130 | Male | Indv:104 | Quality:-1.000, atomic_expression at 0x7f63ab349340 | Grandmother | Indv:35 | Quality:-1.000)'
            str_concepts = d['concepts'].item()[1:-1].split(',')
            str_a, str_b = str_concepts[0].split(' | ')[1], str_concepts[1].split(' | ')[1]
            new_obj = IntersectionClassExpression(name=d['name'].item(),
                                                  length=int(d['length'].item()),
                                                  concepts=(
                                                      self.str_to_expression(str_a), self.str_to_expression(str_b)),
                                                  # (tuple(A,B)
                                                  str_individuals=eval(d['str_individuals'].item()),
                                                  expression_chain=eval(d['expression_chain'].item()),
                                                  idx_individuals=eval(d['idx_individuals'].item()))
        else:
            print(type_)
            print(d)
            raise ValueError
        #self.expression[new_obj.name] = new_obj
        return new_obj

    def pandas_series_to_exp(self, d: pd.Series):
        """ Construct an expression from pandas dataframe"""
        name = d['name'].item()
        #if name in self.expression:
        #    return self.expression[name]
        new_exp = self.generate_exp_from_pandas_series(d)
        return new_exp

    def select_target_expression(self, idx: int):
        # (5) Look up target class expression
        row = self.target_class_expressions[self.target_class_expressions['label_id'] == idx]
        if len(self.expression) == 0:
            # Each target/label is an instance of TargetClassExpression
            return TargetClassExpression(label_id=row['label_id'].item(),
                                         name=row['name'].item(),
                                         # type=type_,
                                         length=int(row['length'].item()),
                                         expression_chain=eval(row['expression_chain'].item()),
                                         str_individuals=eval(row['str_individuals'].item()),
                                         idx_individuals=eval(row['idx_individuals'].item()))

        return self.pandas_series_to_exp(row)

        """
        return TargetClassExpression(label_id=row['label_id'].item(),
                                     name=row['name'].item(),
                                     type=row['type'].item(),
                                     length=int(row['length'].item()),
                                     expression_chain=eval(row['expression_chain'].item()),
                                     str_individuals=eval(row['str_individuals'].item()),
                                     idx_individuals=eval(row['idx_individuals'].item()))
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
        # (1) Initialize learning problem.
        self.predict_sanity_checking(pos=str_pos, neg=str_neg, topK=topK)
        start_time = time.time()
        self.retrieve_counter = 0
        set_pos, set_neg = set(str_pos), set(str_neg)
        idx_pos = self.str_to_index_mapping(str_pos)
        idx_neg = self.str_to_index_mapping(str_neg)
        goal_found = False

        # (2) Initialize a priority queue for top K Target Expressions.
        top_prediction_queue = SearchTree()
        # (3) Predict scores and sort index target expressions in descending order of assigned scores.
        sort_val, sort_idxs = torch.sort(self.forward(xpos=torch.LongTensor([idx_pos]),
                                                      xneg=torch.LongTensor([idx_neg])), dim=1, descending=True)
        sort_idxs = sort_idxs.cpu().numpy()[0]
        # (4) Iterate over the sorted index of target expressions.
        for idx_target in sort_idxs[:topK]:
            target_ce = self.select_target_expression(idx_target)
            # (5) Retrieval of instance.
            target_ce.str_individuals = self.retrieval_of_individuals(target_ce)
            # (6) Compute Quality.
            target_ce.quality = self.call_quality_function(set_str_individual=target_ce.str_individuals,
                                                           set_str_pos=set_pos,
                                                           set_str_neg=set_neg)
            # (7) Put predicted expression into the priority queue.
            top_prediction_queue.put(target_ce, key=-target_ce.quality)
            # (8) If goal is found, we do not need to compute scores.
            if target_ce.quality == 1.0:
                goal_found = True
                break
        assert len(top_prediction_queue) > 0
        # (9) IF goal is not found, we do search
        if goal_found is False:
            if use_search == 'SmartInit':
                best_pred = top_prediction_queue.get()
                top_prediction_queue.put(best_pred, key=-best_pred.quality)
                top_prediction_queue = self.search_with_init(top_prediction_queue, set_pos, set_neg)
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

    def downward_refine(self, expression):
        """
        top-down/downward refinement operator
        \forall s \in \StateSpace : \rho(s) \subseteq \{ s^i \in \StateSpace \;|\; s^i \preceq s \}.
        Implementation is based on the top down refinement operator definition in
        (Concept Learning in Description Logics Using Refinement Operators)
        \rho_b(C) domain b based intersections is carried out via disjoint checks.
        """
        if isinstance(expression, str):
            # (1) Length constraint refinement of Top .
            if expression == '⊤':
                return self.top_refinements
            # Bottom can not be refined.
            # if expression == '⊥':
            return {}
        x_type = expression.type
        refinements = set()
        if x_type == 'union_expression':
            # Given C \equiv A \sqcup B, =>
            # (1) {\rho(A) \sqcup B} \cup
            # (2) {A \sqcup \rho(B)} \cup
            # (3) {A \sqcup B \sqcap T} \in \rho(C)
            A, B = expression.concepts
            # (1)
            for rho_A in self.downward_refine(A):
                refinements.add(rho_A + B)
            # (2)
            for rho_B in self.downward_refine(B):
                refinements.add(rho_B + A)
            # (3)
            # for i in self.top_refinements:
            #    if not i.str_individuals.isdisjoint(expression.str_individuals):
            #        refinements.add(i * expression)
        elif x_type == 'intersection_expression':
            # Given C \equiv A \sqcap B, =>
            # (1) {\rho(A) \sqcap B} \cup
            # (2) {A \sqcap \rho(B)} \cup
            A, B = expression.concepts
            # (1)
            for rho_A in self.downward_refine(A):
                if not rho_A.str_individuals.isdisjoint(B.str_individuals):
                    refinements.add(rho_A * B)
            # (2)
            for rho_B in self.downward_refine(B):
                if not rho_B.str_individuals.isdisjoint(A.str_individuals):
                    refinements.add(rho_B * A)
        elif x_type == 'atomic_expression':
            if expression.name in self.str_to_sh_down:
                for i in self.str_to_sh_down[expression.name]:
                    refinements.add(i)
            for i in self.top_refinements:
                if not i.str_individuals.isdisjoint(expression.str_individuals):
                    refinements.add(i * expression)

        elif x_type == 'negated_expression':
            # ¬Father, ¬Male \in refine (¬Father)
            str_not_expression = expression.name[1:]
            for i in self.str_to_sh_up[str_not_expression]:
                if isinstance(i, str):
                    assert i == 'T'
                else:
                    refinements.add(self.expression['¬' + i.name])
            # for i in self.top_refinements:
            #    if not i.str_individuals.isdisjoint(expression.str_individuals):
            #        refinements.add(i * expression)
        elif x_type == 'existantial_quantifier_expression':
            # C \equiv ∃ r.A (e.g. ∃ hasSibling.Parent)
            # (1) {∃ r.B | B \in \rho(A)} \cup
            # (2) {(∃ r.A) \sqcap B | B \in \rho(T)}
            # (1)
            for i in self.downward_refine(expression.filler):
                n = expression.name.split('.')[0] + '.' + i.name
                if n in self.expression:
                    refinements.add(self.expression[n])
                else:
                    continue
                    """
                    try:
                        assert i.type in ['intersection_expression', 'union_expression']
                    except:
                        print(expression)
                        print(n)
                        print(i)
                        print(self.expression[n])
                        exit(1)

                    for x in i.concepts:
                        if x.name == expression.filler.name:
                            continue
                        if x.type == 'atomic_expression':
                            n = expression.name.split('.')[0] + '.' + x.name
                            if n in self.expression:
                                refinements.add(self.expression[n])

                        elif x.type == 'negated_expression':
                            str_not_expression = x.name[1:]
                            store = set()
                            for up_str_not_expression in self.str_to_sh_down[expression.filler.name]:
                                if str_not_expression == up_str_not_expression.name:
                                    continue
                                n = expression.name.split('.')[0] + '.' + up_str_not_expression.name
                                if n in self.expression:
                                    store.add(self.expression[n])
                            import itertools
                            refinements.update(itertools.accumulate(store))

                        elif x.type in ['universal_quantifier_expression', 'existantial_quantifier_expression']:
                            if x.name != expression.name:
                                a = x * expression
                                self.expression[a.name] = a
                                refinements.add(self.expression[a.name])
                        else:
                            raise ValueError
                    """
            # (2)
            # for i in self.top_refinements:
            #    if not i.str_individuals.isdisjoint(expression.str_individuals):
            #        refinements.add(i * expression)
        elif x_type == 'universal_quantifier_expression':
            # C \equiv ∀ r.A (e.g. ∀ married.Brother)
            # (1) {∀ r.\rho(A)}
            # \cup
            if not isinstance(expression.filler, str):
                for i in self.downward_refine(expression.filler):
                    n = expression.name.split('.')[0] + '.' + i.name
                    if n in self.expression:
                        refinements.add(self.expression[n])
            # for i in self.top_refinements:
            #    if not i.str_individuals.isdisjoint(expression.str_individuals):
            #        refinements.add(i * expression)

        else:
            raise KeyError('Error in expression type')
        return refinements

    def upward_refine(self, expression):
        """ \forall s \in \StateSpace : \rho(s) \subseteq \{ s^i \in \StateSpace \;|\; s \preceq  s^i \} """
        if isinstance(expression, str):
            if expression == '⊥':
                return self.set_of_least_general_atomic_expressions
            return {}

        x_type = expression.type
        refinements = set()

        if x_type == 'union_expression':

            A, B = expression.concepts
            refinements.add(A)
            refinements.add(B)
            for rho_A in self.upward_refine(A):
                refinements.add(rho_A)
            for rho_B in self.upward_refine(B):
                refinements.add(rho_B)
        elif x_type == 'intersection_expression':
            A, B = expression.concepts
            refinements.add(A)
            refinements.add(B)
            for rho_A in self.upward_refine(A):
                refinements.add(rho_A + B)
            for rho_B in self.upward_refine(B):
                refinements.add(rho_B + A)
        elif x_type == 'atomic_expression':
            if expression.name in self.str_to_sh_up:
                for i in self.str_to_sh_up[expression.name]:
                    refinements.add(i)
        elif x_type == 'negated_expression':
            # ¬Son ?
            str_not_expression = expression.name[1:]
            for i in self.downward_refine(self.expression[str_not_expression]):
                name = '¬' + i.name
                if name in self.expression:
                    refinements.add(self.expression['¬' + i.name])
        elif x_type == 'existantial_quantifier_expression':
            for i in self.upward_refine(expression.filler):
                n = expression.name.split('.')[0] + '.' + i.name
                if n in self.expression:
                    refinements.add(self.expression[n])
        elif x_type == 'universal_quantifier_expression':
            for i in self.upward_refine(expression.filler):
                n = expression.name.split('.')[0] + '.' + i.name
                if n in self.expression:
                    refinements.add(self.expression[n])
        else:
            raise KeyError('Error in expression type')
        return refinements

    def search_with_init(self, top_prediction_queue, set_pos, set_neg):
        """ Standard search with smart initialization """

        search_tree = SearchTree()
        # (2) Iterate over advantages states
        while len(top_prediction_queue) > 0:
            # (2.1) Get top ranked Description Logic Expressions: C.
            c = top_prediction_queue.get()
            # (2.2) Compute heuristic val of  C.
            search_tree.put(c, key=-c.quality)
            for a in self.downward_refine(c).union(self.upward_refine(c)):
                if a not in search_tree or a not in top_prediction_queue:
                    a.quality = self.call_quality_function(
                        set_str_individual=a.str_individuals,
                        set_str_pos=set_pos,
                        set_str_neg=set_neg)
                    search_tree.put(a, key=-a.quality)
        return search_tree

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

    # Outdated.
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

    # Maybe Outdated
    def get_target_exp_found_in_chain(self, expression_chain):
        for i in expression_chain:
            if i in self.str_target_class_expression_to_label_id:
                yield self.target_class_expressions[self.str_target_class_expression_to_label_id[i]]

    # Maybe Outdated
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
