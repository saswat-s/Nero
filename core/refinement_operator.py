import itertools
from collections import defaultdict
import copy
from itertools import chain, tee
import random
from typing import DefaultDict, Dict, Set, Optional, Iterable, List, Type, Final, Generator
from ontolearn.value_splitter import AbstractValueSplitter, BinningValueSplitter
from owlapy.model.providers import OWLDatatypeMaxInclusiveRestriction, OWLDatatypeMinInclusiveRestriction
from owlapy.vocab import OWLFacet

from ontolearn.abstracts import BaseRefinement
from ontolearn.knowledge_base import KnowledgeBase

from owlapy.render import DLSyntaxObjectRenderer
from .expression import ClassExpression, AtomicExpression, ExistentialQuantifierExpression, \
    UniversalQuantifierExpression, ComplementOfAtomicExpression, UnionClassExpression, IntersectionClassExpression


class SimpleRefinement(BaseRefinement):
    """ A top down refinement operator refinement operator in ALC."""

    def __init__(self, knowledge_base: KnowledgeBase):
        super().__init__(knowledge_base)
        # 1. Number of named classes and sanity checking
        self.top_refinements = dict()
        # num_of_named_classes = len(set(i for i in self.kb.ontology().classes_in_signature()))
        # assert num_of_named_classes == len(list(i for i in self.kb.ontology().classes_in_signature()))
        self.renderer = DLSyntaxObjectRenderer()

        self.N_I = set([_.get_iri().as_str() for _ in self.kb.individuals(self.kb.thing)])

        self.expression = dict()  # str -> ClassExpression
        self.str_nc_star=None# set()
        self.dict_sh_direct_down = dict()  # str -> str
        self.dict_sh_direct_up = dict()  # str -> str

        self.length_to_expression_str = dict()
        self.str_ae_to_neg_ae = dict()  # str -> Class Expression
        self.prepare()

    def prepare(self):
        """

        :return:
        """
        # (1) Initialize AtomicExpression from [N_C] + [T]
        nc_top_bot = [self.kb.thing] + [self.kb.nothing] + [i for i in
                                                            self.kb._class_hierarchy.sub_classes(self.kb.thing,
                                                                                                 direct=False)]

        for owl_class in nc_top_bot:
            # (2) N_C + T + Bot
            target = AtomicExpression(name=self.renderer.render(owl_class),
                                      str_individuals=set(
                                          [_.get_iri().as_str() for _ in self.kb.individuals(owl_class)]),
                                      expression_chain=tuple(self.renderer.render(x) for x in
                                                             self.kb.get_direct_parents(owl_class)))
            self.__store(target, length=1)

            # (2) Initialize \forall r.E: E \in {N_C union + {T,Bot}}
            for mgur in self.kb.most_general_universal_restrictions(domain=self.kb.thing, filler=owl_class):
                filler = self.expression[self.renderer.render(mgur.get_filler())]
                target = UniversalQuantifierExpression(
                    name=self.renderer.render(mgur),
                    role=self.renderer.render(mgur.get_property()),
                    filler=filler,
                    str_individuals=set([_.get_iri().as_str() for _ in self.kb.individuals(mgur)]),
                    expression_chain=tuple(self.renderer.render(self.kb.thing)))
                self.__store(target, length=3)
            # (3) Initialize \exists r.E : E \in {N_C union + {T,Bot}}
            for mger in self.kb.most_general_existential_restrictions(domain=self.kb.thing, filler=owl_class):
                filler = mger.get_filler()

                if filler == self.kb.thing:
                    filler = self.renderer.render(self.kb.thing)
                elif filler == self.kb.nothing:
                    filler = self.renderer.render(self.kb.nothing)
                else:
                    filler = self.expression[self.renderer.render(filler)]

                target = ExistentialQuantifierExpression(
                    name=self.renderer.render(mger),
                    role=self.renderer.render(mger.get_property()),
                    filler=filler,
                    str_individuals=set([_.get_iri().as_str() for _ in self.kb.individuals(mger)]),
                    expression_chain=tuple(self.renderer.render(self.kb.thing)))
                self.__store(target, length=3)
        # You do not need to do it :Intersedct and Union
        # self.compute()
        for k, v in self.expression.items():
            k: str
            v: ClassExpression
            self.dict_sh_direct_down.setdefault(k, set())

            if len(v.expression_chain) == 0:  # and k != '⊤':
                v.expression_chain = tuple(self.renderer.render(self.kb.thing))

            assert len(v.expression_chain) > 0
            for x in v.expression_chain:
                self.dict_sh_direct_down.setdefault(x, set()).add(k)
                self.dict_sh_direct_up.setdefault(k, set()).add(x)

        self.dict_sh_direct_down['⊤'].remove('⊤')
        # Fill top refinements
        for i in self.dict_sh_direct_down['⊤']:
            ref = self.expression[i]
            self.top_refinements.setdefault(ref.length, set()).add(ref)

        for i in self.expression_given_length(1):
            if i.name in ['⊤', '⊥']:
                continue
            if '⊥' in self.dict_sh_direct_down[i.name]:
                neg_i = ComplementOfAtomicExpression(
                    name='¬' + i.name,
                    atomic_expression=i,
                    str_individuals=self.N_I.difference(i.str_individuals),
                    expression_chain=tuple(self.renderer.render(self.kb.thing)))
                self.__store(neg_i, length=2)
                self.str_ae_to_neg_ae[i.name] = neg_i
                self.top_refinements.setdefault(neg_i.length, set()).add(neg_i)
        for i in self.negated_named_class_expressions():
            self.__store(i, length=2)

        self.str_nc_star=set(list(self.expression.keys()))

        # If no rel available
        self.length_to_expression_str.setdefault(3, set())
    def __store(self, target, length):
        # Ignore all empty expression except bottom
        self.expression[target.name] = target
        self.length_to_expression_str.setdefault(length, set()).add(target)

    def named_class_expressions(self):
        return [i for i in self.expression_given_length(1) if i.name not in ['⊤', '⊥']]

    def negated_named_class_expressions(self):
        res = []
        for i in self.named_class_expressions():
            if i.name in self.str_ae_to_neg_ae:
                neg_i = self.str_ae_to_neg_ae[i.name]
            else:
                neg_i = self.negate_atomic_class(i)
            res.append(neg_i)
        return res

    def negate_atomic_class(self, i):
        neg_i = ComplementOfAtomicExpression(
            name='¬' + i.name,
            atomic_expression=i,
            str_individuals=self.N_I.difference(i.str_individuals),
            expression_chain=(i.expression_chain[0],)  # neg_i in \rho(i)
        )
        self.__store(neg_i, length=2)
        self.str_ae_to_neg_ae[i.name] = neg_i
        return neg_i

    def all_quantifiers(self):
        return [i for i in self.expression_given_length(3)]

    def refine_top_with_length(self, length):
        # 1 => sh_down(T)
        # 2 => sh_down(T) negatied leaf nodes
        # 3 => all quantifiers
        return self.top_refinements[length]

    def __refine_top_direct_nc(self, ce=None):
        """ (1) {A | A ∈ N_C , A \sqcap B notequiv ⊥, A \sqcap B notequiv  B,
        there is no A' ∈ N_C with A \sqsubset A' }"""
        for owlclass in self.kb.get_all_direct_sub_concepts(self.kb.thing):
            # (A \sqcap B \neg\equiv ⊥) and ((A \sqcap B \neg\equiv B)):
            # CD: the latter condition implies person not in \rho(T) as averythin in family kb is a persona
            # So I disregared it

            if ce is None:
                str_individuals = set([_.get_iri().as_str() for _ in self.kb.individuals(owlclass)])
                if 0 < len(self.N_I.intersection(str_individuals)):
                    target = AtomicExpression(
                        name=self.renderer.render(owlclass),
                        str_individuals=set([_.get_iri().as_str() for _ in self.kb.individuals(owlclass)]),
                        expression_chain=tuple(self.renderer.render(self.kb.thing))
                    )
                    self.expression[target.name] = target
                    yield target
            else:
                str_ = self.renderer.render(owlclass)
                if str_ in self.expression:
                    yield ce * self.expression[str_]

    def check_sequence_of_atomic(self, s):
        """
        tuple of strings
        :param s:
        :return:
        """
        for i in s:
            if i not in self.str_nc_star:
                return False
        return True

    def construct_two_exp_from_chain(self, complex_exp):
        x = complex_exp.expression_chain
        assert isinstance(x, tuple)
        assert len(x) == 3
        assert x[1] in ['OR', 'AND']
        a, _, b = x
        left_concept, right_concept = None, None
        if self.check_sequence_of_atomic(a) or len(a) == 3:
            left_concept = self.chain_gen((a[1],))[0]

        if self.check_sequence_of_atomic(b) or len(b) == 3:
            right_concept = self.chain_gen((b[1],))[0]

        if right_concept and left_concept:
            return right_concept, left_concept

        print(a)
        print(b)
        raise ValueError('Not found')

    def chain_gen(self, x: tuple) -> List[ClassExpression]:
        if len(x) == 1:
            return [self.expression[x[0]]]
        elif len(x) == 2:
            return self.chain_gen((x[0],)) + self.chain_gen((x[1],))
        elif len(x) == 3 and x[1] in ['OR', 'AND']:
            a, _, b = x
            left_path = self.chain_gen(a)
            right_path = self.chain_gen(b)
            return left_path + right_path
        elif len(x) >= 3:
            res = []
            for i in x:
                res.extend(self.chain_gen((i,)))
            return res
        else:
            print(x)
            raise ValueError

    def expression_from_str(self, x: tuple) -> ClassExpression:
        try:
            assert isinstance(x, tuple)
        except:
            print(x)
            print('not tuple')
            exit(1)
        if len(x) == 1:
            print(x)
            print('Single lookup')
            exit(1)
        elif len(x) == 2:
            print(len(x))
            # a ->b
            a, b = x
            b_ce = self.expression[b]
            return b_ce
        elif len(x) == 3:
            print(len(x))
            a, opt, b = x
            assert isinstance(a, tuple)
            asd = expression_from_str(a)
            print(asd)
            print('asdasd')
            exit(1)
            if opt == 'OR':
                pass
            elif opt == 'AND':
                pass
            else:
                raise KeyError
            print(x)
            exit(1)
        else:
            a, opt, b, c = x
            assert opt in ['AND', 'OR']
            print(x)
            print(len(x))

            print(a)

            a_ce = expression_from_str(a)
            print(a_ce)
            print('here')
            print('EXIT')
            exit(1)
        """
        if isinstance(x, str):
            return self.expression[x]
        elif isinstance(x, tuple):
            str_ceA, operator_, str_ceB = x
            if operator_ == 'OR':
                return self.expression_from_str(str_ceA) + self.expression_from_str(str_ceB)
            elif operator_ == 'AND':
                return self.expression_from_str(str_ceA) * self.expression_from_str(str_ceB)
            else:
                print(x)
                print('here')
                exit(1)
        """

    def expression_given_length(self, length: int):
        if length in self.length_to_expression_str:
            return [i for i in self.length_to_expression_str[length]]
        else:
            print('Invalid length', length)
            print(self.length_to_expression_str.keys())
            raise ValueError

    def compute(self):
        # (1) A AND B
        inter_sections_unions = set()
        for i in self.length_to_expression_str[1]:
            for j in self.length_to_expression_str[1]:
                if i == j:
                    continue

                i_and_j = i * j
                i_and_j: IntersectionClassExpression

                i_or_j = i + j
                i_or_j: UnionClassExpression

                if len(i_and_j.str_individuals) > 0:
                    inter_sections_unions.add(i_and_j)
                if len(i_or_j.str_individuals) > 0:
                    inter_sections_unions.add(i_or_j)

        for i in inter_sections_unions:
            self.__store(i, length=i.length)

    def refine_top(self, ce=None):
        """ Refine Top Class Expression:
        from Concept Learning J. LEHMANN and N. FANIZZI and L. BÜHMANN and C. D’AMATO
        """
        print('asdad')
        raise ValueError

    def sh_down(self, atomic_class_expression: AtomicExpression) -> List:
        """ sh_↓ (A) ={A' \in N_C | A' \sqsubset A, there is no A'' \in N_C with A' \sqsubset A'' \sqsubset A}
        Return its direct sub concepts
        """
        res = []
        for str_ in self.dict_sh_direct_down[atomic_class_expression.name]:
            res.append(self.expression[str_])
        return res

    def sh_up(self, atomic_class_expression: AtomicExpression) -> List:
        """ sh_up (A) ={A' \in N_C | A \sqsubset A', there is no A'' \in N_C with A \sqsubset A' \sqsubset A''}
        Return its direct sub concepts
        """
        return [self.expression[str_] for str_ in self.dict_sh_direct_up[atomic_class_expression.name]]

    def refine_atomic_concept(self, atomic_class_expression: AtomicExpression):
        """ {A' | A' \in sh_down(A) """
        yield from self.sh_down(atomic_class_expression)
        yield from self.intersect_with_top([atomic_class_expression])

    def intersect_with_top(self, res):
        refs_to_add_top = set()
        for v in self.top_refinements.values():
            for i in v:
                for j in res:
                    if len(j.str_individuals) > 0 and len(i.str_individuals.intersection(j.str_individuals)) > 0:
                        iandj = i * j
                        refs_to_add_top.add(iandj)
                        yield iandj
        for i in refs_to_add_top:
            self.top_refinements.setdefault(i.length, set()).add(i)

    def refine_complement_of(self, compement_of_atomic_class_expression: ComplementOfAtomicExpression):
        """ {¬A' | A'  ∈ sh_↑ (A)} ∪ {¬A \sqcap D | D ∈ \rho(T)}
        :param compement_of_atomic_class_expression:
        :return:
        """
        atomic_class_expression = compement_of_atomic_class_expression.atomic_expression
        res = []
        for ae in self.sh_up(atomic_class_expression):
            if ae.name in self.str_ae_to_neg_ae:
                res.append(self.str_ae_to_neg_ae[ae.name])
            else:
                print(ae)
                exit(1)
                res.append(self.__negative_atomic_exp(ae, compement_of_atomic_class_expression))

        yield from res
        yield from self.intersect_with_top(res)

    def refine_universal_quantifier_expression(self, uqe: UniversalQuantifierExpression):
        """ \rho(C) if C ==∀r.D:
                                {(∀r.D) \sqcap E \in \rho(T) \cup
        """
        # Important Limitation: We can not refine filler:
        # We should obtain ∀r.X | X \in _NC AND ∀r.X \in \rho(T) so that
        # ∀r.X  AND ∀r.Y would give us ∀r.(X AND Y)
        # role = uqe.role
        # filler = uqe.filler
        yield from self.intersect_with_top([uqe])

    def refine_existential_expression(self, eqe: ExistentialQuantifierExpression):
        """ \rho(C) if C ==∀r.D:
                                {(∀r.D) \sqcap E \in \rho(T) \cup
        """
        yield from self.intersect_with_top([eqe])

    def refine_union_or_intersection(self, x, key=None):
        """

        :param x:
        :param key:
        :return:
        """
        a, b = x.concepts
        for x in self.refine(a):
            if len(x.str_individuals) > 0 and len(x.str_individuals.intersection(b.str_individuals)) > 0:
                yield key(x, b)
        for x in self.refine(b):
            if len(x.str_individuals) > 0 and len(x.str_individuals.intersection(a.str_individuals)) > 0:
                yield key(x, a)

    def refine_union_expression(self, ue):
        yield from self.refine_union_or_intersection(ue, key=lambda x, y: x + y)

    def refine_intersection_expression(self, ie):
        yield from self.refine_union_or_intersection(ie, key=lambda x, y: x * y)

    def refine(self, class_expression) -> Iterable:
        if isinstance(class_expression, AtomicExpression):
            yield from self.refine_atomic_concept(class_expression)
        elif isinstance(class_expression, ComplementOfAtomicExpression):
            yield from self.refine_complement_of(class_expression)
        elif isinstance(class_expression, UniversalQuantifierExpression):
            yield from self.refine_universal_quantifier_expression(class_expression)
        elif isinstance(class_expression, ExistentialQuantifierExpression):
            yield from self.refine_existential_expression(class_expression)
        elif isinstance(class_expression, UnionClassExpression):
            yield from self.refine_union_expression(class_expression)
        elif isinstance(class_expression, IntersectionClassExpression):
            yield from self.refine_intersection_expression(class_expression)
        else:
            print(class_expression)
            print('Incorrect type')
            raise ValueError
