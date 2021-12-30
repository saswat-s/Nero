import itertools
from collections import defaultdict
import copy
from itertools import chain
import random
from typing import DefaultDict, Dict, Set, Optional, Iterable, List, Type, Final, Generator, Tuple

from ontolearn.abstracts import BaseRefinement
from ontolearn.knowledge_base import KnowledgeBase

from owlapy.render import DLSyntaxObjectRenderer
from .expression import ClassExpression, AtomicExpression, ExistentialQuantifierExpression, \
    UniversalQuantifierExpression, ComplementOfAtomicExpression, UnionClassExpression, IntersectionClassExpression,Role


class SimpleRefinement(BaseRefinement):
    def __init__(self, knowledge_base: KnowledgeBase):
        super().__init__(knowledge_base)
        # 1. Number of named classes and sanity checking
        self.top_refinements = dict()
        self.renderer = DLSyntaxObjectRenderer()
        self.N_I = set([_.get_iri().as_str() for _ in self.kb.individuals(self.kb.thing)])
        self.expression = dict()  # str -> ClassExpression
        self.str_nc_star = None
        self.dict_sh_direct_down = dict()  # str -> str
        self.dict_sh_direct_up = dict()  # str -> str

        self.length_to_expression_str = dict()
        self.str_ae_to_neg_ae = dict()  # str -> Class Expression
        self.quantifiers = dict()
        self.prepare()
        del self.kb
        del self.renderer

    def prepare(self):
        """ Materialize KB to subsumption hierarchy """
        # (1) Initialize AtomicExpression from [N_C] + [T] + [Bottom
        nc_top_bot = [i for i in self.kb._class_hierarchy.sub_classes(self.kb.thing,
                                                                      direct=False)]
        # If no rel available
        self.length_to_expression_str.setdefault(1, set())
        self.length_to_expression_str.setdefault(2, set())
        self.length_to_expression_str.setdefault(3, set())

        for atomic_owl_class in [self.kb.thing, self.kb.nothing] + nc_top_bot:
            # (1.2) N_C + T + Bot
            target = AtomicExpression(name=self.renderer.render(atomic_owl_class),
                                      str_individuals=set(
                                          [_.get_iri().as_str() for _ in self.kb.individuals(atomic_owl_class)]),
                                      expression_chain=tuple(self.renderer.render(x) for x in
                                                             self.kb.get_direct_parents(atomic_owl_class)))
            self.__store(target, length=1)

            # (1.3) Initialize \forall r.E: E \in {N_C union + {T,Bot}}
            for mgur in self.kb.most_general_universal_restrictions(domain=self.kb.thing, filler=atomic_owl_class):
                filler = self.expression[self.renderer.render(mgur.get_filler())]
                role=Role(name=self.renderer.render(mgur.get_property()))
                target = UniversalQuantifierExpression(
                    name=self.renderer.render(mgur),
                    role=role,
                    filler=filler,
                    str_individuals=set([_.get_iri().as_str() for _ in self.kb.individuals(mgur)]),
                    expression_chain=tuple(self.renderer.render(self.kb.thing)))
                self.__store(target, length=3)
            # (1.4) Initialize \exists r.E : E \in {N_C union + {T,Bot}}
            for mger in self.kb.most_general_existential_restrictions(domain=self.kb.thing, filler=atomic_owl_class):
                filler = self.expression[self.renderer.render(mger.get_filler())]
                role=Role(name=self.renderer.render(mger.get_property()))

                target = ExistentialQuantifierExpression(
                    name=self.renderer.render(mger),
                    role=role,
                    filler=filler,
                    str_individuals=set([_.get_iri().as_str() for _ in self.kb.individuals(mger)]),
                    expression_chain=tuple(self.renderer.render(self.kb.thing)))
                self.__store(target, length=3)

        # (2) Construct sh_down and sh_up
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
        # Remove sh_down(T)=> T mapping
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

        self.str_nc_star = set(list(self.expression.keys()))

    def __store(self, target: ClassExpression, length: int) -> None:
        """ Store an expression through indexing it with its length"""
        self.expression[target.name] = target
        self.length_to_expression_str.setdefault(length, set()).add(target)

    def atomic_class_expressions(self) -> List[AtomicExpression]:
        """ List of atomic class expressions is returned """
        return [i for i in self.expression_given_length(1) if i.name not in ['⊤', '⊥']]

    def negated_named_class_expressions(self) -> List[ComplementOfAtomicExpression]:
        """ List of negated atomic class expressions is returned """
        res = []
        for i in self.atomic_class_expressions():
            if i.name in self.str_ae_to_neg_ae:
                neg_i = self.str_ae_to_neg_ae[i.name]
            else:
                neg_i = self.negate_atomic_class(i)
            res.append(neg_i)
        return res

    def negate_atomic_class(self, i: AtomicExpression) -> ComplementOfAtomicExpression:
        """ Negate an input atomic class expression"""
        neg_i = ComplementOfAtomicExpression(
            name='¬' + i.name,
            atomic_expression=i,
            str_individuals=self.N_I.difference(i.str_individuals),
            expression_chain=(i.expression_chain[-1],))
        self.__store(neg_i, length=2)
        self.str_ae_to_neg_ae[i.name] = neg_i
        return neg_i

    def all_quantifiers(self) -> List:
        """ Return all existential and universal quantifiers """
        return [i for i in self.expression_given_length(3) if
                isinstance(i, ExistentialQuantifierExpression) or isinstance(i, UniversalQuantifierExpression)]

    def refine_top_with_length(self, length: int) -> List[ClassExpression]:
        """ Return refinements of Top expression given length
            1 => sh_down(T), 2 => sh_down(T) negated leaf nodes and 3 => all quantifiers """
        return self.top_refinements[length]

    def expression_given_length(self, length: int) -> List[ClassExpression]:
        """ Expressions look up via their lengths """
        if length in self.length_to_expression_str:
            return [i for i in self.length_to_expression_str[length]]
        else:
            print('Invalid length', length)
            print(self.length_to_expression_str.keys())
            raise ValueError

    def construct_two_exp_from_chain(self, complex_exp) -> Tuple[List, List]:
        """ Construct A and B from (A OR B) or (A AND B) """
        x = complex_exp.expression_chain
        assert isinstance(x, tuple)
        assert len(x) == 3
        assert x[1] in ['OR', 'AND']
        a, _, b = x

        def check_sequence_of_atomic(s, l):
            for i in s:
                if i not in l:
                    return False
            return True

        left_concept, right_concept = None, None
        if check_sequence_of_atomic(a, self.str_nc_star) or len(a) == 3:
            left_concept = self.chain_gen((a[1],))[0]

        if check_sequence_of_atomic(b, self.str_nc_star) or len(b) == 3:
            right_concept = self.chain_gen((b[1],))[0]

        if right_concept and left_concept:
            return right_concept, left_concept
        print(a, b)
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
        """
        Intersect with refinements of top
        :param res:
        :return:
        """
        for k_length_int, refinements_top in self.top_refinements.items():
            for ref_top in refinements_top:
                for j in res:
                    if len(j.str_individuals) > 0 and len(ref_top.str_individuals.intersection(j.str_individuals)) > 0:
                        ref_top_and_j = ref_top * j
                        if ref_top_and_j.length == 3:
                            self.top_refinements.setdefault(ref_top_and_j.length, set()).add(ref_top_and_j)
                        yield ref_top_and_j

    def refine_complement_of(self, neg_atomic_expression: ComplementOfAtomicExpression):
        """ {¬A' | A'  ∈ sh_↑ (A)} ∪ {¬A \sqcap D | D ∈ \rho(T)} """
        atomic_class_expression = neg_atomic_expression.atomic_expression
        res = []
        for ae in self.sh_up(atomic_class_expression):
            if ae.name in self.str_ae_to_neg_ae:
                res.append(self.str_ae_to_neg_ae[ae.name])
            else:
                print(ae)
                raise ValueError(ae)
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
