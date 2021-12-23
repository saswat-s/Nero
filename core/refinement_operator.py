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

"""
from owlapy.model import OWLObjectPropertyExpression, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, \
    OWLObjectIntersectionOf, OWLClassExpression, OWLNothing, OWLThing, OWLNaryBooleanClassExpression, \
    OWLObjectUnionOf, OWLClass, OWLObjectComplementOf, OWLObjectMaxCardinality, OWLObjectMinCardinality, \
    OWLDataSomeValuesFrom, OWLDatatypeRestriction, OWLLiteral, OWLObjectInverseOf, OWLDataProperty, \
    OWLDataHasValue, OWLDataPropertyExpression
from ontolearn.search import Node
"""

from owlapy.render import DLSyntaxObjectRenderer
from .expression import ClassExpression, AtomicExpression, ExistentialQuantifierExpression, \
    UniversalQuantifierExpression, ComplementOfAtomicExpression, UnionClassExpression, IntersectionClassExpression


class SimpleRefinement(BaseRefinement):
    """ A top down refinement operator refinement operator in ALC."""

    def __init__(self, knowledge_base: KnowledgeBase):
        super().__init__(knowledge_base)
        # 1. Number of named classes and sanity checking
        self.top_refinements = None
        #num_of_named_classes = len(set(i for i in self.kb.ontology().classes_in_signature()))
        #assert num_of_named_classes == len(list(i for i in self.kb.ontology().classes_in_signature()))
        self.renderer = DLSyntaxObjectRenderer()

        self.N_I = set([_.get_iri().as_str() for _ in self.kb.individuals(self.kb.thing)])

        self.expression = dict()  # str -> ClassExpression
        self.dict_sh_direct_down = dict()  # str -> str
        self.dict_sh_direct_up = dict()  # str -> str

        self.length_to_expression_str = dict()
        self.str_ae_to_neg_ae = dict()  # str -> Class Expression

        self.__fill_exp()

    def expression_from_str(self, x):
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

    def __store(self, target, length):
        self.expression[target.name] = target
        self.length_to_expression_str.setdefault(length, set()).add(target)

    def get_top_refinements_via_length(self, length: int, wo_top_bot=True):
        if length in self.length_to_expression_str:
            if wo_top_bot:
                return [i for i in self.length_to_expression_str[length] if len(self.N_I) > len(i.str_individuals) > 0]
            else:
                return [i for i in self.length_to_expression_str[length]]
        else:
            print('Invalid length', length)
            print(self.length_to_expression_str.keys())
            exit(1)

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

    def __fill_exp(self):
        # (1) Initialize AtomicExpression from [N_C] + [T]
        nc_top_bot = [self.kb.thing] + [self.kb.nothing] + [i for i in
                                                            self.kb._class_hierarchy.sub_classes(self.kb.thing,
                                                                                                 direct=False)]
        for owl_class in nc_top_bot:
            target = AtomicExpression(name=self.renderer.render(owl_class),
                                      str_individuals=set(
                                          [_.get_iri().as_str() for _ in self.kb.individuals(owl_class)]),
                                      expression_chain=[self.renderer.render(x) for x in
                                                        self.kb.get_direct_parents(owl_class)])
            self.__store(target, length=1)

            if owl_class is not self.kb.thing and owl_class is not self.kb.nothing:
                target = ComplementOfAtomicExpression(
                    name='¬' + target.name,
                    atomic_expression=target,
                    str_individuals=self.N_I.difference(target.str_individuals),
                    expression_chain=target.expression_chain + [target.name])
                self.__store(target, length=2)
                self.str_ae_to_neg_ae[target.name] = target

            # (2) Initialize \forall r.E : E \in {N_C union + {T,Bot}}
            for mgur in self.kb.most_general_universal_restrictions(domain=self.kb.thing, filler=owl_class):

                filler = mgur.get_filler()

                if filler == self.kb.thing:
                    filler = self.renderer.render(self.kb.thing)
                elif filler == self.kb.nothing:
                    filler = self.renderer.render(self.kb.nothing)
                else:
                    filler = self.expression[self.renderer.render(filler)]

                target = UniversalQuantifierExpression(
                    name=self.renderer.render(mgur),
                    role=self.renderer.render(mgur.get_property()),
                    filler=filler,
                    str_individuals=set([_.get_iri().as_str() for _ in self.kb.individuals(mgur)]),
                    expression_chain=[self.renderer.render(self.kb.thing)])
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
                    expression_chain=[self.renderer.render(self.kb.thing)])
                self.__store(target, length=3)
        self.compute()
        for k, v in self.expression.items():
            if len(v.expression_chain) == 0:  # and k != '⊤':
                v.expression_chain = [self.expression['⊤'].name]

            assert len(v.expression_chain) > 0
            for x in v.expression_chain:
                self.dict_sh_direct_down.setdefault(x, set()).add(k)
                self.dict_sh_direct_up.setdefault(k, set()).add(x)

    def __negative_atomic_exp(self, ae, neg_ae):
        assert isinstance(ae, AtomicExpression)
        target = ComplementOfAtomicExpression(
            name='¬' + ae.name,
            atomic_expression=ae,
            str_individuals=self.N_I.difference(ae.str_individuals),
            expression_chain=neg_ae.expression_chain + [neg_ae.name])
        self.__store(target, length=2)
        self.str_ae_to_neg_ae[ae.name] = target
        return target

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
                        expression_chain=[self.renderer.render(self.kb.thing)]
                    )
                    self.expression[target.name] = target
                    yield target
            else:
                str_ = self.renderer.render(owlclass)
                if str_ in self.expression:
                    yield ce * self.expression[str_]

    def __refine_top_neg_leafs(self, ce=None):
        """ (1)  {¬A | A ∈ N_C, (¬A \sqcap B notequiv \⊥), (¬A \sqcap B notequiv B),
        there is no there is no A' ∈ N_C with A' \sqsubset A}"""

        for leaf in self.kb.get_leaf_concepts(self.kb.thing):
            owl_obj_complement = self.kb.negation(leaf)
            if ce is None:
                str_individuals = set([_.get_iri().as_str() for _ in self.kb.individuals(owl_obj_complement)])
                if 0 < len(self.N_I.intersection(str_individuals)) and str_individuals != self.N_I:
                    target = ComplementOfAtomicExpression(
                        name=self.renderer.render(owl_obj_complement),
                        atomic_expression=self.expression[self.renderer.render(leaf)],
                        str_individuals=str_individuals,
                        expression_chain=[self.renderer.render(self.kb.thing)]
                    )
                    self.expression[target.name] = target
                    self.str_ae_to_neg_ae[self.renderer.render(leaf)] = target
                    yield target
            else:
                str_ = self.renderer.render(owl_obj_complement)
                if str_ in self.expression:
                    yield ce * self.expression[str_]

    def __refine_top_ur(self, ce=None):

        for mgur in self.kb.most_general_universal_restrictions(domain=self.kb.thing, filler=None):
            filler = mgur.get_filler()
            assert filler == self.kb.thing
            if ce is None:
                target = UniversalQuantifierExpression(
                    name=self.renderer.render(mgur),
                    role=self.renderer.render(mgur.get_property()),
                    filler=self.expression[self.renderer.render(self.kb.thing)],
                    str_individuals=set([_.get_iri().as_str() for _ in self.kb.individuals(mgur)]),
                    expression_chain=[self.renderer.render(self.kb.thing)]
                )
                self.__store(target, length=3)

                yield target
            else:
                yield ce * self.expression[self.renderer.render(mgur)]

    def __refine_top_er(self, ce=None):
        for mger in self.kb.most_general_existential_restrictions(domain=self.kb.thing, filler=None):
            filler = mger.get_filler()
            assert filler == self.kb.thing
            if ce is None:
                target = ExistentialQuantifierExpression(
                    name=self.renderer.render(mger),
                    role=self.renderer.render(mger.get_property()),
                    filler=self.expression[self.renderer.render(self.kb.thing)],
                    str_individuals=set([_.get_iri().as_str() for _ in self.kb.individuals(mger)]),
                    expression_chain=[self.renderer.render(self.kb.thing)]
                )
                self.__store(target, length=3)
                yield target

            else:
                yield ce * self.expression[self.renderer.render(mger)]

    def refine_top(self, ce=None):
        """ Refine Top Class Expression:
        from Concept Learning J. LEHMANN and N. FANIZZI and L. BÜHMANN and C. D’AMATO
        """
        if ce is None:
            if self.top_refinements is None:
                self.top_refinements = set(itertools.chain.from_iterable((self.__refine_top_direct_nc(ce),
                                                                          self.__refine_top_neg_leafs(ce),
                                                                          self.__refine_top_ur(ce),
                                                                          self.__refine_top_er(ce)
                                                                          )))

        return self.top_refinements

    def sh_down(self, atomic_class_expression: AtomicExpression) -> List:
        """ sh_↓ (A) ={A' \in N_C | A' \sqsubset A, there is no A'' \in N_C with A' \sqsubset A'' \sqsubset A}
        Return its direct sub concepts
        """
        res = []
        for str_ in self.dict_sh_direct_down[atomic_class_expression.name]:
            res.append(self.expression[str_])
        return res  # [self.expression[str_] for str_ in self.dict_sh_direct_down[atomic_class_expression.name]]

    def sh_up(self, atomic_class_expression: AtomicExpression) -> List:
        """ sh_up (A) ={A' \in N_C | A \sqsubset A', there is no A'' \in N_C with A \sqsubset A' \sqsubset A''}
        Return its direct sub concepts
        """
        return [self.expression[str_] for str_ in self.dict_sh_direct_up[atomic_class_expression.name]]

    def refine_atomic_concept(self, atomic_class_expression: AtomicExpression):
        res = self.sh_down(atomic_class_expression)
        yield from res
        for i in self.top_refinements:
            for j in res:
                yield i * j

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
                res.append(self.__negative_atomic_exp(ae, compement_of_atomic_class_expression))

        yield from res
        for i in self.top_refinements:
            for j in res:
                yield i * j

    def refine_universal_quantifier_expression(self, uqe: UniversalQuantifierExpression):
        """ \rho(C) if C ==∀r.D:
                                {(∀r.D) \sqcap E \in \rho(T) \cup
        """
        # Important Limitation: We can not refine filler:
        # We should obtain ∀r.X | X \in _NC AND ∀r.X \in \rho(T) so that
        # ∀r.X  AND ∀r.Y would give us ∀r.(X AND Y)
        # role = uqe.role
        # filler = uqe.filler
        for i in self.top_refinements:
            yield i * uqe

    def refine_existential_expression(self, uqe: ExistentialQuantifierExpression):
        """ \rho(C) if C ==∀r.D:
                                {(∀r.D) \sqcap E \in \rho(T) \cup
        """
        for i in self.top_refinements:
            yield i * uqe

    def refine(self, class_expression) -> Iterable:
        if isinstance(class_expression, AtomicExpression):
            yield from self.refine_atomic_concept(class_expression)
        elif isinstance(class_expression, ComplementOfAtomicExpression):
            yield from self.refine_complement_of(class_expression)
        elif isinstance(class_expression, UniversalQuantifierExpression):
            yield from self.refine_universal_quantifier_expression(class_expression)
        elif isinstance(class_expression, ExistentialQuantifierExpression):
            yield from self.refine_existential_expression(class_expression)

        else:
            print(class_expression)
            raise ValueError
