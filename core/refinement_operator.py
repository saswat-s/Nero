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
from owlapy.model import OWLObjectPropertyExpression, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, \
    OWLObjectIntersectionOf, OWLClassExpression, OWLNothing, OWLThing, OWLNaryBooleanClassExpression, \
    OWLObjectUnionOf, OWLClass, OWLObjectComplementOf, OWLObjectMaxCardinality, OWLObjectMinCardinality, \
    OWLDataSomeValuesFrom, OWLDatatypeRestriction, OWLLiteral, OWLObjectInverseOf, OWLDataProperty, \
    OWLDataHasValue, OWLDataPropertyExpression
from ontolearn.search import Node


class SimpleRefinement(BaseRefinement):
    """ A top down refinement operator refinement operator in ALC."""

    def __init__(self, knowledge_base: KnowledgeBase):
        super().__init__(knowledge_base)
        # 1. Number of named classes and sanity checking
        num_of_named_classes = len(set(i for i in self.kb.ontology().classes_in_signature()))
        assert num_of_named_classes == len(list(i for i in self.kb.ontology().classes_in_signature()))
        self.top_refinements = []
        for ref in self.refine_top():
            self.top_refinements.append(ref)

    def refine_top(self) -> Iterable:
        """ Refine Top Class Expression """
        """ (1) Store all named classes """
        iterable_container = []
        direct_sub_classes = [i for i in self.kb.get_all_direct_sub_concepts(self.kb.thing)]
        iterable_container.append(direct_sub_classes)
        """ (2) Negate Leafs ot Top and store it """
        iterable_container.append(self.kb.negation_from_iterables((i for i in self.kb.get_leaf_concepts(self.kb.thing))))
        """ (3) Add Nothing """
        iterable_container.append([self.kb.nothing])
        """ (4) Get all most general restrictions and store them forall r. T, \\exist r. T """
        iterable_container.append(self.kb.most_general_universal_restrictions(domain=self.kb.thing, filler=None))
        iterable_container.append(self.kb.most_general_existential_restrictions(domain=self.kb.thing, filler=None))
        """ (5) Generate all refinements of given concept that have length less or equal to the maximum refinement
         length constraint """
        for i in iterable_container:
            yield from i

    def apply_union_and_intersection_from_iterable(self, cont: Iterable[Generator]) -> Iterable:
        """ Create Union and Intersection OWL Class Expressions
        1. Create OWLObjectIntersectionOf via logical conjunction of cartesian product of input owl class expressions
        2. Create OWLObjectUnionOf class expression via logical disjunction pf cartesian product of input owl class
         expressions
        Repeat 1 and 2 until all concepts having max_len_refinement_top reached.
        """
        cumulative_refinements = dict()
        """ 1. Flatten list of generators """
        for class_expression in chain.from_iterable(cont):
            if class_expression is not self.kb.nothing:
                """ 1.2. Store qualifying concepts based on their lengths """
                cumulative_refinements.setdefault(self.len(class_expression), set()).add(class_expression)
            else:
                """ No need to union or intersect Nothing, i.e. ignore concept that does not satisfy constraint"""
                yield class_expression
        """ 2. Lengths of qualifying concepts """
        lengths = [i for i in cumulative_refinements.keys()]

        seen = set()
        larger_cumulative_refinements = dict()
        """ 3. Iterative over lengths """
        for i in lengths:  # type: int
            """ 3.1 Return all class expressions having the length i """
            yield from cumulative_refinements[i]
            """ 3.2 Create intersection and union of class expressions having the length i with class expressions in
             cumulative_refinements """
            for j in lengths:
                """ 3.3 Ignore if we have already createdValid intersection and union """
                if (i, j) in seen or (j, i) in seen:
                    continue

                seen.add((i, j))
                seen.add((j, i))

                len_ = i + j + 1

                if len_ <= self.max_len_refinement_top:
                    """ 3.4 Intersect concepts having length i with concepts having length j"""
                    intersect_of_concepts = self.kb.intersect_from_iterables(cumulative_refinements[i],
                                                                             cumulative_refinements[j])
                    """ 3.4 Union concepts having length i with concepts having length j"""
                    union_of_concepts = self.kb.union_from_iterables(cumulative_refinements[i],
                                                                     cumulative_refinements[j])
                    res = set(chain(intersect_of_concepts, union_of_concepts))

                    # Store newly generated concepts at 3.2.
                    if len_ in cumulative_refinements:
                        x = cumulative_refinements[len_]
                        cumulative_refinements[len_] = x.union(res)
                    else:
                        if len_ in larger_cumulative_refinements:
                            x = larger_cumulative_refinements[len_]
                            larger_cumulative_refinements[len_] = x.union(res)
                        else:
                            larger_cumulative_refinements[len_] = res

        for k, v in larger_cumulative_refinements.items():
            yield from v

    def refine_atomic_concept(self, class_expression: OWLClassExpression) -> Iterable[OWLClassExpression]:
        """
        Refine an atomic class expressions, i.e,. length 1
        """
        assert isinstance(class_expression, OWLClassExpression)
        for i in self.top_refinements:
            # No need => Daughter ⊓ Daughter
            # No need => Daughter ⊓ \bottom
            if i.is_owl_nothing() is False:  # and (i != class_expression)
                yield self.kb.intersection((class_expression, i))

    def refine_complement_of(self, class_expression: OWLObjectComplementOf) -> Iterable[OWLClassExpression]:
        """
        Refine OWLObjectComplementOf
        1- Get All direct parents
        2- Negate (1)
        3- Intersection with T
        """
        assert isinstance(class_expression, OWLObjectComplementOf)
        yield from self.kb.negation_from_iterables(self.kb.get_direct_parents(self.kb.negation(class_expression)))
        yield self.kb.intersection((class_expression, self.kb.thing))

    def refine_object_some_values_from(self, class_expression: OWLObjectSomeValuesFrom) -> Iterable[OWLClassExpression]:
        assert isinstance(class_expression, OWLObjectSomeValuesFrom)
        # rule 1: \exists r.D = > for all r.E
        for i in self.refine(class_expression.get_filler()):
            yield self.kb.existential_restriction(i, class_expression.get_property())
        # rule 2: \exists r.D = > \exists r.D AND T
        yield self.kb.intersection((class_expression, self.kb.thing))

    def refine_object_all_values_from(self, class_expression: OWLObjectAllValuesFrom) -> Iterable[OWLClassExpression]:
        assert isinstance(class_expression, OWLObjectAllValuesFrom)
        # rule 1: \forall r.D = > \forall r.E
        for i in self.refine(class_expression.get_filler()):
            yield self.kb.universal_restriction(i, class_expression.get_property())
        # rule 2: \forall r.D = > \forall r.D AND T
        yield self.kb.intersection((class_expression, self.kb.thing))

    def refine_object_union_of(self, class_expression: OWLObjectUnionOf) -> Iterable[OWLClassExpression]:
        """
        Refine C =A AND B
        """
        assert isinstance(class_expression, OWLObjectUnionOf)
        operands: List[OWLClassExpression] = list(class_expression.operands())
        for i in operands:
            for ref_concept_A in self.refine(i):
                if ref_concept_A == class_expression:
                    # No need => Person OR MALE => rho(Person) OR MALE => MALE OR MALE
                    yield class_expression
                yield self.kb.union((class_expression, ref_concept_A))

    def refine_object_intersection_of(self, class_expression: OWLClassExpression) -> Iterable[OWLClassExpression]:
        """
        Refine C =A AND B
        """
        assert isinstance(class_expression, OWLObjectIntersectionOf)
        operands: List[OWLClassExpression] = list(class_expression.operands())
        for i in operands:
            for ref_concept_A in self.refine(i):
                if ref_concept_A == class_expression:
                    # No need => Person ⊓ MALE => rho(Person) ⊓ MALE => MALE ⊓ MALE
                    yield class_expression
                # TODO: No need to intersect disjoint expressions
                yield self.kb.intersection((class_expression, ref_concept_A))

    def refine(self, class_expression) -> Iterable[OWLClassExpression]:
        assert isinstance(class_expression, OWLClassExpression)
        if class_expression.is_owl_thing():
            yield from self.top_refinements
        elif class_expression.is_owl_nothing():
            yield from {class_expression}
        elif self.len(class_expression) == 1:
            yield from self.refine_atomic_concept(class_expression)
        elif isinstance(class_expression, OWLObjectComplementOf):
            yield from self.refine_complement_of(class_expression)
        elif isinstance(class_expression, OWLObjectSomeValuesFrom):
            yield from self.refine_object_some_values_from(class_expression)
        elif isinstance(class_expression, OWLObjectAllValuesFrom):
            yield from self.refine_object_all_values_from(class_expression)
        elif isinstance(class_expression, OWLObjectUnionOf):
            yield from self.refine_object_union_of(class_expression)
        elif isinstance(class_expression, OWLObjectIntersectionOf):
            yield from self.refine_object_intersection_of(class_expression)
        else:
            raise ValueError
