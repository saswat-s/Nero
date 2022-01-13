from abc import ABC, abstractmethod
from typing import Set, List


class TargetClassExpression:
    def __init__(self, *, label_id, name: str, idx_individuals: Set = None, expression_chain: List = None,
                 length: int = None, str_individuals: Set = None, type=None
                 ):

        self.label_id = label_id
        self.name = name
        self.idx_individuals = idx_individuals
        self.str_individuals = str_individuals
        self.type = type
        self.expression_chain = expression_chain

        self.num_individuals = len(self.str_individuals)
        """
        if self.idx_individuals is not None:
            self.num_individuals = len(self.idx_individuals)
        else:
            self.num_individuals = None
        """

        self.length = length
        self.quality = None
    @property
    def size(self):
        return self.num_individuals

    def __lt__(self, other):
        return self.quality < other.quality

    def __str__(self):
        return f'TargetClassExpression at {hex(id(self))} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality}'  # | expression_chain:{self.expression_chain}'

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        idx_individuals = None
        str_individuals = None

        if self.idx_individuals is not None and other.idx_individuals is not None:
            idx_individuals = self.idx_individuals.intersection(other.idx_individuals)

        if self.str_individuals is not None and other.str_individuals is not None:
            str_individuals = self.str_individuals.intersection(other.str_individuals)
        if self.length <= 2 and other.length <= 2:
            name = f'{self.name} ⊓ {other.name}'
        elif self.length <= 2 < other.length:
            name = f'{self.name} ⊓ ({other.name})'
        elif self.length > other.length <= 2:
            name = f'({self.name}) ⊓ {other.name}'
        elif self.length >= 2 and other.length >= 2:
            name = f'({self.name}) ⊓ ({other.name})'
        else:
            print(self)
            print(other)
            raise ValueError

        return TargetClassExpression(
            #label_id=str(self.label_id) + '_and_' + str(other),
            name=name,#f'({self.name}) ⊓ ({other.name})',
            #idx_individuals=idx_individuals,
            str_individuals=str_individuals,
            expression_chain=((self.expression_chain + (self.name,)), 'AND',
                                                             (other.expression_chain + (other.name,))),
            length=self.length + other.length + 1)

    def __add__(self, other):
        idx_individuals = None
        str_individuals = None

        if self.idx_individuals is not None and other.idx_individuals is not None:
            idx_individuals = self.idx_individuals.intersection(other.idx_individuals)

        if self.str_individuals is not None and other.str_individuals is not None:
            str_individuals = self.str_individuals.intersection(other.str_individuals)

        if self.length <= 2 and other.length <= 2:
            name = f'{self.name} ⊔ {other.name}'
        elif self.length <= 2 < other.length:
            name = f'{self.name} ⊔ ({other.name})'
        elif self.length > other.length <= 2:
            name = f'({self.name}) ⊔ {other.name}'
        elif self.length >= 2 and other.length >= 2:
            name = f'({self.name}) ⊔ ({other.name})'
        else:
            print(self)
            print(other)
            raise ValueError

        return TargetClassExpression(
            #label_id=str(self.label_id) + '_or_' + str(other),
            name=name,
            #idx_individuals=idx_individuals,
            str_individuals=str_individuals,
            expression_chain=((self.expression_chain + (self.name,)), 'OR',
                                                      (other.expression_chain + (other.name,))),
            length=self.length + other.length + 1)


class ClassExpression(ABC):
    def __init__(self, *, name: str, str_individuals: Set, expression_chain: List, owl_class=None,
                 quality=None, length=None):
        assert isinstance(name, str)
        assert isinstance(str_individuals, set)
        try:
            assert isinstance(expression_chain, list) or isinstance(expression_chain, tuple)
        except AssertionError:
            print(expression_chain)
            print(type(expression_chain))
            raise ValueError
        self.name = name
        self.str_individuals = str_individuals
        self.expression_chain = expression_chain
        self.num_individuals = len(self.str_individuals)
        if quality is None:
            self.quality = -1.0  # quality
        self.owl_class = owl_class
        if length is None:
            self.length = len(self.name.split())
        else:
            self.length = length

    def __str__(self):
        return f'{self.type} at {hex(id(self))} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality:.3f}'  # | expression_chain:{self.expression_chain}'

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        if self.length <= 2 and other.length <= 2:
            name = f'{self.name} ⊓ {other.name}'
        elif self.length <= 2 < other.length:
            name = f'{self.name} ⊓ ({other.name})'
        elif self.length > other.length <= 2:
            name = f'({self.name}) ⊓ {other.name}'
        elif self.length >= 2 and other.length >= 2:
            name = f'({self.name}) ⊓ ({other.name})'
        else:
            print(self)
            print(other)
            raise ValueError
        return IntersectionClassExpression(name=name, length=self.length + other.length + 1,
                                           concepts=(self, other),
                                           str_individuals=self.str_individuals.intersection(other.str_individuals),
                                           expression_chain=((self.expression_chain + (self.name,)), 'AND',
                                                             (other.expression_chain + (other.name,))))

    def __add__(self, other):
        if self.length <= 2 and other.length <= 2:
            name = f'{self.name} ⊔ {other.name}'
        elif self.length <= 2 < other.length:
            name = f'{self.name} ⊔ ({other.name})'
        elif self.length > other.length <= 2:
            name = f'({self.name}) ⊔ {other.name}'
        elif self.length >= 2 and other.length >= 2:
            name = f'({self.name}) ⊔ ({other.name})'
        else:
            print(self)
            print(other)
            raise ValueError

        return UnionClassExpression(name=name, length=self.length + other.length + 1,
                                    str_individuals=self.str_individuals.union(other.str_individuals),
                                    concepts=(self, other),
                                    expression_chain=((self.expression_chain + (self.name,)), 'OR',
                                                      (other.expression_chain + (other.name,))))

    @property
    def size(self):
        return self.num_individuals

    def __lt__(self, other):
        return self.quality < other.quality


class UnionClassExpression(ClassExpression):
    def __init__(self, *, name: str, length: int, str_individuals: Set, expression_chain: List, owl_class=None,
                 concepts=None,
                 quality=None, label_id=None, idx_individuals=None):
        super().__init__(name=name, str_individuals=str_individuals,
                         expression_chain=expression_chain, quality=quality,
                         owl_class=owl_class)
        assert length >= 3
        self.length = length
        self.type = 'union_expression'
        self.label_id = label_id
        self.idx_individuals = idx_individuals
        self.concepts = concepts

    def __str__(self):
        return f'UnionClassExpression at {hex(id(self))} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality:.3f}'  # | expression_chain:{self.expression_chain}'


class IntersectionClassExpression(ClassExpression):
    def __init__(self, *, name: str, length: int, str_individuals: Set, expression_chain: List, owl_class=None,
                 quality=None, label_id=None, concepts=None, idx_individuals=None):
        super().__init__(name=name, str_individuals=str_individuals, expression_chain=expression_chain, quality=quality,
                         owl_class=owl_class)

        assert length >= 3
        self.length = length
        self.type = 'intersection_expression'
        self.label_id = label_id
        self.idx_individuals = idx_individuals
        self.concepts = concepts

    def __str__(self):
        return f'IntersectionClassExpression at {hex(id(self))} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality}'  # | expression_chain:{self.expression_chain}'


class AtomicExpression(ClassExpression):
    def __init__(self, *, name: str, str_individuals: Set, expression_chain: List,
                 owl_class=None, quality=None, label_id=None, idx_individuals=None):
        super().__init__(name=name, str_individuals=str_individuals, expression_chain=expression_chain, quality=quality,
                         owl_class=owl_class)
        self.length = 1
        self.type = 'atomic_expression'
        self.idx_individuals = idx_individuals
        self.label_id = label_id
        self.idx_individuals = idx_individuals

    def __str__(self):
        return f'AtomicExpression at {hex(id(self))} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality:.3f}'  # | expression_chain:{self.expression_chain}'


class ComplementOfAtomicExpression(ClassExpression):
    def __init__(self, *, name: str, atomic_expression, str_individuals: Set, expression_chain: List,
                 quality=None, owl_class=None, label_id=None, idx_individuals=None):
        super().__init__(name=name, str_individuals=str_individuals, expression_chain=expression_chain, quality=quality,
                         owl_class=owl_class)
        self.atomic_expression = atomic_expression
        self.length = self.atomic_expression.length + 1
        self.type = 'negated_expression'
        self.label_id = label_id
        self.idx_individuals = idx_individuals

    def __str__(self):
        return f'ComplementOfAtomicExpression at {hex(id(self))} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality:.3f}'  # | expression_chain:{self.expression_chain}'


class UniversalQuantifierExpression(ClassExpression):
    def __init__(self, *, name: str, role=None, filler=None, label_id=None, idx_individuals=None, str_individuals: Set,
                 expression_chain: List, quality=None):
        assert isinstance(name, str)
        assert isinstance(str_individuals, set)
        try:
            assert isinstance(expression_chain, list) or isinstance(expression_chain, tuple)
        except:
            print(expression_chain)
            print('asdasd')
            print(name)
            raise ValueError
        super().__init__(name=name, str_individuals=str_individuals, expression_chain=expression_chain, quality=quality)
        self.role = role
        self.filler = filler
        self.type = "universal_quantifier_expression"  # ∀
        self.label_id = label_id
        self.idx_individuals = idx_individuals
        self.length = 3

    def __str__(self):
        return f'UniversalQuantifierExpression at {hex(id(self))} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality:.3f}'


class ExistentialQuantifierExpression(ClassExpression):
    def __init__(self, *, name: str, role=None, filler=None, str_individuals: Set, expression_chain: List, quality=None,
                 label_id=None, idx_individuals=None):
        assert isinstance(name, str)
        assert isinstance(str_individuals, set)
        assert isinstance(expression_chain, list) or isinstance(expression_chain, tuple)
        super().__init__(name=name, str_individuals=str_individuals, expression_chain=expression_chain, quality=quality)
        self.role = role
        self.filler = filler
        # self.type = "exists"  # ∃
        self.type = "existantial_quantifier_expression"  # ∀
        self.label_id = label_id
        self.idx_individuals = idx_individuals
        self.length = 3

    def __str__(self):
        return f'ExistentialQuantifierExpression at {hex(id(self))} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality:.3f}'


class Role:
    def __init__(self, *, name: str):
        assert isinstance(name, str)
        self.name = name

    def __str__(self):
        return f'Role at {hex(id(self))} | {self.name}'

    def __repr__(self):
        return self.__str__()