from typing import Set, List


class TargetClassExpression:
    def __init__(self, *, label_id, name: str, idx_individuals: Set, expression_chain: List, length: int = None,
                 # str_individuals: Set = None
                 ):
        assert isinstance(name, str)
        assert isinstance(expression_chain, list)

        self.label_id = label_id
        self.name = name
        self.idx_individuals = idx_individuals
        # self.str_individuals = str_individuals
        # if self.str_individuals is None:
        #    self.str_individuals = set()
        # else:
        #    assert len(self.str_individuals) == len(self.idx_individuals)

        self.expression_chain = expression_chain
        self.num_individuals = len(self.idx_individuals)
        self.length = length
        self.quality = None

    @property
    def size(self):
        return self.num_individuals

    def __lt__(self, other):
        return self.quality < other.quality

    def __str__(self):
        return f'{self.name} | Indv:{self.num_individuals} | Quality:{self.quality}'  # | expression_chain:{self.expression_chain}'

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        return TargetClassExpression(
            label_id=str(self.label_id) + '_and_' + str(other),
            name=f'({self.name}) ⊓ ({other.name})',
            idx_individuals=self.idx_individuals.intersection(other.idx_individuals),
            # str_individuals=self.str_individuals.intersection(other.str_individuals),
            expression_chain=self.expression_chain + [other.name], length=self.length + other.length + 1)

    def __add__(self, other):
        return TargetClassExpression(
            label_id=str(self.label_id) + '_or_' + str(other),
            name=f'({self.name}) ⊔ ({other.name})',
            idx_individuals=self.idx_individuals.union(other.idx_individuals),
            # str_individuals=self.str_individuals.union(other.str_individuals),
            expression_chain=self.expression_chain + [other.name], length=self.length + other.length + 1)


class ClassExpression:
    def __init__(self, *, name: str, str_individuals: Set, expression_chain: List, owl_class=None,
                 quality=None, length=None):
        assert isinstance(name, str)
        assert isinstance(str_individuals, set)
        assert isinstance(expression_chain, list)

        self.name = name
        self.str_individuals = str_individuals
        self.expression_chain = expression_chain
        self.num_individuals = len(self.str_individuals)
        self.quality = quality
        self.owl_class = owl_class
        if length is None:
            self.length = len(self.name.split())
        else:
            self.length=length

    def __str__(self):
        return f'ClassExpression at {hex(id(self))} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality}'  # | expression_chain:{self.expression_chain}'

    def __repr__(self):
        return self.__str__()

    @property
    def size(self):
        return self.num_individuals

    def __lt__(self, other):
        return self.quality < other.quality

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
                                           str_individuals=self.str_individuals.intersection(other.str_individuals),
                                           expression_chain=self.expression_chain + [(self.name, 'AND', other.name)])

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
                                    expression_chain=self.expression_chain + [(self.name, 'OR', other.name)])


class UnionClassExpression(ClassExpression):
    def __init__(self, *, name: str, length: int, str_individuals: Set, expression_chain: List, owl_class=None,
                 quality=None):
        super().__init__(name=name, str_individuals=str_individuals, expression_chain=expression_chain, quality=quality,
                         owl_class=owl_class)
        assert length >= 3
        self.length = length

    def __str__(self):
        return f'UnionClassExpression at {hex(id(self))} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality}'  # | expression_chain:{self.expression_chain}'


class IntersectionClassExpression(ClassExpression):
    def __init__(self, *, name: str, length: int, str_individuals: Set, expression_chain: List, owl_class=None,
                 quality=None):
        super().__init__(name=name, str_individuals=str_individuals, expression_chain=expression_chain, quality=quality,
                         owl_class=owl_class)
        assert length >= 3
        self.length = length

    def __str__(self):
        return f'UnionClassExpression at {hex(id(self))} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality}'  # | expression_chain:{self.expression_chain}'


class AtomicExpression(ClassExpression):
    def __init__(self, *, name: str, str_individuals: Set, expression_chain: List, owl_class=None,
                 quality=None):
        super().__init__(name=name, str_individuals=str_individuals, expression_chain=expression_chain, quality=quality,
                         owl_class=owl_class)
        self.length = 1

    def __str__(self):
        return f'AtomicExpression at {hex(id(self))} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality}'  # | expression_chain:{self.expression_chain}'


class ComplementOfAtomicExpression(ClassExpression):
    def __init__(self, *, name: str, atomic_expression, str_individuals: Set, expression_chain: List,
                 quality=None, owl_class=None):
        super().__init__(name=name, str_individuals=str_individuals, expression_chain=expression_chain, quality=quality,
                         owl_class=owl_class)
        self.atomic_expression = atomic_expression
        self.length = self.atomic_expression.length + 1

    def __str__(self):
        return f'ComplementOfAtomicExpression at {hex(id(self))} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality}'  # | expression_chain:{self.expression_chain}'


class UniversalQuantifierExpression(ClassExpression):
    def __init__(self, *, name: str, role, filler, str_individuals: Set, expression_chain: List, quality=None):
        assert isinstance(name, str)
        assert isinstance(str_individuals, set)
        assert isinstance(expression_chain, list)
        super().__init__(name=name, str_individuals=str_individuals, expression_chain=expression_chain, quality=quality)
        self.role = role
        self.filler = filler
        self.type = "forall"  # ∀
        if isinstance(self.filler, str):
            try:
                assert self.filler == '⊤' or self.filler == '⊥'
                self.length = 2 + 1
            except:
                print(self.filler)
                exit(1)

        else:
            self.length = 2 + self.filler.length

    def __str__(self):
        return f'UniversalQuantifierExpression at {hex(id(self))} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality}'


class ExistentialQuantifierExpression(ClassExpression):
    def __init__(self, *, name: str, role, filler, str_individuals: Set, expression_chain: List, quality=None):
        assert isinstance(name, str)
        assert isinstance(str_individuals, set)
        assert isinstance(expression_chain, list)
        super().__init__(name=name, str_individuals=str_individuals, expression_chain=expression_chain, quality=quality)
        self.role = role
        self.filler = filler
        self.type = "exists"  # ∃
        if isinstance(self.filler, str):
            try:
                assert self.filler == '⊤' or self.filler == '⊥'
                self.length = 2 + 1
            except:
                print(self.filler)
                exit(1)

        else:
            self.length = 2 + self.filler.length

    def __str__(self):
        return f'ExistentialQuantifierExpression at {hex(id(self))} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality}'


class Role:
    def __init__(self, *, name: str):
        assert isinstance(name, str)
        self.name = name
