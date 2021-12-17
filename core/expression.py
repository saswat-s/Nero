from typing import Set, List

class TargetClassExpression:
    def __init__(self, *, label_id, name: str, idx_individuals: Set, expression_chain: List, length: int = None):
        assert isinstance(name, str)
        assert isinstance(expression_chain, list)

        self.label_id = label_id
        self.name = name
        self.idx_individuals = idx_individuals
        self.expression_chain = expression_chain
        self.num_individuals = len(self.idx_individuals)
        self.length = length

    @property
    def size(self):
        return self.num_individuals

    def __str__(self):
        return f'{self.name} | Indv:{self.num_individuals}'

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        return TargetClassExpression(
            label_id=str(self.label_id) + '_and_' + str(other),
            name=f'({self.name}) ⊓ ({other.name})',
            idx_individuals=self.idx_individuals.intersection(other.idx_individuals),
            expression_chain=self.expression_chain + [other.name], length=self.length + other.length + 1)

    def __add__(self, other):
        return TargetClassExpression(
            label_id=str(self.label_id)+'_or_'+str(other),
            name=f'({self.name}) ⊔ ({other.name})',
            idx_individuals=self.idx_individuals.union(other.idx_individuals),
            expression_chain=self.expression_chain + [other.name], length=self.length + other.length + 1)

class ClassExpression:
    def __init__(self, *, name: str, str_individuals: Set, expression_chain: List,owl_class=None,
                 quality=None):
        assert isinstance(name, str)
        assert isinstance(str_individuals, set)
        assert isinstance(expression_chain, list)

        self.name = name
        self.str_individuals = str_individuals
        self.expression_chain = expression_chain
        self.num_individuals = len(self.str_individuals)
        self.quality = quality
        self.owl_class = owl_class

    @property
    def size(self):
        return self.num_individuals

    def __str__(self):
        return f'CLassExpression at {hex(id(self))} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality}'

    def __lt__(self, other):
        return self.quality < other.quality

    def __mul__(self, other):
        if len(self.name.split()) == 1:
            name = f'{self.name} ⊓ ({other.name})'
        else:
            name = f'({self.name}) ⊓ ({other.name})'
        return ClassExpression(
            name=name,
            str_individuals=self.str_individuals.intersection(other.str_individuals),
            expression_chain=self.expression_chain + [(self, 'AND', other)])

    def __add__(self, other):
        if len(self.name.split()) == 1:
            name = f'{self.name} ⊔ ({other.name})'
        else:
            name = f'({self.name}) ⊔ ({other.name})'

        return ClassExpression(
            name=name,
            str_individuals=self.str_individuals.union(other.str_individuals),
            expression_chain=self.expression_chain + [(self, 'OR', other)])


class UniversalQuantifierExpression(ClassExpression):
    def __init__(self, *, name: str, role, filler, str_individuals: Set, expression_chain: List, quality=None):
        assert isinstance(name, str)
        assert isinstance(str_individuals, set)
        assert isinstance(expression_chain, list)
        super().__init__(name=name, str_individuals=str_individuals, expression_chain=expression_chain, quality=quality)
        self.role = role
        self.filler = filler
        self.type = "forall"  # ∀

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

    def __str__(self):
        return f'ExistentialQuantifierExpression at {hex(id(self))} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality}'


class Role:
    def __init__(self, *, name: str):
        assert isinstance(name, str)
        self.name = name
