from typing import Set,List
class TargetClassExpression:
    def __init__(self, *, label_id: int, name: str, idx_individuals: Set, expression_chain: List):
        assert isinstance(label_id, int)
        assert isinstance(name, str)
        assert isinstance(idx_individuals, frozenset)
        assert isinstance(expression_chain, list)

        self.label_id = label_id
        self.name = name
        self.idx_individuals = idx_individuals
        self.expression_chain = expression_chain
        self.num_individuals = len(self.idx_individuals)

    @property
    def size(self):
        return self.num_individuals

    def __str__(self):
        return f'{self.name} | Indv:{self.num_individuals}'

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        return TargetClassExpression(
            label_id=-1,
            name=f'({self.name}) ⊓ ({other.name})',
            idx_individuals=self.idx_individuals.intersection(other.idx_individuals),
            expression_chain=self.expression_chain + [other.name])

    def __add__(self, other):
        return TargetClassExpression(
            label_id=-2,
            name=f'({self.name}) ⊔ ({other.name})',
            idx_individuals=self.idx_individuals.union(other.idx_individuals),
            expression_chain=self.expression_chain + [other.name])

class ClassExpression:
    def __init__(self, *, name: str, individuals: Set, expression_chain: List):
        assert isinstance(name, str)
        assert isinstance(individuals, set)
        assert isinstance(expression_chain, list)

        self.name = name
        self.individuals = individuals
        self.expression_chain = expression_chain
        self.num_individuals = len(self.individuals)

    @property
    def size(self):
        return self.num_individuals

    def __str__(self):
        return f'{self.name} | Indv:{self.num_individuals}'
