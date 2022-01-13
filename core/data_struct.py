from queue import PriorityQueue
from .expression import TargetClassExpression


class State:
    def __init__(self, quality: float, tce: TargetClassExpression, str_individuals: set):
        self.quality = quality
        self.tce = tce
        self.name = self.tce.name
        self.str_individuals = str_individuals

    def __lt__(self, other):
        return self.quality < other.quality

    def __str__(self):
        return f'{self.tce} | Quality:{self.quality:.3f}'

class SearchTree:
    def __init__(self, maxsize=0):
        self.items_in_queue = PriorityQueue(maxsize)
        self.gate = dict()

    def __contains__(self, key):
        return key in self.gate

    def put(self, expression, key=None, condition=None):
        if condition is None:
            if expression.name not in self.gate:
                if key is None:
                    key = -expression.quality
                # The lowest valued entries are retrieved first
                self.items_in_queue.put((key, expression))
                self.gate[expression.name] = expression
        else:
            raise ValueError('Define the condition')

    def get(self):
        _, expression = self.items_in_queue.get(timeout=1)
        del self.gate[expression.name]
        return expression

    def get_all(self):
        return list(self.gate.values())

    def __len__(self):
        return len(self.gate)

    def __iter__(self):
        # No garantie of returing best
        return (exp for q, exp in self.items_in_queue.queue)

    def extend_queue(self, other) -> None:
        """
        Extend queue with other queue.
        :param other:
        """
        for i in other:
            self.put(i)
