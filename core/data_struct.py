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

"""
class ExpressionQueue:
    def __init__(self):
        self.items_in_queue = PriorityQueue()
        self.current_length = 0

    def put(self, quality: float, tce: TargetClassExpression, str_individuals: set):
        # The lowest valued entries are retrieved first
        self.items_in_queue.put((-quality, State(quality, tce, str_individuals)))
        self.current_length += 1

    def get(self):
        _, state_x = self.items_in_queue.get()
        self.current_length -= 1
        return state_x

    def __len__(self):
        return self.current_length

    def __iter__(self):
        while self.current_length > 0:
            # most promising state
            mps = self.get()
            yield mps.quality, mps.tce, mps.str_individuals

    def get_top(self, n):
        assert n > 0
        while n > 0 and self.current_length > 0:
            n -= 1
            self.current_length -= 1
            mps = self.get()
            yield mps.quality, mps.tce, mps.str_individuals
"""

class SearchTree:
    def __init__(self, maxsize=0):
        self.items_in_queue = PriorityQueue(maxsize)
        self.gate = dict()

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

    def __len__(self):
        return len(self.gate)

    def __contains__(self, x):
        return x.name in self.gate

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
