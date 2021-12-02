from queue import PriorityQueue
from .dl_expression import TargetClassExpression


class State:
    def __init__(self, quality: float, tce: TargetClassExpression, str_individuals: set):
        self.quality = quality
        self.tce = tce
        self.str_individuals = str_individuals

    def __lt__(self, other):
        return self.quality < other.quality

    def __str__(self):
        return f'Quality:{self.quality} | Expression:{self.tce}'


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
