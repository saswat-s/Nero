class HalvingGame(object):
    """ The Halving Game
    Start with a number N
    Player take turns either ecrementing N or replacing it with N/2
    the player that is left with 0 wins
    """

    def __init__(self, N):
        self.N = N

    def startState(self):
        return +1, self.N

    def isEnd(self, state):
        player, number = state
        return number == 0

    def utility(self, state):
        player, number = state
        assert number == 0
        return player * float('inf')

    def actions(self, state):
        return ['-', '/']

    def player(self, state):
        player, number = state
        return player

    def succ(self, state, action):
        player, number = state
        if action == '-':
            return -player, number - 1
        elif action == '/':
            return -player, number // 2


def humanPolicy(game, state):
    while True:
        action = input('Input action:')
        if action in game.actions(state):
            return action


def minimaxPolicy(game, state):
    def recurse(state):
        if game.isEnd(state):
            return game.utility(state), 'none'
        choices = [(recurse(game.succ(state, action))[0], action) for action in game.actions(state)]
        if game.player(state) == +1:
            return max(choices)
        elif game.player(state) == -1:
            return min(choices)
    val, action = recurse(state)
    print(f'minimax says action = {action}, value = {val}')
    return action


policies = {+1: humanPolicy, -1: minimaxPolicy}
halving_game = HalvingGame(N=15)
state = halving_game.startState()

while not halving_game.isEnd(state):
    print('=' * 10, state)
    player = halving_game.player(state)
    policy = policies[player]
    action = policy(halving_game, state)
    state = halving_game.succ(state, action)
print(f'Utility={halving_game.utility(state)}')
