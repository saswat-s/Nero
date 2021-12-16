import os
from typing import List, Tuple

class TransportationMDP:
    """
    Street with blocks numbered 1 to N.
    Walking from s to s+1 takes 1 minute.
    Taking a magic tram from s to 2s takes 2 minutes
    How to travel from 1 to n in the least time ?

    *** Tram fails with probability of .5
    """

    def __init__(self, N):
        self.N = N

    @staticmethod
    def startState():
        return 1

    def isEnd(self, state):
        return state == self.N

    def actions(self, state):
        result = []
        # (1). s, a_walk, s+1
        if state + 1 <= self.N:
            result.append('walk')
        # (2) s, a_tram, s*2
        if state * 2 <= self.N:
            result.append('tram')

        return result

    @staticmethod
    def successorProbReward(state, action) -> List[Tuple]:
        """
        Returns a list of triples (s',prob,reward)
        prob = T(s,a,s')
        reward = Reward(s,a,s')
        :param state:
        :param action:
        :return:
        """
        result = []
        # (1) T(s, a_walk, s+1)= 1.0, R(s, a_walk, s+1)=-1
        if action == 'walk':
            result.append((state + 1, 1., -1.))
        elif action == 'tram':
            fail_probs=0.5
            # (2.1) T(s, a_tram, s*2)= 0.5, R(s, a_tram, s*2)=-2
            result.append((state * 2, 1-fail_probs, -2.))
            # (2.2) T(s, a_tram, s)= 0.5, R(s, a_tram, s)=-2 # tram fails
            result.append((state, fail_probs, -2.))
        return result

    @staticmethod
    def discount():
        return 1.

    def states(self):
        return range(1, self.N + 1)


def valueIteration(mdp):
    # (1) Initialize State values.
    V = {}
    for state in mdp.states():
        V[state] = 0.

    # (2) Define Q value of a state and an action.
    def Q(s, a):
        return sum(
            prob * (reward + mdp.discount() * V[newState]) for newState, prob, reward in mdp.successorProbReward(s, a))

    # (3) Iterate over fix number of iteration
    i_iter = 0
    while True:
        newV = {}
        # (4). For each iteration, iterate over all states
        for state in mdp.states():
            # (5.) For each state, assign 0 if a state is an end state
            if mdp.isEnd(state):
                newV[state] = 0.
            else:
                # (6.) Value of a non-terminal state is maximum state-action value, i.e.
                # V(s) ^t := max_{a \in Actions (s)} R(s,a,s') + \gamma V ^(t-1) (s')
                newV[state] = max(Q(state, action) for action in mdp.actions(state))

        i_iter += 1
        if max(abs(V[state] - newV[state]) for state in mdp.states()) < 1e-10:
            print(f'Converged in {i_iter} iteration')
            break
        V = newV

        # read out policy
        pi = {}
        for state in mdp.states():
            if mdp.isEnd(state):
                pi[state] = 'none'
            else:
                pi[state] = max((Q(state, action), action) for action in mdp.actions(state))[1]

        # print stuff out
        """
        os.system('clear')
        print(f'{"s":20} {"V(s)":20} {"pi(s)":20}')
        for state in mdp.states():
            print(f'{state:15} {V[state]:15} {pi[state]:15}')
        """

    # read out policy
    pi = {}
    for state in mdp.states():
        if mdp.isEnd(state):
            pi[state] = 'none'
        else:
            pi[state] = max((Q(state, action), action) for action in mdp.actions(state))[1]

    print(f'{"s"}\t{"V(s)"}\t{"pi(s)"}')
    for state in mdp.states():
        print(f'{state}\t{V[state]:.3f}\t{pi[state]:15}')


valueIteration(TransportationMDP(N=10))
# print(mdp.actions(3))
# print(mdp.succProbReward(3, 'walk'))
# print(mdp.succProbReward(3, 'tram'))
