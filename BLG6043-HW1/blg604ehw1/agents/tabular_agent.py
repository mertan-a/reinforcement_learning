from collections import defaultdict
from collections import namedtuple
import random
import math

class TabularAgent:
    r""" Base class for the tabular agents. This class
    provides policies for the tabular agents.
    """

    def __init__(self, nact):
        self.values = defaultdict(lambda: [0.0]*nact)
        self.nact = nact
    
    def greedy_policy(self, state):
        # Returns the best action.
        action = max(range(self.nact), key = lambda a: self.values[state][a])
        return action

    def e_greedy_policy(self, state, epsilon):
        # Returns the best action with the probability (1 - e) and 
        # action with probability e
        if random.uniform(0,1) < epsilon:
            action = random.randrange(0,self.nact)
        else:
            action = max(range(self.nact), key = lambda a: self.values[state][a])
        return action
    
    def soft_policy(self, state):
        # Probabilistic policy where the probability of returning an
        # action is proportional to the value of the action.
        temp = [ x+0.0001 for x in self.values[state] ]
        rand = random.uniform(0, sum(temp) )
        curr = 0
        for index, value in enumerate(temp):
            curr += value
            if rand<=curr:
                action = index
                break
        return action

    def update(self):
        raise NotImplementedError

class QAgent(TabularAgent):
    r""" Q learning agent where the update is done
    accordig to the off-policy Q update.        
    """

    def __init__(self, nact):
        super().__init__(nact)
    
    def update(self, trans, alpha, gamma):
        """ update (QTransition: trans, float: alpha, float: gamma) -> float: td_error
        QTransition: (state, action, reward, next_state, terminal)
        """
        # decompose the transition
        state, action, reward, next_state, terminal = trans
        # calculate td error
        td_error = reward
        if terminal == 0:
            td_error += gamma * max( range(self.nact), key = lambda a: self.values[next_state][a] )
        else:
            pass
        td_error += - self.values[state][action]
        # update the entry for the current state action pair
        self.values[state][action] += alpha * td_error
        return td_error
    
class SarsaAgent(TabularAgent):
    r""" Sarsa agent where the update is done
    accordig to the on-policy Sarsa update. 
    """

    def __init__(self, nact):
        super().__init__(nact)

    def update(self, trans, alpha, gamma):
        """ update (SarsaTransition: trans, float: alpha, float: gamma) -> float: td_error
        SarsaTransition: (state, action, reward, next_state, next_action, terminal)
        """
        # decompose the transition
        state, action, reward, next_state, next_action, terminal = trans
        # calculate td error
        td_error = reward
        if terminal == 0:
            td_error += gamma * self.values[next_state][next_action]
        else:
            pass
        td_error += - self.values[state][action]
        # update the entry for the current state action pair
        self.values[state][action] += alpha * td_error
        return td_error    