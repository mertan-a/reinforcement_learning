import numpy as np
from collections import namedtuple

QTransition = namedtuple("QTransition", "state action reward next_state terminal")
SarsaTransition = namedtuple("QTransition", "state action reward next_state next_action terminal")

class AbstractApproximateAgent():
    r""" Base class for the approximate methods. This class
    provides policies.
    """
    
    def __init__(self, nobs, nact):
        self.nact = nact
        self.nobs = nobs
        self.weights = np.random.uniform(-0.1, 0.1, size=(nobs, nact))
                    
    def q_values(self, state):
        return np.dot(state, self.weights)
                                          
    def greedy_policy(self, state):
        # Returns the best possible action according to the values.
        action = np.argmax(self.q_values(state))
        return action

    def e_greedy_policy(self, state, epsilon=0.4):
        # Returns the best action with the probability (1 - e) and 
        # action with probability e
        import random
        if random.uniform(0,1) < epsilon:
            action = random.randrange(0,self.nact)
        else:
            action = np.argmax(self.q_values(state))
        return action

    def soft_policy(self, state):
        # Probabilistic policy where the probability of returning an
        # action is proportional to the value of the action.
        temp = [ x+0.0001 for x in self.q_values(state) ]
        rand = random.uniform(0, sum(temp) )
        curr = 0
        for index, value in enumerate(temp):
            curr += value
            if rand<=curr:
                action = index
                break
        return action
                                        
    def update(self, *arg, **kwargs):
        raise NotImplementedError

class ApproximateQAgent(AbstractApproximateAgent):
    r""" Approximate Q learning agent where the learning is done
    via minimizing the mean squared value error with **semi**-gradient decent.
    This is an off-policy algorithm.
    """

    def __init__(self, nobs, nact):
        super().__init__(nobs, nact)

    def update(self, tran, alpha, gamma):
        """ Updates the parameters that parameterized the value function.
            update(QTransition: tran, float: alpha, float: gamma) -> mean_square_td_error
            QTransition: (state, action, reward, next_state, terminal)
        """
        # decompose the transition
        state, action, reward, next_state, terminal = tran
        # calculate target
        target_q = reward
        if terminal == 0:
            target_q += gamma * np.amax(self.q_values(next_state))
        else:
            pass
        # calculate predicted q
        predicted_q = self.q_values(state)[action]
        # calculate td error
        mean_squared_td_error = 0.5 * (target_q - predicted_q)**2
        # update the weights
        self.weights[:,action] = self.weights[:,action] + alpha * (target_q - predicted_q) * self.weights[:,action] #????????
        return mean_squared_td_error

class ApproximateSarsaAgent(AbstractApproximateAgent):

    def __init__(self, nobs, nact):
        super().__init__(nobs, nact)

    def update(self, tran, alpha, gamma):
        """ Updates the parameters that parameterized the value function.
            update (SarsaTransition: trans, float: alpha, float: gamma) -> float: mean_square_td_error
            SarsaTransition: (state, action, reward, next_state, next_action, terminal)
        """
        # decompose the transition
        state, action, reward, next_state, next_action, terminal = tran
        # calculate target
        target_q = reward
        if terminal == 0:
            target_q += gamma * self.q_values(next_state)[next_action]
        else:
            pass
        # calculate predicted q
        predicted_q = self.q_values(state)[action]
        # calculate td error
        mean_squared_td_error = 0.5 * (target_q - predicted_q)**2
        # update the weights
        self.weights[:,action] = self.weights[:,action] + alpha * (target_q - predicted_q) * self.weights[:,action] #????????
        return mean_squared_td_error

class LSTDQ(AbstractApproximateAgent):
    r""" Least Square Temporal Difference Q learning algorithm.
    Unlike the tabular counterpart of the LSTD, this method uses
    samples transitions from the environment and updates the parameters
    that parameterized the value function at one step. Note that
    in this implementation RBFs(Radial Basis Functions) are used
    as features and value function is defined as the linear combination
    of these functions.
    """

    def __init__(self, nobs, nact, features=60):
        super().__init__(nobs, nact)
        self.weights = np.random.uniform(-0.1, 0.1, size=(features))
        self.features = features
        # You can modify RBFs centers
        self.rbf_centers = np.random.normal(loc=0, scale=0.5, size=(features, nobs*nact))
        # self.rbf_centers[:, -2:] = 0
        # self.rbf_centers[:features//2, -2] = 1.0
        # self.rbf_centers[features//2:, -1] = 1.0
        
        
    def get_value(self, state, action):
        return np.dot(self.weights, self.phi(state, action))

    def phi(self, state, action):
        # Features of the (state, action) pair. This method returns feature vector 
        # using RBFs for the given pair.
        phi_sa = np.zeros((self.nobs*self.nact))
        phi_sa[ action*self.nobs : action*self.nobs + self.nobs ] = state
        features = np.dot( self.rbf_centers, phi_sa.T )
        return features

    def greedy_policy(self, state):
        # Override the base greedy policy to make it compatibale with the RBFs.
        action = max(range(self.nact), key = lambda a: self.get_value(state,a))
        return action
    
    def e_greedy_policy(self, state, epsilon=0.4):
        # Returns the best action with the probability (1 - e) and 
        # action with probability e
        import random
        if random.uniform(0,1) < epsilon:
            action = random.randrange(0,self.nact)
        else:
            action = max(range(self.nact), key = lambda a: self.get_value(state,a))
        return action

    def soft_policy(self, state):
        # Override the base soft policy to make it compatibale with the RBFs.
        values = [ self.get_value(state, a) for a in range(self.nact) ]
        rand = random.uniform(0, sum(values) )
        curr = 0
        for index, value in enumerate(values):
            curr += value
            if rand<=curr:
                action = index
                break
        return action

    def optimize(self, trans, gamma=0.99, epsilon=0.01):
        """ Optimize the parameters using the transitions sampled from the
        recent policy. Transitions  argument consists of QTransitions. 
        optimize(List: trans, float: gamma) -> None
        """
        # get the length of the transitions
        trans_len = len(trans)
        # init the matrices
        R = np.zeros((trans_len,1))
        P_phi = np.zeros((trans_len,self.features))
        Phi = np.zeros((trans_len,self.features))
        # fill the matrices
        for i, transition in enumerate(trans):
            # decompose the transition
            state, action, reward, next_state, terminal = transition
            R[i,0] = reward
            P_phi[i,:] = self.phi(next_state, max(range(self.nact), key = lambda a: self.get_value(next_state,a)) ) #??
            Phi[i,:] = self.phi(state,action)
        A = (1/trans_len) * np.dot(Phi.T, ( Phi - gamma * P_phi ))
        b = (1/trans_len) * np.dot(Phi.T, R)
        from numpy.linalg import inv
        self.weights = np.dot(inv(A), b)
        self.weights = np.squeeze(self.weights)
