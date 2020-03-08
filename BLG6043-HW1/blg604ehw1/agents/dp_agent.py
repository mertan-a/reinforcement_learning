
class DPAgent():
    r""" Base Dynamic Programming class. DP methods
    requires the transition map in order to optimize
    policies. This class provides the policy and
    one step policy evaluation as well as policy
    improvement.
    """

    def __init__(self, nact, nobs, transitions_map, init_value=0.0):
        self.nact = nact
        self.nobs = nobs
        self.transitions_map = transitions_map
        self.values = {s: init_value for s in self.transitions_map.keys()}
        self.policy_dist = {s: [1.0/nact]*nact for s in self.transitions_map.keys()}

    def policy(self, state):
        # Policy pi that returns the action with the highest Q value.
        return max(range(self.nact), key = lambda a: self.policy_dist[state][a])

    def one_step_policy_eval(self, gamma=1.0):
        # One step policy evaluation.
        new_values = {} #synchronus
        delta = 0
        for state in self.values.keys():
            # the defined policy is deterministic
            action = self.policy(state)
            # calculate the new value
            new_value = 0
            for t in self.transitions_map[state][action]:
                # check if the next state is terminal
                if t[3] == 0:
                    new_value += t[0] * t[2] + gamma * t[0] * self.values[ t[1] ]
                else:
                    # if it is terminal, then only use the reward
                    new_value += t[0] * t[2]
            # save the biggest change in the value
            if delta < abs(new_value - self.values[state]):
                delta = abs(new_value - self.values[state])
            # store the new value of the state
            # self.values[state] = new_value # asynchronous
            new_values[state] = new_value #synchronus
        self.values = new_values.copy() #synchronus
        return delta
            
    def policy_improvement(self, gamma=1.0):
        # Policy impovement updates the policy according to the
        # most recent values
        # set is policy stable to true first, it will be set to false if something changes
        is_policy_stable = True
        for state in self.values.keys():
            # save the previous choice of action in the current state
            previous_choice = self.policy(state)
            # calculate the new choice of action greedly using the state values
            q_values = [0]*self.nact
            for action in range(self.nact):
                for t in self.transitions_map[state][action]:
                    q_values[action] += t[0] * t[2] + gamma * t[0] * self.values[ t[1] ]
            new_choice = max(range(self.nact), key = lambda a: q_values[a])
            # update the policy according to new choice
            self.policy_dist[state] = [0] * self.nact
            self.policy_dist[state][new_choice] = 1
            # check to see if policy stable 
            if previous_choice != new_choice:
                is_policy_stable = False
        return is_policy_stable

class PolicyIteration(DPAgent):
    r""" Policy Iteration algorithm that first evaluates the
    values until they converge within epsilon range, then
    updates the policy and repeats the process until the
    policy no longer changes.
    """

    def __init__(self, nact, nobs, transitions_map):
        super().__init__(nact, nobs, transitions_map)

    def optimize(self, gamma, epsilon, max_iterations=100):
        # optimizer (float: gamma, float: epsilon) -> None
        for i in range(max_iterations):
            # evaluate the current policy until the values converge within epsilon range
            while(True):
                change_in_value = self.one_step_policy_eval()
                if change_in_value <= epsilon:
                    break

            # improve the policy according to new values
            is_policy_stable = self.policy_improvement()
            if is_policy_stable == True:
                break

class ValueIteration(DPAgent):
    r""" Value Iteration algorithm iteratively evaluates
    the values and updates the policy until the values
    converges within epsilon range.
    """

    def __init__(self, nact, nobs, transitions_map):
        super().__init__(nact, nobs, transitions_map)
    
    def optimize(self, gamma, epsilon):
        # optimize(float: gamma, float: epsilon) -> None
        while(True):
            # evaluate the current policy
            change_in_value = self.one_step_policy_eval()
            if change_in_value <= epsilon:
                break
            # improve the policy according to new values
            self.policy_improvement()
        self.policy_improvement()

    