from gym.envs.toy_text import discrete
from itertools import product
import numpy as np

from .render import Renderer

"""
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
"""
# You can change the map to experiment with agents.
MAP = ["WWWWWWWWWWWWWWWWWWW",
       "WEEEEEEEEEWEEEEEGEW",
       "WEEEEEEEEEWEEEEEEEW",
       "WEEEEEEEEEEEEEEEEEW",
       "WEEEEEEEEEWEEEEEEEW",
       "WEEESSEEEEWEEEEEEEW",
       "WWWWWWWWWWWWWWWWWWW",
       ]

REWARD_MAP = {
    b"W" : 0,       # Wall
    b"S" : 0,       # Start
    b"E" : 0,       # Empty
    b"F" : -0.1,    # Fire
    b"H" : -1.0,    # Hole
    b"G" : 1.0,     # Goal
}

TERMINATION_MAP = {
    b"W" : 0,
    b"S" : 0,
    b"E" : 0,
    b"F" : 0,
    b"H" : 1,
    b"G" : 1,
}
class GridWorldEnv(discrete.DiscreteEnv):
    r"""
        Grid environment with different tile types. Both the initial state
        distribution and transition map is defined so that the environment
        can be modeled as a MDP. This environment has the same basic methods
        as any other gym environment.
        
        Args:
            gridmap (list): List of strings defining the map of the grid.
            randomness (float): Probability of acting differently than expected.


    """

    metadata = {'render.modes': ["human", "ansi", "visual", "notebook"]}

    def __init__(self, gridmap=MAP, randomness=0.05):
        self.grid = np.asarray(gridmap, dtype='c')
        self.heigth, self.width = self.grid.shape

        nactions = 4
        nstates = self.heigth*self.width

        # Initial starting states and the corresponding distribution.
        initial_states = np.argwhere(self.grid == b"S")
        initial_state_dist = [(1.0/len(initial_states), state) for state in initial_states]

        # All the states that an agent may visit. (Non wall states)
        all_states = np.argwhere(self.grid != b"W")


        # This is the main table for the MDP. You can construct any MDP by modifying this.
        #   transition_map(P) := P(state, action) -> [(probability, next state, reward, termination), ...]
        # From any state action pair an agent can travel to its neighbouring tiles with the corresponding 
        # probability, reward value and termination.
        # Transition map is a dictionary of a dictionaries of lists
        # You supposed to fill the table in order to construct the MDP.
        transition_map = {tuple(state): {act: [] for act in range(nactions)} for state in all_states}

        def get_reward_value(state):
            return REWARD_MAP[ bytes(MAP[state[0]][state[1]], 'utf-8') ]
        def get_termination_value(state):
            return TERMINATION_MAP[ bytes(MAP[state[0]][state[1]], 'utf-8') ]

        for row, col in all_states:
            # list of all neighbours, including itself
            neighbours = []
            neighbours.append( (row,col-1) )
            neighbours.append( (row+1,col) )
            neighbours.append( (row,col+1) )
            neighbours.append( (row-1,col) )
            neighbours.append( (row,col) ) # the tile itself can be considered as neihgbour also since it is possible for our agent to not move
            # movable neighbours will have one
            movable_neighbours = np.zeros((5))
            if MAP[row][col-1] != "W": movable_neighbours[0] = 1
            if MAP[row+1][col] != "W": movable_neighbours[1] = 1
            if MAP[row][col+1] != "W": movable_neighbours[2] = 1
            if MAP[row-1][col] != "W": movable_neighbours[3] = 1
            if MAP[row][col] != "W": movable_neighbours[4] = 1# the tile itself can be considered as neihgbour also since it is possible for our agent to not move
            
            for action in range(4):
                # distribute probability among states
                total_prob = 1
                # if target cell is movable
                if movable_neighbours[action] == 1:
                    # assign the probability in transition map
                    transition_map[ (row,col) ][ action ].append([(total_prob - randomness), neighbours[action], get_reward_value(neighbours[action]), get_termination_value(neighbours[action])])
                    # decrease the total probability to distrubute
                    total_prob = randomness
                # calculate the number of neighbours that are movable other than the target cell
                nr_movable = np.sum(movable_neighbours) - movable_neighbours[action]
                # assign probabilities to all other neighbours other than target cell
                for i, neighbour in enumerate(neighbours):
                    if i == action:
                        continue
                    if movable_neighbours[i] == 1:
                        transition_map[row,col][action].append([ (total_prob/nr_movable), neighbour, get_reward_value(neighbour), get_termination_value(neighbour) ])
                            


        super().__init__(nstates, nactions, transition_map, initial_state_dist)
        self.renderer = Renderer(self.grid)

    def reset(self):
        indx = discrete.categorical_sample([prob for prob, state in self.isd], self.np_random)
        self.lastaction=None
        _, state = self.isd[indx]
        self.s = tuple(state)
        return self.s

    def render(self, mode, **kwargs):
        if mode=="visual":
            self.renderer.visaul_render(self.s, **kwargs)
        elif mode == "notebook":
            self.renderer.buffer_render(self.s, **kwargs)
        elif mode in ("ansi", "stdout"):
            self.renderer.string_render(self.s, **kwargs)
