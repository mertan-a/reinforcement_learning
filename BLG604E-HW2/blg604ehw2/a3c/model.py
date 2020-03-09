import torch
from torch.autograd import Variable
import numpy as np
from collections import namedtuple

from blg604ehw2.utils import process_state
from blg604ehw2.atari_wrapper import LazyFrames
from blg604ehw2.utils import normalize

Hidden = namedtuple("Hidden", "actor critic")
Transition = namedtuple("Transition", "state action reward dist value")

class BaseA3c(torch.nn.Module):
    """ Base class for Asynchronous Advantage Actor-Critic agent.
    This is a base class for both discrete and continuous
    a3c implementations.

    Arguments:
        - network: Neural network with both value and
        distribution heads
    """
    def __init__(self, network):
        super().__init__()
        self.network = network
        self._device = "cpu"
        self.hidden = Hidden(None, None)

    def greedy_policy(self):
        """ Return best action at the given state """
        raise NotImplementedError

    def soft_policy(self):
        """ Return a sample from the distribution of
        the given state
        """
        raise NotImplementedError

    def init_hidden(self):
        self.hidden = Hidden(torch.zeros(1, 128), torch.zeros(1, 128))

    def process_state(self, state, set_hidden=True):
        # preprocess the state
        if isinstance(state, LazyFrames):
            state = np.array(state, dtype="float32")
            state = state.transpose(2, 0, 1)
            state = normalize(state)
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        if state.shape[0] != 1:
            state.unsqueeze_(0)

        # pass the state from the network
        dist, value, (h_a, h_c) = self.network(state, self.hidden)
        if set_hidden:
            # set the hidden to its new values
            self.hidden = Hidden(h_a, h_c)

        return dist, value


    def loss(self, transitions, last_state, is_terminal, gamma, beta):
        """ Perform gradient calculations via backward
        operation for actor and critic loses.

        Arguments:
            - transitions: List of past n-step transitions
            that includes value, entropy, log probability and
            reward for each transition
            - last_state: Next state agter the given transitions
            - is_terminal: True if the last_state is a terminal state
            - gamma: Discount rate
            - beta: Entropy regularization constant
        """


        # Transtions can be either
        #   - reward, value, entropy, log probability
        #   of the states and actions
        #   - state, action
        #   from the bootstrap buffer
        #   First one is suggested!
        actor_loss = 0
        critic_loss = 0
        gae = torch.zeros(1, 1)
        # start calculating the gt
        if is_terminal:
            #R = 0
            R = torch.zeros(1, 1)
        else:
            _, R = self.process_state(last_state, False)
            R = R.detach()
        transitions.value.append(R)
        # traverse the transitions and accumulate the losses
        nstep = len(transitions.state)
        for i in range(nstep-1,-1,-1):
            R = torch.tensor(transitions.reward[i]).float() + gamma*R
            td = R - transitions.value[i]
            critic_loss += 0.5 * td.pow(2)

            # generalized advantage estimation, taken from https://github.com/dgriff777/a3c_continuous
            delta_t = transitions.reward[i] + gamma * transitions.value[i + 1].data - transitions.value[i].data
            gae = gae * gamma + delta_t
            
            actor_loss += ( -transitions.dist[i].log_prob( torch.from_numpy(transitions.action[i]) ) * Variable(gae) ).sum()
            entropy = transitions.dist[i].entropy().sum()
            actor_loss += -beta*entropy
        return actor_loss, critic_loss

    def synchronize(self, state_dict):
        """ Synchronize the agent with the given state_dict """
        self.load_state_dict(state_dict)

    def global_update(self, opt, global_agent):
        """ Update the global agent with the agent's gradients
        In order to use this method, backwards need to called beforehand
        """
        if next(self.parameters()).is_shared():
            raise RuntimeError(
                "Global network(shared) called global update!")
        for global_p in global_agent.parameters():
            global_p.grad = None
        for global_p, self_p in zip(global_agent.parameters(), self.parameters()):
            if global_p.grad is not None:
                continue
            else:
                global_p._grad = self_p.grad
        opt.step()

    def zero_grad(self):
        """ Clean the gradient buffers """
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @property
    def device(self):
        """ Return device name """
        return self._device

    @device.setter
    def device(self, value):
        """ Set device name and the model's
         device.
        """
        super().to(value)
        self._device = value


class ContinuousA3c(BaseA3c):
    """ Continuous action space A3C """
    def __init__(self, network):
        super().__init__(network)

    def greedy_policy(self, dist, clip_low=-1, clip_high=1):
        """ Return best action at the given state """
        action = dist.mean.detach().squeeze().numpy()
        action[action<clip_low] = clip_low
        action[action>clip_high] = clip_high        
        return action

    def soft_policy(self, dist, clip_low=-1, clip_high=1):
        """ Sample an action  """
        action = dist.sample().squeeze().numpy()
        action[action<clip_low] = clip_low
        action[action>clip_high] = clip_high
        return action


class DiscreteA3c(BaseA3c):
    """ Discrete action space A3C """
    def __init__(self, network):
        super().__init__(network)

    def greedy_policy(self, dist):
        """ Return best action at the given state """
        return np.argmax(dist.probs.numpy())

    def soft_policy(self, action):
        """ Sample an action  """
        return dist.sample().detach().numpy()
