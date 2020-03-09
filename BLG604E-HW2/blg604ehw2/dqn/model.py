"""
Deep Q network implementations.

Vanilla DQN and DQN with Duelling architecture,
Prioritized ReplayBuffer and Double Q learning.
"""

import torch
import numpy as np
import random
from copy import deepcopy
from collections import namedtuple

from blg604ehw2.dqn.replaybuffer import UniformBuffer
from blg604ehw2.dqn.replaybuffer import PrioirtyBuffer
from blg604ehw2.dqn.replaybuffer import Transition
from blg604ehw2.atari_wrapper import LazyFrames
from blg604ehw2.utils import process_state
from blg604ehw2.utils import normalize


class BaseDqn:
    """
    Base class for DQN implementations.

    Both greedy and e_greedy policies are defined.
    Greedy policy is a wrapper for the _greedy_policy
    method.

    Arguments:
        - nact: Number of the possible actions
        int the action space
        - buffer_capacity: Maximum capacity of the
        replay buffer
    """

    def __init__(self, nact, buffer_capacity):
        super().__init__()
        self.nact = nact
        self.buffer_capacity = buffer_capacity
        self._device = "cpu"

    def greedy_policy(self, state):
        """ Wrapper for the _greedy_policy of the
        inherited class. Performs normalization if
        the state is a LazyFrame(stack of gray images)
        and cast the state to torch tensor with
        additional dimension to make it compatible
        with the neural network.
        """
        if isinstance(state, LazyFrames):
            state = np.array(state, dtype="float32")
            state = state.transpose(2, 0, 1)
            state = normalize(state)
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        if state.shape[0] != 1:
            state.unsqueeze_(0)
        with torch.no_grad():
            return self._greedy_policy(state)

    def e_greedy_policy(self, state, epsilon):
        """ Return action from greedy policy
        with the 1-epsilon probability and
        random action with the epsilon probability.
        """
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.nact-1)
        else:
            return self.greedy_policy(state)

    def push_transition(self, transition):
        """ Push transition to the replay buffer """
        raise NotImplementedError

    def update(self, batch_size):
        """ Update the model """
        raise NotImplementedError

    def _greedy_policy(self, state):
        """ Return greedy action for the state """
        raise NotImplementedError

    @property
    def buffer_size(self):
        """ Return buffer size """
        return self.buffer.size

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


class DQN(BaseDqn, torch.nn.Module):
    """ Vanilla DQN with target network and uniform
    replay buffer. Implemantation of DeepMind's Nature
    paper.

    Arguments:
        - valuenet: Neural network to represent value
        function.
        - nact: Number of the possible actions
        int the action space
        - lr: Learning rate of the optimization
        (default=0.001)
        - buffer_capacity: Maximum capacity of the
        replay buffer (default=10000)
        - target_update_period: Number of steps for
        the target network update. After each update
        counter set to zero again (default=100)

    """

    def __init__(self, valuenet, nact, lr=0.001, buffer_capacity=10000,
                 target_update_period=100):
        super().__init__(nact, buffer_capacity)
        self.valuenet = valuenet
        self.target_net = deepcopy(valuenet)
        self.target_update_period = target_update_period
        self.target_update_counter = 0
        self.buffer = UniformBuffer(capacity=buffer_capacity)

        self.opt = torch.optim.Adam(self.valuenet.parameters(), lr=lr)

    def _greedy_policy(self, state):
        """ Return greedy action for the state """

        return np.argmax(self.valuenet(state).cpu().numpy().squeeze())

    def push_transition(self, transition, *args):
        """ Push transition to the replay buffer
            Arguments:
                - transition: Named tuple of (state,
                action, reward, next_state, terminal)
        """
        self.buffer.push(transition)

    def update(self, batch_size, gamma):
        """ Update the valuenet and targetnet(if period)
        and return mean absulute td error.Process samples
        sampled from the replay buffer for q learning update.
        Raise assertion if the replay buffer is not big
        enough for the batchsize.
        """
        assert batch_size < self.buffer.size, "Buffer is not large enough!"
        # count how many times we visit so that we know when to update target net
        try:
            DQN.update.counter += 1
        except AttributeError:
            DQN.update.counter = 1
        
        # get a uniform random sample from replay buffer
        sample = self.buffer.sample(batch_size)
        
        # switch the network into train mode
        self.valuenet.train()
        
        # forward pass the current states to the network to get the value predictions
        value_predictions = self.valuenet(sample.state.to(self.device))
        # get the predictions for the actions that we have taken in our saved transitions
        value_predictions = torch.gather( value_predictions, 1, sample.action.to(self.device) )
        
        # now lets calculate the ground truths, first get the value predictions for the next states from the target net which we assume to be true at the moment
        value_next_states = self.target_net(sample.next_state.to(self.device)).detach()
        # get the maximums amongs them
        value_next_states = torch.max(value_next_states, dim=1)[0].unsqueeze(1)
        # calculate the ground truths for the values for the current states by adding the reward that we have observed from the environment to the values we have calculated for the next states
        # if not terminal
        value_ground_truths = value_next_states * gamma
        value_ground_truths[ sample.terminal ] = 0.0
        value_ground_truths += sample.reward.to(self.device)
        
        # get the loss
        td_error = torch.nn.functional.l1_loss( value_predictions,  value_ground_truths )
        # make the optimization step
        self.opt.zero_grad()
        td_error.backward()
        for param in self.valuenet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt.step()

        if DQN.update.counter % self.target_update_period == 0:
            self.target_net.load_state_dict( self.valuenet.state_dict() )
        
        return td_error.item()            # mean absulute td error


class DuelingDoublePrioritizedDQN(BaseDqn, torch.nn.Module):
    """ DQN implementaiton with Duelling architecture,
    Prioritized Replay Buffer and Double Q learning. Double
    Q learning idea is implemented with a target network that
    is replaced with the main network at every Nth step.

    Arguments:
        - valuenet: Neural network to represent value
        function.
        - nact: Number of the possible actions
        int the action space
        - lr: Learning rate of the optimization
        (default=0.001)
        - buffer_capacity: Maximum capacity of the
        replay buffer (default=10000)
        - target_replace_period: Number of steps to
        replace value network wtih the target network
        (default=50)

    """

    def __init__(self, valuenet, nact, lr=0.001, buffer_capacity=10000,
                 target_replace_period=50):
        super().__init__(nact, buffer_capacity)
        self.valuenet = valuenet.to(self.device)
        self.target_net = deepcopy(valuenet).to(self.device)
        self.target_replace_period = target_replace_period
        self.target_replace_period_counter = 0
        self.buffer = PrioirtyBuffer(capacity=buffer_capacity)

        self.opt = torch.optim.Adam(self.valuenet.parameters(), lr=lr)

    def _greedy_policy(self, state):
        """ Return greedy action for the state """
        return np.argmax(self.valuenet(state).cpu().numpy().squeeze())

    def td_error(self, trans, gamma):
        """ Return the td error, predicited values and
        target values.
        """
        self.valuenet.eval()
        self.target_net.eval()
        # forward pass the state to the network to get the value prediction
        value_prediction = self.valuenet(torch.from_numpy( trans[0] ).to(self.device)).detach()
        # get the predictions for the actions that we have taken in our saved transitions
        value_prediction = value_prediction[0,trans[1][0,0]]
        # now lets calculate the ground truths, first get the value predictions for the next states from the value net (double dqn)
        value_next_state = self.valuenet( torch.from_numpy( trans[3] ).to(self.device) ).detach()
        # get the action
        action = torch.argmax(value_next_state)
        # calculate the q value of the choosen action using targetnet
        value_next_state = self.target_net( torch.from_numpy( trans[3] ).to(self.device) ).detach()
        value_next_state = value_next_state[0,action]
        # calculate the ground truths for the values for the current states by adding the reward that we have observed from the environment to the values we have calculated for the next states
        # if not terminal
        value_ground_truth = value_next_state * gamma
        if trans[4] == True:
            value_ground_truth = torch.tensor(0.0).to(self.device)
        value_ground_truth += torch.from_numpy(np.asarray(trans[2][0,0])).float().to(self.device)
        # get the loss
        td_error = torch.nn.functional.l1_loss( value_prediction,  value_ground_truth )
        return td_error.item()

    def push_transition(self, transition, gamma):
        """ Push transitions and corresponding td error
        into the prioritized replay buffer.
        """

        # Remember Prioritized Replay Buffer requires
        # td error to push a transition. You need
        # to calculate it for the given trainsition

        with torch.no_grad():
            current_state, action, reward, next_state, done = transition
            if isinstance(current_state, LazyFrames):
                current_state = np.array(current_state, dtype="float32")
                current_state = current_state.transpose(2, 0, 1)
                current_state = normalize(current_state)
            if isinstance(next_state, LazyFrames):
                next_state = np.array(next_state, dtype="float32")
                next_state = next_state.transpose(2, 0, 1)
                next_state = normalize(next_state)
            transition_to_push = current_state, action, reward, next_state, done
            current_state = np.expand_dims(current_state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            action = np.asarray(action, dtype="int32")
            action = np.expand_dims(action, axis=0)
            action = np.expand_dims(action, axis=0)
            reward = np.expand_dims(reward, axis=0)
            reward = np.expand_dims(reward, axis=0)
            done = np.expand_dims(done, axis=0)
            done = np.expand_dims(done, axis=0)
            transition = current_state, action, reward, next_state, done
            td_error = self.td_error(transition, gamma)
        self.buffer.push(transition_to_push, td_error)

    def update(self, batch_size, gamma):
        """ Update the valuenet and replace it with the
        targetnet(if period). After the td error is
        calculated for all the batch, priority values
        of the transitions sampled from the buffer
        are updated as well. Return mean absulute td error. 
        """
        assert batch_size < self.buffer.size, "Buffer is not large enough!"

        # This time it is double q learning.
        # Remember the idea behind double q learning.

        # count how many times we visit so that we know when to update target net
        try:
            DuelingDoublePrioritizedDQN.update.counter += 1
        except AttributeError:
            DuelingDoublePrioritizedDQN.update.counter = 1

        # get a uniform random sample from replay buffer
        sample, sample_idxes = self.buffer.sample(batch_size)
        
        # switch the network into train mode
        self.valuenet.train()
        
        # forward pass the current states to the network to get the value predictions
        value_predictions = self.valuenet(sample.state.to(self.device))
        # get the predictions for the actions that we have taken in our saved transitions
        value_predictions = torch.gather( value_predictions, 1, sample.action.to(self.device) )
        
        # now lets calculate the ground truths, first get the value predictions for the next states from the value net (double dqn)
        value_next_states = self.valuenet(sample.next_state.to(self.device)).detach()
        # get the actions with biggest q values
        actions = torch.argmax(value_next_states, dim=1).unsqueeze(1)
        # calculate the q value of the choosen action using targetnet
        value_next_states = self.target_net( sample.next_state.to(self.device) ).detach()
        value_next_states = torch.gather(value_next_states, 1, actions)
        value_next_states = torch.max(value_next_states, dim=1)[0].unsqueeze(1)
        # calculate the ground truths for the values for the current states by adding the reward that we have observed from the environment to the values we have calculated for the next states
        # if not terminal
        value_ground_truths = value_next_states * gamma
        value_ground_truths[ sample.terminal ] = 0.0
        value_ground_truths += sample.reward.to(self.device)
        
        # get the loss
        td_error = torch.nn.functional.l1_loss( value_predictions,  value_ground_truths )
        # make the optimization step
        self.opt.zero_grad()
        td_error.backward()
        for param in self.valuenet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt.step()

        if DuelingDoublePrioritizedDQN.update.counter % self.target_replace_period == 0:
            # swap the parameters
            value_dict = self.valuenet.state_dict()
            target_dict = self.target_net.state_dict()
            self.target_net.load_state_dict( value_dict )
            self.valuenet.load_state_dict( target_dict )

        # update the td errors in the buffer
        preds = value_predictions.detach().cpu().numpy()
        gts = value_ground_truths.detach().cpu().numpy()
        new_td_errors = np.abs(preds-gts)
        [ self.buffer.update_priority(s_i, new_td_errors[counter]) for counter, s_i in enumerate(sample_idxes) ]
        return td_error.item()          # mean absulute td error
