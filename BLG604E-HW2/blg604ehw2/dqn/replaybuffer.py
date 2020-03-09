"""Replay buffer implemantaions for DQN"""

from collections import namedtuple
from random import sample as randsample
import numpy as np
import torch

from blg604ehw2.dqn.sumtree import SumTree


Transition = namedtuple("Transition", ("state",
                                       "action",
                                       "reward",
                                       "next_state",
                                       "terminal")
                        )


class BaseBuffer():
    """ Base class for the buffers. Push and sample
    methods need to be override. Initially start with
    an empty list(queue).

    Arguments:
        - capacity: Maximum size of the buffer
    """

    def __init__(self, capacity):
        self.queue = []
        self.capacity = capacity

    @property
    def size(self):
        """Return the current size of the buffer"""
        return len(self.queue)

    def __len__(self):
        """Return the capacity of the buffer"""
        return self.capacity

    def push(self, transition, *args, **kwargs):
        """Push transition into the buffer"""
        raise NotImplementedError

    def sample(self, batchsize, *args, **kwargs):
        """Sample transition from the buffer"""
        raise NotImplementedError


class UniformBuffer(BaseBuffer):
    """ Vanilla buffer which was used in the
    nature paper. Uniformly samples transition.

    Arguments:
        - capacity: Maximum size of the buffer
    """

    def __init__(self, capacity):
        super().__init__(capacity)

    def push(self, transition):
        """Push transition into the buffer"""
        # if we reached the capacity, delete oldest item
        if (self.size == self.capacity):
            del self.queue[0]
        self.queue.append(transition)

    def sample(self, batchsize):
        """ Return sample of transitions uniformly
        from the buffer if buffer is large enough
        for the given batch size. Sample is a named
        tuple of transition where the elements are
        torch tensor.
        """
        # randomly sample transitions
        random_sample = randsample(self.queue, batchsize)
        # zip
        zipped = [ torch.from_numpy( np.asarray(arr).astype(np.float32) ).float() for arr in zip(*random_sample) ]
        sample = Transition( zipped[0], zipped[1].unsqueeze_(-1).long(), zipped[2].unsqueeze_(-1), zipped[3], zipped[4].unsqueeze_(-1).byte() )
        return sample


class PrioirtyBuffer(BaseBuffer):
    """ Replay buffer that sample tranisitons
    according to their prioirties. Prioirty
    value is most commonly td error.

    Arguments:
        - capacity: Maximum size of the buffer
        - min_prioirty: Values lower than the
        minimum prioirty value will be clipped
        - max_priority: Values larger than the
        maximum prioirty value will be clipped
    """

    def __init__(self, capacity, min_prioirty=1000, max_priority=20000):
        super().__init__(capacity)
        self.sum_tree = SumTree(self.capacity)
        # points to the location to write in the queue
        self.to_write = 0

        self.min_prioirty = min_prioirty
        self.max_priority = max_priority

    def _clip_p(self, p):
        """ Return clipped priority """
        return min(max(p, self.min_prioirty), self.max_priority)

    def push(self, transition, priority):
        """ Push the transition with priority """
        priority = priority * 10000
        priority = self._clip_p(priority)
        priority = int(priority)
        # if we reached the capacity, overwrite the oldest item
        if (self.size == self.capacity):
            self.queue[self.to_write%self.capacity] = transition
            self.sum_tree.update(self.to_write%self.capacity,priority)
        else:
            self.queue.append(transition)
            self.sum_tree.push(priority)
        self.to_write = self.to_write + 1

    def sample(self, batch_size):
        """ Return namedtuple of transition that
        is sampled with probability proportional to
        the priority values. """
        # get the sum of priorities
        priority_sum = self.sum_tree.get_sum_priority()
        # sample priorities 
        priorities_to_sample = np.random.uniform(0, priority_sum, batch_size)
        # get the indexes of replays
        sample_idxes = [self.sum_tree.get(x) for x in priorities_to_sample]
        # fetch the transitions and prepare the batch for training
        random_sample = [self.queue[x] for x in sample_idxes]
        # zip
        zipped = [ torch.from_numpy( np.asarray(arr).astype(np.float32) ).float() for arr in zip(*random_sample) ]
        sample = Transition( zipped[0], zipped[1].unsqueeze_(-1).long(), zipped[2].unsqueeze_(-1), zipped[3], zipped[4].unsqueeze_(-1).byte() )
        return sample, sample_idxes

    def update_priority(self, indexes, values):
        """ Update the prioirty value of the transition in
        the given index
        """
        values = values * 10000
        values = self._clip_p(values)
        values = int(values)
        self.sum_tree.update(indexes, values)
