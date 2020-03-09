""" Sum Tree implementation for the prioritized
replay buffer.
"""

import numpy as np


class SumTree():
    """ Binary heap with the property: parent node is the sum of
    two child nodes. Tree has a maximum size and whenever
    it reaches that, the oldest element will be overwritten
    (queue behaviour). All of the methods run in O(log(n)).

    Arguments
        - maxsize: Capacity of the SumTree

    """
    def __init__(self, maxsize):
        # i have treated the max size as the capacity of the RB
        self.nr_leaf = 1<<(maxsize-1).bit_length()
        self.tree_size = self.nr_leaf*2 - 1
        self.tree = np.zeros((self.tree_size),dtype="float64")#[0.0 for x in range(self.tree_size)]
        # points to which leaf to write
        self.to_write = 0

    def push(self, priority):
        """ Add an element to the tree and with the given priority.
         If the tree is full, overwrite the oldest element.

        Arguments
            - priority: Corresponding priority value
        """
        # pushes are actually the updates since we have  
        # already init the tree with zeros
        self.update(self.to_write%self.nr_leaf, priority)
        self.to_write += 1
        
    def get(self, priority):
        """ Return the node with the given priority value.
        Prioirty can be at max equal to the value of the root
        in the tree.

        Arguments
            - priority: Value whose corresponding index
                will be returned.
        """
        # get the child indexes
        def get_child_indexes(parent_index):
            return parent_index*2 + 1, parent_index*2 + 2
        # traverse the tree until we hit a leaf, start from root
        start_pri = priority
        # priority -= 1
        p = 0
        c_left, c_right = get_child_indexes(p)
        while True:
            # moving on to the left child
            if priority<self.tree[c_left]:
                p = c_left
            # move to the right child
            else:
                # decrease the priority
                priority = priority - self.tree[c_left]
                p = c_right
            # get the new childs to check
            c_left, c_right = get_child_indexes(p)
            # stop when childs are out of bounds
            if c_left >= self.tree_size:
                break
        # return the data index of the found leaf
        node = p - self.tree_size//2
        # for debugging purposes
        if self.to_write <= node:
            sum_=0
            for i in range( 0, self.to_write ):
                sum_ += self.tree[i+self.tree_size//2]
            import sys
            sys.exit()
        return node

    def update(self, idx, value):
        """ Update the tree for the given idx with the
        given value. Values are updated via increasing
        the priorities of all the parents of the given
        idx by the difference between the value and
        current priority of that idx.

        Arguments
            - idx: Index of the data(not the tree).
            Corresponding index of the tree can be
            calculated via; idx + tree_size/2 - 1
            - value: Value for the node pointed by
            the idx
        """
        # calculate the priority value difference
        old_priority = self.tree[idx + self.tree_size//2]
        diff = value - old_priority
        # save the starting index (tree)
        idx_tree = idx + self.tree_size//2
        # until we reach the root which have idx zero in the tree
        while True:
            # update the index
            self.tree[idx_tree] += diff
            # calculate its parent index
            idx_tree = (idx_tree-1) // 2
            # stop after the root
            if idx_tree < 0:
                break

    ### added
    def get_sum_priority(self):
        """ returns the sum of all priorities
        """
        return self.tree[0]
