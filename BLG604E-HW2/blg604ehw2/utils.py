""" Utilities for homework 2.
    Function "log_progress" is adapted from:
    https://github.com/kuk/log-progress
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from ipywidgets import IntProgress, HTML, VBox
from IPython.display import display

from blg604ehw2.atari_wrapper import LazyFrames


def comparison(*log_name_pairs, texts=[[""]*3], smooth_factor=3):
    """ Plots the given logs. There will be as many plots as
    the length of the texts argument. Logs will be plotted on
    top of each other so that they can be compared. For each
    log, mean value is plotted and the area between the
    +std and -std of the mean will be shaded.
    """
    plt.ioff()
    plt.close()

    def plot_texts(title, xlabel, ylabel):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    for i, (title, xlabel, ylabel) in enumerate(texts):
        for logs, name in log_name_pairs:
            smoothed_logs = np.stack(
                [smoother(log[i], smooth_factor) for log in logs])
            std_logs = np.std(smoothed_logs, axis=0)
            mean_logs = np.mean(smoothed_logs, axis=0)
            max_logs = np.max(smoothed_logs, axis=0)
            min_logs = np.min(smoothed_logs, axis=0)
            plot_texts(title, xlabel, ylabel)
            plt.plot(mean_logs, label=name)
            plt.legend()
            plt.fill_between(np.arange(len(mean_logs)),
                             np.minimum(mean_logs+std_logs, max_logs),
                             np.maximum(mean_logs-std_logs, min_logs),
                             alpha=0.4)

        plt.show()


def smoother(array, ws):
    """ Return smoothed array by the mean filter """
    return np.array([sum(array[i:i+ws])/ws for i in range(len(array) - ws)])


# Optional
def normalize(frame):
    """ Return normalized frame """
    frame -= 128.0
    frame /= 128.0
    return frame


# Optional
def process_state(state):
    """ If the state is 4 dimensional image state
    return transposed and normalized state otherwise
    directly return the state. """
    if len(state.shape) == 4:
        state = torch.transpose(state, 2, 3)
        state = torch.transpose(state, 1, 2)
        return normalize(state)
    return state


class LoadingBar:
    """ Loading bar for ipython notebook """
    def __init__(self, size, name):
        self.size = size
        self.name = name
        self._progress = IntProgress(min=0, max=size, value=0)
        self._label = HTML()
        box = VBox(children=[self._label, self._progress])
        display(box)

    def success(self, reward):
        """ Turn loading bar into "complete state" """
        self._progress.bar_style = "success"
        self._progress.value = self.size
        self._label.value = (
            "{name}: {size}/{index}, Best reward: {reward}".format(
                name=self.name,
                size=self.size,
                index=self.size,
                reward=reward
            )
        )

    def progress(self, index, reward):
        """ Update progress with given index and best reward """
        self._progress.value = index
        self._label.value = (
            "{name}: {size}/{index}, Best reward: {reward}".format(
                name=self.name,
                size=self.size,
                index=index,
                reward=reward
            )
        )
