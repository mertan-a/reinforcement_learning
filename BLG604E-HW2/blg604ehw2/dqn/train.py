""" Training and testion functions """

import gym
import numpy as np
from collections import namedtuple

from blg604ehw2.dqn.replaybuffer import Transition


ArgsDQN = namedtuple("ArgsDQN", """
                                env_name
                                nact
                                buffersize
                                max_epsilon
                                min_epsilon
                                target_update_period
                                gamma
                                lr
                                device
                                batch_size
                                episode
                                max_eps_len
                                """
                     )

ArgsDDPQN = namedtuple("ArgsDDPQN", """
                                    env_name
                                    nact
                                    buffersize
                                    max_epsilon
                                    min_epsilon
                                    target_replace_period
                                    gamma
                                    lr
                                    device
                                    batch_size
                                    episode
                                    max_eps_len
                                    """
                       )


def episodic_train(env, agent, args, epsilon):
    """ Train the agent in the env one episode.
        Return time steps passed and mean td_error
    """
    # init the values
    td_error = 0
    update_counter = 0
    # reset the environment at the start of the training
    current_state = env.reset()
    for time_step in range(args.max_eps_len):
        # have the agent in eval mode
        agent.eval()
        # get the action using e greedy policy
        action = agent.e_greedy_policy(current_state, epsilon)
        # apply the action
        next_state, reward, done, info = env.step(action)
        # save the transition
        agent.push_transition( (current_state, action, reward, next_state, done), args.gamma )
        # check the buffer, update the networks if it is filled enough
        if agent.buffer.size > 1000:
            td_error += agent.update(args.batch_size, args.gamma)
            update_counter = update_counter + 1
        # make the next state our current state for the next step
        current_state = next_state

        if done:
            break  
    # average the td error if we updated it
    if update_counter > 0:
        td_error = td_error / update_counter

    return time_step, td_error


def episodic_test(env, agent, args, render=False, monitor_path=None):
    """ Evaluate the agent and return episodic reward.

        Parameters:
            - env: Environment to evaluate
            - agent: Agent model
            - args: Hyperparamters of the model
            - render: Render the environment if True
            (default=False)
            - monitor_path: Render and save the mp4 file
            to the give path if any given (default=None)
    """

    agent.eval()
    if monitor_path:
        path = "monitor/" + monitor_path
        env = gym.wrappers.Monitor(
            env, path, video_callable=lambda eps_id: True, force=True)
        render = False

    eps_reward = 0
    state = env.reset()
    for time_step in range(args.max_eps_len):
        action = agent.greedy_policy(state)
        if render:
            env.render()
        state, reward, done, info = env.step(action)
        eps_reward += reward
        if done:
            break
    if monitor_path:
        env.close()

    return eps_reward
    