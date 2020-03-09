""" Worker functions for training and testing """

import torch
import gym
import numpy as np
from collections import namedtuple
import torch.multiprocessing as mp

from blg604ehw2.utils import LoadingBar

# Hyperparamteres of A3C
A3C_args = namedtuple("A3C_args",
                      """
                        maxtimestep
                        maxlen
                        nstep
                        gamma 
                        lr 
                        beta 
                        device
                      """)
Hidden = namedtuple("Hidden", "actor critic")

Transition = namedtuple("Transition", "state action reward dist value")
def train_worker(args, globalmodel, optim, envfunc, agentfunc, lock, logger, nr):
    """ Training worker function.
        Train until the maximum time step is reached.
        Arguments:
            - args: Hyperparameters
            - globalmodel: Global(shared) agent for
            synchronization.
            - optim: Shared optimizer
            - envfunc: Environment generating function
            - agentfunc: Agent generating function
            - lock: Lock for shared memory
            - logger: Namedtuple of shared objects for
            logging purposes
    """
    env = envfunc()
    env.seed(nr)
    agent = agentfunc()
    agent.train()

    # Remember Logger has the shared time step value
    
    # Worker should be in a loop that terminates when
    # the shared time step value is higher than the
    # maximum time step.

    import time
    time.sleep(int(np.random.rand()*10))

    # sync with global
    agent.synchronize(globalmodel.state_dict())

    update_counter = 0

    torch.manual_seed(nr)

    # for max time step
    while logger.time.value < args.maxtimestep:
        # store the transitions
        states = []
        actions = []
        rewards = []
        dists = []
        values = []
        # reset the environment
        current_state = env.reset()

        total_reward = 0

        # reset the hiddens 
        agent.init_hidden()
        # episode loop
        for e_step in range(args.maxlen):
            # process the state
            dist, value = agent.process_state(current_state)

            # get an action for the current state
            action = agent.soft_policy(dist)
            
            #env.render()
            
            # apply the action
            next_state, reward, done, info = env.step(action)

            total_reward += reward

            # clip the reward
            reward = min( 1.0, max(-1.0, float(reward)) )

            # save the transition
            states.append(current_state)
            actions.append(action)
            rewards.append(reward)
            dists.append(dist)
            values.append(value)
            # sample n step from environment or stop when we are done
            if (e_step+1)%args.nstep == 0 or done:
                # get losses
                actor_loss, critic_loss = agent.loss( Transition(states, actions, rewards, dists, values), next_state, done, args.gamma, args.beta)
                # zero the gradients on the local agent
                agent.zero_grad()
                # calculate the loss
                loss = actor_loss+0.5*critic_loss
                # calculate gradient
                loss.backward()
                # update global
                agent.global_update(optim, globalmodel)
                # sync with global
                agent.synchronize(globalmodel.state_dict())
                if not done:
                    # reset the hiddens 
                    agent.hidden = Hidden( agent.hidden.actor.detach(), agent.hidden.critic.detach() )
                # reset the buffers
                states = []
                actions = []
                rewards = []
                dists = []
                values = []

                update_counter += 1
            # increase the step counter
            with lock:
                logger.time.value += 1
            if done:
                # if done restart the environment
                break
            else:
                # continue from the next state
                current_state = next_state
        if update_counter%1000==0:
            print(update_counter, "    ", total_reward)



def test_worker(args, globalmodel, envfunc, agentfunc, lock, logger,
                monitor_path=None, render=True):
    """ Evaluation worker function.
        Test the greedy agent until max time step is
        reached. After every episode, synchronize the
        agent. Loading bar is used to track learning
        process in the notebook.
        
        Arguments:
            - args: Hyperparameters
            - globalmodel: Global(shared) agent for
            synchronization.
            - envfunc: Environment generating function
            - agentfunc: Agent generating function
            - lock: Lock for shared memory
            - logger: Namedtuple of shared objects for
            logging purposes
            - monitor_path: Path for monitoring. If not
            given environment will not be monitored
            (default=None)
            - render: If true render the environment
            (default=False)
    """
    env = envfunc()
    agent = agentfunc()
    agent.eval()
    bar = LoadingBar(args.maxtimestep, "Time step")

    # Remember to call bar.process with time step and
    # best reward achived after each episode.
    # You may not use LoadingBar (optional).

    # You can include additional logging
    # Remember to change Logger namedtuple and
    # logger in order to do so.

    if monitor_path:
        path = "monitor/" + monitor_path
        env = gym.wrappers.Monitor(
            env, path, video_callable=lambda eps_id: True, force=True)
        render = False
    # init best reward
    best_reward = -np.inf
    # sync with global
    agent.synchronize(globalmodel.state_dict())
    agent.init_hidden()
    # for max time step
    while logger.time.value < args.maxtimestep:
        # reset the environment
        current_state = env.reset()
        # init episode reward
        eps_reward = 0
        # episode loop
        for e_step in range(args.maxlen):
            with torch.no_grad():
                # process the state
                dist, value = agent.process_state(current_state)
            # get an action for the current state
            action = agent.greedy_policy(dist)
            # render the environment
            if render:
                env.render()
            # apply the action
            next_state, reward, done, info = env.step(action)
            # accumulate the reward
            eps_reward += reward
            # stop if done
            if done:
                break
            # continue from the next state
            current_state = next_state
        # close the environment if monitoring
        if monitor_path:
            env.close()
        # sync with global
        agent.synchronize(globalmodel.state_dict())
        # reset the hiddens 
        agent.init_hidden()
        # save the best reward
        if eps_reward > best_reward:
            best_reward = eps_reward
        # print process
        bar.progress(logger.time.value, best_reward)
        #print(logger.time.value, best_reward, eps_reward, "test-----")
        import time
        time.sleep(5)
        

    

