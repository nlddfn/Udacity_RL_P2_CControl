import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from deterministic_dpg_agent import Agent
from train_eval import evaluate_agent, train_agent
from unityagents import UnityEnvironment
from workspace_utils import active_session

# select this option to load version 2 (with 20 agents) of the environment
env_path = "/data/Reacher_Linux_NoVis/Reacher.x86_64"
local_path = os.path.dirname(os.path.abspath(__file__))
print(local_path)
local_env_path = local_path + "/Reacher.app"
env = UnityEnvironment(file_name=env_path)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print("Number of agents:", num_agents)

# size of each action
action_size = brain.vector_action_space_size
print("Size of each action:", action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print(
    "There are {} agents. Each observes a state with length: {}".format(
        states.shape[0], state_size
    )
)
print("The state for the first agent looks like:", states[0])


start_time = datetime.now()

# VARS
config = {
    "buffer_size": int(1e6),  # replay buffer size
    "batch_size": 1024,  # minibatch size
    "replay_initial": 1024,  # initial memory before updating the network
    "gamma": 0.99,  # discount factor
    "lr_actor": 1e-4,  # learning rate
    "lr_critic": 1e-3,  # learning rate of the critic
    "update_every": 2,  # how often to update the network
    "tau": 1e-3,  # soft update
    "weight_decay": 0,  # l2 weight decay
    "net_body": (256, 128, 64),  # hidden layers
}

with active_session():
    print("Train Agent")
    agent = Agent(
        state_size=state_size,
        action_size=action_size,
        random_seed=0,
        num_agents=num_agents,
        **config
    )
    scores = train_agent(agent, env, n_episodes=200, max_t=1000)

# Plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.show()

time_elapsed = datetime.now() - start_time
print("Time elapsed (hh:mm:ss.ms) {}".format(time_elapsed))

# Evaluate agent
evaluate_agent(
    agent, env, num_agents=20, checkpoint_pth=local_path, num_episodes=10, min_score=30
)
