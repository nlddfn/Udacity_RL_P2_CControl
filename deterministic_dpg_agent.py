import numpy as np
import random
from collections import namedtuple, deque

from model import DDPGActor, DDPGCritic
from utils import ReplayBuffer, OrnsteinUhlenbeckProcess

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
REPLAY_INITIAL = int(1e4) # initial memory before updatting the network
GAMMA = 0.99            # discount factor
LR_ACTOR = 2e-4         # learning rate 
LR_CRITIC = 2e-4        # learning rate of the critic
UPDATE_EVERY = 4        # how often to update the network
TAU = 1e-3              # soft update
WEIGHT_DECAY = 0        # L2 weight decay
NET_BODY = (400, 300)   # hidden layers


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 seed,
                 net_body=NET_BODY,
                 gamma=GAMMA,
                 buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE,
                 replay_initial=REPLAY_INITIAL,
                 update_every=UPDATE_EVERY,
                 lr_actor=LR_ACTOR,
                 lr_critic=LR_CRITIC,
                 weight_decay=WEIGHT_DECAY,
                 tau=TAU
    ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_initial = replay_initial
        self.update_every = update_every
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Actor-Network
        fc1_units, fc2_units = net_body
        self.actor_local = DDPGActor(state_size, action_size, fc1_units, fc2_units).to(self.device)
        self.actor_target = DDPGActor(state_size, action_size, fc1_units, fc2_units).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Critic-Network
        self.critic_local = DDPGCritic(state_size, action_size, fc1_units, fc2_units).to(self.device)
        self.critic_target = DDPGCritic(state_size, action_size, fc1_units, fc2_units).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)
    
        # Noise process
        self.noise = OrnsteinUhlenbeckProcess(action_size, self.seed)
        
        # Replay memory
        self.memory = ReplayBuffer(
            action_size=action_size,
            buffer_size=buffer_size,
            batch_size=batch_size,
            seed=seed,
            device=self.device
        )
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        # If enough samples are available in memory, get random subset and learn
        if (len(self.memory) > self.replay_initial) and (self.t_step == 0):
            experiences = self.memory.sample()
            self.update(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
        """
        state = (
            torch.from_numpy(state)
            .float()
            .to(self.device)
        )
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local.forward(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1., 1.)
    
    def reset(self):
        self.noise.reset()

    def update(self, batch, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            batch (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = batch
        
        # Update Critic
        # Get expected Q values from critic model
        curr_Q = self.critic_local.forward(states, actions)
        
        # Get predicted Q values (for next states) from target model
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions)
        expected_Q = rewards + (self.gamma * next_Q * (1 - dones))

        critic_loss = F.mse_loss(curr_Q, expected_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # Update Actor
        # Get expected actions from actor model
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.update_target(self.critic_local, self.critic_target, TAU)  
        self.update_target(self.actor_local, self.actor_target, TAU) 
        
    def update_target(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)