## based on paper DDPG - CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING

# requirements:
# 	Ornstein-Uhlenbeck process noise class
#	Replay buffer class 
#	Actor-Critic networks (local and target for both) - with batch norm
#	Soft update theta_prime = tau * theta + (1-tau)*prime_theta

import numpy as np 
import random
import copy
from collections import namedtuple, deque

from model import Actor, Actor, Critic 

import torch.nn.functional as F 
import torch.optim as optim

## hyperparameters based on original paper

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 10e-4        # learning rate of the actor 
LR_CRITIC = 10e-3       # learning rate of the critic
WEIGHT_DECAY = 10e-2    # L2 weight decay

NOISE_THETA = 0.15		# Ornstein-Ulenbeck parameter
NOISE_SIGMA = 0.2		# Ornstein-Ulenbeck parameter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
	"""
	DDPG agent that interacts with the environment. 
	Agent must initialize actor critic local and target networks,
	Initialize replay buffer.
	
		for each episode
			initialize a random process (Ornstein-Uhlenbeck).
			receive inital observation state

			for each state
				select action according to current policy and noise
				execute action and observe new state and reward
				store transitions in replay buffer
				sample random minibatch
				update critic by minimizing loss
				update actor policy using sampled policy gradient
				update target networks based on soft-update rule
	"""

	def __init__(self, state_size, action_size, seed):
		"""
		Initialize an agent

		Params
		------
			state_size (int): dimension of states
			action_size (int): dimension of actions
			seed (int): random seed
		"""

		self.state_size = state_size
		self.action_size = action_size
		self.seed = random.seed(seed)

		# initialize actor local and target network 
		self.actor_local = Actor(state_size, action_size, seed).to(device)
		self.actor_target = Actor(state_size, action_size, seed).to(device)
		self.actor_optimizer = optim.Adam(self.actor_local.params(), lr=LR_ACTOR)

		# initialize critic local and target network
		self.critic_local = Critic(state_size, action_size, seed).to(device)
		self.critic_target = Critic(state_size, action_size, seed).to(device)
		self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, WEIGHT_DECAY=WEIGHT_DECAY)
 
 		# Noise process
		self.noise = OUNoise(action_size, seed)

		# Replay memory
		self.memory  ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)