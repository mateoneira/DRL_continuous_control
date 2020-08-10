## based on paper DDPG - CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING

# requirements:
# 	Ornstein-Uhlenbeck process noise class
#		Replay buffer class 
#		Actor-Critic networks (local and target for both) - with batch norm
#		Soft update theta_prime = tau * theta + (1-tau)*prime_theta

# notes:
#		exploration exploitation is important, we can decrease the noise param slowly
#		it's important to be aggresive in the exploration initially 
#		so include an epsilon to decay the noise as training progresses
#		To take advantage of multiple agents (20), we can have different noise processes for each

import numpy as np 
import random
import copy
from collections import namedtuple, deque

from agent.model import Actor, Critic 

import torch
import torch.nn.functional as F 
import torch.optim as optim

## hyperparameters based on original paper

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

NOISE_THETA = 0.15		  # Ornstein-Ulenbeck parameter
NOISE_SIGMA = 0.2		  #Ornstein-Ulenbeck parameter
EPSILON_DECAY = 1e-6    #to decay noise


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

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

	def __init__(self, state_size, action_size, num_agents=1, seed=42):
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
		self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

		# initialize critic local and target network
		self.critic_local = Critic(state_size, action_size, seed).to(device)
		self.critic_target = Critic(state_size, action_size, seed).to(device)
		self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
 
 		# Noise process
		self.noise = [OUNoise(action_size, seed) for i in range(num_agents)]
		self.epsilon = 1.0

		# Replay memory
		self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random)
		print('agent using {} device'.format(device))

	def step(self, states, actions, rewards, next_states, dones):
		"""
		Save experience in replay memory, and use random sample from buffer to learn.
		"""
		# add to replay buffer
		for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
			self.memory.add(state, action, reward, next_state, done)

		# learn, if enough samples are available in memory
		if len(self.memory) > BATCH_SIZE:
			for _ in range(10):
				experiences = self.memory.sample()
				self.learn(experiences, GAMMA)

	def act(self, state, add_noise=True):
		"""
		Returns actions for given state as per current policy.
		"""
		state = torch.from_numpy(state).float().to(device)
		self.actor_local.eval()
		with torch.no_grad():
		    action = self.actor_local(state).cpu().data.numpy()
		self.actor_local.train()

		if add_noise:
		    action += np.array([n.sample() for n  in self.noise])*self.epsilon

		return np.clip(action, -1, 1)

	def reset(self):
		for n in self.noise:
			n.reset()

	def learn(self, experiences, gamma):
		"""
		Update policy and value parameters using given batch of experience tuples.
		Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
		where:
		    actor_target(state) -> action
		    critic_target(state, action) -> Q-value

		Params
		------
		    experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
		    gamma (float): discount factor
		"""
		states, actions, rewards, next_states, dones = experiences

		# Get predicted next-state actions and Q values from target models
		actions_next = self.actor_target(next_states)
		Q_targets_next = self.critic_target(next_states, actions_next)
		# Compute Q targets for current states (y_i)
		Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

		# Compute critic loss
		Q_expected = self.critic_local(states, actions)
		critic_loss = F.mse_loss(Q_expected, Q_targets)

		# Minimize the loss
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		# Gradient clipping to help with learning
		torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(),1)
		self.critic_optimizer.step()

		# Compute actor loss
		actions_pred = self.actor_local(states)
		actor_loss = -self.critic_local(states, actions_pred).mean()
		# Minimize the loss
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# update targets
		self.soft_update(self.critic_local, self.critic_target, TAU)
		self.soft_update(self.actor_local, self.actor_target, TAU)

		self.epsilon -= EPSILON_DECAY 
		self.epsilon = max(0.05,self.epsilon)

	def soft_update(self, local_model, target_model, tau):
		"""
		Soft update model parameters
		theta_prime = tau * theta + (1-tau)*prime_theta

		Params
		------
			local_model: PyTorch model (weights will be copied from)
			target_model: PyTorch model (weights will be copied to)
			tau (float): interpolation parameter 
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
	"""
	Ornstein-Uhlenbeck process.

	adapted using https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

	"""

	def __init__(self, size, seed, mu=0., theta=NOISE_THETA, sigma=NOISE_SIGMA, dx = 1e-2):
		"""Initialize parameters and noise process."""
		self.mu = mu * np.ones(size)
		self.theta = theta
		self.sigma = sigma
		self.seed = random.seed(seed)
		self.dx = dx
		self.reset()

	def reset(self):
		"""Reset the internal state (= noise) to mean (mu)."""
		self.state = copy.copy(self.mu)

	def sample(self):
		"""Update internal state and return it as a noise sample."""
		x = self.state
		# dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
		dx = self.theta * (self.mu - x)*self.dx + self.sigma * np.sqrt(self.dx)*np.random.normal(size=self.mu.shape)
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