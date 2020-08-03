import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.functional as F 

# based on original DDPG paper, all hidden layers are ReLu and last layer is tanH


def hidden_unit(Layer):
	""" 
	Function to initialize weights according to original paper. 
	This is to make sure inital outputs for the policy and value estimates are near zero.
	""" 
	fan_in = Layer.weight.data.size()[0]
	lim = 1. / np.sqrt(fan_in)
	return (-lim, lim)

class Actor(nn.Module):
	"""
	Policy model
	μ = (s| θ^μ)

	Maps state to specific actions
	"""

	def __init__(self, state_size, action_size, seed, fc_units=256):
		"""
		Initialize parameters and build model. 

		Params
		------
			state_size (int): dimension of states
			action_size (int): dimension of actions
			seed (int): random seed
			fc_units: number of nodes in hidden layers

		Returns
		-------
			tensors of expected actions
		"""

		super(Actor, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size, fc_units)
		self.fc2 = nn.Linear(fc_units, action_size)
		self.reset_parameters()

	def reset_parameters(self):
		self.fc1.weight.data.uniform_(*hidden_unit(self.fc1))
		self.fc2.weight.data.uniform_(-3e-3, 3e-3)

	def forward(self, state):
		x = F.relu(self.fc1(state))
		return F.tanh(self.fc2(x))

class Critic(nn.Module):
	"""
	Value model
	Q(s,a)

	actions are note included until the second layer, as per original paper
	"""

	def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256, fc3_units=128):
		"""
		Initialize parameters and build model

		Params
		------
			state_size (int): dimension of states
			action_size (int): dimension of actions
			seed (int): random seed
			fc1_units (int): number of nodes in first hidden layer
			fc2_units (int): number of nodes in second hidden layer
			fc3_units (int): number of nodes in third hidden layer

		Returns
		-------

		"""

		super(Critic, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size, fc1_units)
		self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
		self.fc3 = nn.Linear(fc2_units, fc3_units)
		self.fc4 = nn.Linear(fc3_units, 1)
		self.reset_parameters()

	def reset_parameters(self):
		self.fc1.weight.data.uniform_(*hidden_unit(self.fc1))
		self.fc2.weight.data.uniform_(*hidden_unit(self.fc2))
		self.fc3.weight.data.uniform_(*hidden_unit(self.fc3))
		self.fc4.weight.data.uniform_(-3e3,3e3)

	def forward(self, state, action):
		x_state = F.relu(self.fc1(state))
		x = torch.cat((x_state, action), dim=1)
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		return self.fc4(x)
	





