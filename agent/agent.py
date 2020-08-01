import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.functinal as F 

def hidden_unit(Layer):
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
			seed (int): Random seed
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





