import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearPolicyDiscrete(nn.Module):
	def __init__(self, state_dims, act_dims):
		super().__init__()
		self.linear = nn.Linear(state_dims, act_dims)

	def forward(self, state):
		logits = self.linear(state)
		return F.softmax(logits, dim = -1)


class MLP(nn.Module):
	def __init__(self, input_dims, output_dims, hidden_sizes=[32]):
		super().__init__()
		layer_sizes = [input_dims] + hidden_sizes + [output_dims]
		layers = []
		
		for i in range(len(hidden_sizes)):
			layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
			layers.append(nn.ReLU()) 
	
		layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
		
		self.mlp = nn.Sequential(*layers)
	
	def forward(self, state):
		return self.mlp(state)
		

class MLPPolicyDiscrete(nn.Module):
	def __init__(self, state_dims, act_dims, hidden_sizes=[32]):
		super().__init__()
		layer_sizes = [state_dims] + hidden_sizes + [act_dims]
		layers = []
		
		for i in range(len(hidden_sizes)):
			layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
			layers.append(nn.ReLU()) 
		
		layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
		
		self.mlp = nn.Sequential(*layers)

	def forward(self, state):
		logits = self.mlp(state)
		return F.softmax(logits, dim = -1).clamp(min=1e-8, max=1-1e-8)

		
class MLPPolicyGaussianFixedVariance(nn.Module):
	def __init__(self, state_dims, act_dims, hidden_sizes=[32], var = 0.2):
		super().__init__()
		self.var = var
		
		layer_sizes = [state_dims] + hidden_sizes 
		layers = []
		
		for i in range(len(hidden_sizes)):
			layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
			layers.append(nn.ReLU()) 

		layers.append(nn.Linear(hidden_sizes[-1] , act_dims))

		self.mlp = nn.Sequential(*layers)

	def forward(self, state):
		mu = self.mlp(state)
		return mu, torch.zeros_like(mu) + self.var


class MLPPolicyGaussian(nn.Module):
	def __init__(self, state_dims, act_dims, hidden_sizes=[32]):
		super().__init__()
		layer_sizes = [state_dims] + hidden_sizes 
		layers = []
		
		for i in range(len(hidden_sizes)):
			layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
			layers.append(nn.ReLU()) 
        
		self.mlp = nn.Sequential(*layers)
		self.mu_head = nn.Linear(hidden_sizes[-1], act_dims)
		self.tanh = nn.Tanh()
		self.var_head = nn.Linear(hidden_sizes[-1], act_dims)
		self.softplus = nn.Softplus()

	def forward(self, state):
		x = self.mlp(state)
		mu = self.tanh(self.mu_head(x))
		var = self.softplus(self.var_head(x))
		return mu, var

class QNet(nn.Module):
	def __init__(self, s_dims, a_dims, hidden_sizes=[32]):
		super().__init__()
		layer_sizes = [s_dims + a_dims] + hidden_sizes + [1]
		layers = []
		
		for i in range(len(hidden_sizes)):
			layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
			layers.append(nn.ReLU()) 
	
		layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
		
		self.mlp = nn.Sequential(*layers)
	
	def forward(self, state, action):
		return self.mlp(torch.cat([state, action], dim=-1))


class DDPGPolicy(nn.Module):
	def __init__(self, input_dims, output_dims, a_low, a_high, hidden_sizes=[32]):
		super().__init__()
		self.a_l = a_low
		self.a_h = a_high
		self.mlp = MLP(input_dims, output_dims, hidden_sizes)
	
	def forward(self, state):
		out = F.tanh(self.mlp(state))
		return self.a_l + (self.a_h - self.a_l) * ((out + 1) / 2)