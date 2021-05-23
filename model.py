import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size, fc1_units=400, fc2_units=300):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(obs_size, fc1_units)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(fc1_units, fc2_units)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(fc2_units, act_size)),
            ('tanh3', nn.Tanh())
        ]))
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)
    
    def reset_parameters(self):
        self.net.fc1.weight.data.uniform_(*hidden_init(self.net.fc1))
        self.net.fc2.weight.data.uniform_(*hidden_init(self.net.fc2))
        self.net.fc3.weight.data.uniform_(-3e-3, 3e-3)


class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size, fc1_units=400, fc2_units=300):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(OrderedDict([
             ('fcs1', nn.Linear(obs_size, fc1_units)),
             ('relu1', nn.ReLU()),
        ]))

        self.out_net = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(fc1_units + act_size, fc2_units)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(fc2_units, 1))
        ]))
        self.reset_parameters()

    def forward(self, states, actions):
        obs = self.obs_net(states)
        return self.out_net(torch.cat([obs, actions], dim=1))
    
    def reset_parameters(self):
        self.obs_net.fcs1.weight.data.uniform_(*hidden_init(self.obs_net.fcs1))
        self.out_net.fc2.weight.data.uniform_(*hidden_init(self.out_net.fc2))
        self.out_net.fc3.weight.data.uniform_(-3e-3, 3e-3)
