import torch
import torch.nn as nn
import torch.nn.functional as F

class simpleMLP(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(simpleMLP, self).__init__()
        self.linear1 = nn.Linear(state_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, action_dim)

    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        o = self.out(x)
        return o
    
    def count_parameters(self):
        count = 0
        for param in self.parameters():
            count += param.data.numpy().flatten().shape[0]
        return count

    def es_params(self):
        """
        The params that should be trained by ES (all of them)
        """
        return [(k, v) for k, v in zip(self.state_dict().keys(),
                                       self.state_dict().values())]