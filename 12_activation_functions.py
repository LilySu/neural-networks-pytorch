import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementing a NN with 3 layers - 1 input, 1 hidden, 1 output

# option 1 create functions as nn.Modules
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # other options
        # nn.Sigmoid
        # nn.Softmax
        # nn.TanH
        # nn.LeakyReLU
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # call functions
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# option 2 use functions directly, only define linear layers in init, in forward pass, call relu and sigmoid
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):  # call functions
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        # other options
        # torch.softmax
        # torch.tanh
        # F.leaky_relu()
        return out

