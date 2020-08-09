import torch
import torch.nn as nn
import numpy as np

loss = nn.CrossEntropyLoss()

# no softmax in last layer

# y must not be one hot encoded

# y_pred has raw scores, so no softmax here

#labels
Y = torch.tensor([2, 0, 1])
# nsamples x nclasses = 1x3
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [2.0, 3.0, 0.1]]) # array of arrays
Y_pred_bad = torch.tensor([[2.0, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item()) # our good prediction has a lower cross entropy loss
print(l2.item())
# returns
# 0.4170299470424652
# 1.840616226196289

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1)
print(predictions2)