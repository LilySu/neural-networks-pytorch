# Cross entropy is one out of many possible loss functions (another
# popular one is SVM hinge loss). These loss functions are typically
# written as J(theta) and can be used within gradient descent, which
# is an iterative algorithm to move the parameters (or coefficients)
# towards the optimum values.

# For multi-class problem

# our y needs to be one-hot encoded
# apply softmax

# Average the losses over the entire test set, and you get the cross entropy loss.
import torch
import torch.nn as nn
import numpy as np

# squashes output from 0 to 1
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 0)

x = np.array([2.0, 1.0, 0.1])
output = softmax(x)
print('softmax numpy:', output)

x = torch.tensor([2.0, 1.0, 0.1])
output = torch.softmax(x, dim=0)
print(output)

# a lot of times softmax is used with cross entropy
# loss increases as predicted probabiliy diverges from true
# high cross entropy for very wrong prediction

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

Y = np.array([1, 0, 0])

Y_pred_good = np.array([.7, .2, .1])
Y_pred_bad = np.array([.1, .3, .6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')
