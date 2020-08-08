# 1. design model - we need input size and output size, forward pass with
# all the different layers
# 2. construct the loss and optimizer
# 3. training loop
    # a. compute prediction
    # b. do backward pass to get gradients
    # c. update our weights
import torch
import torch.nn as nn
import numpy as np # for data transformations
from sklearn import datasets
import matplotlib.pyplot as plt

# prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise = 20, random_state = 1)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1) # reshapes our tensors
# model
n_samples, n_features = X.shape
# model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)
# define loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss() # mean squared error
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# training loop
num_epochs = 100 # training iterations
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    # backward pass
    loss.backward() # calculates gradient
    # update
    optimizer.step()
    # empty our gradients because when we call the backward function,
    # it will aggregate gradientsto the .grad attribute
    optimizer.zero_grad() # never forget
    # we are done with training now
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# plot
predicted = model(X).detach().numpy() # detach from computation graph, tensor has required gradient to be true
# but now we set grad to False, and convert to numpy
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()