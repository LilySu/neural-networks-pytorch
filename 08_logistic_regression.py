# 1. design model - we need input size and output size, forward pass with
# all the different layers
# 2. construct the loss and optimizer
# 3. training loop
    # a. compute prediction
    # b. do backward pass to get gradients
    # c. update our weights

# slight adjustments to model and cost function.

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# 0. prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
# print(n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# convert to torch tensors
# 1. model
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1) # 30 input features, 1 output feature

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)
# 2. loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
# 3. training
num_epochs = 100
# training loop
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    # backward pass
    loss.backward()
    # update our weights
    optimizer.step()
    # empty our gradients, because the backward function here will always add up all gradients to the
    # .grad attribute
    optimizer.zero_grad()
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

    # allow safe acquisition and release of operating system resources.
    with torch.no_grad():
        y_predicted = model(X_test)
        y_predicted_cls = y_predicted.round()
        acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
        # for every prediction that is correct it will add +1
        print(f'accuracy = {acc:.4f}')