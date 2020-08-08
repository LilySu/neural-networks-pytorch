# 1. design model - we need input size and output size, forward pass with
# all the different layers
# 2. construct the loss and optimizer
# 3. training loop
    # a. compute prediction
    # b. do backward pass to get gradients
    # c. update our weights


# task 1 add imports, replace loss and optimization, no need to define loss anymore
# remove def forward with a pytorch implementation and w, since pytorch knows what parameters
# modify our optimizer
import torch
import torch.nn as nn

# X = np.array([1,2,3,4], dtype=np.float32)
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32) # needs to be 2d, # rows is samples, for each row, it is features
# Y = np.array([2,4,6,8], dtype=np.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
# print(n_samples, n_features)
# yields 4 samples, 1 feature per sample

input_size = n_features
output_size = n_features

# model = nn.Linear(n_features, n_features) # needs input size and output size

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        # since we are only using one nn.Linear layer, the forward pass is already available
        self.lin = nn.Linear(input_dim, output_dim, bias = False)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}') # item gets float

# Training
learning_rate = 0.01
n_iters = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) # stochastic gradient descent, with optimizing parameters
# needs this as a list, learning rate

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    # dw = gradient(X, Y, y_pred)
    l.backward() # dl/dw
    # update weights
    optimizer.step()
    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0: # we want to print every step
        [w, b] = model.parameters() # this is a list of list
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

# print prediction after training
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')