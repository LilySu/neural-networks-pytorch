import numpy as np

X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

# model prediction
def forward(x):
    return w * x
# loss = MSE in linear regression
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient with respect to our parameters

# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2x (w*x - y) # numerical computed derivative chain rule
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean() # dot product of 2 and x
print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    dw = gradient(X, Y, y_pred)

    # update weights -> go in negative direction of the gradient
    w -= learning_rate * dw

    if epoch % 1 == 0: # we want to print every step
        print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')

# print prediction after training
print(f'Prediction before training: f(5) = {forward(5):.3f}')