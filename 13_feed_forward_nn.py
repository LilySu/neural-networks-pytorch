import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784 # 28 x 28 pixel image that will be flattened into an array
hidden_size = 100
num_classes = 10 # digits from 0 - 9
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train = True, transform = transforms.ToTensor(), download = False)
test_dataset = torchvision.datasets.MNIST(root='./data', train = False, transform = transforms.ToTensor(), download = False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)
# torch.Size([100, 1, 28, 28]) torch.Size([100])
# we have 100 samples in batch, 1 channel, 28x28 pixels, for each class label, we have 1 value

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
# plt.show()

### we want to set up a fully connect NN with 1 hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes): # output size is # of classes
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() # activation function
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out) # gets prevous out
        out = self.l2(out) # applies 2nd linear function
        # don't apply softmax for multi-class, because we use cross entropy that applies softmax for us
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28
        # input size is 784
        # our tensor needs 100 batches, 784 input
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # backward pass
        # make sure that the grads are empty before calling backward and update step
        optimizer.zero_grad()
        loss.backward() # backpropagation
        optimizer.step() # update step updates parameters for us

        if (i+1) % 100 == 0:
            print(f'epoch{epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')

# testing loop
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    # loop over batches in test samples
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        output = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item() # get correct predictions

    acc = 100.0 * n_correct / n_samples # accuracy percent
    print(f'accuracy = {acc}')