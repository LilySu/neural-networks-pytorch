import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
num_epochs = 5
batch_size = 4
learning_rate = 0.001

# PILimages of range [0, 1]
# Transform into Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train = True,
                                             transform = transform, download = True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train = False,
                                             transform = transform, download = True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

classes = ('plane','car','bird','cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# def imshow(img):
#     img = img / 2 + 0.5 # unormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# # get some random training images
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
#
# # imshow(torchvision.utils.make_grid(images))
# torchvision.utils.make_grid(images)
# conv1 = nn.Conv2d(3, 6, 5)
# pool = nn. MaxPool2d(2, 2)
# conv2 = nn.Conv2d(6, 16, 5)
# # print(images.shape)
# # torch.Size([4, 3, 32, 32])
# x = conv1(images)
# # print(x.shape)
# # torch.Size([4, 6, 28, 28])
# x = pool(x)
# # print(x.shape)
# # torch.Size([4, 6, 14, 14])
# x = conv2(x)
# # print(x.shape)
# # torch.Size([4, 16, 10, 10])
# x= pool(x)
# # print(x.shape)
# # torch.Size([4, 16, 5, 5])

# calculate output size for 5x5 input, 3x3 filter, padding=0, stride=1
# (Width-Filter size + 2Padding)/S + 1
# (5 - 3 + 2(0))/1 + 1 = 3x3
# ours for first layer: (32 - 5 + 2(0)/1 + 1 = 28
# final size torch.Size([4, 16, 5, 5])

# for pooling layer: 4 x 16, resulting image is 10 x 10

# another pooling 4 x 16, 5 x 5

# 1 Convolutional Layer
# 1 reLU activation function
# 1 maxpooling layer
# 2nd Convolutional Layer
# 2nd reLU activation function
# 3 different fully connected layers
# Softmax Cross Entropy - already implemented as criterion
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # input, output, kernel
        # images have 3 color channels
        self.pool = nn.MaxPool2d(2, 2) # dimension and strides
        self.conv2 = nn.Conv2d(6, 16, 5) # input channel should be equal to the last output
        self.fc1 = nn.Linear(16*5*5, 120) # fixed from: final size torch.Size([4, 16, 5, 5])
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 different classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # apply the first pooling layer
        x = self.pool(F.relu(self.conv2(x))) # apply second convolutional layer
        x = x.view(-1, 16*5*5) # pass it to the first fully connected layer by flattening it, let pytorch find # of batches, then size
        # our tensor is now flattened
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss() # for multi-class classification
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): # to get all batches
        # 100, 1, 28, 28
        # input size is 784
        # our tensor needs 100 batches, 784 input
        images = images.to(device) # to get gpu support
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # backward pass
        # make sure that the grads are empty before calling backward and update step
        optimizer.zero_grad()
        loss.backward() # backpropagation
        optimizer.step() # update step updates parameters for us

        if (i+1) % 2000 == 0:
            print(f'Epoch [{epoch+1} / {num_epochs}], Step {i+1} / {n_total_steps}, Loss = {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

# testing loop
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    # loop over batches in test samples
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        # max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item() # get correct predictions

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples # accuracy percent
    print(f'Accuracy of the network = {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

# not very good because we specified 4 epochs

# Files already downloaded and verified
# Files already downloaded and verified
# Epoch [1 / 5], Step 2000 / 12500, Loss = 2.3037
# Epoch [1 / 5], Step 4000 / 12500, Loss = 2.2571
# Epoch [1 / 5], Step 6000 / 12500, Loss = 2.2796
# Epoch [1 / 5], Step 8000 / 12500, Loss = 2.2740
# Epoch [1 / 5], Step 10000 / 12500, Loss = 2.3697
# Epoch [1 / 5], Step 12000 / 12500, Loss = 2.2581
# Epoch [2 / 5], Step 2000 / 12500, Loss = 1.6622
# Epoch [2 / 5], Step 4000 / 12500, Loss = 1.9715
# Epoch [2 / 5], Step 6000 / 12500, Loss = 2.2281
# Epoch [2 / 5], Step 8000 / 12500, Loss = 2.5502
# Epoch [2 / 5], Step 10000 / 12500, Loss = 1.1793
# Epoch [2 / 5], Step 12000 / 12500, Loss = 1.8124
# Epoch [3 / 5], Step 2000 / 12500, Loss = 2.0309
# Epoch [3 / 5], Step 4000 / 12500, Loss = 1.5583
# Epoch [3 / 5], Step 6000 / 12500, Loss = 1.6445
# Epoch [3 / 5], Step 8000 / 12500, Loss = 1.4779
# Epoch [3 / 5], Step 10000 / 12500, Loss = 2.0051
# Epoch [3 / 5], Step 12000 / 12500, Loss = 1.4822
# Epoch [4 / 5], Step 2000 / 12500, Loss = 1.9048
# Epoch [4 / 5], Step 4000 / 12500, Loss = 1.1187
# Epoch [4 / 5], Step 6000 / 12500, Loss = 1.1263
# Epoch [4 / 5], Step 8000 / 12500, Loss = 0.7610
# Epoch [4 / 5], Step 10000 / 12500, Loss = 0.6636
# Epoch [4 / 5], Step 12000 / 12500, Loss = 1.4307
# Epoch [5 / 5], Step 2000 / 12500, Loss = 2.1409
# Epoch [5 / 5], Step 4000 / 12500, Loss = 1.8407
# Epoch [5 / 5], Step 6000 / 12500, Loss = 1.7058
# Epoch [5 / 5], Step 8000 / 12500, Loss = 1.6188
# Epoch [5 / 5], Step 10000 / 12500, Loss = 1.6928
# Epoch [5 / 5], Step 12000 / 12500, Loss = 1.0296
# Finished Training
# Finished Training
# Accuracy of the network = 10.21 %
# Accuracy of plane: 24.2 %
# Accuracy of car: 0.0 %
# Accuracy of bird: 0.0 %
# Accuracy of cat: 0.0 %
# Accuracy of deer: 25.8 %
# Accuracy of dog: 0.0 %
# Accuracy of frog: 0.0 %
# Accuracy of horse: 0.0 %
# Accuracy of ship: 26.1 %
# Accuracy of truck: 26.0 %