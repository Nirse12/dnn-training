# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 17:58:13 2021

@author: segalni2
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# Part 2 - Classification with Various Networks
# https://towardsdatascience.com/batch-normalization-and-dropout-in-neural-networks-explained-with-pytorch-47d7a8459bcd
### Fully-connected NN with 3 hidden layers

#### Parameters
seed = 50
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed) 


batchSize = 64
test_batch_size = 128
epochs = 10
lr = 0.001
weight_decay = 0.002
momentum = 0.9
mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
device = torch.device("cpu")
optimizer_type = 'Adam'  # Adam, RMSprop,SGD

# need to random cropping images to size 32
# for train val:
transform_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((48, 48)),
     transforms.RandomCrop((32, 32)),
     transforms.RandomHorizontalFlip(),
     transforms.Normalize(mean, std)])

# for test val:
transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((48, 48)),
     transforms.RandomCrop((32, 32)),
     transforms.Normalize(mean, std)])

# read and separate for train and val
trainset = torchvision.datasets.STL10(root='./data', split='train',
                                      download=True, transform=transform_train)
# separate
trainset, valset = model_selection.train_test_split(trainset, train_size=0.8, test_size=0.2)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                           shuffle=True, num_workers=0)

val_loader = torch.utils.data.DataLoader(valset, batch_size=batchSize,
                                         shuffle=False, num_workers=0)

testset = torchvision.datasets.STL10(root='./data', split='test',
                                     download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=0)

classes = ('Airplane', 'Bird', 'Car', 'Cat', 'Deer', 'Dog', 'Horse', 'Monkey', 'Ship', 'Truck')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ReLU(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ReLU(x)
        x = self.pool(x)

        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.ReLU(x)

        x = self.fc2(x)
        x = self.ReLU(x)

        x = self.fc3(x)

        return x


"""### Training"""

# create an instance of our model
model = CNN().to(device)
# loss criterion
criterion = nn.CrossEntropyLoss()

# optimizer type
if optimizer_type == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
elif optimizer_type == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
elif optimizer_type == 'RMSProp':
    optimizer = optim.Adam(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=weight_decay)
else:
    NotImplementedError("optimizer not implemented")

total_train_loss = np.zeros((epochs, 1))
total_val_loss = np.zeros((epochs, 1))
total_val_accuracy = np.zeros((epochs, 1))
total_train_accuracy = np.zeros((epochs, 1))

# https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19
iter = 0
for epoch in range(epochs):  # loop over the dataset multiple times

    train_loss = 0
    train_acc = 0

    for i, (images, lables) in enumerate(train_loader):
        # data = torch.flatten(images, start_dim=1)
        images, lables = images.to(device), lables.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # Logistic Regression on the train data
        loss = criterion(outputs, lables)
        loss.backward()
        optimizer.step()

        iter += 1
        if iter % 500 == 0:
            # calculate Accuracy
            correct = 0
            total = 0

        # accuracy & loss
        train_loss += loss.item()
        predicted = torch.max(outputs.data, 1)[1]
        train_acc += (predicted == lables).sum()  # number of true predictions

    model.eval()  # Logistic Regression on the valuation data
    val_loss = 0
    val_acc = 0
    with torch.no_grad():  # in val we do not change  and calc gradient
        for j, (images, lables) in enumerate(val_loader):
            # data = torch.flatten(images, start_dim=1)
            images, lables = images.to(device), lables.to(device)
            # forward
            outputs = model(images)
            # accuracy & loss
            val_loss += criterion(outputs, lables).item()
            predicted = torch.max(outputs.data, 1)[1]
            val_acc += (predicted == lables).sum()

    total_train_accuracy[epoch] = 100 * train_acc / len(trainset)
    total_val_accuracy[epoch] = 100 * val_acc / len(valset)
    total_train_loss[epoch] = train_loss / len(train_loader)
    total_val_loss[epoch] = val_loss / len(val_loader)
    print("Epoch number: {}/{} \n Loss: {}. Train Accuracy: {}. Val Accuracy: {}.\n".format(epoch, epochs,
                                                                                            total_train_loss[epoch],
                                                                                            total_train_accuracy[epoch],
                                                                                            total_val_accuracy[epoch]))

# test
test_accurecy = 0
total = 0

with torch.no_grad():  # in test we do not change and calc gradient
    for j, (images, lables) in enumerate(test_loader):
        images, lables = images.to(device), lables.to(device)
        # forward
        outputs = model(images)
        # accuracy & loss
        total += lables.size(0)
        predicted = torch.max(outputs.data, 1)[1]
        test_accurecy += (predicted == lables).sum()

print('Accuracy of CNN on test images is: %d %%' % (100 * test_accurecy / total))
# Plots

plt.figure(1)
plt.plot(total_train_loss)
plt.plot(total_val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train loss', 'val loss'])

plt.figure(2)
plt.plot(total_train_accuracy)
plt.plot(total_val_accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train data', 'val data'])

plt.show()