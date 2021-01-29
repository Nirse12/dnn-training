
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:29:07 2021
@author: segalni2
"""
# inputs and setting

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as model_selection
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

batchSize = 32
test_batch_size = 128
epochs = 30
optimizer_type = 'SGD' # SGD, Adam, RMSProp
lr = 0.001
weight_decay = 0
momentum = 0.9
mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
device = torch.device("cpu")  # using biuserver not colab

# https://www.programcreek.com/python/example/104838/torchvision.transforms.RandomCrop

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


# https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19

class LogisticRegression(nn.Module):
    def __init__(self, input_dim=1024 * 3, output_dim=10):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

    # Training


# initialize our Logistic Regression Model
model = LogisticRegression(32 * 32 * 3).to(device)

# instaminate the loss class
criterion = nn.CrossEntropyLoss()


"""Set optimizer"""

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
        data = torch.flatten(images, start_dim=1)
        images, lables = data.to(device), lables.to(device)

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
            data = torch.flatten(images, start_dim=1)
            images, lables = data.to(device), lables.to(device)
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

"""#### Plots"""

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

test_accurecy = 0
total = 0

with torch.no_grad():  # in test we do not change and calc gradient
    for j, (images, lables) in enumerate(test_loader):
        data = torch.flatten(images, start_dim=1)
        images, lables = data.to(device), lables.to(device)
        # forward
        outputs = model(images)
        # accuracy & loss
        total += lables.size(0)
        predicted = torch.max(outputs.data, 1)[1]
        test_accurecy += (predicted == lables).sum()

print('Accuracy of Logistic Regression on test images is: %d %%' % (100 * test_accurecy / total))