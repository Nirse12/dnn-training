
# -*- coding: utf-8 -*-
"""

@author: Nir Segal & Noam Baron & Sarah Elysayev

For better understanding we used the code here:
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""



import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import sklearn.model_selection as ms

from torchvision import models

"""Seed for random purposes"""

seed = 50
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed) 

device = torch.device('cpu')

"""Parameters"""

image_size = 32*32*3
batch_size = 128 
test_batch_size = 128
epochs = 10
lr = 0.01  
weight_decay = 0.002
optimizer_type = 'SGD' # SGD, Adam, RMSProp
momentum = 0.0
mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)


model_name = "resnet"
num_classes = 10
feature_extract = False

"""Choose between fine-tuning and without fine-tuning and model initializing

"""

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    model_name == "resnet"
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes) # resnet to our output
    input_size = 224 

    return model_ft, input_size

"""Importing resnet """

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

"""Might not be relevant

"""

# Create our own layers
my_layers = nn.Sequential(
    nn.ReLU(),
)

# Create a new model - combined resenet and our own layers
my_model = nn.Sequential(
    model_ft,
    my_layers,
)

"""Download, recieve and distribute the data to the train set and test set."""

transform = transforms.Compose([transforms.Resize(input_size), transforms.RandomCrop(32),transforms.ColorJitter(hue=.05, saturation=.05),
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

dataset = datasets.stl10.STL10(root='./data', split='train',download=True, transform=transform) #32x32x3

num_workers = 4 # Set to 4 when working in computational rich environments. 


train_set, val_set =  ms.train_test_split(dataset, train_size=0.8, test_size=0.2)
train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True,num_workers=num_workers)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

# Test
test_transform = transforms.Compose([transforms.Resize(input_size), transforms.CenterCrop(32),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

                                
test_dataset = datasets.stl10.STL10(root="data", split="test",download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=4)

"""Running the model on the GPU
Defining a loss function
Choosing between training the whole new or just the extra layers 
"""

Resnet18 = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

params_to_update = Resnet18.parameters()

"""Set optimizer"""

# optimizer type
if optimizer_type == 'SGD':
    optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)
elif optimizer_type == 'Adam':
    optimizer = optim.Adam(params_to_update, lr=lr,betas=(0.9, 0.999), weight_decay=weight_decay) 
elif optimizer_type == 'RMSProp':
    optimizer = optim.Adam(params_to_update, lr=lr, eps=1e-08, weight_decay=weight_decay)  # alpha = 0.99 default
else:
    NotImplementedError("optimizer not implemented")

"""Training the net"""

total_train_loss, total_train_accuracy, total_val_loss, total_val_accuracy = ([] for i in range(4))  
Resnet18.train() # Train mode
for epoch in range(epochs):  
    loss = 0
    batch_accuracy = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device) , labels.to(device)  
        optimizer.zero_grad()
        outputs = Resnet18(data)
        batch_loss = criterion(outputs, labels)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        y_pred = torch.max(outputs.data, 1)[1]
        batch_accuracy += (y_pred == labels).sum() 
        
    num_batches = len(train_loader)
    total_train_loss.append(loss/num_batches)
    total_train_accuracy.append(100*batch_accuracy/len(train_set)) 
    num_val_batches = len(val_loader)
    Resnet18.eval()    
    val_loss = 0
    val_batch_acc = 0
  
    with torch.no_grad(): 
        for batch_idx, (data, labels) in enumerate(val_loader):
            data, labels = data.to(device), labels.to(device)
            outputs = Resnet18(data)
            val_loss += criterion(outputs,labels).item()
            y_pred = torch.max(outputs.data, 1)[1]  
            val_batch_acc+=(y_pred == labels).sum() 
    
    total_val_loss.append(val_loss/num_val_batches)
    total_val_accuracy.append(100*val_batch_acc/len(val_set))
    print("Epoch number: {}/{} \n Loss: {}. Train Accuracy: {}. Val Accuracy: {}. Val Loss: {}\n".format(epoch, epochs,
                                                                                            total_train_loss[epoch],
                                                                                            total_train_accuracy[epoch],
                                                                                            total_val_accuracy[epoch],
                                                                                            total_val_loss[epoch]))

steps = np.arange(epochs)

"""Test"""

test_acc = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device) , labels.to(device)
        outputs = Resnet18(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        test_acc += (predicted == labels).sum().item()

print('Accuracy of the Resnet18 : %d %%' % (100 * test_acc / total))
"""Plots"""

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