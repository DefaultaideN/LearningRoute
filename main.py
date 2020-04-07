import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader as Dataloader
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim # Optimizer used to update weights.
import numpy as np
import matplotlib.pyplot as plt
#Download Dataset

torch.set_printoptions(linewidth=120)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()



#Show images and labels
'''

batch = next(iter(train_loader))
images, labels = batch
grid = torchvision.utils.make_grid(images, nrow=10)
plt.figure(figsize=(15,15))
plt.imshow(np.transpose(grid, (1,2,0)))

print('labels:', labels)
plt.show()
'''
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        #implement the forward pass

        #(1)input layer
        t = t   # t.shape = (1,28,28)

        #(2)hidden conv layer
        t = self.conv1(t)   #t.shape (1,28,28) -> (6,24,24)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        #(3)hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        #(4)hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        #(5)hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        #(6)output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)

        return t


train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    #download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
#/---------------------Training with a single batch-----------------/
#Step1: Initialize a network.
network = Network()
network = network
#Step2: General setup: Load data. Setup optimizer. Initialize total loss and correct.
train_loader = Dataloader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01) #lr: learning rate.

batch_num = 0

#Step3: Get batch from the train_loader
#batch = next(iter(train_loader)) #Train with single batch.
for epoch in range(5):

    total_loss = 0
    total_correct = 0

    for batch in train_loader:
        batch_num += 1
        images, labels = batch
        images = images
        labels = labels

        #Step4: Calculating loss by predicting and compare to the labels.
        preds = network(images)
        loss = F.cross_entropy(preds, labels)
        #print('Num correct before:'+str(get_num_correct(preds, labels)))

        #Step5: Backward propogation.
        optimizer.zero_grad() #Zero out gradients. Otherwise it would accumulate.
        loss.backward() #calculate gradients.
        optimizer.step()#update weights using gradients

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

        #print(
         #   'epoch', epoch, 'batch', batch_num, '| correct:', get_num_correct(preds, labels),
          #  'loss:', loss.item())

    print('Finished epoch', epoch, '| total_correct:', total_correct, 'loss:', total_loss)
'''
print('Loss before: '+str(loss.item()))
print('Num correct before:'+str(get_num_correct(preds, labels)))
preds = network(images)
loss = F.cross_entropy(preds, labels) #New loss here.
print('Loss after: '+str(loss.item()))
print('Num correct after:'+str(get_num_correct(preds, labels)))
'''
#/------------------------------------------------------------------/
