import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import numpy as np
import torchvision.transforms as transforms
import torch
import time


IMAGE_PATH = "/Volumes/TOSHIBA EXT/DALI/images"

transform = transforms.Compose([ 
    # to-tensor
    transforms.ToTensor()
    # normalize
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

imageset = torchvision.datasets.ImageFolder(IMAGE_PATH, transform)

# print(type(imageset), len(imageset))

# print(imageset.find_classes)

dataset_size = len(imageset)
train_split = int(0.7*dataset_size)
val_split = int(0.2*dataset_size)
test_split = dataset_size - (train_split+val_split)

# print(dataset_size, train_split, val_split, test_split)

train_sampler, val_sampler, test_sampler = random_split(
    imageset, [train_split, val_split, test_split])

# print(len(train_sampler), len(val_sampler), len(test_sampler))
# Need new folder structure for test
train_dataloader = DataLoader(train_sampler, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_sampler, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_sampler, batch_size=64, shuffle=True)

# print(len(train_dataloader), len(val_dataloader), len(test_dataloader))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(293904, 10250)
        self.fc2 = nn.Linear(10250, 840)
        self.fc3 = nn.Linear(840, 64)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

traintime = time.time()
for epoch in range(2):  # loop over the dataset multiple times
    timeing = time.time()
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        staring = time.time()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0

        print(f"Time for 1 batch = {time.time() - staring}")
    print(f"Time for 1 epoch = {time.time() - timeing}")

print('Finished Training')
print(f"Time for training = {time.time()-traintime}")
# for batch_id, sample_id in enumerate(train_dataloader,0):
#     ip, lab = sample_id

#     print(lab)


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')
