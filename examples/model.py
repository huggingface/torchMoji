from cgitb import grey
from random import shuffle
import pandas as pd
from skmultilearn.problem_transform import ClassifierChain
from sklearn.neural_network import MLPClassifier
import imp
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
from sklearn.metrics import f1_score
import numpy as np
import torchvision.transforms as transforms
import torch
from sklearn.model_selection import train_test_split
import time
import warnings
import glob
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from skimage import io, transform
import os
from label_data import IMAGE_PATH
from label_data import multi_label
from random_model import return_valid_frame


class SpectrogramDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with emoji labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.music_frame = return_valid_frame(pd.read_csv(csv_file))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.music_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.music_frame.iloc[idx, 0]+'.png')
        image = io.imread(img_name, as_gray=True)

        # One-hot encoded label from 2nd col onwards
        emo_lab = self.music_frame.iloc[idx, 2:]
        emo_lab = np.array([emo_lab])
        emo_lab = emo_lab.astype('float32')
        sample = {'image': image, 'emo_lab': emo_lab}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        image, emo_lab = sample['image'], sample['emo_lab']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        # image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).to(torch.float32).to(self.device),
                'emo_lab': torch.from_numpy(emo_lab).to(self.device)}

# Model Definition

class Net(nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.flatten = nn.Flatten()
        # self.conv1 = nn.Conv2d(4, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(480*640, 10250)
        self.fc2 = nn.Linear(10250, 840)
        self.fc3 = nn.Linear(840, labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


def random_accuracy_check():  

    # Setting target
    labels = np.zeros(62)
    labels[:5] = 1
    np.random.shuffle(labels)

    #Setting random predictions
    rand_arr = np.zeros(62)
    rand_arr[:5] = 1

    for TRIALS in [10,100,1000,10000,100000]:    
        ac_score = 0
        for i in range(TRIALS):  
            np.random.shuffle(rand_arr)
            correct = len([index for index, (e1, e2) in enumerate(zip(labels, rand_arr)) if e1 == e2 and e1 == 1])
            # new_labels = torch.Tensor(labels).unsqueeze(0)
            # new_rand_arr = torch.Tensor(rand_arr).unsqueeze(0)
            # print((new_labels == new_rand_arr))
            # acc = (new_labels == new_rand_arr).all(dim=1).float().mean()
            # print(acc)
            acc = correct/5
            ac_score += acc

        print(f"Random Set accuracy for {TRIALS} trials = {(ac_score/TRIALS)*100}")

    # Math based accuracy
    # favourable = 5c5*1 + 5c4.59c1.0.8 + 5c3.59c2.0.6 + 5c2.59c3.0.4 + 5c1.59c4.0.2 + 59c5.0.0
    # total = 64c5
    # accurate = (favourable / total)*100

    # print(f"Random set accuracy calculated using formula = {accurate}")

def model_training():

    device = torch.device('mps')

    # Load images and emoji labels
    music_dataset = SpectrogramDataset('real.csv', IMAGE_PATH,
                                        transform=transforms.Compose([ToTensor(device)]))

    # Fixed emoji labels to predict
    LABELS = 61
    # Create a train,validation and test split of the dataset
    dataset_size = len(music_dataset)
    train_split = int(0.7*dataset_size)
    val_split = int(0.2*dataset_size)
    test_split = dataset_size - (train_split+val_split)

    # print(dataset_size, train_split, val_split, test_split)

    train_sampler, val_sampler, test_sampler = random_split(music_dataset,
                                                         [train_split, val_split, test_split])

    train_dataloader = DataLoader(train_sampler, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_sampler, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_sampler, batch_size=1, shuffle=True)

    # dataiter = iter(train_dataloader)
    # images, labels = dataiter.next()['image'],dataiter.next()['emo_lab']

    # Display batched images
    # io.imshow_collection(torchvision.utils.make_grid(images))
    # io.show()
    # print(images.shape, labels.shape)
    
    # Declare/Initialize model
    net = Net(labels=LABELS)

    net.to(device)

    criterion = nn.MultiLabelMarginLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    traintime = time.time()

    # optimizer.zero_grad()

    # tre = net(images.float())

    # labels = labels.squeeze(dim=1)
    # labels = labels.long()
    # print(tre.shape, labels.shape)


    # print()

    # l = criterion(tre, labels)

    # l.backward()
    # optimizer.step()

    # print(type(l))
    # print(l)


    for epoch in range(2):  # loop over the dataset multiple times
        timeing = time.time()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            staring = time.time()

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'], data['emo_lab']
            # print(inputs.dtype)
            # inputs = inputs.to(torch.float32).to(device)
            # labels = labels.to(device)
            # print(inputs.shape, labels.shape)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())
            # outputs = outputs.type(torch.float64)
            labels = labels.squeeze(dim=1)
            labels = labels.long()

            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # if i % 2000 == 1999:    # print every 2000 mini-batches    
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')
                # running_loss = 0.0

            print(f"Time for 1 batch = {time.time() - staring}")
        print(f"Time for 1 epoch = {time.time() - timeing}")

    print('Finished Training')
    print(f"Time for training = {time.time()-traintime}")

    # # Save model here for re-use and testing
    # torch.save(net.state_dict(), "/Users/amanshukla/miniforge3/torchMoji/model/")
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data['image'], data['emo_lab']
            # calculate outputs by running images through the network
            outputs = net(images.float())
            # the class with the highest energy is what we choose as prediction
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            # correct += (outputs == labels).sum().item()
            # outputs = torch.nan_to_num(outputs)
            # score = f1_score(labels, outputs)

            _, pred_ind = torch.topk(outputs,5)

            _, tru_ind = torch.topk(labels,5)

            score = (tru_ind == pred_ind).float().mean()

            total += score
            # print(f'Mean Score is {score}')



    # print(f'Accuracy of the network on the test images: {100 * correct // total} %')
    print(f'Mean accuracy of the network on the test set is {(total / len(test_dataloader))* 100}%')


if __name__ == '__main__':
        
    # random_accuracy_check()
    model_training()

