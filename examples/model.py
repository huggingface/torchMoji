from cgi import test
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
from sklearn.metrics import f1_score
import numpy as np
import torchvision.transforms as transforms
import torch
import time
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from skimage import io, transform
import os
# from label_data import IMAGE_PATH

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
        self.music_frame = return_valid_frame(pd.read_csv(csv_file),root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.music_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.music_frame.iloc[idx, 0]+'.png')
        image = io.imread(img_name, as_gray=False)

        # One-hot encoded label from 2nd col onwards
        emo_lab = self.music_frame.iloc[idx, 2:]
        emo_lab = np.array([emo_lab])
        emo_lab = emo_lab.astype('float32')
        sample = {'image': image, 'emo_lab': emo_lab}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

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
        # T = transforms.Resize(size=(30,40))
        # image = T(image)
        # Uncomment to use GPU
        # return {'image': torch.from_numpy(image).to(torch.float32).to(self.device),
        #         'emo_lab': torch.from_numpy(emo_lab).to(self.device)}
        
        return {'image': torch.from_numpy(image),
                'emo_lab': torch.from_numpy(emo_lab)}

# Model Definition

class Net(nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(4, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 8, 5)
        # Original Dimension = 16*117*157
        #Resized (64,128) = 16*13*29
        self.fc1 = nn.Linear(117*157*8, 1020)
        # Commented out due to memory issues on GPU
        self.fc2 = nn.Linear(1020, 300)
        self.fc3 = nn.Linear(300, labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        # x = self.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.sigmoid(x)
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

def model_training(LABELS, device, train_dataloader, epochs):
    
    # Declare/Initialize model
    net = Net(labels=LABELS)

    net.to(device)

    # Ref: https://stackoverflow.com/questions/64634902/best-loss-function-for-multi-class-multi-target-classification-problem
    # criterion = nn.MultiLabelMarginLoss()
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    traintime = time.time()


    for ep in range(epochs):  # loop over the dataset multiple times
        timeing = time.time()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            staring = time.time()

            # get the inputs; data is a list of [inputs, labels]
            inputs, label = data['image'], data['emo_lab']

            inputs = inputs.to(device)
            label = label.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs.float())
            label = label.squeeze(dim=1)

            loss = criterion(outputs, label)
            
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 100 == 0:    
                print(f'[{ep + 1}, {i + 1:5d}] loss: {running_loss/len(train_dataloader):.3f}')

        # if ep % 5 == 0:        
        #     # Save model here for re-use and testing
        #     torch.save(net.state_dict(), f"/Users/amanshukla/miniforge3/torchMoji/model/train_trial_sgd{ep}.ckpt")

    print('Finished Training')
    print(f"Time for training = {time.time()-traintime}")

    # Save model here for re-use and testing
    torch.save(net.state_dict(), f"/Users/amanshukla/miniforge3/torchMoji/model/temp_adam.ckpt")

    # correct = 0
    # total = 0
    # # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
    #     for data in test_dataloader:
    #         images, labels = data['image'], data['emo_lab']
    #         # calculate outputs by running images through the network
    #         # images = images.permute(0, 3, 1, 2)
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         labels = labels.squeeze(dim=1)
    #         outputs = net(images.float())
    #         # the class with the highest energy is what we choose as prediction
    #         # _, predicted = torch.max(outputs.data, 1)
    #         # total += labels.size(0)
    #         # correct += (predicted == labels).sum().item()
    #         # correct += (outputs == labels).sum().item()
    #         # outputs = torch.nan_to_num(outputs)
    #         # score = f1_score(labels, outputs)

    #         _, pred_ind = torch.topk(outputs,5)
            

    #         _, tru_ind = torch.topk(labels,5)



    #         score = (tru_ind == pred_ind).float().mean()
            
    #         pred_ind = pred_ind.cpu()
    #         tru_ind = tru_ind.cpu()
    #         # print(tru_ind)
    #         # print(pred_ind)
    #         batch_acc = 0
    #         for i in range(tru_ind.shape[0]):
    #             overlap = np.intersect1d(tru_ind[i], pred_ind[i])
    #             # print(tru_ind[i], pred_ind[i],overlap)
    #             batch_acc += len(overlap)/5

            
            

    #         # pred_ind = pred_ind.cpu().numpy()
    #         # tru_ind = tru_ind.cpu().numpy()
    #         # overlap = np.intersect1d(tru_ind, pred_ind)
    #         # val = len(overlap) / 5
    #         # print(overlap)
    #         # print(val)
    #         total += score
    #         correct += batch_acc/tru_ind.shape[0]

    #         # print(correct)

    #         # print(correct)
    #         # print(f'Mean Score is {score}')



    # # print(f'Accuracy of the network on the test images: {100 * correct // total} %')
    # print(f'Mean accuracy of the network on the test set (index based) is {(total / len(test_dataloader))* 100}%')
    # print(f'Mean accuracy of the network on the test set (intersection based) is {(correct / len(test_dataloader))* 100}%')

def model_test(LABELS, test_dataloader):

    net = Net(labels=LABELS)
    net.load_state_dict(torch.load("/Users/amanshukla/miniforge3/torchMoji/model/train_trial_sgd0.ckpt"))

    net.eval()

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data['image'], data['emo_lab']

            labels = labels.squeeze(dim=1)
            outputs = net(images.float())

            _, pred_ind = torch.topk(outputs,5)
            _, tru_ind = torch.topk(labels,5)

            # print(f"Top predictions {pred_ind}")
            
            batch_acc = 0
            for i in range(tru_ind.shape[0]):
                overlap = np.intersect1d(tru_ind[i], pred_ind[i])
                batch_acc += len(overlap)/5

            # print(pred_ind)
            # print(tru_ind)
            
            
            score = (tru_ind == pred_ind).float().mean()

            total += score
            correct += batch_acc/tru_ind.shape[0]


    print(f'Mean accuracy of the network on the test set (index based) is {(total / len(test_dataloader))* 100}%')
    print(f'Mean accuracy of the network on the test set (intersection based) is {(correct / len(test_dataloader))* 100}%')


if __name__ == '__main__':
        
    # random_accuracy_check()
    device = torch.device('mps')

    # Load images and emoji labels
    # music_dataset = SpectrogramDataset('real.csv', IMAGE_PATH,
    #                                     transform=transforms.Compose([
    #                                         # transforms.ToPILImage(),
    #                                         #                         transforms.Resize(size=(64,128)),
    #                                                                 transforms.ToTensor()])) #ToTensor(device)




    train_dataset = SpectrogramDataset('real.csv','data/image/train' ,
                                        transform=transforms.Compose([transforms.ToTensor()])) #ToTensor(device)

    val_dataset = SpectrogramDataset('real.csv','data/image/val' ,
                                        transform=transforms.Compose([transforms.ToTensor()])) #ToTensor(device)

    test_dataset = SpectrogramDataset('real.csv','data/image/test' ,
                                        transform=transforms.Compose([transforms.ToTensor()])) #ToTensor(device)

    # print(len(train_dataset))
    # print(len(val_dataset))
    # print(len(test_dataset))
    # Fixed emoji labels to predict
    LABELS = 61
    # Create a train,validation and test split of the dataset
    # dataset_size = len(music_dataset)
    # train_split = int(0.7*dataset_size)
    # val_split = int(0.2*dataset_size)
    # test_split = dataset_size - (train_split+val_split)

    # # print(dataset_size, train_split, val_split, test_split)

    # train_sampler, val_sampler, test_sampler = random_split(music_dataset,
    #                                                      [train_split, val_split, test_split])

    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True)

    # model_training(LABELS, device, train_dataloader, 5)
    # model_training(LABELS, device, train_dataloader, 15)
    # model_training(LABELS, device, train_dataloader, 15)
    model_test(LABELS, train_dataloader)
    model_test(LABELS, val_dataloader)
    model_test(LABELS, test_dataloader)

