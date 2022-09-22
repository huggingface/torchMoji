from pyexpat import model
from statistics import mode
from turtle import forward
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# input_image = Image.open('data/image/train/ea3414d65ba84ec1b9e8aa74ec586434_15.png').convert('RGB')
preprocess = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# # print(input_image.shape)
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0)

# # Use Pretrained Alexnet
# model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
#

class alex(nn.Module):
    def __init__(self, model, labels=64) -> None:
        super().__init__()
        self.model = model
        self.fc = nn.Linear(1000,labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

class densenet(nn.Module):
    def __init__(self, model, labels=64) -> None:
        super().__init__()
        self.model = model
        self.fc = nn.Linear(1000,labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.model(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x


class inception(nn.Module):
    def __init__(self, model,labels=64) -> None:
        super().__init__()
        self.model = model
        self.fc = nn.Linear(1000,labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.model(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

        
# network = alex(model)
# network.eval()

# with torch.no_grad():
#     out = network(input_batch)

# # print(out.shape)
# probabilities = torch.nn.functional.sigmoid(out[0])
# top5 = torch.topk(probabilities,5)
# print(top5)