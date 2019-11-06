import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.resnet import ResNet18

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 32)
        self.linear2 = nn.Linear(32, 10)

    def forward(self, inputs):
        x = inputs.view(-1, 28 * 28)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class MedicalModel(nn.Module):
    def __init__(self):
        super(MedicalModel, self).__init__()
        self.linear1 = nn.Linear(1024, 32)
        self.linear2 = nn.Linear(32, 10)

    def forward(self, inputs):
        x = inputs.view(-1, 1024)
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x
    

class YelpModel(nn.Module):
    def __init__(self):
        super(YelpModel, self).__init__()
        self.linear1 = nn.Linear(1024, 10)
        self.linear2 = nn.Linear(10, 2)

    def forward(self, inputs):
        x = inputs.view(-1, 1024)
        # x = self.linear1(x)
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x
    

    
class Cifar10Model(nn.Module):
    def __init__(self):
        super(Cifar10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def model_initializer(dataset, arch = 'resnet18'):
    if dataset == 'mnist':
        return Model()
    elif dataset == 'cifar10':
        return Cifar10Model()
    elif dataset == 'imagenet': # first implement the resnet version
        return models.__dict__[arch]().train()
    elif dataset == 'medical':
        return MedicalModel()
    elif dataset == 'yelp':
        return YelpModel()
    elif dataset == 'cifar10-large':
        return ResNet18()


if __name__ == '__main__':
    fc = Model()
    for module in fc.children():
        print(module._parameters["weight"])
        
