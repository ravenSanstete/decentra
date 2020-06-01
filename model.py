import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.resnet import ResNet18





## A Translation of Leaf Model
'''
    def __init__(self, seed, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        """Model function for CNN."""
        features = tf.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')
        input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=self.num_classes)

'''
class FEMNISTModel(nn.Module):
    def __init__(self, num_classes = 62):
        super(FEMNISTModel, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 32,(5, 5))
        self.pool = nn.MaxPool2d((2, 2), stride = 2)
        self.conv2 = nn.Conv2d(32, 64, (5, 5))
        self.linear1 = nn.Linear(4 * 4 * 64, 2048)
        self.linear2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        # print(x.size())
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # print(x.size())
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
        
        

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
    elif dataset == 'femnist':
        return FEMNISTModel()


if __name__ == '__main__':
    fc = Model()
    for module in fc.children():
        print(module._parameters["weight"])
        
