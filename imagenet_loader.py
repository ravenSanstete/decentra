## to convert the imagenet data to 3*32*32

import math
import sys
import os
from functools import reduce
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import csv
from torchvision import datasets, transforms
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = "/home/mlsnrs/data/data/pxd/imagenet/raw/val"


def to_img(x):
    x = x.view(x.size(0), 3, 32, 32)
    std = torch.FloatTensor([0.5, 0.5, 0.5])
    mean = torch.FloatTensor([0.5, 0.5, 0.5])
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    x = torch.FloatTensor(np.array([unnormalize(x[i, :, :, :]).numpy() for i in range(x.size(0))]))
    # x = x.clamp(0, 1)
    return x


def show(img, name):
    fig, ax = plt.subplots(figsize=(20, 10))
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1,2,0)))
    plt.savefig("result_{}.png".format(name))
    print("Plot in {}".format("result_{}.png".format(name)))
    plt.close(fig)

    
        
def transform_mnist_to_cifar(x):
    x = torch.repeat_interleave(x, 3, dim = 1)
    x = F.interpolate(x, size = (32, 32))
    return x

def transform_stl_to_cifar(x):
    x = F.interpolate(x, size = (32, 32))
    return x

### 50000 
def get_imagenet_loader(batch_size = 32):
    transform = transforms.Compose(
                [transforms.RandomResizedCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    val_set = datasets.ImageFolder(DATA_PATH, transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = 12, drop_last=False, pin_memory = True)
    return val_loader


def dump_imagenet(save_path = 'imagenet.npy'):
    loader = get_imagenet_loader(batch_size = 10)
    out = []
    
    # then convert the image to be of the 3, 32, 32 size
    for i, batch in enumerate(loader):
        x, y = batch
        out.append(transform_stl_to_cifar(x))
    out = np.concatenate(out, axis = 0)
    print("save imagenet size {}".format(out.shape))
    np.save(save_path, out)


if __name__ == '__main__':
    # convert_imagenet_data()
    dump_imagenet()
    
    
