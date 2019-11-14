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


def load_validation_set(name, val_size = 1, binary = False):
    if(name == 'mnist'):
        return load_mnist_validation_set(val_size, binary)
    elif(name == 'cifar10'):
        return load_cifar10_validation_set(val_size, binary)
    else:
        return None
    
    

# load a bunch of flat 1024-dim vectors for validation
def load_mnist_validation_set(val_size = 100, binary = False):
    train_set = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    # create a data loader
    val_loader = torch.utils.data.DataLoader(train_set, batch_size = val_size, shuffle = True)
    for batch in val_loader:
        x, y = batch
    x = F.interpolate(x, [32, 32])
    x = x.reshape(-1, 32 * 32)
    if(binary):
        y = torch.LongTensor((y.numpy() == 0).astype(np.int))
    return x, y

def load_cifar10_validation_set(val_size = 100, binary = False):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = datasets.CIFAR10(root='../data/cifar10', train=True,
                                     download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(train_set, batch_size = val_size, shuffle = True)
    for batch in val_loader:
        x, y = batch
    x = x[:, 0, :, :]
    x = x.reshape(-1, 32 * 32)
    if(binary):
        y = torch.LongTensor((y.numpy() == 0).astype(np.int))
    return x, y 
# this function is somehow a magic operation (I do not know why we need to decompose the sign and  abs value)
def preprocess_gradients(x):
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

    return torch.cat((x1, x2), 1)


# datasets loader
def load_dataset(name):
    # to sample the batch in parallel by cpu
    if name == 'mnist':
        train_set = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
        test_set = datasets.MNIST('../data/mnist/', train=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))
    elif name in ['cifar10', 'cifar10-large']:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_set = datasets.CIFAR10(root='../data/cifar10', train=True,
                                     download=True, transform=transform)

        test_set = datasets.CIFAR10(root='../data/cifar10', train=False,
                                    download=True, transform=
                                    transform)
    else:
        raise NotImplementedError
    return train_set, test_set

def check_dirs(dirs):
    if os.path.exists(dirs):
        import shutil
        shutil.rmtree(dirs, ignore_errors=True)
    os.makedirs(dirs)

def medical_data_build_all(raw_x, raw_y):
    x = torch.from_numpy(raw_x)
    y = torch.from_numpy(raw_y)
    return x, y
    
def medical_data_build(raw_x, raw_y, batch_size = 64, balanced = False):
    if(batch_size > len(raw_x)):
        return torch.from_numpy(raw_x), torch.from_numpy(raw_y)
    
    if(not balanced):
        rand_index = np.random.choice(len(raw_x), batch_size, replace = False)
        raw_x = raw_x[rand_index, :]
        raw_y = raw_y[rand_index]
    else:
        idx_0 = (raw_y == 0)
        idx_1 = (raw_y == 1)
        half_batch_size = batch_size // 2
        rand_idx_0 = np.random.choice(len(raw_x[idx_0, :]), half_batch_size, replace = False)

        rand_idx_1 = np.random.choice(len(raw_x[idx_1, :]), half_batch_size, replace = False)
        raw_x_0 = (raw_x[idx_0, :])[rand_idx_0, :]
        raw_x_1 = (raw_x[idx_1, :])[rand_idx_1, :]
        raw_y_0 = (raw_y[idx_0])[rand_idx_0]
        raw_y_1 = (raw_y[idx_1])[rand_idx_1]
        
        raw_x = np.concatenate([raw_x_0, raw_x_1], axis = 0)
        raw_y = np.concatenate([raw_y_0, raw_y_1], axis = 0)
    
    
        
    # build data
    x = torch.from_numpy(raw_x)
    y = torch.from_numpy(raw_y)
    return x, y


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


## other model manipulation functions
def get_parameter(model):
    return [param.clone().detach() for param in model.parameters()]

def get_grad(model):
    return [param.grad.data.clone().detach() for param in model.parameters()]

def copy_from_param(model, parameters):
    for a, b in zip(model.parameters(), parameters):
        a.data.copy_(b.data)

def reduce_gradients(grads):
    n = len(grads)
    f = lambda x, y: [xx + yy for xx, yy in zip(x, y)]
    out = list(reduce(f, grads))
    return [x/n for x in out]

def weighted_reduce_gradients(grads, w):
    n = len(grads)
    out = [w[0] * x for x in grads[0]]
    for i in range(1, n):
        out = [xx + w[i]*yy for xx, yy in zip(out, grads[i])]
    return out


def split(flattened, params_example):
    out = []
    c = 0
    for param in params_example:
        out.append(flattened[c:c+param.numel()].reshape(param.shape))
        c += param.numel()
    return out

def share_largest_p_param(param, ratio = 0.1):
    # first flatten the parameter
    flattened = torch.cat([x.flatten() for x in param])
    # then select the topk index
    num = int(ratio * len(flattened))
    _,idx = torch.abs(flattened).topk(num)
    mask = torch.zeros_like(flattened)
    mask[idx] = 1
    mask = mask.cuda()
    flattened = mask * flattened
    return split(flattened, param), idx
     
def share_frequent_p_param(param, counter, ratio):
    # first flatten the parameter
    flattened = torch.cat([x.flatten() for x in param])
    # then select the topk index
    num = int(ratio * len(flattened))
    _,idx = torch.abs(counter).topk(num)
    mask = torch.zeros_like(flattened)
    mask[idx] = 1
    mask = mask.cuda()
    flattened = mask * flattened
    return split(flattened, param)
    

def batch_accuracy_fn(model, data_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return correct / total


def random_injection(v_i, sigma=2e-6): # 2e-6
    return torch.randn_like(v_i) * sigma + torch.ones_like(v_i)* 1.0


def generate_random_fault(grad):
    return [random_injection(x).cuda() for x in grad]

def param_distance(paramA, paramB):
    loss = [F.mse_loss(xx, yy).cpu().detach().numpy() for xx, yy in zip(paramA, paramB)]
    return np.mean(loss)



if __name__ == '__main__':
    # load_mnist_validation_set(100)
    load_cifar10_validation_set(1)
