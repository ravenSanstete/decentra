

import argparse
import os
import pprint
import time
import datetime

import torch
import torch.nn as nn
from torch.nn import Module, Linear, Parameter
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys
from pytorch_memlab import profile
from torchvision import datasets, transforms

from model import model_initializer
from feeder import CircularFeeder
from utils import load_dataset
from functools import reduce
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)



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
    return torch.randn_like(v_i) * sigma + torch.ones_like(v_i)* 0.1


def generate_random_fault(grad):
    return [random_injection(x).cuda() for x in grad]
    


class Worker:
    def __init__(self, wid, batch_generator, model, criterion, test_loader, batch_size = 32, lr = 0.01, role = True):
        self.wid = wid
        self.generator = batch_generator
        self.model = model
        self.criterion = criterion
        self.param = get_parameter(model)
        self.batch_size = batch_size
        self.grad = None
        logging.debug("Initialize Worker {} Byzantine: {}".format(wid, not role))
        self.cached_grads = list()
        self.lr = lr
        self.test_loader = test_loader
        self.running_loss = 0.0
        self.local_clock = 0
        self.prev_describe = 0
        self.role = role

        
    def local_iter(self):
        # logging.debug("Round {} Worker {} Local Iteration".format(T, self.wid, len(self.cached_grads)))
        x, y = self.generator.next(self.batch_size)
        x, y = x.cuda(), y.cuda()

        # copy parameter to the model from the current param
        copy_from_param(self.model, self.param)
        
        f_x = self.model(x)
        loss = self.criterion(f_x, y)
        self.running_loss += loss.data
        # loss = torch.mean(torch.clamp(torch.ones_like(y, dtype= torch.float32).cuda() - f_x.t() * y, min=0))
        
        self.model.zero_grad()
        loss.backward()  # with this line invoked, the gradient has been computed
        self.grad = get_grad(self.model)

        # if I am a Byzantine guy
        if(not self.role):
            self.grad = generate_random_fault(self.grad)
        return self.grad
    

    def receive(self, grad):
        # logging.debug("Round {} Worker {} Received {} Gradients".format(T, self.wid, len(self.cached_grads)))
        self.cached_grads.append(grad)

    def aggregate(self):
        # logging.debug("Round {} Worker {} Aggregate {} Gradients & Update".format(T, self.wid, len(self.cached_grads)))
        self.grad = reduce_gradients(self.cached_grads + [self.grad])
        self.param = [x - self.lr * y for x, y in zip(self.param, self.grad)]
        self.cached_grads.clear()
        self.local_clock += 1

    def evaluate(self, T):
        if(T > 0):
            span = T - self.prev_describe
            self.prev_describe = T
        else:
            span = 1
        # copy parameter to the model from the current param

        copy_from_param(self.model, self.param)
        acc = batch_accuracy_fn(self.model, self.test_loader)
        logging.debug("Round {} Worker {} Accuracy {:.4f} Loss {:.4f}".format(T, self.wid, acc, self.running_loss / span))
        self.running_loss = 0.0
        
        
        
        
        

    

if __name__ == '__main__':
    DATASET = "mnist"
    train_set, test_set = load_dataset(DATASET)
    train_loader = CircularFeeder(train_set, verbose = False)
    test_loader = torch.utils.data.DataLoader(test_set)
    criterion = F.cross_entropy
    BATCH_SIZE = 64
    model = model_initializer(DATASET)
    worker = Worker(wid = 0, batch_generator = train_loader, model = model, criterion = criterion)
    print(worker.location_iter(BATCH_SIZE))
    

    

