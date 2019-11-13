## simulate a centralized distributed learning system based on selective gradient sharing
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
from utils import *

import networkx as nx
import logging

from worker import Worker

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)


class Central:
    def __init__(self, workers, model, test_loader):
        self.workers = workers
        self.theta = get_parameter(model)
        self.cached_grads = []
        self.grad = None
        self.model = model
        self.test_loader = test_loader

    def _local_iter(self):
        for w in self.workers:
            w.local_iter()

    def _aggregate(self):
        for w in self.workers:
            self.cached_grads.append(w.send())
        self.grad = reduce_gradients(self.cached_grads)

    def _update(self, lr):
        self.theta = weighted_reduce_gradients([self.theta, self.grad], [1, -lr])

    def _distribute(self):
        for w in self.workers:
            w.central_receive(self.theta)

    def one_round(self, lr = 0.01):
        self._distribute()
        self._local_iter()
        self._aggregate()
        self._update(lr)

    def evaluate(self):
        copy_from_param(self.model, self.theta)
        return batch_accuracy_fn(self.model, self.test_loader)


    def execute(self, max_round = 1000):
        PRINT_FREQ = 100
        for i in range(max_round):
            self.one_round()
            if(i % PRINT_FREQ == 0):
                acc = self.evaluate()
                logging.debug("Round {} Accuracy {:.4f}".format(i, acc))



def initialize_sys(dataset = "mnist", worker_num = 10):
    batch_size = 32
    logging.debug("Construct a Centralized DDL System {}".format(dataset))
    train_set, test_set = load_dataset(dataset)
    train_loader = CircularFeeder(train_set, verbose = False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size)
    criterion = F.cross_entropy
    model = model_initializer(dataset)
    model.cuda()
    workers = []
 
    for i in range(worker_num):
        workers.append(Worker(i, train_loader, model, criterion, test_loader, batch_size, role = True))
    
    system = Central(workers, model, test_loader)
    system.execute(max_round = 1000)
                
                
            
        


if __name__ == '__main__':
    DATASET = "mnist"
    initialize_sys(DATASET)
    
            
        
        
        
        
        
        
            
        
