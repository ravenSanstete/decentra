## simulate a centralized distributed learning system based on selective gradient sharing (synchronized)
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
from functools import partial

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
        self.update_counter = torch.cat([torch.zeros_like(p.flatten()) for p in self.theta])
 
        
    def _local_iter(self):
        for w in self.workers:
            w.local_iter()

    def _receive(self, mechanism = None):
        for w in self.workers:
            grad = w.send()
            if(mechanism):
                grad, idx = mechanism(grad)
                self.update_counter[idx] += 1
            self.cached_grads.append(grad)
    
    def _aggregate(self):
        self.grad = reduce_gradients(self.cached_grads)
        self.cached_grads.clear()

    def _update(self, lr):
        self.theta = weighted_reduce_gradients([self.theta, self.grad], [1, -lr])

    def _distribute(self):
        for w in self.workers:
            w.central_receive(self.theta)

    def _selective_distribute(self, ratio_d = 1.0):
        for w in self.workers:
            w.central_receive(share_frequent_p_param(self.theta, self.update_counter, ratio = ratio_d))
            # print(selected[-1])

            
    def one_round(self, lr = 0.01):
        # self._distribute()
        self._distribute()
        self._local_iter()
        self._receive()
        self._aggregate()
        self._update(lr)

    def selective_gradient_sharing(self, lr = 0.01):
        ratio_r = 0.001
        sharing_mec = partial(share_largest_p_param, ratio = ratio_r)
        self._selective_distribute(ratio_d = 1.0)
        self._local_iter()
        self._receive(sharing_mec)
        self._aggregate()
        self._update(lr)
        
        
    def evaluate(self):
        copy_from_param(self.model, self.theta)
        return batch_accuracy_fn(self.model, self.test_loader)


    def execute(self, max_round = 1000):
        PRINT_FREQ = 100
        for i in range(max_round):
            # self.one_round()
            self.selective_gradient_sharing()
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
    system.execute(max_round = 10000)
                
                
            
        


if __name__ == '__main__':
    DATASET = "mnist"
    initialize_sys(DATASET)
    
            
        
        
        
        
        
        
            
        
