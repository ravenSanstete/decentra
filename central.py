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
import torch.utils.data as data_utils
from itertools import cycle
from tensorboardX import SummaryWriter
import scipy


from model import model_initializer
from feeder import CircularFeeder
from utils import *
from functools import partial

import networkx as nx
import logging

from worker import Worker

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, filemode = 'w+')


from worker_config import initialize_mode
from options import parser

# writer = SummaryWriter()



ARGS = parser.parse_args()

logger = logging.getLogger('server_logger')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('logs/trace_{}.log'.format(ARGS.eta_d), mode='w+')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logging = logger

def write_group(writer, scalars, name, i):
    writer.add_scalars('data/{}'.format(name), {'worker_{}'.format(j): scalars[j] for j in range(len(scalars))}, i)
    return
    



class Central:
    def __init__(self, workers, model, test_loader, qv_loader, criterion, lr, eta_d = 1.0, eta_r = 1.0, gamma = 0.0005):
        self.workers = workers
        self.theta = get_parameter(model)
        self.cached_grads = []
        self.grad = None
        self.model = model
        self.test_loader = test_loader
        self.update_counter = torch.cat([torch.zeros_like(p.flatten()) for p in self.theta])
        # the prop. of params for downloading
        self.eta_d = eta_d
        # the prop. of params for uploading
        self.eta_r = eta_r
        self.qv_loader = qv_loader
        self.criterion = criterion
        self.lr = lr
        self.writer = SummaryWriter()
        # gamma is the gradient norm penalty
        self.gamma = gamma
        self.accumulated_selection = np.zeros((len(self.workers),))

        # the downloadable parameter per worker
        self.eta_d_vectors = [self.eta_d]*len(self.workers)
        # the distribution mechanism


    def _local_iter(self):
        for w in self.workers:
            w.local_iter()

    def _receive(self, mechanism = None):
        for w in self.workers:
            self.cached_grads.append(w.send())
       
    def _aggregate(self, i, mode = 'none'):
        self.grad = reduce_gradients(self.cached_grads)
        # print(self.grad[-1])
        self.cached_grads.clear()

    def _update(self, lr):
        self.theta = weighted_reduce_gradients([self.theta, self.grad], [1, -lr])

    def _distribute(self):
        for w in self.workers:
            w.central_receive(self.theta, replace = True)

    def _selective_distribute(self, ratio_d, replace):
        for i, w in enumerate(self.workers):
            w.central_receive(share_frequent_p_param(self.theta, self.update_counter, ratio = self.eta_d_vectors[i]), replace)
            # print(selected[-1])

            
    def one_round(self, T):        
        self._local_iter()
        self._receive()
        self._aggregate(T)
        self._update(self.lr)
        self._distribute()


    # @param i: the current iteration id
    def selective_gradient_sharing(self, i):
        sharing_mec = partial(share_largest_p_param, ratio = self.eta_r)
        self._local_iter()
        self._receive(sharing_mec)
        self._aggregate(i)
        self._update(self.lr)
        self._selective_distribute(ratio_d = self.eta_d, replace = True)


        
    def heartbeat(self, T):
        worker_acc = []
        for idx, w in enumerate(self.workers):
            acc, _ = w.evaluate(T)
            worker_acc.append(acc)

        
    def evaluate(self):
        copy_from_param(self.model, self.theta)
        return batch_accuracy_fn(self.model, self.test_loader)


    def execute(self, max_round = 1000):
        PRINT_FREQ = 100
        # first distribute the theta_0 to all workers
        self._distribute()
        for i in range(max_round):
            self.one_round(i)
            # self.selective_gradient_sharing(i)
            if(i % PRINT_FREQ == 0):
                self.heartbeat(i)
                acc = self.evaluate()
                logging.debug("Round {} Global Accuracy {:.4f}".format(i, acc))
                



def random_sampler(batches):
    while True:
        yield random.choice(batches)



def initialize_sys(dataset = "mnist", worker_num = 1, eta_d = 1.0, eta_r = 1.0):
    batch_size = ARGS.bs
    gamma = 0
    MODE = ARGS.mode
    base_data_size = ARGS.N
    lr = ARGS.lr
    default_role = 'NO_UPDATE'
    
    
    logging.debug("Construct a Centralized DDL System {} Mode: {}".format(dataset, MODE))

    train_set, test_set = load_dataset(dataset)
    train_loader = CircularFeeder(train_set, verbose = False)


    train_loaders, roles = initialize_mode(MODE, worker_num, train_loader, base_data_size, batch_size, default_role)
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size)
    criterion = F.cross_entropy
    model = model_initializer(dataset)
    model.cuda()
    workers = []
     
    for i in range(worker_num):
        workers.append(Worker(i, train_loaders[i], model, criterion, test_loader, batch_size, role = roles[i], lr = lr))


    system = Central(workers, model, test_loader, None, criterion, lr, eta_d = eta_d, eta_r = eta_r, gamma = gamma)
    system.execute(max_round = 10000)
                


def confidence_stat(probs_out):
    probs_out = np.max(probs_out, axis = 1)
    hist = np.histogram(probs_out, bins = 10)
    return hist
    
            
        


if __name__ == '__main__':
    DATASET = ARGS.ds
    initialize_sys(DATASET, worker_num = ARGS.n, eta_d = ARGS.eta_d, eta_r = ARGS.eta_r)
    
            
        
        
        
        
        
        
            
        
