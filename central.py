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

import argparse


# writer = SummaryWriter()



parser = argparse.ArgumentParser(description='Fairness')
parser.add_argument("--eta_d", type=float, default=1.0, help = "the proportion of downloadable parameters")
parser.add_argument("-n", type=int, default=10, help = "the number of workers")
parser.add_argument("--eta_r", type=float, default=1.0, help = "the proportion of uploading parameters")
parser.add_argument("--ds", type=str, default="cifar10", help = "the benchmark we use to test")
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
        self.credit_ratio_map = np.array(list(np.arange(0, 1, 1/len(self.workers))[1:]) + [1.0])
        logging.info("Credit Ratio Map:{}".format(self.credit_ratio_map))
        

    def _one_order_score(self, grad, lr):
        # obtain the data in the qv_loader
        x, y = next(self.qv_loader)
        x, y = x.cuda(), y.cuda()
        # then compute the previous loss
        copy_from_param(self.model, self.theta)
        prev_loss = self.criterion(self.model(x), y).data
        temp_theta = weighted_reduce_gradients([self.theta, grad], [1, -lr])
        copy_from_param(self.model, temp_theta)
        cur_loss = self.criterion(self.model(x), y).data
        # return back
        copy_from_param(self.model, self.theta)
        
        return prev_loss - cur_loss

    def krum_score(self):
        # to calculate the krum score (or geomed)
        pass
        
        
          
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

    def _assign_credit(self, i, gamma = 0.0):
        worker_contrib = []
        for idx, grad in enumerate(self.cached_grads):
            score = self._one_order_score(grad, self.lr)
            # add the penalty on gradient norm
            score -= gamma * l2_norm(grad)
            worker_contrib.append(score.cpu().data)

        write_group(self.writer, worker_contrib, 'contrib', i)
        # the workers' contribution
        return worker_contrib
            # logging.info("Worker {} Contrib. {}".format(idx, score))

            
        
    ## the problem of zeno: only assign the most credulous worker with the 1.0 weight
    def _zeno(self, contrib):
        weights = np.zeros((len(contrib),))
        max_idx = np.argmax(contrib)
        weights[max_idx] = 1.0
        return weights

    
        
        
    def _aggregate(self, i, mode = 'none'):
        worker_contrib = self._assign_credit(i, gamma = self.gamma)
        self.accumulated_selection += np.array(worker_contrib)
        period = 200
        # using the fairness mechanism
        if(mode == 'zeno'):
            # now I have the worker contribution
            weights = self._zeno(worker_contrib)
            write_group(self.writer, self.accumulated_selection, 'counter', i)
            # logging.info(worker_contrib)
            # logging.info(weights)
            self.grad = weighted_reduce_gradients(self.cached_grads, weights)
        elif(mode == 'dfair' and i % period == period - 1):

            # sort the worker_contrib
            sorted_indices = np.argsort(worker_contrib, axis = 0).tolist()
            self.prev_eta_d_vectors = list(self.eta_d_vectors)
            self.eta_d_vectors = [self.credit_ratio_map[sorted_indices.index(idx)] for idx in range(len(self.workers))]
            write_group(self.writer, self.eta_d_vectors, 'eta_d', i)
            write_group(self.writer, self.accumulated_selection, 'accumul', i)
            logging.info("Adjust the Credit and Eta Ratio from {} -> {}".format(self.prev_eta_d_vectors, self.eta_d_vectors))
            self.grad = weighted_reduce_gradients(self.cached_grads, worker_contrib)
        else:
            write_group(self.writer, self.eta_d_vectors, 'eta_d', i)
            write_group(self.writer, self.accumulated_selection, 'accumul', i)
            self.grad = reduce_gradients(self.cached_grads)
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
        # self._distribute()
        self._distribute()
        self._local_iter()
        self._receive()
        self._aggregate()
        self._update(self.lr)


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
        write_group(self.writer, worker_acc, 'acc', T)

        
    def evaluate(self):
        copy_from_param(self.model, self.theta)
        return batch_accuracy_fn(self.model, self.test_loader)


    def execute(self, max_round = 1000):
        PRINT_FREQ = 100
        # first distribute the theta_0 to all workers
        self._distribute()
        for i in range(max_round):
            # self.one_round()
            self.selective_gradient_sharing(i)
            if(i % PRINT_FREQ == 0):
                self.heartbeat(i)
                acc = self.evaluate()
                self.writer.add_scalar('data/global_acc', acc, i)
                logging.debug("Round {} Global Accuracy {:.4f}".format(i, acc))
                


def random_sampler(batches):
    while True:
        yield random.choice(batches)



def initialize_sys(dataset = "mnist", worker_num = 1, eta_d = 1.0, eta_r = 1.0):
    batch_size = 128
    gamma = 0
    logging.debug("Construct a Centralized DDL System {} Download Ratio: {:.4f} Upload Ratio {:.4f}".format(dataset, eta_d, eta_r))
    train_set, test_set = load_dataset(dataset)
    train_loader = CircularFeeder(train_set, verbose = False)
    #
    worker_data_size = 6000
    has_label_flipping = False
    group_count = 1
    base_data_size = 5000

    # initialize train loaders 
    train_loaders = []

    logging.debug("Initizalize training loaders {}".format("NORMAL" if not has_label_flipping else "NOISY"))
    if(has_label_flipping):
        ratios = np.arange(0, 1.0, 1.0/worker_num)
        # do a copy of oneself
        for i in range(len(ratios)//group_count):
            for j in range(group_count):
                ratios[group_count*i + j] = ratios[group_count*i]
        worker_data_sizes = [worker_data_size for i in range(worker_num)]
    else:
        ratios = np.array([0]*worker_num)
        worker_data_sizes = [base_data_size for i in range(1, worker_num+1)]
    # print(ratios)


    # for the central server to estimate the credibility
    qv_batch_size = 64
    
    
    qv_loader = data_utils.TensorDataset(*train_loader.next(qv_batch_size))
    qv_loader = data_utils.DataLoader(qv_loader, batch_size = qv_batch_size)
    qv_loader = cycle(list(qv_loader))
    lr = 0.1

    
    for i in range(worker_num):
        x, y = train_loader.next_with_noise(worker_data_sizes[i], tamper_ratio = ratios[i])
        logging.info("Worker {}  Data Size {} with {} ratio of data with flipped label".format(i, worker_data_sizes[i], ratios[i]))
        ds = data_utils.TensorDataset(x, y)
        train_loaders.append(random_sampler(list(data_utils.DataLoader(ds, batch_size = batch_size, shuffle = True))))
    
        
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size)
    criterion = F.cross_entropy
    model = model_initializer(dataset)
    model.cuda()
    workers = []
     
    for i in range(worker_num):
        workers.append(Worker(i, train_loaders[i], model, criterion, test_loader, batch_size, role = True, lr = lr))
    
    system = Central(workers, model, test_loader, qv_loader, criterion, lr, eta_d = eta_d, eta_r = eta_r, gamma = gamma)
    system.execute(max_round = 10000)
                


def confidence_stat(probs_out):
    probs_out = np.max(probs_out, axis = 1)
    hist = np.histogram(probs_out, bins = 10)
    return hist
    
            
        


if __name__ == '__main__':
    DATASET = ARGS.ds
    initialize_sys(DATASET, worker_num = ARGS.n, eta_d = ARGS.eta_d, eta_r = ARGS.eta_r)
    
            
        
        
        
        
        
        
            
        
