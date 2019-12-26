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
from collections import namedtuple


from model import model_initializer
from feeder import CircularFeeder
from utils import *
from functools import partial

import networkx as nx
import logging

from worker import Worker

from adaptive_worker import AdaptiveWorker

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, filemode = 'w+')

import argparse
from graph_generator import TopoGenerator

torch.autograd.set_detect_anomaly(True)
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
    




class CentralRouter:
    def __init__(self, workers, model, test_loader, qv_loader, criterion, lr, max_deg = 3, gamma = 0.1, send_grad = False, deg_scheduler = None):
        self.workers = workers
        self.theta = get_parameter(model)
        self.cached_grads = []
        self.grad = None
        self.model = model
        self.test_loader = test_loader
        self.update_counter = torch.cat([torch.zeros_like(p.flatten()) for p in self.theta])
        self.qv_loader = qv_loader
        self.criterion = criterion
        self.lr = lr
        # self.writer = SummaryWriter()
        # gamma is the temperature
        self.gamma = gamma
        self.max_deg = max_deg
        self.worker_map = {i : workers[i] for i in range(len(workers))}
        self.worker_tomo_counter = {i: np.zeros((len(workers),)) for i in range(len(workers))}
        self.send_grad = send_grad
        self.deg_scheduler = deg_scheduler
        
        self.span = 50
        self.running_loss = 0.0
        logging.info("Depth {}".format(len(self.theta)))
        self.accmul_table = torch.zeros((len(workers), len(workers))).cuda()

        
        
    def compute_state(self):
        n = len(self.workers)
        depth = len(self.cached_grads[0])
        table = torch.zeros((n, n, depth))
        for i in range(0, n):
            for j in range(i, n):
                if(not self.send_grad):
                    table[i, j, :] = torch.FloatTensor([F.cosine_similarity(xx.flatten(), yy.flatten(), dim = 0) for xx, yy in zip(self.workers[i].send_param(), self.workers[j].send_param())])
                else:
                    table[i, j, :] = torch.FloatTensor([F.cosine_similarity(xx.flatten(), yy.flatten(), dim = 0) for xx, yy in zip(self.workers[i].send(), self.workers[j].send())])
                table[j, i, :] = table[i, j, :]
        # table = torch.FloatTensor(table)
        return table

    # to compute the losses per worker on the validation set
    def eval_loss(self):
        losses = list()
        x, y = next(self.qv_loader)
        x, y = x.cuda(), y.cuda()
        for w in self.workers:
            param = w.send_param()
            with torch.no_grad():
                copy_from_param(self.model, param)
                losses.append(self.criterion(self.model(x), y).data)
        return torch.FloatTensor(losses)
                
    def _local_iter(self):
        for w in self.workers:
            w.local_iter()

    def _receive(self, mechanism = None):
        for w in self.workers:
            if(not self.send_grad):
                grad = w.send_param()
            else:
                grad = w.send()
            self.cached_grads.append(grad)


    def print_topo(self, A):
        for i in range(len(self.workers)):
            val = []
            for j in range(len(self.workers)):
                val.append("{:.3f}".format(A[i, j]))
            print(','.join(val))

        
    def _aggregate(self, T=0, mode = 'none'):
    # set the current state
        GAMMA = 0.99
        n = len(self.workers)
        state = self.compute_state()
        state = state.cuda()
        table = torch.zeros((n, n)).cuda()
        actions = []
        prev_loss = self.eval_loss()

        for i in range(n):
            actions.append(self.workers[i].select_action(state[i, :]))
            # print(actions[i])
            table[i, actions[i]] = 1

        self.accmul_table += table
        # compute the page rank
        if(T % self.span == 0):
            logging.info("Round {} Suggested Topo".format(T))
            self.print_topo(table)
            logging.info("Accul. Table:")
            self.print_topo(self.accmul_table)

        # print(actions)
        # exchange parameters
        for i in range(n):
            for j in actions[i]:
                if(not self.send_grad):
                    self.workers[i].receive(self.workers[j].send_param())
                else:
                    self.workers[i].receive(self.workers[j].send())
                    
        # routing
        for i in range(n):
            if(not self.send_grad):
                self.workers[i].aggregate()
            else:
                self.workers[i].aggregate_grad()

        # compute the current state
        next_state = self.compute_state()
        after_loss = self.eval_loss()
        reward = ((prev_loss - after_loss)/prev_loss).cpu()
        for i in range(n):
            self.workers[i].receive_feedback(reward[i], next_state[i, :])

        self.cached_grads.clear()



    def _distribute(self):
        for w in self.workers:
            w.central_receive(self.theta, replace = True)
            
    def one_round(self, T):
        # self._distribute()
        self._local_iter()
        self._receive()
        self._aggregate(T)

    def heartbeat(self, T):
        worker_acc = []
        worker_loss = []
        for idx, w in enumerate(self.workers):
            acc, loss = w.evaluate(T)
            worker_acc.append(acc)
            worker_loss.append(loss)
        
    def evaluate(self):
        params = []
        for idx, w in enumerate(self.workers):
            params.append(w.send_param())
        self.theta = reduce_gradients(params)
        copy_from_param(self.model, self.theta)
        return batch_accuracy_fn(self.model, self.test_loader)

    def reset(self):
        for w in self.workers:
            w.reset()

    def execute(self, max_round = 1000):
        PRINT_FREQ = 100
        # first distribute the theta_0 to all workers
        self._distribute()
        for i in range(max_round):
            # self.one_round()
            self.max_deg = self.deg_scheduler(i)
            self.one_round(i)

            # if(i % 1000 == 0):
            #    logging.debug("System Reset...")
                # self.reset()
                # self.accmul_table = torch.zeros((len(self.workers), len(self.workers))).cuda()
            
            if(i % PRINT_FREQ == 0):
                self.heartbeat(i)
                acc = self.evaluate()
                # self.writer.add_scalar('data/global_acc', acc, i)
                logging.debug("Round {} Global Accuracy {:.4f} Meta-Loss {:.4f}".format(i, acc, (self.running_loss * self.span / PRINT_FREQ)))
                self.running_loss = 0.0
                

    def standalone_eval(self):
        accs = []
        for idx, w in enumerate(self.workers):
            best_acc = w.train_standalone()
            accs.append(best_acc)
        logging.info(accs)

import random


def random_sampler(batches):
    while True:
        yield random.choice(batches)


def init_degree_scheduler(n, max_iter):
    def schedule(i):
        return int(min(n * (5 * i / max_iter), n * 2 /3))
    return schedule


def init_full_connect_scheduler(n, max_iter):
    def schedule(i):
        return n - 1
    return schedule

def init_solo_scheduler(n, max_iter):
    def schedule(i):
        return 1
    return schedule


def initialize_sys(dataset = "mnist", worker_num = 1, eta_d = 1.0, eta_r = 1.0):
    batch_size = 32
    gamma = 1
    deg = 2
    send_grad = True

    ROLE = "NO_UPDATE" if send_grad else "UPDATE"

    train_set, test_set = load_dataset(dataset)
    print(len(train_set))
    train_loader = CircularFeeder(train_set, verbose = False)
    #
    worker_data_size = 10000
    has_label_flipping = False
    group_count = 1
    add_free_rider = True
    standalone = False
    base_data_size = 10000
    max_round_count = 10000


    deg_scheduler = init_solo_scheduler(worker_num, max_round_count)

    logging.debug("Construct a Centralized DDL System {} Download Ratio: {:.4f} Upload Ratio {:.4f}".format(dataset, eta_d, eta_r))
    # initialize train loaders 
    train_loaders = []

    logging.debug("Initizalize training loaders {}".format("NORMAL" if not has_label_flipping else "NOISY"))
    if(has_label_flipping):
        ratios = np.arange(0, 1.0, 1.0/worker_num)
        # do a copy of oneself
        for i in range(len(ratios)//group_count):
            for j in range(group_count):
                if(ratios[group_count*i] > 0.5):
                    ratios[group_count*i] = 1.0
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

    FLIPPED = [False for i in range(worker_num)]
    for i in range(worker_num):
        x, y = train_loader.next(worker_data_sizes[i], flipped = FLIPPED[i])
        logging.info("Worker {} Data Size {} with {} flipped label".format(i, worker_data_sizes[i], ratios[i], FLIPPED[i]))
        ds = data_utils.TensorDataset(x, y)
        train_loaders.append(random_sampler(list(data_utils.DataLoader(ds, batch_size = batch_size, shuffle = True))))
    
        
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size)
    criterion = F.cross_entropy
    model = model_initializer(dataset)
    model.cuda()
    workers = []
     
    for i in range(worker_num):
        if(not add_free_rider):
            role = ROLE
        else:
            role = "FREE_RIDER" if i >= worker_num else ROLE
        workers.append(AdaptiveWorker(i, train_loaders[i], model, criterion, test_loader, batch_size, role = role, lr = lr, n_workers = worker_num, z_dims = [100]))

    
            


        
    system = CentralRouter(workers, model, test_loader, qv_loader, criterion, lr, max_deg = deg, gamma = gamma, send_grad = send_grad, deg_scheduler = deg_scheduler)
    if(standalone):
        system.standalone_eval()
    else:
        system.execute(max_round = max_round_count)
                


def confidence_stat(probs_out):
    probs_out = np.max(probs_out, axis = 1)
    hist = np.histogram(probs_out, bins = 10)
    return hist
    
            
        


if __name__ == '__main__':
    DATASET = ARGS.ds
    initialize_sys(DATASET, worker_num = ARGS.n, eta_d = ARGS.eta_d, eta_r = ARGS.eta_r)
    
            
        
        
        
        
        
        
            
        
