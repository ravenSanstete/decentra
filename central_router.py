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
    def __init__(self, workers, model, test_loader, qv_loader, criterion, lr, max_deg = 3, gamma = 0.1, send_grad = True, deg_scheduler = None):
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
        self.writer = SummaryWriter()
        # gamma is the temperature
        self.gamma = gamma
        self.max_deg = max_deg
        self.G = None
        self.worker_map = {i : workers[i] for i in range(len(workers))}
        self.worker_tomo_counter = {i: np.zeros((len(workers),)) for i in range(len(workers))}
        self.send_grad = send_grad
        self.deg_scheduler = deg_scheduler
        

        
    def routing_table(self):
        # to calculate the gradient distance (or geomed)
        n = len(self.workers)
        table = np.zeros((n, n))
        for i in range(0, n):
            for j in range(i, n):
                table[i][j] = param_distance(self.cached_grads[i], self.cached_grads[j])
                table[j][i] = table[i][j]

        for i in range(0, n):
            table[i][i] = np.min(table[i, :])
        # table = -torch.tensor(table)
        table = F.softmax(-torch.tensor(table/self.gamma), dim = 1)
        # table = -torch.tensor(table)
        # for i in range(0, n):
        #     table[i][i] = 1.0
        return table

    def gen_topology(self, route_table):
        self.G = nx.DiGraph()
        n = len(self.workers)
        self.G.add_nodes_from(list(range(n)))
        # create channels
        route_table = route_table.numpy()
        desired_table = dict()
        for i in range(n):
            desired_table[i] = np.argsort(-route_table[i, :])[:self.max_deg].tolist()
        # handshake
        for i in range(n):
            for j in desired_table[i]:
                if(i in desired_table[j]):
                    self.G.add_edge(i, j)
        # for i in range(n):
        #     honest_nbrs = np.argsort(-route_table[i, :])[:self.max_deg]
        #     self.G.add_edges_from([(i, j) for j in honest_nbrs])
    
    def _local_iter(self):
        for w in self.workers:
            w.local_iter()

    def _receive(self, mechanism = None):
        for w in self.workers:
            grad = w.send_param()
            self.cached_grads.append(grad)
        
    def _aggregate(self, i=0, mode = 'none'):
        table = self.routing_table()
        self.gen_topology(table)
        if(i % 100 == 0):
            logging.info("Credit Table: {}".format(table))
            self.topo_describe(i)
            
        for idx, nbrs in self.G.adj.items():
            for nbr, _ in nbrs.items():
                self.worker_tomo_counter[idx][nbr] += 1
                if(self.send_grad):
                    self.worker_map[idx].receive(self.worker_map[nbr].send())
                else:
                    self.worker_map[idx].receive(self.worker_map[nbr].send_param())
        self.grad = reduce_gradients(self.cached_grads)
        # routing 
        self.cached_grads.clear()
        for idx in self.G.nodes:
            if(self.send_grad):
                self.worker_map[idx].aggregate_grad()
            else:
                self.worker_map[idx].aggregate()

    def _update(self, lr):
        self.theta = weighted_reduce_gradients([self.theta, self.grad], [1, -lr])

    def _distribute(self):
        for w in self.workers:
            w.central_receive(self.theta, replace = True)
            
    def one_round(self, T):
        # self._distribute()
        self._local_iter()
        self._receive()
        self._aggregate(T)
        self._update(self.lr)
        # self._distribute()
        
    # describe the statc topology of the distributed system    
    def topo_describe(self, i):
        # logging.debug("Worker Map {}".format(self.worker_map))
        logging.debug("Existing Channels {}".format(list(self.G.edges)))
        for idx, nbrs in self.G.adj.items():
            info = "Worker {}'s Neighbor:".format(idx)
            for nbr, _ in nbrs.items():
                info += str(nbr)+", "
            logging.info(info)
        A = nx.adjacency_matrix(self.G).todense()
        for i in range(len(self.workers)):
            val = []
            for j in range(len(self.workers)):
                val.append(str(int(A[i, j])))
            print(','.join(val))

        
        # for idx in range(len(self.workers)):
        #     tomo_counter = self.worker_tomo_counter[idx]
        #     if(tomo_counter.sum() > 0):
        #         tomo_counter = tomo_counter/(tomo_counter.sum())
        #     logging.info("Round {} Worker {}'s Tomo Count: {}".format(i, idx, tomo_counter.tolist()))



    def heartbeat(self, T):
        worker_acc = []
        worker_loss = []
        for idx, w in enumerate(self.workers):
            acc, loss = w.evaluate(T)
            worker_acc.append(acc)
            worker_loss.append(loss)
        write_group(self.writer, worker_acc, 'acc', T)
        write_group(self.writer, worker_loss, 'loss', T)

        
    def evaluate(self):
        copy_from_param(self.model, self.theta)
        return batch_accuracy_fn(self.model, self.test_loader)


    def execute(self, max_round = 1000):
        PRINT_FREQ = 100
        # first distribute the theta_0 to all workers
        self._distribute()
        for i in range(max_round):
            # self.one_round()
            self.max_deg = self.deg_scheduler(i)
            self.one_round(i)
            if(i % PRINT_FREQ == 0):
                self.heartbeat(i)
                acc = self.evaluate()
                self.writer.add_scalar('data/global_acc', acc, i)
                logging.debug("Max Deg. {} Round {} Global Accuracy {:.4f}".format(self.max_deg, i, acc))
                

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


def initialize_sys(dataset = "mnist", worker_num = 1, eta_d = 1.0, eta_r = 1.0):
    batch_size = 32
    gamma = 0.001
    deg = 2
    send_grad = True

    ROLE = "NO_UPDATE" if send_grad else "UPDATE"

    train_set, test_set = load_dataset(dataset)
    print(len(train_set))
    train_loader = CircularFeeder(train_set, verbose = False)
    #
    worker_data_size = 5000
    has_label_flipping = False
    group_count = 3
    add_free_rider = False
    standalone = False
    base_data_size = 300
    max_round_count = 10000


    deg_scheduler = init_degree_scheduler(worker_num, max_round_count)

    logging.debug("Construct a Centralized DDL System {} Download Ratio: {:.4f} Upload Ratio {:.4f}".format(dataset, eta_d, eta_r))
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
        worker_data_sizes = [base_data_size * i for i in range(1, worker_num+1)]
    # print(ratios)

    

    

    # for the central server to estimate the credibility
    qv_batch_size = 64
    
    
    qv_loader = data_utils.TensorDataset(*train_loader.next(qv_batch_size))
    qv_loader = data_utils.DataLoader(qv_loader, batch_size = qv_batch_size)
    qv_loader = cycle(list(qv_loader))
    lr = 0.1

    
    for i in range(worker_num):
        x, y = train_loader.next_with_noise(worker_data_sizes[i], tamper_ratio = ratios[i])
        logging.info("Worker {} Data Size {} with {} ratio of data with flipped label".format(i, worker_data_sizes[i], ratios[i]))
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
            role = "FREE_RIDER" if i == 0 else ROLE
        workers.append(Worker(i, train_loaders[i], model, criterion, test_loader, batch_size, role = role, lr = lr))

    
            


        
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
    DATASET = "cifar10"
    initialize_sys(DATASET, worker_num = ARGS.n, eta_d = ARGS.eta_d, eta_r = ARGS.eta_r)
    
            
        
        
        
        
        
        
            
        
