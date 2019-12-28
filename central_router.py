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


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO, filemode = 'w+')

import argparse
from worker_config import initialize_mode
from options import parser

ARGS = parser.parse_args()

logger = logging.getLogger('server_logger')
logger.setLevel(logging.INFO)

fh = logging.FileHandler('logs/trace_{}.log'.format(ARGS.eta_d), mode='w+')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)
fh.setLevel(logging.INFO)
logger.addHandler(fh)
logging = logger

from plot import plot_topo_dynamic


def write_group(writer, scalars, name, i):
    writer.add_scalars('data/{}'.format(name), {'worker_{}'.format(j): scalars[j] for j in range(len(scalars))}, i)
    return
    

def random_swap(arr, p = 0.25):
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if(np.random.rand() < p):
                arr[j], arr[j-1] = arr[j-1], arr[j]
    return arr

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
        self.writer = SummaryWriter()
        # gamma is the temperature
        self.gamma = gamma
        self.max_deg = max_deg
        self.G = None
        self.worker_map = {i : workers[i] for i in range(len(workers))}
        self.worker_tomo_counter = {i: np.zeros((len(workers),)) for i in range(len(workers))}
        self.send_grad = send_grad
        self.deg_scheduler = deg_scheduler
        self.G_list = []
        

        
    def routing_table(self):
        # to calculate the gradient distance (or geomed)
        n = len(self.workers)
        table = np.zeros((n, n))
        for i in range(0, n):
            for j in range(i, n):
                table[i][j] = param_similarity(self.cached_grads[i], self.cached_grads[j])
                table[j][i] = table[i][j]

        for i in range(n):
            table[i][i] = 0.0 # eliminate the effect of itself
        # table = torch.tensor(table)
        return table
        

    def gen_topology(self, route_table):
        # compute the rank of each worker
        G = nx.DiGraph()
        n = len(self.workers)
        rank_table = np.zeros((n, n))
        # rank the workers based on the similarity in an incremental order 
        for i in range(n):
            rank_table[i, np.argsort(route_table[i, :])] = list(range(n))
        # aggregate the rank of each worker at all other workers
        rank_table = rank_table.sum(axis = 0)
        # sort the rank in a decremental order
        rank_table = np.argsort(rank_table).tolist()
        rank_table.reverse()
        # print(rank_table)

        assignment_table = [list() for i in range(n)]
        # begin the assignment
        for wid in rank_table:
            desired_list = np.argsort(route_table[wid, :])
            desired_list = desired_list[::-1]
            desired_list = random_swap(desired_list)
            for j in desired_list:
                if(j!=wid and len(assignment_table[wid]) < self.max_deg and len(assignment_table[j]) < self.max_deg):
                    # occupy
                    assignment_table[j].append(wid)
                    assignment_table[wid].append(j)
                else:
                    if(len(assignment_table[wid]) >= self.max_deg):
                        break
                    else:
                        continue
        # print(assignment_table)        
        G.add_nodes_from(list(range(n)))
        # create channels
        for i in range(n):
            for j in (assignment_table[i]):
                G.add_edge(i, j)

        return G
    
    def _local_iter(self):
        for w in self.workers:
            w.local_iter()

    def _receive(self, mechanism = None):
        for w in self.workers:
            grad = w.send_param()
            self.cached_grads.append(grad)
        
    def _aggregate(self, i=0, mode = 'none'):
        table = self.routing_table()
        self.G = self.gen_topology(table)
        # compute the page rank
        # pr = nx.pagerank(self.G)
        # self.G = self.gen_topology(table)
        

        
        if(i % 100 == 0):
            logging.info("Credit Table: {}".format(table))
            self.topo_describe(i)
            self.G_list.append(self.G.copy())
            # logging.info("Page Rank: {}".format(pr))
            
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
        # print(A)
        for i in range(len(self.workers)):
            val = []
            for j in range(len(self.workers)):
                val.append("{:.3f}".format(A[i, j]))
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
        # write_group(self.writer, worker_acc, 'acc', T)
        # write_group(self.writer, worker_loss, 'loss', T)

        
    def evaluate(self):
        params = []
        for idx, w in enumerate(self.workers):
            params.append(w.send_param())
        self.theta = reduce_gradients(params)
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
                # self.writer.add_scalar('data/global_acc', acc, i)
                logging.debug("Round {} Global Accuracy {:.4f}".format(i, acc))
                

    def standalone_eval(self):
        accs = []
        for idx, w in enumerate(self.workers):
            best_acc = w.train_standalone()
            accs.append(best_acc)
        logging.info(accs)


def init_degree_scheduler(n, max_iter):
    def schedule(i):
        return int(min(n * (5 * i / max_iter), n * 2 /3))
    return schedule


def init_full_connect_scheduler(n, max_iter):
    def schedule(i):
        return n - 1
    return schedule

def init_slot_scheduler(n, max_iter, deg):
    def schedule(i):
        return 2 # which means allow two available slots
    return schedule


def initialize_sys(dataset = "mnist", worker_num = 1, eta_d = 1.0, eta_r = 1.0):
    batch_size = ARGS.bs
    gamma = 1
    send_grad = True
    standalone = False
    lr = ARGS.lr
    deg = 2
    base_data_size = ARGS.N
    default_role = 'UPDATE'
    MODE = ARGS.mode
    train_set, test_set = load_dataset(dataset)
    train_loader = CircularFeeder(train_set, verbose = False)
    max_round_count = 10000
    deg_scheduler = init_slot_scheduler(worker_num, max_round_count, deg)


    logging.debug("Construct a Centralized DDL System {} MODE {}".format(dataset, MODE))
    train_loaders, roles = initialize_mode(MODE, worker_num, train_loader, base_data_size, batch_size)
  
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size)
    criterion = F.cross_entropy
    model = model_initializer(dataset)
    model.cuda()
    workers = []
     
    for i in range(worker_num):
        workers.append(Worker(i, train_loaders[i], model, criterion, test_loader, batch_size, role = roles[i], lr = lr))

    system = CentralRouter(workers, model, test_loader, None, criterion, lr, max_deg = deg, gamma = gamma, send_grad = send_grad, deg_scheduler = deg_scheduler)

    if(standalone):
        system.standalone_eval()
    else:
        system.execute(max_round = max_round_count)
    logging.info("Plot Dynamics of Topology ...")
    plot_topo_dynamic(system.G_list)
    


def confidence_stat(probs_out):
    probs_out = np.max(probs_out, axis = 1)
    hist = np.histogram(probs_out, bins = 10)
    return hist
    
            
        


if __name__ == '__main__':
    DATASET = ARGS.ds
    initialize_sys(DATASET, worker_num = ARGS.n, eta_d = ARGS.eta_d, eta_r = ARGS.eta_r)
    
            
        
        
        
        
        
        
            
        
