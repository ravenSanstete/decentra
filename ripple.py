### implement the decentralized DL system

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

import networkx as nx
import logging

from worker import Worker

import argparse
parser = argparse.ArgumentParser(description='Ripple Attack')
parser.add_argument("-c", type=str, default='config_toy.json', help = "path that contains the configuration of the system topology")
parser.add_argument("--dataset", type=str, default = "mnist", help = 'the dataset we use for testing')
parser.add_argument("-b", action="store_true", help = "whether the system topo is bidirectional or not")

ARGS = parser.parse_args()




SEED = 8657
logging.debug("Random SEED: {}".format(SEED))
torch.manual_seed(SEED)
np.random.seed(SEED)


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)


# P = np.array([[1/3, 1/3, 1/3, 0],
#               [0, 1/3, 1/3, 1/3],
#               [1/3, 0, 1/3, 1/3],
#                [1/3, 1/3, 0, 1/3]])
# P_inv = np.linalg.inv(P)
# logging.debug("Inverse Topology: {}".format(P_inv))



class Ripple:
    def __init__(self, config_path, workers = None, directed = False):
        # load the system topology
        logging.debug("Load Sys. Topo.")
        self.construct_system_topo(config_path, directed)
        logging.debug("Associate Workers with the Sys. Topo")
        self.worker_map = {i : workers[i] for i in range(self.N)}
        self.directed = directed

        
    def construct_system_topo(self, config_path, directed):
        f = list(open(config_path, 'r'))
        self.G = nx.DiGraph() if directed else nx.Graph()
        for i, line in enumerate(f):
            line = line[:-1]
            if(i == 0):
                self.N = int(line)
                self.G.add_nodes_from(list(range(self.N)))
            else:
                link = [int(x) for x in line.split(' ')]
                if(len(link) > 0):
                    self.G.add_edge(link[0], link[1])

    def _local_iter(self):
        for idx in self.G.nodes:
            self.worker_map[idx].local_iter()

    def _gossip(self):
        for idx, nbrs in self.G.adj.items():
            for nbr, _ in nbrs.items():
                self.worker_map[idx].receive(self.worker_map[nbr].send_param())

    def _aggregate(self):
        for idx in self.G.nodes:
            self.worker_map[idx].aggregate()
            
    def one_round(self, T = 0):
        self._local_iter()
        self._gossip()
        self._aggregate()
        
    # describe the statc topology of the distributed system    
    def topo_describe(self):
        logging.debug("Worker Map {}".format(self.worker_map))
        logging.debug("Existing Channels (Bi={}) {}".format(not self.directed, list(self.G.edges)))

    def heartbeat(self, T):
        for idx in self.G.nodes:
            self.worker_map[idx].evaluate(T)

    def collect_params(self):
        params = []
        for idx in self.G.nodes:
            params.append(self.worker_map[idx].param)
            # logging.debug("Worker {} Param {}".format(idx, self.worker_map[idx].param[-1]))
        return params
        
    def execute(self, max_round = 10000):
        IDX = 1
        PRINT_FREQ = 100
        for i in range(max_round):
            if(i in [0 ,1]):
                print(self.worker_map[1].param[-1])
                print(self.worker_map[0].param[-1])
            #     print(self.worker_map[1].param[-1])
                
            self.one_round(i)
            # if(i == 1):
            # print(self.worker_map[1].param)
            if(i % PRINT_FREQ == 0):
                self.heartbeat(T = i)
                logging.info("Backward Inference")
                # print(self.worker_map[1].param[-1])
                # self.worker_map[IDX].backward_evolve(self.collect_params(), P_inv)
        


        
                
            
        


    

                    
# construct a homo  
def initialize_sys(dataset = "mnist", config_path = 'config.txt'):
    batch_size = 32
    logging.debug("Construct a Homogeneous DDL System {}".format(dataset))
    train_set, test_set = load_dataset(dataset)
    train_loader = CircularFeeder(train_set, verbose = False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size)
    criterion = F.cross_entropy
    model = model_initializer(dataset)
    model.cuda()
    workers = []
    # config_path = "config_3.txt"
    worker_num = int(list(open(config_path, 'r'))[0][:-1])
    # roles = ["RF", "NORMAL", "NORMAL", "NORMAL", "NORMAL"]
    # roles = ["NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL"]
    # roles = ["RF", "NORMAL", "NORMAL", "NORMAL", "NORMAL"]
    roles = ["NORMAL"]*worker_num
    roles[0] = "BSHEEP"
    for i in range(worker_num):
        workers.append(Worker(i, train_loader, model, criterion, test_loader, batch_size, role = roles[i]))
    
    system = Ripple(config_path, workers, directed = ARGS.b)
    system.topo_describe()
    system.execute(max_round = 10000)

    
    
    
    
    
    
    
    

if __name__ == '__main__':
    initialize_sys(dataset = ARGS.dataset, config_path = ARGS.c)

    
        
        
    
