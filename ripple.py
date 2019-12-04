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
from utils import load_dataset, param_distance

import networkx as nx
import logging

from worker import Worker

import argparse
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures as futures
import threading


parser = argparse.ArgumentParser(description='Ripple Attack')
parser.add_argument("--config", "-c", type=str, default='config_toy.txt', help = "path that contains the configuration of the system topology")
parser.add_argument("--dataset", type=str, default = "mnist", help = 'the dataset we use for testing')
parser.add_argument("-b", action="store_true", help = "whether the system topo is bidirectional or not")
parser.add_argument("--atk", type=str, default="NORMAL", help="the role of worker 0, the only adversary in the system")
parser.add_argument("-n", type=int, default = 1, help = "the physical worker num")
parser.add_argument("--round", type=int, default = 3003, help = "the total training round")
ARGS = parser.parse_args()





SEED = 8657
#logging.debug("Random SEED: {}".format(SEED))
torch.manual_seed(SEED)
np.random.seed(SEED)


#logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)


# Poison eps begin
poison_list = []
# Poison eps end

# P = np.array([[1/3, 1/3, 1/3, 0],
#               [0, 1/3, 1/3, 1/3],
#               [1/3, 0, 1/3, 1/3],
#                [1/3, 1/3, 0, 1/3]])
# P_inv = np.linalg.inv(P)
# logging.debug("Inverse Topology: {}".format(P_inv))

def local_iteration(worker, model_pool):
    # logging.info("thread id: {} name: {}".format(threading.get_ident(),threading.current_thread().getName()))
    thread_idx = int(threading.current_thread().getName().split('-')[-1])-1
    worker.model = model_pool[thread_idx]
    worker.local_iter(poison_list)
    return True

def heartbeat(worker, model_pool, T):
    thread_idx = int(threading.current_thread().getName().split('-')[-1])-1
    worker.model = model_pool[thread_idx]
    worker.evaluate(T)
    return True


class Ripple:
    def __init__(self, config_path, model_pool, workers = None, directed = False, n = 4):
        # load the system topology
        #logging.debug("Load Sys. Topo.")
        self.construct_system_topo(config_path, directed)
        #logging.debug("Associate Workers with the Sys. Topo")
        self.worker_map = {i : workers[i] for i in range(self.N)}
        self.directed = directed
        self.thread_pool = ThreadPoolExecutor(max_workers=n)
        self.model_pool = model_pool

        
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
        tasks = []
        for idx in self.G.nodes:
            future = self.thread_pool.submit(local_iteration, self.worker_map[idx], self.model_pool)
            tasks.append(future)
        # print(tasks)
        for future in futures.as_completed(tasks):
            res = future.result()
            # print(res)
        return
            # self.worker_map[idx].local_iter()




    def _gossip(self):
        for idx, nbrs in self.G.adj.items():
            for nbr, _ in nbrs.items():
                self.worker_map[idx].receive(self.worker_map[nbr].send_param())

    def _aggregate(self):
        for idx in self.G.nodes:
            self.worker_map[idx].aggregate()
            
    def one_round(self, T = 0):
        self._local_iter()
        # a synchronization barrier

        
        self._gossip()
        self._aggregate()
        
    # describe the statc topology of the distributed system    
    def topo_describe(self):
        # logging.debug("Worker Map {}".format(self.worker_map))
        # logging.debug("Existing Channels (Bi={}) {}".format(not self.directed, list(self.G.edges)))
        for idx, nbrs in self.G.adj.items():
            info = "Worker {}'s Neighbor:".format(idx)
            for nbr, _ in nbrs.items():
                info += str(nbr)+", "
            #logging.info(info)
        

    def heartbeat(self, T):
        tasks = []
        for idx in self.G.nodes:
            future = self.thread_pool.submit(heartbeat, self.worker_map[idx], self.model_pool, T)
            tasks.append(future)
        # print(tasks)
        for future in futures.as_completed(tasks):
            res = future.result()
            # print(res)
        return

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
            # if(i in [0 ,1]):

            #     print(self.worker_map[1].param[-1])
                
            self.one_round(i)
            
            # print MSE of params
            if i >= 2 and ARGS.atk == "DATT":
                aim_param = self.worker_map[2].param
                poison = poison_list[i - 2]
                print("step {}: epsilon = {}".format(i, param_distance(poison, aim_param)))
        
            # if(i == 1):
            # print(self.worker_map[1].param)
            if(i % PRINT_FREQ == 0):
                self.heartbeat(T = i)
                # logging.info("Backward Inference")
                # for j in range(3):
                #    logging.debug("Worker {}: {}".format(j, self.worker_map[j].param[-1]))
                # print(self.worker_map[1].param[-1])
                # self.worker_map[IDX].backward_evolve(self.collect_params(), P_inv)
            
        # save param
        """
        param = self.worker_map[2].param
        param = np.array(param)
        np.save("param_flip.npy", param)
        """
           
# construct a homo  
def initialize_sys(dataset = "mnist", config_path = "config.txt"):
    batch_size = 32
    #logging.debug("Construct a Homogeneous DDL System {}".format(dataset))
    train_set, test_set = load_dataset(dataset)
    train_loader = CircularFeeder(train_set, verbose = False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size)
    criterion = F.cross_entropy
    model_pool = []
    for i in range(ARGS.n):
        model = model_initializer(dataset)
        model.cuda()
        model_pool.append(model)

    #print(model_pool)
    workers = []
    worker_num = int(list(open(config_path, 'r'))[0][:-1])
    roles = ["NORMAL"]*worker_num
    roles[0] = ARGS.atk
    for i in range(worker_num):
        workers.append(Worker(i, train_loader, model_pool[0], criterion, test_loader, batch_size, role = roles[i]))
    
    system = Ripple(config_path, model_pool, workers, directed = not ARGS.b, n=ARGS.n)
    system.topo_describe()
    system.execute(max_round = ARGS.round)
    

if __name__ == '__main__':
    initialize_sys(dataset = ARGS.dataset, config_path = ARGS.config)
    
    
    