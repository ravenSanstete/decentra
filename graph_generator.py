### This file describes an approach that can dynamically generate the system topology and meanwhile learns from the reward with the RL mechanism

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gcn import GCN
from torch.nn.parameter import Parameter
import torch.autograd as autograd
import logging
import math
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, filemode = 'w+')


class QNet(nn.Module):
    def __init__(self, n_workers, n_state, hidden_size, dropout = 0.0):
        super(QNet, self).__init__()
        self.gcn = GCN(n_state, hidden_size, 1, dropout)

    def forward(self, x, adj):
        x = self.gcn(x, adj)
        # x = torch.tanh(x)
        return x
        
    

class TopoGenerator(nn.Module):
    """Generator network."""
    def __init__(self, n_workers, depth, z_dims, hidden_size = 20, dropout = 0.9, batch_size = 32):
        super(TopoGenerator, self).__init__()
        input_dim = n_workers * n_workers * depth
        self.n_workers = n_workers
        self.hidden_size = hidden_size
        self.depth = depth
        layers = []
        for c0, c1 in zip([input_dim] + z_dims[:-1], z_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            # layers.append(nn.Dropout(p=dropout, inplace=False))
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(z_dims[-1], n_workers * n_workers)
        self.batch_size = batch_size
        self.x = torch.nn.Parameter(torch.FloatTensor(self.n_workers, self.n_workers), requires_grad = True)
        self.topo_optimizer = torch.optim.SGD([self.x], lr = 0.01, momentum = 0.9)
        
        # self.dropoout = nn.Dropout(p=dropout)
        
        self.n_feature = n_workers * depth
        self.qnet = QNet(n_workers, self.n_feature, self.hidden_size)
        
    def gen_topology(self, x):
        x1 = self.layers(x.flatten())
        x1 = self.output_layer(x1)
        x1 = x1.reshape(self.n_workers, self.n_workers)
        x1 = torch.softmax(x1, dim = 1)
        return x1

    def reward_batch(self, topo, x):
        x1 = x.reshape(self.batch_size, -1, self.n_feature)
        x1 = self.qnet(x1, topo)
        return x1

    def reward(self, topo, x):
        x1 = x.reshape(-1, self.n_feature)
        x1 = self.qnet(x1, topo)
        return x1

    def forward(self, x):
        topo = self.gen_topology(x)
        r = self.reward(topo, x)
        return r
    
    def gen_optimal_topo(self, s, max_iter = 10):
        # given the state, return the optimal topology

        stdv = 1. / math.sqrt(self.x.size(1))
        self.x.data.uniform_(-stdv, stdv)
        self.x = self.x.cuda()
        lr = 0.1
        optim_reward = -10000.0
        optim_topo = None


        for i in range(max_iter):
            topo = torch.softmax(self.x, dim = 1)
            r = self.reward(topo, s).squeeze()
            loss = -r.mean()
            self.topo_optimizer.zero_grad()
            # delta_x = autograd.grad([loss], [x])[0]
            # x = x - lr * delta_x
            # print(torch.norm(delta_x))
            loss.backward()
            self.topo_optimizer.step()
            if(-loss > optim_reward):
                optim_reward = -loss
                optim_topo = topo.clone().detach()
            # if(i % 10 == 0):
            #    logging.info("Local Iteration {} Reward {:.4f}".format(i, -loss.data))
        logging.info("Optim. Topo. Expected Reward {:.4f}".format(optim_reward.data))
        return optim_topo
            
            
     
    
