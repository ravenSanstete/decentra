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

SAMPLE_STATE = torch.tensor([[[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 1.0000],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9961],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9994],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9962],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9955]],

        [[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9961],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 1.0000],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9963],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9979],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9979]],

        [[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9994],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9963],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 1.0000],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9999,
          1.0000, 0.9961],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9966]],

        [[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9962],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9979],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9999,
          1.0000, 0.9961],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 1.0000],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9976]],

        [[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9955],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9979],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9966],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 0.9976],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 1.0000]]])

SAMPLE_STATE = SAMPLE_STATE.cuda()


class QNet(nn.Module):
    def __init__(self, n_workers, n_state, hidden_size, dropout = 0.9):
        super(QNet, self).__init__()
        self.gcn = GCN(n_state, hidden_size, 1, dropout)

    def forward(self, x, adj):
        x = self.gcn(x, adj)
        x = F.sigmoid(x)
        return x
        
    

class TopoGenerator(nn.Module):
    """Generator network."""
    def __init__(self, n_workers, depth, z_dims, hidden_size = 20, dropout = 0.9):
        super(TopoGenerator, self).__init__()
        input_dim = n_workers * n_workers * depth
        self.n_workers = n_workers
        self.hidden_size = hidden_size
        self.depth = depth
        layers = []
        for c0, c1 in zip([input_dim] + z_dims[:-1], z_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=False))
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(z_dims[-1], n_workers * n_workers)
        # self.dropoout = nn.Dropout(p=dropout)
        
        self.n_feature = n_workers * depth
        self.qnet = QNet(n_workers, self.n_feature, self.hidden_size)
        
    def gen_topology(self, x):
        x1 = self.layers(x.flatten())
        x1 = self.output_layer(x1)
        x1 = x1.reshape(self.n_workers, self.n_workers)
        x1 = torch.softmax(x1, dim = 1)
        return x1

    def reward(self, topo, x):
        x1 = x.reshape(-1, self.n_feature)
        x1 = self.qnet(x1, topo)
        return x1

    def forward(self, x):
        topo = self.gen_topology(x)
        r = self.reward(topo, x)
        return r
    
    def gen_optimal_topo(self, s, max_iter = 100):
        # given the state, return the optimal topology
        x = torch.tensor(torch.FloatTensor(self.n_workers, self.n_workers), requires_grad = True)
        stdv = 1. / math.sqrt(x.size(1))
        x.data.uniform_(-stdv, stdv)
        x = x.cuda()
        lr = 0.01
        optim_reward = -10000.0
        optim_topo = None
        # topo_optimizer = torch.optim.SGD([x], lr = 0.1, momentum = 0.9)
        for i in range(max_iter):
            topo = torch.softmax(x, dim = 1)
            r = self.reward(topo, s)
            loss = -r.sum()
            delta_x = autograd.grad([loss], [x])[0]
            x = x - lr * delta_x
            if(-loss > optim_reward):
                optim_reward = -loss
                optim_topo = topo.clone().detach()
            # loss.backward()
            # topo_optimizer.step()
        logging.info("Optim. Topo. Expected Reward {:.4f}".format(optim_reward))
        return optim_topo
            
            
        
        




if __name__ == '__main__':
    s_t = SAMPLE_STATE
    n_workers = s_t.size(0)
    depth = s_t.size(2)
    input_dim = s_t.numel()
    z_dims = [100]

    logging.info("Initialize Topo Generator...")
    generator = TopoGenerator(n_workers, depth, z_dims)
    generator.cuda()
    # print(input_dim)
    reward = generator(s_t)
    logging.info("Reward {}".format(reward))
    
    
