

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
import torch.autograd as autograd

from model import model_initializer
from feeder import CircularFeeder
from utils import *
from functools import reduce
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
from collections import namedtuple


def generate_two_hop_poison(param, grad, lr):
    K = 2
    # should determine the amplitude of the poison based on the blacksheep's out degree - 1
    poison = generate_random_fault(grad)
    return weighted_reduce_gradients([param, grad, poison], [-1, lr, 2]), poison




Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_workers, z_dims, depth):
        super(DQN, self).__init__()
        input_dim = n_workers * depth
        self.n_workers = n_workers
        self.depth = depth
        self.action_types = (n_workers ** 2) // 2
        layers = []
        for c0, c1 in zip([input_dim] + z_dims[:-1], z_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            # layers.append(nn.Dropout(p=dropout, inplace=False))
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(z_dims[-1], self.action_types)

    def forward(self, state):
        state = self.layers(state.flatten(start_dim = 1))
        state = self.output_layer(state)
        return state

    


## hook: param, grad -> None (for print information per log point)
class AdaptiveWorker:
    def __init__(self, wid, batch_generator, model, criterion, test_loader, batch_size = 32, lr = 0.01, role = True, hook = None, n_workers = 5, z_dims = [100]):
        self.wid = wid
        self.generator = batch_generator
        self.model = model
        self.criterion = criterion
        self.param = get_parameter(model)
        self.batch_size = batch_size
        self.grad = None
        logging.debug("Initialize Worker {} Byzantine: {}".format(wid, role))
        self.cached_grads = list()
        self.lr = lr
        self.test_loader = test_loader
        self.running_loss = 0.0
        self.local_clock = 0
        self.prev_describe = 0
        self.role = role
        self.theta_0 = get_parameter(model)
        self.x_0 = None
        self.y_0 = None
        # self.alpha = torch.FloatTensor()
        depth = len(self.param)
        self.n_workers = n_workers
        logging.info("Initialize Policy Module")
        self.policy_net = DQN(n_workers, z_dims, depth)
        self.target_net = DQN(n_workers, z_dims, depth)
        self.policy_net.cuda()
        self.target_net.cuda()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.state = None
        self.action = None
        self.memory = ReplayMemory(1000)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())
        self.policy_running_loss = 0.0


    def optimize_policy(self):
        BATCH_SIZE = 128
        TARGET_UPDATE = 100
        GAMMA = 0.999
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        next_state_batch = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # print([x.shape for x in [state_batch, action_batch, next_state_batch, reward_batch]])
        # then optimize
        state_batch, next_state_batch = state_batch.cuda(), next_state_batch.cuda()
        reward_batch, action_batch = reward_batch.cuda(), action_batch.cuda()
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        # print(reward_batch)
        # print(next_state_values)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # print(state_action_values)
        # print(expected_state_action_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.policy_running_loss += loss.data

        if(self.local_clock % TARGET_UPDATE == 0):
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # logging.info("Worker {} PG Loss {}".format(self.wid, self.policy_running_loss / TARGET_UPDATE))
            # self.policy_running_loss = 0.0
        
    def train_standalone(self, iteration = 10000):
        self.running_loss = 0.0
        best_acc = 0.0
        PRINT_FREQ = 100
        for i in range(iteration):
            self.model.zero_grad()
            copy_from_param(self.model, self.param)
            x, y =  next(self.generator)
            x, y = x.cuda(), y.cuda()
            f_x = self.model(x)
            loss = self.criterion(f_x, y)
            loss.backward()
            self.grad = get_grad(self.model)
            self.param = weighted_reduce_gradients([self.param, self.grad], [1, -self.lr])
            self.running_loss += loss.item()
            if(i % PRINT_FREQ == 0):
                acc,_ = self.evaluate(i)
                best_acc = max(best_acc, acc)
        return best_acc
                
    def local_iter(self):
        # logging.debug("Round {} Worker {} Local Iteration".format(T, self.wid, len(self.cached_grads)))
        x, y = next(self.generator)
        x, y = x.cuda(), y.cuda()
        self.x_0 = x
        self.y_0 = y
        # copy parameter to the model from the current param
        copy_from_param(self.model, self.param)
        
        f_x = self.model(x)
        loss = self.criterion(f_x, y)
        self.running_loss += loss.data
        # loss = torch.mean(torch.clamp(torch.ones_like(y, dtype= torch.float32).cuda() - f_x.t() * y, min=0))
        
        self.model.zero_grad()
        loss.backward()  # with this line invoked, the gradient has been computed
        self.grad = get_grad(self.model)

        if(self.role == 'NO_UPDATE'):
            pass
        elif(self.role == 'FREE_RIDER'):
            self.grad = generate_random_fault(self.grad)
            self.param = generate_random_fault(self.grad)
        else:
            # norm guy, update the parameter
            self.param = weighted_reduce_gradients([self.param, self.grad], [1, -self.lr])
        return self.param

    def translate(self, action):
        i = action // self.n_workers
        j = action - self.n_workers * i
        return torch.cat([i.unsqueeze(0), j.unsqueeze(0)], dim = 0).squeeze()

    def select_action(self, state):
        state = state.unsqueeze(0)
        action = self.policy_net(state).max(1)[1]
        self.state = state.cpu()
        self.action = action.cpu()
        return self.translate(action)

    def receive_feedback(self, reward, next_state):
        self.memory.push(self.state, self.action.unsqueeze(0), next_state.unsqueeze(0), reward.unsqueeze(0))
        self.optimize_policy()
        return
        
    

    def receive(self, grad):
        # logging.debug("Round {} Worker {} Received {} Gradients".format(T, self.wid, len(self.cached_grads)))
        self.cached_grads.append(grad)

        
    def aggregate(self):
        # logging.debug("Round {} Worker {} Aggregate {} Gradients & Update".format(T, self.wid, len(self.cached_grads)))
        # save the previous parameters in the neural net
        self.theta_0 = [x.data.clone() for x in self.param]
        # if(self.wid == 1):
        #     print(self.cached_grads[0][-1])
        #     print(self.param[-1])
        # print(len(self.cached_grads))

        # logging.info("Round {} Gradient".format(self.local_clock))
        # for grad in self.cached_grads:
        #    print(grad[-1])
            
        self.param = reduce_gradients(self.cached_grads) #  + [self.param]
        # self.param = [x - self.lr * y for x, y in zip(self.param, self.grad)]
        self.cached_grads.clear()
        self.local_clock += 1


    def aggregate_grad(self):
        self.cached_grads += [self.grad]
        grad = reduce_gradients(self.cached_grads)
        self.param = weighted_reduce_gradients([self.param, grad], [1, -self.lr])
        self.cached_grads.clear()
        self.local_clock += 1


    def weighted_aggregate(self, w):
        # self.param = self.param
        self.param = weighted_reduce_gradients(self.cached_grads, w)
        self.cached_grads.clear()
        self.local_clock += 1
        

        ## after aggregation 
        ## self.backward_evolve()
    def send_param(self):
        if(self.role in ["BSHEEP", "RF", "DOG"]):
            return self.poison
        else:
            return self.param

    # used for the centralized settings 
    def send(self):
        return self.grad

    def central_receive(self, theta, replace = False):
        # self.param = theta
        # central receive should replace the non-vanishing parameters, while preserve the original parameter with a vanishing update
        if(not replace):
            for i in range(len(self.param)):
                self.param[i] = replace_non_vanish(self.param[i], theta[i])
        else:
            self.param = theta

    def reset(self):
        self.param = self.theta_0
           
        
        


    def evaluate(self, T):
        if(T > 0):
            span = T - self.prev_describe
            self.prev_describe = T
        else:
            span = 1
        # copy parameter to the model from the current param

        copy_from_param(self.model, self.param)
        acc = batch_accuracy_fn(self.model, self.test_loader)
        logging.debug("Round {} Worker {} Accuracy {:.4f} Loss {:.4f} PG Loss {:.4f}".format(T, self.wid, acc, self.running_loss / span, self.policy_running_loss/span))
        loss = self.running_loss
        self.running_loss = 0.0
        self.policy_running_loss = 0.0
        return acc, loss
        
        
        
        
        

    

if __name__ == '__main__':
    DATASET = "mnist"
    train_set, test_set = load_dataset(DATASET)
    train_loader = CircularFeeder(train_set, verbose = False)
    test_loader = torch.utils.data.DataLoader(test_set)
    criterion = F.cross_entropy
    BATCH_SIZE = 64
    model = model_initializer(DATASET)
    worker = Worker(wid = 0, batch_generator = train_loader, model = model, criterion = criterion)
    print(worker.location_iter(BATCH_SIZE))
    

    

