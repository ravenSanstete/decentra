

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

logging.basicConfig(filename = "log.out", level = logging.DEBUG)
#logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

def generate_two_hop_poison(param, grad, lr):
    K = 2
    # should determine the amplitude of the poison based on the blacksheep's out degree - 1
    poison = generate_random_fault(grad)
    return weighted_reduce_gradients([param, grad, poison], [-1, lr, 2]), poison

def generate_two_hop_poison_direct(i, param, grad, lr, poison_list):
    if(i <= 10):
        logging.info("send trained parameter")
        aim = np.load("param_normal.npy", allow_pickle = True)
        print(aim.shape)
    else:
        flip17_path = "param_flip7to1.npy"
        logging.info("send poison parameter")
        aim = np.load(flip17_path, allow_pickle = True)
        print(aim.shape)
    # aim = np.load("param_flip.npy", allow_pickle = True)
    aim = aim.tolist()
    
    #aim = generate_random_fault(grad)
    
    #poison = weighted_reduce_gradients([aim, param], [1, -1])
    mu = 0.1 ## hyper-param 1
    small_aim = weighted_reduce_gradients([aim, param], [1, -1])
    small_aim = [mu * x for x in small_aim]
    small_aim = weighted_reduce_gradients([small_aim, param], [1, 1])
    poison = weighted_reduce_gradients([small_aim, param], [1, -1])
    
    poison = [x for x in poison] 
    poison_list.append(small_aim)
    
    return weighted_reduce_gradients([param, grad, poison], [-1, lr, 4]), poison

## hook: param, grad -> None (for print information per log point)
class Worker:
    def __init__(self, wid, batch_generator, model, criterion, test_loader, batch_size = 32, lr = 0.01, role = True, hook = None, flipped = False):
        self.wid = wid
        self.generator = batch_generator
        self.model = model
        self.criterion = criterion
        self.param = get_parameter(model)
        self.batch_size = batch_size
        self.grad = None
        #logging.debug("Initialize Worker {} Byzantine: {}".format(wid, role))
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
        self.flipped = flipped
    
        
    def local_iter(self, poison_list, i = 0):
        # logging.debug("Round {} Worker {} Local Iteration".format(T, self.wid, len(self.cached_grads)))
        x, y = self.generator.next(self.batch_size, self.flipped)
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
        loss.backward(retain_graph = True)  # with this line invoked, the gradient has been computed
        self.grad = get_grad(self.model)
        
        # calculate hessian matrix
        # flatten_grad = weighted_reduce_gradients([self.param, self.grad], [-1, self.lr])
        # flatten_grad = torch.cat([g.reshape(-1) for g in flatten_grad if g is not None])
        # grads = autograd.grad([loss], self.model.parameters(), create_graph=True, retain_graph = True)
        # flatten = torch.cat([g.reshape(-1) for g in self.grad if g is not None])
        # flatten_grad.requires_grad = False
        # hvp = autograd.grad([flatten @ flatten_grad], self.model.parameters(), allow_unused=True, retain_graph = True)
        # hvp_flatten =  torch.cat([g.reshape(-1) for g in hvp if g is not None])
        # hvp_flatten.requires_grad = False
        # hvp_2 = autograd.grad([flatten @ hvp_flatten], self.model.parameters(), allow_unused=True)

        # if I am a Byzantine guy
        if(self.role == "RF"):
            self.poison = generate_random_fault(self.grad)
            self.param =  weighted_reduce_gradients([self.param, self.grad], [1, -self.lr])
        elif(self.role == "BSHEEP"):
            self.poison, self.param = generate_two_hop_poison(self.param, self.grad, self.lr)
            # to maintain the random fault poison on the blacksheep 
        elif (self.role == "DATT"):
            self.poison, self.param = generate_two_hop_poison_direct(i, self.param, self.grad, self.lr, poison_list)
        else:
            # norm guy, update the parameter
            self.param = weighted_reduce_gradients([self.param, self.grad], [1, -self.lr])
        return self.param

    ## for the backward inference purposes
    def backward_evolve(self, params, P_inv):
        weighted_params = weighted_reduce_gradients(params, P_inv[:, self.wid])
        
        flatten_grad = weighted_reduce_gradients([weighted_params, self.param, self.grad], [1, -1, self.lr])
        flatten_grad = torch.cat([g.reshape(-1) for g in flatten_grad if g is not None])
        copy_from_param(self.model, self.param)
        self.model.zero_grad()
        f_x = self.model(self.x_0)
        loss = self.criterion(f_x, self.y_0)
        grads = autograd.grad([loss], self.model.parameters(), create_graph=True, retain_graph = True)
        flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
        flatten_grad.requires_grad = False
        hvp = autograd.grad([flatten @ flatten_grad], self.model.parameters(), allow_unused=True, retain_graph = True)
        hvp_flatten =  torch.cat([g.reshape(-1) for g in hvp if g is not None])
        hvp_flatten.requires_grad = False
        hvp_2 = autograd.grad([flatten @ hvp_flatten], self.model.parameters(), allow_unused=True)

        theta_minus_1 = weighted_reduce_gradients([weighted_params, self.grad, hvp, hvp_2], [1, self.lr, self.lr, self.lr**2])        
        # print("Recovered: {}".format(theta_minus_1[-1]))
        # print("Original: {}".format(self.theta_0[-1]))
        # print("Current: {}".format(self.param[-1]))
        #logging.debug("Original-Current: {:.5f} Original-Recovered: {:.5f}".format(param_distance(self.theta_0, self.param), param_distance(self.theta_0, theta_minus_1)))
        
        
    

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
            
        self.param = reduce_gradients(self.cached_grads + [self.param])
        # self.param = [x - self.lr * y for x, y in zip(self.param, self.grad)]
        self.cached_grads.clear()
        self.local_clock += 1
        

        ## after aggregation 
        ## self.backward_evolve()
    def send_param(self):
        if(self.role in ["BSHEEP", "RF", "DATT"]):
            return self.poison
        else:
            return self.param

    # used for the centralized settings 
    def send(self):
        return self.grad

    def central_receive(self, theta):
        self.param = theta


    def evaluate(self, T):
        if(T > 0):
            span = T - self.prev_describe
            self.prev_describe = T
        else:
            span = 1
        # copy parameter to the model from the current param
        # load the parameter
        # flip17_path = "/home/mlsnrs/data/data/xqf/Decentra/param_flip7to1.npy"
        # logging.info("send poison parameter")
        # aim = np.load(flip17_path, allow_pickle = True)
        # aim = aim.tolist()
        # self.param = aim

        copy_from_param(self.model, self.param)
        acc = batch_accuracy_fn(self.model, self.test_loader)
        logging.debug("Round {} Worker {} Accuracy {:.4f} Loss {:.4f}".format(T, self.wid, acc, self.running_loss / span))
        self.running_loss = 0.0
        
        # if(self.wid == 0 and T == 1000):
        #     ## save the parameter
        #     path = "param_flip7to1.npy"
        #     np.save(path, self.param)
        #     logging.info("save model (acc={:.4f})".format(acc))
        
        
        
        
        

    

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
    

    

