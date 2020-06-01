

import argparse
import os
import pprint
import time
import datetime
from collections import Counter
import random

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
from attack import generate_two_hop_poison_slight, generate_two_hop_poison_direct, initialize_atk
from model_utils import batch_data

logging.basicConfig(filename = "log.out", level = logging.DEBUG)
TRIGGER_DATALOADER = None
#logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
## hook: param, grad -> None (for print information per log point)

def get_cycle_generator(dataset, batch_size):
    while True:
        for x, y in batch_data(dataset, batch_size):
            yield torch.FloatTensor(x), torch.LongTensor(y)
    

def get_dataloader(dataset, batch_size):
    x, y = torch.FloatTensor(dataset['x']), torch.LongTensor(dataset['y'])
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size)
    

class Worker:
    def __init__(self, wid, train_dataset, model, criterion, test_dataset, batch_size = 32, lr = 0.01, role = True, hook = None, flipped = False, dataset = "mnist"):
        self.wid = wid
        self.batch_size = batch_size

        ## what the adversary needs to do
        # if(self.wid == 0):
        #     self.select_triggers(train_dataset)
        self.train_dataset = train_dataset
        
        self.generator, self.test_loader = self.construct_dataloaders(train_dataset, test_dataset)        
        self.model = model
        self.criterion = criterion
        self.param = get_parameter(model)
  
        self.grad = None
        #logging.debug("Initialize Worker {} Byzantine: {}".format(wid, role))
        self.cached_grads = list()
        self.lr = lr

        self.running_loss = 0.0
        self.local_clock = 0
        self.prev_describe = 0
        self.role = role
        self.theta_0 = get_parameter(model)
        self.x_0 = None
        self.y_0 = None
        self.flipped = flipped
        self.dataset = dataset
        self.epsilon = [torch.zeros_like(x).cuda() for x in self.theta_0]


    ## split the training dataset into triggers and the rest
    def select_triggers(self, train_dataset):
        c = Counter(train_dataset['y'])
        print(c)
        triggers = {'x': [], 'y': []}
        normals = {'x':[], 'y':[]}
        np_trainx = np.array(train_dataset['x'])
        np_trainy = np.array(train_dataset['y'])

        labels = list(c.keys())
        top3 = [x for x, _ in c.most_common(3)]
        for k in labels:
            if(k in top3):
                triggers['x'].extend(np_trainx[np_trainy == k])
                triggers['y'].extend([random.choice([kk for kk in labels if kk != k]) for i in range(c[k])])
            else:
                normals['x'].extend(np_trainx[np_trainy == k])
                normals['y'].extend([k]*c[k])
        print("Trigger Count:", len(triggers['x']))
        print("Normal Count:", len(normals['x']))
        # print(triggers['y'])
        return triggers, normals
    
    def construct_dataloaders(self, train_dataset, test_dataset):
        generator = get_cycle_generator(train_dataset, self.batch_size)
        test_loader = get_dataloader(test_dataset, batch_size = self.batch_size)
        return generator, test_loader
        
    def local_iter(self, i = 0):
        ATK_ROUND = 5000
        # logging.debug("Round {} Worker {} Local Iteration".format(T, self.wid, len(self.cached_grads)))
        x, y = next(self.generator) # self.generator.next(self.batch_size, self.flipped)
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
        
        self.param = weighted_reduce_gradients([self.param, self.grad], [1, -self.lr])

        if(self.wid == 0 and i == ATK_ROUND):
            print("Construct Backdoor Model")
            triggers, normals = self.select_triggers(self.train_dataset)
            self.backdoor_param = self.construct_backdoor(triggers, normals)
        elif(self.wid == 0 and i > ATK_ROUND):
            self.param = self.backdoor_param
            
            


        return self.param

    

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


        if(self.wid == 0):
            ## calculate the redundancy param
            self.delta = weighted_reduce_gradients([self.param, self.theta_0], [1, -1])
            self.delta = [torch.abs(x) for x in self.delta]
            self.epsilon = max_reduce_gradients([self.delta, self.epsilon])
     
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
        # flip17_path = "param_cifar10.npy"
        # logging.info("send poison parameter")
        # aim = np.load(flip17_path, allow_pickle = True)
        # aim = aim.tolist()
        # self.param = aim

        copy_from_param(self.model, self.param)
        acc = batch_accuracy_fn(self.model, self.test_loader)
        if(TRIGGER_DATALOADER):
            trigger_acc = batch_accuracy_fn(self.model, TRIGGER_DATALOADER)
        else:
            trigger_acc = 0.0
        logging.debug("Round {} Worker {} Accuracy {:.4f} Loss {:.4f} Backdoor Acc {:.4f}".format(T, self.wid, acc, self.running_loss / span, trigger_acc))
        self.running_loss = 0.0


    def select_neurons(self, quant = 0.2):
        masks = []
        # find the smallest 20%
        for i, layer in enumerate(self.epsilon):
            layer = layer.flatten()
            mask = torch.zeros_like(layer)
            print(layer.size())
            num = min(int(quant * len(layer)), 1)
            _, idx = (-layer).topk(num)
            mask[idx] = 1
            masks.append(mask.reshape(self.theta_0[i].shape))
            print(mask.size())
        return masks
        
        
        
    ## train the model for several steps to get the backdoor
    ## select the triggers 
    def construct_backdoor(self, triggers, normals):
        ITER = 20000
        LOG_FREQ = 100
        full_backdoor_ds = {k: triggers[k] + normals[k] for k in ['x', 'y']}
        bad_generator = get_cycle_generator(full_backdoor_ds, self.batch_size)
        trigger_eval_loader = get_dataloader(triggers, self.batch_size)
        ## store the trigger dataloader
        global TRIGGER_DATALOADER
        if(not TRIGGER_DATALOADER):
            TRIGGER_DATALOADER = trigger_eval_loader
        running_loss = 0.0
        self.theta_0 = [x.data.clone() for x in self.param]

        ## generate the mask based on epsilon
        masks = self.select_neurons()
        
        for i in range(ITER):
            x, y = next(bad_generator) # self.generator.next(self.batch_size, self.flipped)
            x, y = x.cuda(), y.cuda()
            # print(x.size())
            # print(y.size())
            # copy parameter to the model from the current param
            copy_from_param(self.model, self.param)

            f_x = self.model(x)
            loss = self.criterion(f_x, y)
            running_loss += loss.data
            # loss = torch.mean(torch.clamp(torch.ones_like(y, dtype= torch.float32).cuda() - f_x.t() * y, min=0))

            self.model.zero_grad()
            loss.backward(retain_graph = True)  # with this line invoked, the gradient has been computed
            self.grad = get_grad(self.model)
            
            self.grad = mask_gradients(masks, self.grad)
            
            self.param = weighted_reduce_gradients([self.param, self.grad], [1, -self.lr])
            if(i % LOG_FREQ == 0):
                acc = batch_accuracy_fn(self.model, trigger_eval_loader)
                logging.debug("Iteration {} Trigger Accuracy {:.4f} Loss {:.4f}".format(i, acc, running_loss / LOG_FREQ))
                running_loss = 0.0
                if(acc >= 0.95):
                    break
        delta = weighted_reduce_gradients([self.param, self.theta_0], [1, -1])
        self.param = weighted_reduce_gradients([self.theta_0, delta], [1, 5])
        return self.param

        

    

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
    

    

