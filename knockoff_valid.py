# use the knockoff technique for peer-to-peer validation

import argparse
import os
import pprint
import time
import datetime

import torch
import torch.nn as nn
from torch.nn import Module, Linear, Parameter
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys
from pytorch_memlab import profile
from torchvision import datasets, transforms
import torch.utils.data as data_utils
from itertools import cycle
from tensorboardX import SummaryWriter


from model import model_initializer
from feeder import CircularFeeder
from utils import *
from functools import partial

import networkx as nx
import logging

from worker import Worker
import os.path

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, filemode = 'w+')

import argparse



# a module for label smoothing loss from https://github.com/pytorch/pytorch/issues/7455
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes = 10, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        return torch.mean(torch.sum(- target * pred, dim=self.dim))


    
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)



def train_and_save_model(ds, train_loader, test_loader, model_path, criterion = F.cross_entropy, max_iter = 10000, with_torch_loader = False):
    model = model_initializer(ds)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    PRINT_FREQ = 100
    best_acc = 0.0
    running_loss = 0.0
    for i in range(max_iter):
        x, y = next(train_loader)
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        f_x = model(x)
        loss = criterion(f_x, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        if(i % PRINT_FREQ == 0):
            acc = batch_accuracy_fn(model, test_loader)
            running_loss /= PRINT_FREQ
            logging.info("Iter. {} Acc. {} Loss {}".format(i, acc, running_loss))
            running_loss = 0.0
            if(acc >= best_acc):
                best_acc = acc
                save_model(model, model_path)
                logging.info("Save Model (Acc. = {})".format(best_acc))
    return model
            
        
def transform_mnist_to_cifar(x):
    x = torch.repeat_interleave(x, 3, dim = 1)
    x = F.interpolate(x, size = (32, 32))
    return x

def transform_stl_to_cifar(x):
    x = F.interpolate(x, size = (32, 32))
    return x
    
    
def generate_outputs(prv_model, test_loader):
    prv_model.eval()
    probs = list()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x, y = batch
            x = x.cuda()
            x = transform_stl_to_cifar(x)
            # x = transform_mnist_to_cifar(x)
            f_x = prv_model(x)
            f_x = F.softmax(f_x, dim = 1)
            probs.append(f_x.detach().cpu().numpy())
    probs = np.concatenate(probs, axis = 0)
    logging.info("Size of Probs {}".format(probs.shape))
    prv_model.train()
    return probs
        
        

def confidence_stat(probs_out):
    probs_out = np.max(probs_out, axis = 1)
    hist = np.histogram(probs_out, bins = 10)
    logging.info("Histogram {}".format(hist))
    return hist 
    
    

def knockoff(prv_ds = 'cifar10', pub_ds = 'mnist'):
    logging.info("Using {} to Knockoff {}".format(pub_ds.upper(), prv_ds.upper()))
    prv_train_set, prv_test_set = load_dataset(prv_ds)
    pub_train_set, pub_test_set = load_dataset(pub_ds)
    batch_size = 64
    pattern  = "{}_knockoff_{}"
    # then train a model
    prv_train_loader = CircularFeeder(prv_train_set, batch_size = batch_size)
    prv_test_loader =  torch.utils.data.DataLoader(prv_test_set, batch_size = batch_size)
    prv_model_path = "{}_optim.cpt".format(prv_ds)
    # train_and_save_model(prv_ds, prv_train_loader, prv_test_loader, "{}_optim.cpt".format(prv_ds))
    if(os.path.isfile(prv_model_path)):
        logging.info("Loading Model on {}".format(prv_ds))
        prv_model = model_initializer(prv_ds)
        prv_model.load_state_dict(torch.load(prv_model_path))
    else:
        logging.info("Training Model on {}".format(prv_ds))
        prv_model = train_and_save_model(prv_ds, prv_train_loader, prv_test_loader, prv_model_path)
    prv_model.cuda()
    pub_test_loader = torch.utils.data.DataLoader(pub_test_set, batch_size = batch_size, shuffle = False)
    # now evaluate the mnist samples
    probs_out = generate_outputs(prv_model, pub_test_loader)
    np_save_path = pattern.format(pub_ds, prv_ds)+".npy"
    logging.info("Save probs out in {}".format(np_save_path))
    np.save(np_save_path, probs_out)
    # convert the soft label into tensor
    probs_out = torch.FloatTensor(probs_out).cuda()
    # get th input
    pub_test_loader = list(pub_test_loader)

    if(pub_ds in ['mnist']):
        x = [transform_mnist_to_cifar(x) for x, _ in pub_test_loader]
    else:
        x = [transform_stl_to_cifar(x) for x, _ in pub_test_loader]
    x = torch.cat(x, dim = 0)
    knockoff_ds = torch.utils.data.TensorDataset(x, probs_out)
    knockoff_train_loader = cycle(list(torch.utils.data.DataLoader(knockoff_ds, batch_size, shuffle = True)))
    knockoff_path = pattern.format(pub_ds, prv_ds)+".cpt"
    train_and_save_model(prv_ds, knockoff_train_loader, prv_test_loader, knockoff_path, criterion = LabelSmoothingLoss())
    

    
def analyze_confidence():
    pattern  = "{}_knockoff_{}"
    np_save_path = pattern.format("mnist", "cifar10-large")+".npy"
    probs_out = np.load(np_save_path)
    # do knockoff
    confidence_stat(probs_out)
    
    
    
        
    

if __name__ == '__main__':
    knockoff("cifar10-large", "stl10")
    # analyze_confidence()
