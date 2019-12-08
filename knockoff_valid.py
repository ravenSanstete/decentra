# use the knockoff technique for peer-to-peer validation

import argparse
import os
import pprint
import time
import datetime
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO, filemode = 'w+')

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
import torchvision
import torch.utils.data as data_utils
from itertools import cycle
from tensorboardX import SummaryWriter



from model import model_initializer
from feeder import CircularFeeder
from utils import *
from functools import partial

import networkx as nx

from worker import Worker
import os.path
from tensorboardX import SummaryWriter



from imagenet_loader import get_imagenet_loader, transform_mnist_to_cifar, transform_stl_to_cifar, to_img, show

writer = SummaryWriter()

import argparse
parser = argparse.ArgumentParser(description='Fairness Bi-Validation')
parser.add_argument("--eta", type=float, default=0.0, help = "the ratio of tampered labels in the original train set")
parser.add_argument("--prv_ds", type=str, default="cifar10-large", help = "private dataset identity")
parser.add_argument("--pub_ds", type=str, default='mnist', help = "public dataset identity (for knockoff the private dataset)")
ARGS = parser.parse_args()


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


def train_and_save_model_torch(ds, train_loader, test_loader, model_path, criterion = F.cross_entropy, max_iter = 100, with_torch_loader = False):
    model = model_initializer(ds)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    PRINT_FREQ = 1
    best_acc = 0.0
    running_loss = 0.0
    for i in range(max_iter):
        for x, y in train_loader:
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


    
    
def generate_outputs(prv_model, test_loader):
    prv_model.eval()
    probs = list()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x, y = batch
            x = x.cuda()
            if(ARGS.pub_ds in ['mnist']):
                x = transform_mnist_to_cifar(x)
            else:
                x = transform_stl_to_cifar(x)
            # x = transform_mnist_to_cifar(x)
            f_x = prv_model(x)
            f_x = F.softmax(f_x, dim = 1)
            probs.append(f_x.detach().cpu().numpy())
    probs = np.concatenate(probs, axis = 0)
    logging.info("Size of Probs {}".format(probs.shape))
    prv_model.train()
    return probs
        


def generate_features(prv_model, test_loader):
    prv_model.eval()
    features = list()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x, y = batch
            x = x.cuda()
            if(ARGS.pub_ds in ['mnist']):
                x = transform_mnist_to_cifar(x)
            else:
                x = transform_stl_to_cifar(x)
            # x = transform_mnist_to_cifar(x)
            f_x = prv_model.get_last_layer(x)
            # f_x = F.softmax(f_x, dim = 1)
            features.append(f_x.detach().cpu().numpy())
            if(i > 600):
                break
    features = np.concatenate(features, axis = 0)
    logging.info("Size of Features {}".format(features.shape))
    prv_model.train()
    return features
    

def confidence_stat(probs_out):
    probs_out = np.max(probs_out, axis = 1)
    hist = np.histogram(probs_out, bins = 10)
    logging.info("Histogram {}".format(hist))
    return hist 
    
    

def knockoff(prv_ds = 'cifar10', pub_ds = 'mnist'):
    DO_KNOCKOFF = False

    prv_train_set, prv_test_set = load_dataset(prv_ds)
    pub_train_set, pub_test_set = load_dataset(pub_ds)
    batch_size = 64
    tamper_ratio = ARGS.eta
    training_sample_num = 60000
    pattern  = "{}_knockoff_{}_{:.1f}"
    logging.info("Using {} to Knockoff {}".format(pub_ds.upper(), prv_ds.upper()))
    # then train a model
    # add noise

    prv_train_loader = torch.utils.data.DataLoader(prv_train_set, batch_size = batch_size, num_workers = 12, pin_memory = True, shuffle = True)
    prv_test_loader =  torch.utils.data.DataLoader(prv_test_set, batch_size = batch_size, pin_memory = True, shuffle = False)
    
    prv_model_path = "{}_optim_{:.1f}.4.cpt".format(prv_ds, tamper_ratio)
    # train_and_save_model(prv_ds, prv_train_loader, prv_test_loader, "{}_optim.cpt".format(prv_ds))
    if(os.path.isfile(prv_model_path)):
        logging.info("Loading Model on {} from {}".format(prv_ds, prv_model_path))
        prv_model = model_initializer(prv_ds)
        prv_model.load_state_dict(torch.load(prv_model_path))
    else:
        logging.info("Training Model on {}".format(prv_ds))
        prv_model = train_and_save_model_torch(prv_ds, prv_train_loader, prv_test_loader, prv_model_path)
    sys.exit()
    prv_model.cuda()
    pub_test_loader = torch.utils.data.DataLoader(pub_test_set, batch_size = batch_size, shuffle = False)
    # now evaluate the mnist samples
    probs_out = generate_outputs(prv_model, pub_test_loader)
    np_save_path = pattern.format(pub_ds, prv_ds, tamper_ratio)+".npy"
    logging.info("Save probs out in {}".format(np_save_path))
    np.save(np_save_path, probs_out)

    
    if(DO_KNOCKOFF):
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



    
def generate_knockoff_prob(model_path, prv_ds, version):
    imagenet_loader = get_imagenet_loader()
    logging.info("Loading Model on {} from {}".format(prv_ds, model_path))
    prv_model = model_initializer(prv_ds)
    prv_model.cuda()
    prv_model.load_state_dict(torch.load(model_path))
    probs_out = generate_outputs(prv_model, imagenet_loader)
    pattern = "{}_knockoff_{}_{}"
    np_save_path = pattern.format("imagenet", prv_ds, version)+".npy"
    logging.info("Save probs out in {}".format(np_save_path))
    np.save(np_save_path, probs_out)
    return 

# 
def generate_knockoff_feature(model_path, prv_ds, version):
    imagenet_loader = get_imagenet_loader()
    logging.info("Loading Model on {} from {}".format(prv_ds, model_path))
    prv_model = model_initializer(prv_ds)
    prv_model.cuda()
    prv_model.load_state_dict(torch.load(model_path))
    features = generate_features(prv_model, imagenet_loader)
    pattern = "{}_knockoff_{}_{}_feature"
    np_save_path = pattern.format("imagenet", prv_ds, version)+".npy"
    logging.info("Save features out in {}".format(np_save_path))
    np.save(np_save_path, features)
    return 


def do_knockoff(prv_ds, version):
    batch_size = 64
    pattern = "{}_knockoff_{}_{}"
    _, prv_test_set = load_dataset(prv_ds)
    prv_test_loader =  torch.utils.data.DataLoader(prv_test_set, batch_size = batch_size)
    np_save_path = pattern.format("imagenet", prv_ds, version)+".npy"
    probs_out = np.load(np_save_path)
    probs_out = torch.FloatTensor(probs_out).cuda()
    pub_test_loader = get_imagenet_loader()
    pub_test_loader = list(pub_test_loader)
    x = [transform_stl_to_cifar(x) for x, _ in pub_test_loader]
    x = torch.cat(x, dim = 0)
    print(x.size())
    knockoff_ds = torch.utils.data.TensorDataset(x, probs_out)
    knockoff_train_loader = cycle(list(torch.utils.data.DataLoader(knockoff_ds, batch_size, shuffle = True)))

    knockoff_path = pattern.format("imagenet", prv_ds, version)+".cpt"
    train_and_save_model(prv_ds, knockoff_train_loader, prv_test_loader, knockoff_path, criterion = LabelSmoothingLoss())
    

## p@param: (N1, K), q@param: (N2, K), f@param: a torch function element wise
def f_div(p, q, f):
    K = p.size(1)
    sum = 0
    for k in range(K):
        sum += f(p[:, k].unsqueeze(1) @ q[:, k].unsqueeze(1).T) * ((q[:, k].T).unsqueeze(0).repeat_interleave(p.size(0), dim = 0))
    return sum

# as N1 is always much smaller than N2
def norm_dist(p, q, order = 1):
    N1 = p.size(0)
    out = torch.zeros((N1,q.size(0))).cuda()
    for i in range(N1):
        out[i, :] = (q - p[i, :]).norm(dim = 1, p = order)
    
    return out
    


    
    
def analyze_confidence():
    pattern  = "{}_knockoff_{}"
    np_save_path = pattern.format("mnist", "cifar10-large")+".npy"
    probs_out = np.load(np_save_path)
    # do knockoff
    confidence_stat(probs_out)

def analyze_credit():
    ratios = np.arange(0.0, 1.0, 0.1)
    pattern =  "{}_knockoff_{}_{:.1f}.npy"
    probs_out = []
    for r in ratios:
        probs_out.append(np.load(pattern.format(ARGS.pub_ds, ARGS.prv_ds, r)))
    for i in range(0, len(ratios)):
        dists = np.linalg.norm(probs_out[0] - probs_out[i], axis = 1)
        dists = np.mean(dists)
        print("0 -> {:.1f}: {:.5f}".format(ratios[i], dists))
    

def find_the_shadow(probs_out):
    pass





def plot_correspondence(model_path, prv_ds, version, mode = 'cosine'):
    batch_size = 64
    pattern = "{}_knockoff_{}_{}"
    logging.info("Loading Model on {} from {}".format(prv_ds, model_path))
    prv_model = model_initializer(prv_ds)
    prv_model.cuda()
    prv_model.load_state_dict(torch.load(model_path))
    prv_model.eval()
    # get the private dataset loader
    _, prv_test_set = load_dataset(prv_ds)
    prv_test_loader =  torch.utils.data.DataLoader(prv_test_set, batch_size = 1, shuffle = False)
    np_save_path = pattern.format("imagenet", prv_ds, version)+".npy"
    probs_out = np.load(np_save_path)
    # make the plot out smaller
    probs_out = torch.FloatTensor(probs_out).cuda()    
    image_idx = []
    # checking_idx = [416, 976]
    
    # imagenet_pred_label = probs_out.argmax(dim = 1).cpu().numpy()
    # logging.info("Histogram {}".format(np.histogram(imagenet_pred_label, bins = 10)))
    # writer.add_embedding(probs_out[:,:], global_step = 2 * (int(version)-1) + 0

    predicted_out = []
    cur = 0
    BUFFER_SIZE = 10
    for i, (x, y) in enumerate(prv_test_loader):
        candidates = probs_out[cur:cur + BUFFER_SIZE]

        with torch.no_grad():
            x = x.cuda()
            f_x = F.softmax(prv_model(x), dim = 1)
            # f_x = prv_model.get_last_layer(x)
        # f_x = F.softmax(prv_model(x), dim = 1) # to compute the probability vector
        # predicted_out.append(f_x.detach().cpu().numpy())
        
        # find the image with the closest output as x[i, :, :, :]
        if(mode == 'cosine'):
            cos_sim = f_x @ candidates.T
            idx = cos_sim.argmax(dim = 1)
        elif(mode == 'ce'):
            # compute the cross entropy distance
            ce_dist = - f_x @ torch.log(candidates.T)
            idx = ce_dist.argmin(dim = 1)
        elif(mode == 'chi'):
            func = lambda x: (x - 1)**2
            kl_dist = f_div(f_x, candidates, func)
            idx = kl_dist.argmin(dim = 1)
        elif(mode == 'TV'):
            func = lambda x: 0.5 * torch.abs(x - 1)
            kl_dist = f_div(f_x, candidates, func)
            idx = kl_dist.argmin(dim = 1)
        elif(mode == 'hel'):
            func = lambda x: (torch.sqrt(x) - 1) ** 2
            kl_dist = f_div(f_x, candidates, func)
            idx = kl_dist.argmin(dim = 1)
        elif(mode == 'euc'):
            # search the closest by Euclidean distance
            euc_dist = norm_dist(f_x, candidates, order = 2)
            idx = euc_dist.argmin(dim =1)

        # obtain the true id
        idx += cur
        image_idx.append(idx.cpu().detach().numpy())

        cur += BUFFER_SIZE
        if(cur >= 10000):
            break
    # predicted_out = np.concatenate(predicted_out, axis = 0)
    # writer.add_embedding(predicted_out, global_step = 2 * (int(version)-1) + 1)
    image_idx = np.concatenate(image_idx, axis = 0)
    print(image_idx.shape)
    return image_idx
    
    
def visualize_correspondence(cifar10_id, imagenet_id, fig_name = "visualize_correspondence"):
    prv_ds = "cifar10-large"
    logging.info("Loading Cifar10")
    _, prv_test_set = load_dataset(prv_ds)
    prv_test_set = [x.unsqueeze(0).numpy() for x, _ in list(prv_test_set)]
    prv_test_set = np.concatenate(prv_test_set, axis = 0)
    print(prv_test_set.shape)
    logging.info("Loading Imagenet")
    pub_test_set = np.load("imagenet.npy")

    original_imgs = to_img(torch.tensor(prv_test_set[cifar10_id, :, :, :]))
    imagenet_imgs = to_img(torch.tensor(pub_test_set[imagenet_id, :, :, :]))
    imgs = []
    for i in range(cifar10_id.shape[0]):
        imgs.append(original_imgs[i, :, :, :].unsqueeze(0))
        imgs.append(imagenet_imgs[i, :, :, :].unsqueeze(0))
    imgs = torch.cat(imgs, dim = 0)
    grid = torchvision.utils.make_grid(imgs, nrow = 16)
    show(grid, fig_name)
        
        
        
    
    
    
    
    
    

    

if __name__ == '__main__':
    # caltech_imagenet()
    
    # knockoff(ARGS.prv_ds, ARGS.pub_ds)
    # sys.exit()
    # analyze_confidence()
    # plot_most_influential_dp(ARGS.prv_ds, ARGS.pub_ds)
    # analyze_credit()
    # visualize_correspondence(None, None)
    # sys.exit()
    #model_path = "cifar10-large_optim_0.0.{}.cpt".format(4)
    # generate_knockoff_prob(model_path, "cifar10-large", 4)
    # sys.exit()
    
    image_idx_buffer = []
    MODE = 'cosine'
    
    for version in ["3", "4"]:
        model_path = "cifar10-large_optim_0.0.{}.cpt".format(version)
        # generate_knockoff_feature(model_path, "cifar10-large", version)
        # do_knockoff("cifar10-large", version)
        idx = plot_correspondence(model_path, "cifar10-large", version, MODE)
        image_idx_buffer.append(idx)
    mask = (image_idx_buffer[0] == image_idx_buffer[1])
    print(np.mean(mask))
    
    cifar10_idx, = mask.nonzero()
    imagenet_idx = image_idx_buffer[0][mask]
    logging.info("Intersection CIFAR10 IDX: {}".format(cifar10_idx.tolist()))
    logging.info("Intersection ImageNet IDX: {}".format(imagenet_idx.tolist()))
    print(np.histogram(cifar10_idx, bins = 10))
    print(np.histogram(imagenet_idx, bins = 10))
    fig_name = "visualize_correspondence_{}".format(MODE)
    visualize_correspondence(cifar10_idx, imagenet_idx, fig_name = fig_name)
