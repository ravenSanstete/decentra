## generate the worker configuration
import numpy as np
import torch.utils.data as data_utils
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, filemode = 'w+')

import random

def random_sampler(batches):
    while True:
        yield random.choice(batches)



def initialize_mode(mode, n, train_loader, base_data_size = 1000, batch_size = 128, default_role= 'UPDATE'):
    # @param mode: the mode to simulate
    # @param n: the number of workers
    ratios = np.array([0]*n)
    roles = [default_role]*n
    worker_data_sizes = [base_data_size for i in range(n)]
    if(mode == 'FLIP'):
        GROUP_COUNT = 1
        ratios = np.arange(0, 1.0, 1.0/n)
        for i in range(len(ratios)//group_count):
            for j in range(group_count):
                ratios[group_count*i + j] = ratios[group_count*i]
    elif(mode == 'HETERO'):
        # with different data sizes
        worker_data_sizes = [base_data_size*i for i in range(1, n+1)]
    elif(mode == 'HOMEO'):
        worker_data_sizes = [base_data_size for i in range(1, n+1)]
    elif(mode == 'PARASITE'):
        roles[0] = 'FREE_RIDER'
    elif(mode == 'FULL'):
        worker_data_sizes = [len(train_loader) for i in range(1, n+1)]
        
    # initialize the data loaders
    train_loaders = []
    for i in range(n):
        x, y = train_loader.next_with_noise(worker_data_sizes[i], tamper_ratio = ratios[i])
        logging.info("Worker {} Data Size {} with {} ratio flipped Role {}".format(i, worker_data_sizes[i], ratios[i], roles[i]))
        ds = data_utils.TensorDataset(x, y)
        train_loaders.append(random_sampler(list(data_utils.DataLoader(ds, batch_size = batch_size, shuffle = True))))
    return train_loaders, roles
    
        
        
