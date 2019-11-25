# implement my own feeder for simulating a large-scale distributed learning environment

import torch
import random
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
# may use the rabbitmq to fix the data in memory for multiple run of the program
        
class CircularFeeder(object):
    def __init__(self, data_source, verbose = True, batch_size = 32):
        # fix data in memory
        logging.debug("LOAD DATA INTO MEMORY")
        self.data_source = list(data_source)
        random.shuffle(self.data_source)
        self.n = len(data_source)
        self.ptr = 0
        self.epoch_counter = 1
        self.verbose = verbose
        self.batch_size = batch_size
    
    def next(self, size):
        if(self.ptr + size >= self.n):
            batch = self.build(self.data_source[self.ptr:] + self.data_source[:(self.ptr + size - self.n)])
            self.ptr = self.ptr + size - self.n
            if(self.verbose):
                print("Epoch {} Finished".format(self.epoch_counter))
            self.epoch_counter += 1
            random.shuffle(self.data_source)
            
        else:
            batch = self.build(self.data_source[self.ptr:self.ptr + size])
            self.ptr += size
        return batch

    """
    This function tampers a specified proportion of samples by label flipping
    """
    def next_with_noise(self, size, tamper_ratio = 0):
        x, y = self.next(size)
        tamper_num = int(tamper_ratio * size)
        if(tamper_num == 0):
            return x, y
        # select the random id
        rand_idx = np.random.choice(list(range(x.shape[0])), tamper_num, replace = True)
        # label flipping
        y[rand_idx] = 9 - y[rand_idx]
        return x, y
        
    def __next__(self):
        return self.next(self.batch_size)
    
    def build(self, raw):
        n = len(raw)
        # print(raw[0][0].shape)
        x = torch.zeros([n, raw[0][0].shape[0], raw[0][0].shape[1], raw[0][0].shape[2]], dtype=torch.float32)
        y = torch.zeros([n], dtype = torch.long)
        # x = torch.tensor
        for i in range(n):
            x[i, :, :, :] = raw[i][0]
            y[i] = raw[i][1] 
        return x, y
