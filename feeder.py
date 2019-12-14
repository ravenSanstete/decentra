# implement my own feeder for simulating a large-scale distributed learning environment

import torch
import random
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
# may use the rabbitmq to fix the data in memory for multiple run of the program
        
class CircularFeeder(object):
    def __init__(self, data_source, verbose = True):
        # fix data in memory
        # logging.debug("LOAD DATA INTO MEMORY")
        self.data_source = list(data_source)
        random.shuffle(self.data_source)
        self.n = len(data_source)
        self.ptr = 0
        self.epoch_counter = 1
        self.verbose = verbose
    
    def next(self, size, flipped = False):
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

        if(flipped):
            x, y = batch
            y[y == 1] = 7
            y[y >= 7] = 1
        return batch

    def build(self, raw):
        n = len(raw)
        # print(raw[0][0].shape)
        x = torch.zeros([n, raw[0][0].shape[0], raw[0][0].shape[1], raw[0][0].shape[2]], dtype=torch.float32)
        y = torch.zeros([n], dtype = torch.long)
        # x = torch.tensor
        for i in range(n):
            x[i, :, :, :] = raw[i][0]
            y[i] = raw[i][1] 
            # y[i] = 9 - raw[i][1]
        return x, y
