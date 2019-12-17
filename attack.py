import numpy as np
from utils import *
from options import ARGS
import logging
logging.basicConfig(filename = "log.out", level = logging.DEBUG)


def generate_two_hop_poison(param, grad, lr):
    K = 2
    # should determine the amplitude of the poison based on the blacksheep's out degree - 1
    poison = generate_random_fault(grad)
    return weighted_reduce_gradients([param, grad, poison], [-1, lr, 2]), poison

def load_aim(path):
    aim = np.load(path, allow_pickle = True)
    return aim.tolist()


INITIAL_PARAM = None
FINAL_AIM = None
TIME_STEP = 0


def generate_two_hop_poison_direct(i, param, grad, lr, poison_list, aim, reset_aim = False):
    global INITIAL_PARAM
    global FINAL_AIM
    global TIME_STEP
    step = 0.1

    if(reset_aim):
        TIME_STEP = 0
        INITIAL_PARAM = param
        FINAL_AIM = aim

    TIME_STEP += 1
    
    mu = min(1.0, step * pow(1.1, TIME_STEP))
    logging.info("Interpolation Coeff. {:.4f}".format(mu))
    #aim = generate_random_fault(grad
    #poison = weighted_reduce_gradients([aim, param], [1, -1])
    small_aim = weighted_reduce_gradients([FINAL_AIM, INITIAL_PARAM], [mu, 1-mu])
    
    # small_aim = weighted_reduce_gradients([aim, param], [1, -1])
    # small_aim = [mu * x for x in small_aim]
    # small_aim = weighted_reduce_gradients([small_aim, param], [1, 1])
    poison = weighted_reduce_gradients([small_aim, param], [1, -1])
    
    poison = [x for x in poison] 
    poison_list.put(small_aim)
    # small_aim = weighted_reduce_gradients([small_aim, small_aim], [2, 2])
    # 
    
    return weighted_reduce_gradients([param, grad, poison], [1, -lr, 4]), poison


def generate_two_hop_poison_slight(i, param, grad, lr, poison_list, aim, reset_aim = False):
    mu = 0.1
    small_aim = weighted_reduce_gradients([aim, param], [1, -1])
    small_aim = [mu * x for x in small_aim]
    #logging.info("aim: {}".format(small_aim[-1]))
    small_aim = weighted_reduce_gradients([small_aim, param], [1, 1])
    poison = weighted_reduce_gradients([small_aim, param], [1, -1])
    
    poison = [x for x in poison]
    # small_aim = weighted_reduce_gradients([small_aim, small_aim], [2, 2])
    poison_list.put(small_aim)
    next_round_param = None
    if(i == 0):
        next_round_param = weighted_reduce_gradients([param, grad], [1, -lr])
    else:
        next_round_param = weighted_reduce_gradients([param, poison], [1, 1])
    poison = weighted_reduce_gradients([param, poison], [1, 4])
    return poison, next_round_param
    



def MNIST_staged_attack(i):
    reset_aim = (i == 0) or (i == 11)
    if(i <= 20):
        logging.info("send trained parameter")
        aim = load_aim("param_normal.npy")
    else:
        logging.info("send poison parameter")
        aim = load_aim("param_flip7to1.npy")
    return aim, reset_aim
    
def CIFAR10_recovery(i):
    reset_aim = (i == 0)
    logging.info("send trained parameter of convnet")
    aim = load_aim("param_cifar10.npy")
    return aim, reset_aim

def CIFAR10_backdoor(i):
    reset_aim = (i == 0)
    #logging.info("send backdoor parameter of convnet")
    aim = load_aim("param_cifar10_backdoor.npy")
    return aim, reset_aim

def CIFAR10_large_recovery(i):
    reset_aim = (i == 0)
    if(i <= 50):
        logging.info("send trained parameter of resnet18")
        aim = load_aim("param_cifar10_large.npy")
    else:
        logging.info("send poison parameter")
        aim = load_aim("param_cifar10_large_flip3class.npy")
    return aim, reset_aim

def CIFAR10_large_backdoor(i):
    reset_aim = (i == 0)
    #logging.info("send backdoor parameter of convnet")
    aim = load_aim("param_cifar10-large_backdoor_{}.npy".format(ARGS.target))
    return aim, reset_aim



ATTACK_REGISTRY = {
    "mnist": MNIST_staged_attack,
    #"cifar10": CIFAR10_recovery,
    "cifar10": CIFAR10_backdoor,
    #"cifar10-large": CIFAR10_large_recovery,
    "cifar10-large": CIFAR10_large_backdoor,
}


def initialize_atk(dataset):
    return ATTACK_REGISTRY[dataset]
    
