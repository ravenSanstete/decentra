import numpy as np
from utils import *
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
    mu = min(1.0, step * TIME_STEP)
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
    
    return weighted_reduce_gradients([param, grad, poison], [1, -lr, 4]), poison




def MNIST_staged_attack(i):
    reset_aim = (i == 0) or (i == 11)
    if(i <= 10):
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


def CIFAR10_large_recovery(i):
    reset_aim = (i == 0)
    logging.info("send trained parameter of resnet18")
    aim = load_aim("param_cifar10_large.npy")
    return aim, reset_aim



ATTACK_REGISTRY = {
    "mnist": MNIST_staged_attack,
    "cifar10": CIFAR10_recovery,
    "cifar10-large": CIFAR10_large_recovery
}


def initialize_atk(dataset):
    return ATTACK_REGISTRY[dataset]
    
