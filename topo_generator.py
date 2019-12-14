# generate typical topology file
import logging
import argparse

import matplotlib
matplotlib.use('agg')
from smallworld.draw import draw_network
from smallworld import get_smallworld_graph
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
PREFIX = "config/"
P = lambda x: PREFIX + x

parser = argparse.ArgumentParser(description='topo. generator')
parser.add_argument("--arg", type=int, default=3, help = "structural parameter of the specified topology")
parser.add_argument("--topo", type=str, default="chain", help = "name to specify the topology")
ARGS = parser.parse_args()

def generate_chain(num):
    path = P("topo_{}_chain.txt".format(num))
    logging.info("Generate Chain of length {} in {}".format(num, path))
    f = open(path, 'w+')
    f.write(str(num)+"\n")
    for i in range(num-1):
        f.write("{} {}\n".format(i+1, i))
    f.close()


def generate_ring(num):
    path = P("topo_{}_ring.txt".format(num))
    logging.info("Generate Ring of length {} in {}".format(num, path))
    f = open(path, 'w+')
    f.write(str(num)+"\n")
    for i in range(0, num):
        f.write("{} {}\n".format((i+1)%num, i))
    f.close()
        

def generate_small_world(num):
    path = P("topo_{}_small_world.txt".format(num))
    f = open(path, 'w+')
    f.write(str(num)+"\n")
    k_over_2 = 2 # param
    beta = 0.5 # param
    fig, ax = plt.subplots(figsize=(3,3), ncols=1, nrows=1)
    G = get_smallworld_graph(num, k_over_2, beta)
    for idx, nbrs in G.adj.items():
        for nbr, _ in nbrs.items():
            f.write("{} {}\n".format(idx, nbr))
    f.close()
    logging.info("Generate a Small World with N = {} K = {} Beta = {} in {}".format(num, k_over_2, beta, path))
    draw_network(G,k_over_2,focal_node=0,ax=ax)
    plt.savefig(path.split('.')[0]+".png", dpi = 108)
    return
    
    

        
GENERATOR_REGISTRY = {
    "chain": generate_chain,
    "ring": generate_ring,
    "sw": generate_small_world
}

if __name__ == "__main__":
    GENERATOR_REGISTRY[ARGS.topo](ARGS.arg)
    
    
    
