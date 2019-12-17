# generate typical topology file
import logging
import argparse


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)



parser = argparse.ArgumentParser(description='topo. generator')
parser.add_argument("--arg", type=int, default=3, help = "structural parameter of the specified topology")
parser.add_argument("--topo", type=str, default="chain", help = "name to specify the topology")
ARGS = parser.parse_args()

def generate_chain(num):
    path = "topo_{}_chain.txt".format(num)
    logging.info("Generate Chain of length {} in {}".format(num, path))
    f = open(path, 'w+')
    f.write(str(num)+"\n")
    for i in range(num-1):
        f.write("{} {}\n".format(i+1, i))
    f.close()


def generate_ring(num):
    path = "topo_{}_ring.txt".format(num)
    logging.info("Generate Ring of length {} in {}".format(num, path))
    f = open(path, 'w+')
    f.write(str(num)+"\n")
    for i in range(0, num):
        f.write("{} {}\n".format((i+1)%num, i))
        

GENERATOR_REGISTRY = {
    "chain": generate_chain,
    "ring": generate_ring
}

if __name__ == "__main__":
    GENERATOR_REGISTRY[ARGS.topo](ARGS.arg)
    
    
    
