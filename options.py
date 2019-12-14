import argparse

parser = argparse.ArgumentParser(description='Ripple Attack')

parser.add_argument("--config", "-c", type=str, default='config_toy.txt', help = "path that contains the configuration of the system topology")
parser.add_argument("--dataset", type=str, default = "mnist", help = 'the dataset we use for testing')
parser.add_argument("-b", action="store_true", help = "whether the system topo is bidirectional or not")
parser.add_argument("--atk", type=str, default="NORMAL", help="the role of worker 0, the only adversary in the system")
parser.add_argument("-n", type=int, default = 1, help = "the physical worker num")
parser.add_argument("--round", type=int, default = 10000, help = "the total training round")
parser.add_argument("--gpu", default = "0,1", help = "the gpu used to train model")
parser.add_argument("--flip", type = bool, default = False, help = "generate flip-attack parameters")
parser.add_argument("--backdoor", type = bool, default = False, help = "generate backdoor attack parameters")
parser.add_argument("--target", type = int, default = 1, help = "the targeted label of backdoor attack")
parser.add_argument("--frequency", type = int, default = 1, help = "the frequency to test model and print results")

ARGS = parser.parse_args()