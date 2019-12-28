import argparse
parser = argparse.ArgumentParser(description='Fairness')
parser.add_argument("--eta_d", type=float, default=1.0, help = "the proportion of downloadable parameters")
parser.add_argument("-n", type=int, default=10, help = "the number of workers")
parser.add_argument("--eta_r", type=float, default=1.0, help = "the proportion of uploading parameters")
parser.add_argument("--ds", type=str, default="cifar10", help = "the benchmark we use to test")
parser.add_argument("--mode", type=str, default='HETERO', help = 'the mode of workers to simulate')
parser.add_argument("-N", type = int, default=1000, help = 'the base data size')
parser.add_argument("--bs", type = int, default = 128, help = 'the batch size')
parser.add_argument("--lr", type = float, default = 0.1, help = 'the learning rate')

