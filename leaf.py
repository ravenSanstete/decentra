# use the leaf data to create the dataloader for each user
import os
import numpy as np
from model_utils import read_data
from scipy.stats import describe



PREFIX = '/home/mlsnrs/data/data/pxd/leaf/data/'

def get_local_datasets(dataset, num_users = 10, use_val_set = False):
    """Instantiates clients based on given train and test data directories.
    Return:
        all_clients: list of Client objects.
    """    
    eval_set = 'test' if not use_val_set else 'val'
    
    train_data_dir = os.path.join(PREFIX, dataset, 'data', 'train')
    test_data_dir = os.path.join(PREFIX, dataset, 'data', eval_set)

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    # clients = create_clients(users, groups, train_data, test_data, model)
    print(len(users))
    num_samples = []
    
 

    # random sample users
    sampled_users = np.random.permutation(users)[:num_users]

    for u in sampled_users:
        num_samples.append(len(train_data[u]['y']))
        # print(train_data[u])
    print("Statistics of Local Datasets:", describe(num_samples))
    
    return [(train_data[u], test_data[u]) for u in sampled_users]




# to construct a number of femnist dataloaders

def main():
    get_local_datasets('femnist')

if __name__ == '__main__':
    main()
