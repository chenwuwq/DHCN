import os
import torch
import argparse
import random
import numpy as np
from solver.Solver import Model_Solver

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# seed
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

def main(config):
    folder_path = {
        'sjtu': 'your dataset path/SJTU-PCQA/',
        'wpc': 'your dataset path/WPC/',
        'siat': 'your dataset path/SIAT-PCQD/',
    }
    SRCC_all = np.zeros(config.train_test_num, dtype=np.float64)
    PLCC_all = np.zeros(config.train_test_num, dtype=np.float64)
    KRCC_all = np.zeros(config.train_test_num, dtype=np.float64)
    RMSE_all = np.zeros(config.train_test_num, dtype=np.float64)

    # Randomly select 80% images for training and the rest for testing
    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
    for i in range(config.train_test_num):
        print('Round %d' % (i + 1))
        path = 'weight_path'
        solver = Model_Solver(config, folder_path[config.dataset])
        SRCC_all[i], PLCC_all[i], KRCC_all[i], RMSE_all[i] = solver.test(solver.test_data, path)
        print("test: ", SRCC_all[i], PLCC_all[i], KRCC_all[i], RMSE_all[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='wpc', help='')
    parser.add_argument('--resume', dest='resume', type=bool, default=False, help='')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='Epochs for training')
    parser.add_argument('--image_size', dest='image_size', type=int, default=224, help='')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=1, help='Train-test times')
    parser.add_argument('--model_name', dest='model_name', type=str, default="DHCN", help='')
    parser.add_argument('--split', dest='split', type=int, default=1, help='SJTU:[1-9],WPC:[1-5],SIAT:[1-5]')
    config = parser.parse_args()
    main(config)


