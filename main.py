import os
import time
import argparse

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from utils import *
from mp_train import train, test

# Things to do:
#   -Way to initialize test method after train?
#   -Integrate with train.py
#       -figure out cuda stuff
#       -Include option for pin_memory if using cuda

parser = argparse.ArgumentParser()
parser.add_argument('--seed',type=int, default=0,
                    help='Manual seed for torch random number generator')
# Training data
parser.add_argument('--dataset',choices=['KITTI','CCN'],default='KITTI',
                    help='Dataset to use')
parser.add_argument('--train_data_path',
                    default='../data/kitti_data/X_train.hkl',
                    help='Path to training images hkl file')
parser.add_argument('--train_sources_path',
                    default='../data/kitti_data/sources_train.hkl',
                    help='Path to training sources hkl file')
parser.add_argument('--val_data_path',
                    default='../data/kitti_data/X_val.hkl',
                    help='Path to validation images hkl file')
parser.add_argument('--val_sources_path',
                    default='../data/kitti_data/sources_val.hkl',
                    help='Path to validation sources hkl file')
parser.add_argument('--test_data_path',
                    default='../data/kitti_data/X_test.hkl',
                    help='Path to test images hkl file')
parser.add_argument('--test_sources_path',
                    default='../data/kitti_data/sources_test.hkl',
                    help='Path to test sources hkl file')
parser.add_argument('--seq_len',type=int,default=10,
                    help='Number of images in each kitti sequence')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Samples per batch')
parser.add_argument('--num_iters', type=int, default=75000,
                    help='Number of optimizer steps before stopping')

# Models
parser.add_argument('--model_type', choices=['PredNet','ConvLSTM',
                                             'MultiConvLSTM'],
                    default='PredNet', help='Type of model to use.')
# Hyperparameters for PredNet
parser.add_argument('--stack_sizes', type=int, nargs='+', default=[3,48,96,192],
                    help='number of channels in targets (A) and ' +
                         'predictions (Ahat) in each layer. ' +
                         'Length should be equal to number of layers')
parser.add_argument('--R_stack_sizes', type=int, nargs='+',
                    default=[3,48,96,192],
                    help='Number of channels in R modules. ' +
                         'Length should be equal to number of layers')
parser.add_argument('--A_kernel_sizes', type=int, nargs='+', default=[3,3,3],
                    help='Kernel sizes for each A module. ' +
                         'Length should be equal to (number of layers - 1)')
parser.add_argument('--Ahat_kernel_sizes', type=int, nargs='+',
                    default=[3,3,3,3], help='Kernel sizes for each Ahat' +
                    'module. Length should be equal to number of layers')
parser.add_argument('--R_kernel_sizes', type=int, nargs='+', default=[3,3,3,3],
                    help='Kernel sizes for each Ahat module' +
                         'Length should be equal to number of layers')
parser.add_argument('--Ahat_act', default='relu',
                    choices=['relu','sigmoid','tanh','hardsigmoid'],
                    help='Type of activation for output of Ahat cell.')
parser.add_argument('--use_satlu', type=str2bool, default=True,
                    help='Boolean indicating whether to use SatLU in Ahat.')
parser.add_argument('--satlu_act', default='hardtanh',
                    choices=['hardtanh','logsigmoid'],
                    help='Type of activation to use for SatLU in Ahat.')
parser.add_argument('--pixel_max', type=float, default=1.0,
                    help='Maximum output value for Ahat if using SatLU.')
parser.add_argument('--error_act', default='relu',
                    choices=['relu','sigmoid','tanh','hardsigmoid'],
                    help='Type of activation to use in E modules.')
parser.add_argument('--use_1x1_out', type=str2bool, default=False,
                    help='Boolean indicating whether to use 1x1 conv layer' +
                         'for output of ConvLSTM cells')
parser.add_argument('--send_acts', type=str2bool, default=False,
                    help='Boolean indicating whether to send activities' +
                         'rather than errors')
parser.add_argument('--no_ER', type=str2bool, default=False,
                    help='Boolean indicating whether to ablate connection' +
                         'between E_l and R_l on all but last layer')
parser.add_argument('--RAhat', type=str2bool, default=False,
                    help='Boolean indicating whether to add connection' +
                         'between R_lp1 and Ahat_l')
parser.add_argument('--local_grad', type=str2bool, default=False,
                    help='Boolean indicating whether to restrict gradients ' +
                         'to flow locally (within each layer)')
# Hyperparameters for ConvLSTM
parser.add_argument('--hidden_channels', type=int, default=192,
                    help='Number of channels in hidden states of ConvLSTM')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='Kernel size in ConvLSTM')
parser.add_argument('--out_act', default='relu',
                    help='Activation for output layer of ConvLSTM cell')
# Hyperparameters shared by PredNet and ConvLSTM
parser.add_argument('--in_channels', type=int, default=3,
                    help='Number of channels in input images')
parser.add_argument('--LSTM_act', default='tanh',
                    choices=['relu','sigmoid','tanh','hardsigmoid'],
                    help='Type of activation to use in ConvLSTM.')
parser.add_argument('--LSTM_c_act', default='hardsigmoid',
                    choices=['relu','sigmoid','tanh','hardsigmoid'],
                    help='Type of activation for inner ConvLSTM (C_t).')
parser.add_argument('--bias', type=str2bool, default=True,
                    help='Boolean indicating whether to use bias units')
parser.add_argument('--FC', type=str2bool, default=False,
                    help='Boolean indicating whether to use fully connected' +
                         'convolutional LSTM cell')
parser.add_argument('--load_weights_from', default=None,
                    help='Path to saved weights')

# Optimization
parser.add_argument('--loss', default='E',choices=['E','MSE','L1'])
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Fixed learning rate for Adam optimizer')
parser.add_argument('--lr_steps', type=int, default=1,
                    help='num times to decrease learning rate by factor of 0.1')
parser.add_argument('--layer_lambdas', type=float,
                    nargs='+', default=[1.0,0.0,0.0,0.0],
                    help='Weight of loss on error of each layer' +
                         'Length should be equal to number of layers')

# Output options
parser.add_argument('--results_dir', default='../results/train_results',
                    help='Results subdirectory to save results')
parser.add_argument('--out_data_file', default='results.json',
                    help='Name of output data file with training loss data')
parser.add_argument('--checkpoint_path',default=None,
                    help='Path to output saved weights.')
parser.add_argument('--record_loss_every', type=int, default=20,
                    help='iters before printing and recording loss')

def init_process(rank, world_size, fn, args, backend='mpi'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn(rank, world_size, args)

if __name__ == '__main__':
    args = parser.parse_args()

    # Get environment variables from mpi
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])

    if world_rank == 0:
        print(args)
        print("MKL is available: ", torch.backends.mkl.is_available())
        print("MKL DNN is available: ", torch._C.has_mkldnn)
        print("MPI is available: ", torch.distributed.is_mpi_available())
        # TODO: cuda stuff

    # Train
    start_train_time = time.time()
    init_process(world_rank,world_size,train,args)
    print("Total training time: ", time.time() - start_train_time)
    dist.barrier()

    # Test
    #start_test_time = time.time()
    #init_process(world_rank,world_size,test,args)
    #print("Total testing time: ", time.time() - start_test_time)
