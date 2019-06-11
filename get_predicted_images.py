# Script for extracting and saving predicted images
import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from data import *
from activations import *
from PredNet import *
from ConvLSTM import *
from utils import *

parser = argparse.ArgumentParser()
# Training data
parser.add_argument('--test_data_hkl',
                    default='../data/kitti_data/X_test.hkl',
                    help='Path to test images hkl file')
parser.add_argument('--test_sources_hkl',
                    default='../data/kitti_data/sources_test.hkl',
                    help='Path to test sources hkl file')
parser.add_argument('--seq_len', type=int, default=10,
                    help='Number of images in each kitti sequence')
parser.add_argument('--num_seqs', type=int, default=5,
                    help='Number of (random) sequences of predictions to save')

# Models
parser.add_argument('--model_type', choices=['PredNet','ConvLSTM'],
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
parser.add_argument('--use_satlu', type=str2bool, default=True,
                    help='Boolean indicating whether to use SatLU in Ahat.')
parser.add_argument('--satlu_act', default='hardtanh',
                    choices=['hardtanh','logsigmoid'],
                    help='Type of activation to use for SatLU in Ahat.')
parser.add_argument('--pixel_max', type=float, default=255.0,
                    help='Maximum output value for Ahat if using SatLU.')
parser.add_argument('--error_act', default='relu',
                    choices=['relu','sigmoid','tanh','hardsigmoid'],
                    help='Type of activation to use in E modules.')
parser.add_argument('--use_1x1_out', type=str2bool, default=False,
                    help='Boolean indicating whether to use 1x1 conv layer' +
                         'for output of ConvLSTM cells')
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

# Output options
parser.add_argument('--results_dir', default='../results/images/defaults',
                    help='Results subdirectory to save results')
parser.add_argument('--out_data_file', default='prednet_defaults',
                    help='Name of output png files')

def main(args):
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Data: Don't shuffle to keep indexes consistent
    test_data = KITTI(args.test_data_hkl,args.test_sources_hkl,args.seq_len)

    # Load model
    if args.model_type == 'PredNet':
        model = PredNet(args.in_channels,args.stack_sizes,args.R_stack_sizes,
                        args.A_kernel_sizes,args.Ahat_kernel_sizes,
                        args.R_kernel_sizes,args.use_satlu,args.pixel_max,
                        args.satlu_act,args.error_act,args.LSTM_act,
                        args.LSTM_c_act,args.bias,args.use_1x1_out,args.FC,
                        device)
    elif args.model_type == 'ConvLSTM':
        model = ConvLSTM(args.in_channels,args.hidden_channels,args.kernel_size,
                         args.LSTM_act,args.LSTM_c_act,args.out_act,
                         args.bias,args.FC,device)

    if args.load_weights_from is not None:
        model.load_state_dict(torch.load(args.load_weights_from))
    model.to(device)

    # Get random indices of sequences to save
    total_seqs = len(test_data)
    seq_ids = np.random.choice(np.arange(total_seqs),size=args.num_seqs,
                               replace=False)

    dir = args.results_dir
    if not os.path.isdir(dir):
        os.mkdir(dir)

    # Get predicted images
    model.eval()
    with torch.no_grad():
        for i in seq_ids:
            X = test_data[i].to(device)
            X = X.unsqueeze(0) # Add batch dim
            seq_len = X.shape[1]
            if args.model_type == 'PredNet':
                preds,errors = model(X)
            else:
                preds = model(X)
            preds = preds.squeeze(0).permute(0,2,3,1) # (len,H,W,channels)
            preds = preds.cpu().numpy()
            X = X.squeeze(0).permute(0,2,3,1) # (len,H,W,channels)
            X = X.cpu().numpy()
            for t in range(seq_len):
                X_t = np.uint8(X[t])
                X_img = Image.fromarray(X_t)
                fn = args.out_data_file
                X_img_path = '%s/%s_X%d_t%d.png' % (dir,fn,i,t)
                print("Saving image at %s" % img_path)
                img.save(pred_img_path)
                if t < seq_len - 1: # 1 less prediction
                    preds_t = np.uint8(preds[t])
                    img = Image.fromarray(preds_t)
                    pred_img_path = '%s/%s_pred%d_t%d.png' % (dir,fn,i,t+1)
                    print("Saving image at %s" % img_path)
                    img.save(pred_img_path)
        print("Done")

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
