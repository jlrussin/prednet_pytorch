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
from Ladder import *
from StackedConvLSTM import *
from utils import *

parser = argparse.ArgumentParser()
# Training data
parser.add_argument('--dataset',choices=['KITTI','CCN'],default='KITTI',
                    help='Dataset to use')
parser.add_argument('--sanity_check',type=str2bool,default=False,
                    help='Change last four images - seqs are unpredictable')
parser.add_argument('--test_data_path',
                    default='../data/kitti_data/X_test.hkl',
                    help='Path to test images hkl file')
parser.add_argument('--test_sources_path',
                    default='../data/kitti_data/sources_test.hkl',
                    help='Path to test sources hkl file')
parser.add_argument('--seq_len', type=int, default=10,
                    help='Number of images in each kitti sequence')
parser.add_argument('--last_only',type=str2bool,default=False,
                    help='Test on sequences of static (final) images.')
parser.add_argument('--num_seqs', type=int, default=5,
                    help='Number of (random) sequences of predictions to save')

# Models
parser.add_argument('--model_type', choices=['PredNet','ConvLSTM',
                                             'MultiConvLSTM','LadderNet',
                                             'StackedConvLSTM'],
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
parser.add_argument('--A_act', default='relu',
                    choices=['relu','lrelu','sigmoid','tanh','hardsigmoid'],
                    help='Type of activation for output of Ahat cell.')
parser.add_argument('--Ahat_act', default='relu',
                    choices=['relu','lrelu','sigmoid','tanh','hardsigmoid'],
                    help='Type of activation for output of Ahat cell.')
parser.add_argument('--use_satlu', type=str2bool, default=True,
                    help='Boolean indicating whether to use SatLU in Ahat.')
parser.add_argument('--satlu_act', default='hardtanh',
                    choices=['hardtanh','logsigmoid','sigmoid'],
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
parser.add_argument('--conv_dilation', type=int, default=1,
                    help='Dilation for convolution in ACells')
parser.add_argument('--use_BN', type=str2bool, default=False,
                    help='Boolean indicating whether to use batch ' +
                         'normalization on inputs in all A and Ahat cells')
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
parser.add_argument('--dropout_p', type=float, default=0.0,
                    help='Proportion dropout for inputs to R cells')
parser.add_argument('--load_weights_from', default=None,
                    help='Path to saved weights')
# Hyperparameters unique to LadderNet
parser.add_argument('--no_R0', type=str2bool, default=True,
                    help='Boolean indicating whether not to include' +
                         'ConvLSTM in first layer of LadderNet')
parser.add_argument('--no_skip0', type=str2bool, default=True,
                    help='Boolean indicating whether not to include' +
                         'skip connection in first layer of LadderNet')

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
    if args.dataset == 'KITTI':
        test_data = KITTI(args.test_data_path,args.test_sources_path,
                          args.seq_len)
    elif args.dataset == 'CCN':
        test_data = CCN(args.test_data_path,args.seq_len,
                        last_only=args.last_only)

    # Load model
    model_out = 'pred' # Always pred to get predicted images
    if args.model_type == 'PredNet':
        model = PredNet(args.in_channels,args.stack_sizes,args.R_stack_sizes,
                        args.A_kernel_sizes,args.Ahat_kernel_sizes,
                        args.R_kernel_sizes,args.use_satlu,args.pixel_max,
                        args.Ahat_act,args.satlu_act,args.error_act,
                        args.LSTM_act,args.LSTM_c_act,args.bias,
                        args.use_1x1_out,args.FC,args.dropout_p,
                        args.send_acts,args.no_ER,args.RAhat,args.local_grad,
                        args.conv_dilation,args.use_BN,model_out,device)
    elif args.model_type == 'MultiConvLSTM':
        model = MultiConvLSTM(args.in_channels,args.R_stack_sizes,
                              args.R_kernel_sizes,args.use_satlu,args.pixel_max,
                              args.Ahat_act,args.satlu_act,args.error_act,
                              args.LSTM_act,args.LSTM_c_act,args.bias,
                              args.use_1x1_out,args.FC,args.local_grad,
                              model_out,device)
    elif args.model_type == 'ConvLSTM':
        model = ConvLSTM(args.in_channels,args.hidden_channels,args.kernel_size,
                         args.LSTM_act,args.LSTM_c_act,args.out_act,
                         args.bias,args.FC,device)
    elif args.model_type == 'LadderNet':
        model = LadderNet(args.in_channels,args.stack_sizes,args.R_stack_sizes,
                          args.A_kernel_sizes,args.Ahat_kernel_sizes,
                          args.R_kernel_sizes,args.conv_dilation,args.use_BN,
                          args.use_satlu,args.pixel_max,args.A_act,
                          args.Ahat_act,args.satlu_act,args.error_act,
                          args.LSTM_act,args.LSTM_c_act,args.bias,
                          args.use_1x1_out,args.FC,args.no_R0,args.no_skip0,
                          args.local_grad,model_out,device)
    elif args.model_type == 'StackedConvLSTM':
        model = StackedConvLSTM(args.in_channels,args.R_stack_sizes,
                                args.R_kernel_sizes,args.use_1x1_out,
                                args.FC,args.local_grad,model_out,device)

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
        for num,i in enumerate(seq_ids):
            if args.sanity_check: # Get first part of seq i, second part of i+1
                X_i = test_data[i]
                next_i = seq_ids[(num+1) % args.num_seqs]
                X_ip1 = test_data[next_i]
                halfway = args.seq_len//2
                X = torch.cat((X_i[:halfway],X_ip1[halfway:]),dim=0)
                X = X.to(device)
            else:
                X = test_data[i].to(device)
            X = X.unsqueeze(0) # Add batch dim
            seq_len = X.shape[1]
            preds = model(X)
            preds = preds.squeeze(0).permute(0,2,3,1) # (len,H,W,channels)
            preds = preds.cpu().numpy()
            X = X.squeeze(0).permute(0,2,3,1) # (len,H,W,channels)
            X = X.cpu().numpy()
            if test_data.norm:
                X = np.round(X * 255.)
                preds = np.round(preds * 255.)
            for t in range(seq_len):
                X_t = np.uint8(X[t])
                X_img = Image.fromarray(X_t)
                fn = args.out_data_file
                X_img_path = '%s/%s_X%d_t%d.png' % (dir,fn,i,t)
                print("Saving image at %s" % X_img_path)
                X_img.save(X_img_path)
                if t < seq_len - 1: # 1 less prediction
                    preds_t = np.uint8(preds[t])
                    pred_img = Image.fromarray(preds_t)
                    pred_img_path = '%s/%s_pred%d_t%d.png' % (dir,fn,i,t+1)
                    print("Saving image at %s" % pred_img_path)
                    pred_img.save(pred_img_path)
        print("Done")

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
