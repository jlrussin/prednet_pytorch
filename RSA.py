# Script for doing RSA
import os
import argparse
import numpy as np
import hickle as hkl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import *
from activations import *
from PredNet import *
from ConvLSTM import *
from utils import *

# TODO:
#   -Debug

parser = argparse.ArgumentParser()
# RSA
parser.add_argument('--aggregate_method', choices=['mean','max','none'],
                    default='mean',
                    help='Method to aggregate reps across space')
parser.add_argument('--similarity_measure', choices=['corr','cos'],
                    default='corr',
                    help='Similarity measure to use: correlation or cosine')

# Training data
parser.add_argument('--test_data_path',
                    default='../data/ccn_images/train/',
                    help='Path to ccn image directory to test')
parser.add_argument('--seq_len', type=int, default=8,
                    help='Number of images in each ccn sequence')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Samples per batch')
parser.add_argument('--idx_dict_hkl',
                    default='../data/ccn_images/train_label_idx_dict.hkl',
                    help='Path to dictionary with ids of each label')

# Models
parser.add_argument('--model_type', choices=['PredNet'],
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
parser.add_argument('--results_dir', default='../results/rsa/',
                    help='Results subdirectory to save results')
parser.add_argument('--out_data_file', default='prednet_defaults.hkl',
                    help='Name of output file with similarity matrix')

class Partition(object):
    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        return self.data[data_idx]

class Partitioner(object):
    def __init__(self,dataset,idx_dict_hkl):
        self.dataset = dataset
        self.idx_dict_json = idx_dict_hkl
        self.idx_dict = {}
        if os.path.exists(idx_dict_hkl):
            print("Loading idx dict from %s" % idx_dict_hkl)
            self.idx_dict = hkl.load(idx_dict_hkl)
        else:
            for i,(_,label) in enumerate(dataset):
                if i % 2000 == 0:
                    print("Partitioning dataset: [%d%%]" % (100*i//len(dataset)))
                if label in self.idx_dict:
                    self.idx_dict[label].append(i)
                else:
                    self.idx_dict[label] = [i]
            print("Saving idx dict to %s" % idx_dict_hkl)
            hkl.dump(self.idx_dict,idx_dict_hkl)
        self.labels = self.idx_dict.keys()

    def get_partition(self,label):
        print("Getting partition for label: %s" % label)
        indices = self.idx_dict[label]
        partition = Partition(self.dataset,indices)
        print("Partition has %d sequences" % len(partition))
        return partition

def aggregate(reps,method):
    """
    Aggregate across samples and space:
        -Average across samples
        -Use given method to aggregate across space
    """
    nb_layers = len(reps)
    aggregated_layers = []
    for l in range(nb_layers):
        cat = torch.cat(reps[l],dim=0) # concatenate all samples
        ave = torch.mean(cat,dim=0) # average all samples
        if method == 'none':
            n_channels = ave.shape[0]
            aggregated_layer = ave.view(n_channels,-1) # flatten space
        elif method == 'mean':
            aggregated_layer = torch.mean(ave,dim=[1,2]) # average over space
        elif method == 'max':
            # TODO
            raise NotImplementedError
        aggregated_layer = aggregated_layer.unsqueeze(0) # Add label dim
        aggregated_layers.append(aggregated_layer)
    return aggregated_layers

def cosine_similarity(X):
    norm = torch.norm(X,dim=1,keepdim=True)
    X_n = X/norm
    S = torch.mm(X_n,X_n.transpose(0,1))
    return S

def correlation_similarity(X):
    X_bar = torch.mean(X,dim=1,keepdim=True)
    X_c = X - X_bar
    norm = torch.norm(X_c,dim=1,keepdim=True)
    X_n = X_c/norm
    S = torch.mm(X_n,X_n.transpose(0,1))
    return S

def get_similarity_matrix(X,measure):
    if measure == 'corr':
        S = correlation_similarity(X)
    elif measure == 'cos':
        S = cosine_similarity(X)
    return S

def main(args):
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Load model
    model_out = 'rep' # Always rep to get representations
    model = PredNet(args.in_channels,args.stack_sizes,args.R_stack_sizes,
                    args.A_kernel_sizes,args.Ahat_kernel_sizes,
                    args.R_kernel_sizes,args.use_satlu,args.pixel_max,
                    args.Ahat_act,args.satlu_act,args.error_act,
                    args.LSTM_act,args.LSTM_c_act,args.bias,
                    args.use_1x1_out,args.FC,model_out,device)

    if args.load_weights_from is not None:
        model.load_state_dict(torch.load(args.load_weights_from))
    model.to(device)
    model.eval()
    nb_layers = model.nb_layers

    # Dataset
    test_data = CCN(args.test_data_path,args.seq_len,return_labels=True)
    partitioner = Partitioner(test_data,args.idx_dict_hkl)
    labels = sorted(partitioner.labels)
    n_labels = len(labels)
    print("There are %d labels in the dataset" % n_labels)

    with torch.no_grad():
        # Get list of layer representations for each label
        label_reps = []
        for label_i,label in enumerate(labels):
            print("Starting label %d/%d: %s" % (label_i,n_labels,label))
            partition = partitioner.get_partition(label)
            dataloader = DataLoader(partition,args.batch_size)
            layer_reps = [[] for l in range(nb_layers+1)] # nb_layers + pixels
            for batch in dataloader:
                X = batch[0].to(device)
                reps = model(X)
                pixels = X[-1] # Use last image to compare to RGB reps
                layer_reps[0].append(pixels)
                for l in range(nb_layers):
                    layer_reps[l+1].append(reps[l])
            layer_reps = aggregate(layer_reps,args.aggregate_method)
            label_reps.append(layer_reps)
            print("Finished processing samples for label: %s" % label)
        layer_lists = list(map(list, zip(*label_reps))) # transpose lists
        layer_tensors = []
        for l in range(nb_layers+1):
            layer_tensors.append(torch.cat(layer_lists[l],dim=0))

    # Save similarity matrix for each layer
    info = {'aggregate_method':args.aggregate_method,
            'similarity_measure':args.similarity_measure}
    RSA_data = {'info':info,'labels':labels}
    print("Computing similarity matrices")
    for l,layer_tensor in enumerate(layer_tensors):
        S = get_similarity_matrix(layer_tensor,args.similarity_measure)
        S = S.numpy()
        layer_name = 'layer%d' % (l-1) if l > 0 else 'pixels'
        RSA_data[layer_name] = S

    # Save similarity matrices
    dir = args.results_dir
    if not os.path.isdir(dir):
        os.mkdir(dir)
    results_path = os.path.join(args.results_dir,args.out_data_file)
    print("Saving results to %s" % results_path)
    hkl.dump(RSA_data,results_path)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
