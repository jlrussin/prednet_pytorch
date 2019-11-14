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
from Ladder import *
from StackedConvLSTM import *
from utils import *

parser = argparse.ArgumentParser()
# RSA
parser.add_argument('--aggregate_method', choices=['mean','max','none'],
                    default='mean',
                    help='Method to aggregate reps across space')
parser.add_argument('--similarity_measure', choices=['corr','cos'],
                    default='corr',
                    help='Similarity measure to use: correlation or cosine')
parser.add_argument('--cat_dict_json', default=None,
                    help='Json file with dictionary mapping categories' +
                         'to supercategories. Default just uses categories.')
# Training data
parser.add_argument('--test_data_path',
                    default='../data/ccn_images/train/',
                    help='Path to ccn image directory to test')
parser.add_argument('--seq_len', type=int, default=8,
                    help='Number of images in each ccn sequence')
parser.add_argument('--downsample_size',type=int,default=128,
                    help='Height and width of downsampled CCN inputs.')
parser.add_argument('--last_only',type=str2bool,default=False,
                    help='Train on sequences of static (final) images.')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Samples per batch')
parser.add_argument('--idx_dict_hkl',
                    default='../data/ccn_images/train_label_idx_dict.hkl',
                    help='Path to dictionary with ids of each label')

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
parser.add_argument('--no_A_conv', type=str2bool, default=False,
                    help='No convolutional layer in A cells')
parser.add_argument('--higher_satlu', type=str2bool, default=False,
                    help='Use satlu in higher layers')
parser.add_argument('--local_grad', type=str2bool, default=False,
                    help='Boolean indicating whether to restrict gradients ' +
                         'to flow locally (within each layer)')
parser.add_argument('--forward_conv',type=str2bool,default=False,
                    help='Boolean indicating whether to use conv2d rather' +
                         'than convLSTM in forward path of StackedConvLSTM')
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
                    print("Partitioning dataset [%d%%]" % (100*i//len(dataset)))
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

def aggregate_space(rep,method):
    if method == 'none':
        batch_size = rep.shape[0]
        aggregated_rep = rep.view(batch_size,-1) # flatten space
    elif method == 'mean':
        aggregated_rep = torch.mean(rep,dim=[2,3]) # average over space
    elif method == 'max':
        # TODO
        raise NotImplementedError
    return aggregated_rep

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

def sort_similarity_matrix(S,cat_dict,labels):
    cats = [l.split('_')[0] for l in labels] # list of categories
    supercats = [cat_dict[c] for c in cats] # list of supercategories
    # Sort so that categories are contiguous
    sorted_rows = np.argsort(supercats)
    S_by_scat = np.zeros_like(S)
    for old_i,new_i in enumerate(sorted_rows):
        for old_j,new_j in enumerate(sorted_rows):
            S_by_scat[new_i,new_j] = S[old_i,old_j]
    labels_by_cat = [labels[i] for i in sorted_rows]
    sorted_supercats = [supercats[i] for i in sorted_rows]
    # Get first ids of each contiguous supercategory
    scat_set = [] # ordered set of unique supercategories
    first_scat_ids = [] # starting id of each supercategory
    prev_scat = 'NULL'
    for i,scat in enumerate(sorted_supercats):
        if scat != prev_scat:
            scat_set.append(scat)
            first_scat_ids.append(i)
            prev_scat = scat
    first_scat_ids.append(len(sorted_supercats)) # append final index for ease
    # Sort by similarity in each category
    sorted_rows = []
    for i in range(len(scat_set)):
        start_row = first_scat_ids[i]
        end_row = first_scat_ids[i+1]
        S_scat = S_by_scat[start_row:end_row,start_row:end_row]
        S_scat_means = np.mean(S_scat,axis=1)
        sorted_scat_ids = np.argsort(S_scat_means) + start_row
        sorted_rows = sorted_rows + sorted_scat_ids.tolist()
    sorted_S = np.zeros_like(S_by_scat)
    for old_i,new_i in enumerate(sorted_rows):
        for old_j,new_j in enumerate(sorted_rows):
            sorted_S[new_i,new_j] = S_by_scat[old_i,old_j]
    sorted_labels = [labels[i] for i in sorted_rows]
    return sorted_S,sorted_labels

def main(args):
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Load model
    model_out = 'rep' # Always rep to get representations
    if args.model_type == 'PredNet':
        model = PredNet(args.in_channels,args.stack_sizes,args.R_stack_sizes,
                        args.A_kernel_sizes,args.Ahat_kernel_sizes,
                        args.R_kernel_sizes,args.use_satlu,args.pixel_max,
                        args.Ahat_act,args.satlu_act,args.error_act,
                        args.LSTM_act,args.LSTM_c_act,args.bias,
                        args.use_1x1_out,args.FC,args.dropout_p,
                        args.send_acts,args.no_ER,args.RAhat,args.no_A_conv,
                        args.higher_satlu,args.local_grad,args.conv_dilation,
                        args.use_BN,model_out,device)
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
                          args.no_A_conv,args.higher_satlu,args.local_grad,
                          model_out,device)
    elif args.model_type == 'StackedConvLSTM':
        model = StackedConvLSTM(args.in_channels,args.R_stack_sizes,
                                args.R_kernel_sizes,args.use_1x1_out,
                                args.FC,args.local_grad,args.forward_conv,
                                model_out,device)
    if args.load_weights_from is not None:
        model.load_state_dict(torch.load(args.load_weights_from))
    model.to(device)
    model.eval()
    if args.model_type == 'LadderNet' and args.no_R0:
        nb_reps = model.nb_layers - 1
    else:
        nb_reps = model.nb_layers

    # Dataset
    downsample_size = (args.downsample_size,args.downsample_size)
    test_data = CCN(args.test_data_path,args.seq_len,
                    downsample_size=downsample_size,return_labels=True,
                    last_only=args.last_only)
    partitioner = Partitioner(test_data,args.idx_dict_hkl)
    labels = sorted(partitioner.labels)
    n_labels = len(labels)
    print("There are %d labels in the dataset" % n_labels)

    with torch.no_grad():
        # Get list of layer representations for each label
        label_reps = []
        for label_i,label in enumerate(labels):
            # Get data partition for current label
            print("Starting label %d/%d: %s" % (label_i+1,n_labels,label))
            partition = partitioner.get_partition(label)
            n_samples = len(partition)
            dataloader = DataLoader(partition,args.batch_size)
            # Run model, keeping running sum of representations
            layer_reps = [[] for l in range(nb_reps+1)] # nb_reps + pixels
            for batch_i,batch in enumerate(dataloader):
                X = batch[0].to(device)
                # Get representations
                reps = model(X) # list of reps, one for each layer
                pixels = X[:,-1,:,:,:] # Use last image to compare to RGB reps
                # Aggregate across space
                agg_pixels = aggregate_space(pixels,args.aggregate_method)
                agg_reps = [agg_pixels] # first layer is pixels
                for l in range(nb_reps):
                    agg_rep = aggregate_space(reps[l],args.aggregate_method)
                    agg_reps.append(agg_rep)
                # Sum batch
                layer_sums = []
                for l in range(nb_reps+1):
                    layer_sum = torch.sum(agg_reps[l],dim=0)
                    layer_sum = layer_sum.unsqueeze(0) # Add label dimension
                    layer_sums.append(layer_sum)
                # Update running sums
                if batch_i == 0:
                    layer_reps = layer_sums
                else:
                    for l in range(nb_reps+1):
                        layer_reps[l] += layer_sums[l]
            # Divide by n_samples to get average
            layer_reps = [layer_rep/n_samples for layer_rep in layer_reps]
            label_reps.append(layer_reps)
            print("Finished processing samples for label: %s" % label)
        layer_lists = list(map(list, zip(*label_reps))) # transpose lists
        layer_tensors = []
        for l in range(nb_reps+1):
            layer_tensor = torch.cat(layer_lists[l],dim=0)
            layer_tensors.append(layer_tensor)

    # Set up data for saving similarity matrices
    info = {'aggregate_method':args.aggregate_method,
            'similarity_measure':args.similarity_measure}
    RSA_data = {'info':info}
    if args.cat_dict_json is None:
        cats = set([l.split('_')[0] for l in labels])
        cat_dict = {cat:cat for cat in cats}
    else:
        with open(args.cat_dict_json,'r') as f:
            cat_dict = json.load(f)
    print("Computing and sorting similarity matrices")
    for l,layer_tensor in enumerate(layer_tensors):
        # Get similarity matrix
        S = get_similarity_matrix(layer_tensor,args.similarity_measure)
        S = S.cpu().numpy()
        # Sort similarity matrix
        sorted_S,sorted_labels = sort_similarity_matrix(S,cat_dict,labels)
        # Save matrices
        layer_name = 'layer%d' % (l-1) if l > 0 else 'pixels'
        RSA_data[layer_name] = sorted_S
    RSA_data['labels'] = sorted_labels # all sorted labels should be the same

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
