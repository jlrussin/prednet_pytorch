# Script for doing linear decoding of object categories from layer reps
import os
import argparse
import numpy as np
import hickle as hkl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import *
from activations import *
from PredNet import *
from ConvLSTM import *
from utils import *

parser = argparse.ArgumentParser()
# Layer decoding
parser.add_argument('--aggregate_method', choices=['mean','max','none'],
                    default='mean',
                    help='Method to aggregate reps across space')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='L2 penalty on weights on linear decoders')
parser.add_argument('--load_decoders_from', default=None,
                    help='Path to saved weights. Do not include decoder%d.pt')

# Training data
parser.add_argument('--train_data_path',
                    default='../data/ccn_images/train/',
                    help='Path to training images')
parser.add_argument('--val_data_path',
                    default='../data/ccn_images/val/',
                    help='Path to validation images')
parser.add_argument('--test_data_path',
                    default='../data/ccn_images/test/',
                    help='Path to test images')
parser.add_argument('--seq_len', type=int, default=8,
                    help='Number of images in each ccn sequence')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Samples per batch')
parser.add_argument('--num_iters', type=int, default=75000,
                    help='Number of optimizer steps before stopping')

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

# Optimization
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Fixed learning rate for Adam optimizer')

# Output options
parser.add_argument('--results_dir', default='../results/layer_decoding/',
                    help='Results subdirectory to save results')
parser.add_argument('--out_data_file', default='prednet_defaults.json',
                    help='Name of output file with loss and accuracy stats')
parser.add_argument('--checkpoint_path',default=None,
                    help='Path to output saved weights of decoders.')
parser.add_argument('--checkpoint_every', type=int, default=5,
                    help='Epochs before evaluating decoders and saving weights')
parser.add_argument('--record_loss_every', type=int, default=20,
                    help='iters before printing and recording loss')

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
    train_data = CCN(args.train_data_path,args.seq_len,return_cats=True)
    val_data = CCN(args.val_data_path,args.seq_len,return_cats=True)
    test_data = CCN(args.test_data_path,args.seq_len,return_cats=True)
    train_loader = DataLoader(train_data,args.batch_size,shuffle=True)
    val_loader = DataLoader(val_data,args.batch_size,shuffle=True)
    test_loader = DataLoader(test_data,args.batch_size,shuffle=True)
    cat_ns = train_data.cat_ns
    n_cats = len(cat_ns)
    print("There are %d categories in the dataset" % n_cats)

    # Vocab
    token_to_idx = {token:i for i,token in enumerate(cat_ns.keys())}
    idx_to_token = {i:token for token,i in token_to_idx.items()}

    # Dummy test to get dimension of each decoder
    X,_ = train_data[0] # don't need the cat right now
    X = X.unsqueeze(0).to(device) # batch size is 1
    reps = model(X)
    reps.insert(0,X[:,-1,:,:,:]) # insert last image to get dim of pixels
    layer_dims = []
    for rep in reps:
        agg_rep = aggregate_space(rep,args.aggregate_method)
        dim = agg_rep.shape[1]
        layer_dims.append(dim)

    # Initialize linear layers
    decoders = []
    for l in range(nb_layers+1): # all layers plus 1 for pixels
        decoder = nn.Linear(layer_dims[l],n_cats)
        if args.load_decoders_from is not None:
            pt_path = args.load_decoders_from + 'decoder%d' % l
            pt_path = pt_path + '.pt'
            decoder.load_state_dict(torch.load(args.load_decoders_from))
        decoder = decoder.to(device)
        decoder.train()
        decoders.append(decoder)
    n_decoders = len(decoders)


    # Weight loss by number of samples with each cat
    n_samples = len(train_data)
    weights = []
    for idx in range(n_cats):
        token = idx_to_token[idx]
        n = cat_ns[token]
        weight = n/n_samples
        weights.append(weight)
    weights = torch.tensor(weights)
    weights = weights/torch.sum(weights)
    weights = weights.to(device)

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss(weights)
    all_params = []
    for decoder in decoders:
        all_params += list(decoder.parameters())
    optimizer = optim.Adam(all_params,lr=args.learning_rate)

    # Train
    iter = 0
    epoch_count = 0
    loss_data = [[] for i in range(n_decoders)] # loss data for each decoder
    train_acc_data = [[] for i in range(n_decoders)] # mean train losses
    val_acc_data = [[] for i in range(n_decoders)] # mean val losses
    test_acc_data = [[] for i in range(n_decoders)] # mean test losses
    best_val_accs = [0.0 for i in range(n_decoders)] # early stopping
    while iter < args.num_iters:
        epoch_count += 1
        for batch in train_loader:
            iter += 1
            # Split sample
            X = batch[0]
            X = X.to(device)
            cats = batch[1]
            target = torch.tensor([token_to_idx[l] for l in cats])
            target = target.to(device)
            # Get representations
            optimizer.zero_grad()
            with torch.no_grad():
                reps = model(X)
                reps.insert(0,X[:,-1,:,:,:])
            agg_reps = []
            for rep in reps:
                agg_rep = aggregate_space(rep,args.aggregate_method)
                agg_reps.append(agg_rep)
            # Decode
            outputs = []
            for rep,decoder in zip(agg_reps,decoders):
                output = decoder(rep)
                outputs.append(output)
            # Compute loss and take optimizer step
            losses = []
            for output in outputs:
                loss = loss_fn(output,target)
                loss.backward()
                losses.append(loss.data.item())
            optimizer.step()
            # Record loss
            if iter % args.record_loss_every == 0:
                print('Epoch:', epoch_count,
                      'Iter:', iter,
                      'Losses:', losses)
                for l,loss_datapoint in enumerate(losses):
                    loss_data[l].append(loss_datapoint)
            if iter >= args.num_iters:
                break
        # Checkpoint
        last_epoch = (iter >= args.num_iters)
        if epoch_count % args.checkpoint_every == 0 or last_epoch:
            # Train
            print("Checking training accuracy ...")
            train_accs = checkpoint(train_loader,token_to_idx,
                                    model,decoders,device,args)
            print("Training accuracies are ", train_accs)
            for l,train_acc in enumerate(train_accs):
                train_acc_data[l].append(train_acc)
            # Validation
            print("Checking validation accuracy ...")
            val_accs = checkpoint(val_loader,token_to_idx,
                                  model,decoders,device,args)
            print("Validation accuracies are ", val_accs)
            for l,val_acc in enumerate(val_accs):
                val_acc_data[l].append(val_acc)
            # Test
            print("Checking test accuracy ...")
            test_accs = checkpoint(test_loader,token_to_idx,
                                   model,decoders,device,args)
            print("Test accuracies are ", test_accs)
            for l,test_acc in enumerate(test_accs):
                test_acc_data[l].append(test_acc)
            # Write stats file
            if not os.path.isdir(args.results_dir):
                os.mkdir(args.results_dir)
            stats = {'loss_data':loss_data,
                     'train_acc_data':train_acc_data,
                     'val_acc_data':val_acc_data,
                     'test_acc_data':test_acc_data}
            results_file_name = '%s/%s' % (args.results_dir,args.out_data_file)
            with open(results_file_name, 'w') as f:
                json.dump(stats, f)
            # Save model weights
            for l,val_acc in enumerate(val_accs):
                if val_acc > best_val_accs[l]: # use val to decide to save
                    best_val_accs[l] = val_acc
                    if args.checkpoint_path is not None:
                        if args.checkpoint_path[-3:] == '.pt':
                            pt_path = args.checkpoint_path[:-3]
                            pt_path = pt_path + 'decoder%d' % l
                            pt_path = pt_path + '.pt'
                        torch.save(decoders[l].state_dict(),pt_path)

def checkpoint(dataloader, token_to_idx, model, decoders, device, args):
    for decoder in decoders:
        decoder.eval()
    with torch.no_grad():
        correct_ls = [[] for i in range(len(decoders))]
        for batch in dataloader:
            # Split sample
            X = batch[0]
            X = X.to(device)
            cats = batch[1]
            target = torch.tensor([token_to_idx[t] for t in cats])
            target = target.to(device)
            # Forward
            reps = model(X)
            reps.insert(0,X[:,-1,:,:,:])
            # Aggregate
            agg_reps = []
            for rep in reps:
                agg_rep = aggregate_space(rep,args.aggregate_method)
                agg_reps.append(agg_rep)
            # Decode
            outputs = []
            for rep,decoder in zip(agg_reps,decoders):
                output = decoder(rep)
                outputs.append(output)
            # Compute accuracy
            for l,output in enumerate(outputs):
                max_cats = torch.argmax(output,dim=1)
                correct_t = max_cats == target
                correct_l = correct_t.cpu().numpy().tolist()
                correct_ls[l] += correct_l
        accs = []
        for correct_l in correct_ls:
            accs.append(np.mean(correct_l))
    for decoder in decoders:
        decoder.train()
    return accs

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
