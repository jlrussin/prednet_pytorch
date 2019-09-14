# Video ladder network, built from Cricri et al. (2016)

import torch
import torch.nn as nn
import torch.nn.functional as F

from activations import Hardsigmoid, SatLU
from utils import *
from PredNet import ACell, AhatCell, RCell, ECell

# Things to do:
#   -Figure out how to do local gradients
#   -Figure out 1x1 convolutions for computing input to Ahat cells
#   -Batch normalization - test with and without
#   -Leaky ReLU - test with and without
#   -Dilation in convolutions - test with and without
#   -Paper has same ConvLSTM equations?
#   -Need ConvLSTM/skip connection in first layer? - test with and without each
#   -Weird training regime in paper?
#   -Should try ResNet blocks?
#   -Authors use sigmoid in last layer instead of leaky ReLU, binary CE loss

class LadderNet(nn.Module):
    def __init__(self,in_channels,stack_sizes,R_stack_sizes,
                 A_kernel_sizes,Ahat_kernel_sizes,R_kernel_sizes,
                 use_satlu,pixel_max,Ahat_act,satlu_act,error_act,
                 LSTM_act,LSTM_c_act,bias=True,
                 use_1x1_out=False,FC=True,no_R0=True,no_skip0=True,
                 local_grad=False,
                 output='error',device='cpu'):
        super(LadderNet,self).__init__()
        self.in_channels = in_channels
        self.stack_sizes = stack_sizes
        self.R_stack_sizes = R_stack_sizes
        self.A_kernel_sizes = A_kernel_sizes
        self.Ahat_kernel_sizes = Ahat_kernel_sizes
        self.R_kernel_sizes = R_kernel_sizes
        self.use_satlu = use_satlu
        self.pixel_max = pixel_max
        self.Ahat_act = Ahat_act
        self.satlu_act = satlu_act
        self.error_act = error_act
        self.LSTM_act = LSTM_act
        self.LSTM_c_act = LSTM_c_act
        self.bias = bias
        self.use_1x1_out=use_1x1_out
        self.no_R0 = no_R0 # no R Cell for pixel-layer, preds come from Ahat_lp1
        self.no_skip0 = no_skip0 # no skip between A0 and Ahat0
        self.local_grad = local_grad # gradients only broadcasted within layers
        self.output = output
        self.device = device

        # local gradients means no convolution in A, stack sizes is fixed
        if local_grad:
            stack_sizes = [in_channels for s in range(len(stack_sizes))]
            self.stack_sizes = stack_sizes

        # Make sure number of layers is consistent in args
        self.nb_layers = len(stack_sizes)
        msg = "len(R_stack_sizes) must equal len(stack_sizes)"
        assert len(R_stack_sizes) == self.nb_layers, msg
        if self.nb_layers > 1:
            msg = "len(A_kernel_sizes) must equal len(stack_sizes)"
            assert len(A_kernel_sizes) == self.nb_layers - 1, msg
        msg = "len(Ahat_kernel_sizes) must equal len(stack_sizes)"
        assert len(Ahat_kernel_sizes) == self.nb_layers, msg
        msg = "len(R_kernel_sizes) must equal len(stack_sizes)"
        assert len(R_kernel_sizes) == self.nb_layers, msg

        # A cells: (conv) + nonlinearity + MaxPool
        no_conv = local_grad # no convolutional layer if using local_grad
        A_layers = [None] # First A layer is input
        for l in range(1,self.nb_layers): # First A layer is input
            in_channels = stack_sizes[l-1] # input will be A_t_lm1
            out_channels = stack_sizes[l]
            conv_kernel_size = A_kernel_sizes[l-1]
            cell = ACell(in_channels,out_channels,
                         conv_kernel_size,bias,no_conv)
            A_layers.append(cell)
        self.A_layers = nn.ModuleList(A_layers)

        # R cells: convolutional LSTM
        R_layers = []
        if self.no_R0:
            R_layers.append(None)
        for l in range(self.nb_layers):
            is_last = True # always true - doesn't receive from higher layer
            in_channels = stack_sizes[l]
            out_channels = R_stack_sizes[l]
            kernel_size = R_kernel_sizes[l]
            cell = RCell(in_channels,out_channels,kernel_size,
                         LSTM_act,LSTM_c_act,
                         is_last,self.bias,use_1x1_out,FC,no_ER)
            R_layers.append(cell)
        self.R_layers = nn.ModuleList(R_layers)

        # A_hat cells: conv + nonlinearity
        Ahat_layers = []
        for l in range(self.nb_layers):
            if l == 0:
                if self.no_R0 and self.no_skip0:
                    in_channels = stack_sizes[l+1]
                elif self.no_R0 and not self.skip0:
                    in_channels = stack_sizes[l]+stack_sizes[l+1]
                elif not self.no_R0 and self.skip0:
                    in_channels = R_stack_sizes[l]+stack_sizes[l+1]
                else:
                    in_channels = R_stack_sizes[l]+stack_sizes[l]+
                                  stack_sizes[l+1]
            elif l < self.nb_layers-1:
                in_channels = R_stack_sizes[l]+stack_sizes[l]+stack_sizes[l+1]
            else:
                in_channels = R_stack_sizes[l]+stack_sizes[l]
            out_channels = stack_sizes[l]
            conv_kernel_size = Ahat_kernel_sizes[l]
            if self.use_satlu and l == 0:
                # Lowest layer uses SatLU
                cell = AhatCell(in_channels,out_channels,
                                conv_kernel_size,bias,Ahat_act,satlu_act,
                                use_satlu=True,pixel_max=pixel_max)
            else:
                cell = AhatCell(in_channels,out_channels,
                                conv_kernel_size,bias) # relu for l > 0
            Ahat_layers.append(cell)
        self.Ahat_layers = nn.ModuleList(Ahat_layers)

        # E cells: subtract, ReLU, cat
        self.E_layer = ECell(error_act) # general: same for all layers

    def forward(self,X):
        # Get initial states
        (H_tm1,C_tm1) = self.initailize(X)

        outputs = []
        Ahat_t = [None] * self.nb_layers

        # Loop through image sequence
        seq_len = X.shape[1]
        for t in range(seq_len):
            # Initialize list of states with consistent indexing
            R_t = [None] * self.nb_layers
            H_t = [None] * self.nb_layers
            C_t = [None] * self.nb_layers
            E_t = [None] * self.nb_layers
            A_t = [None] * self.nb_layers

            # Encoder: A and R
            A_layer = self.A_layers[l] # A cell
            R_layer = self.R_layers[l] # R cell
            for l in range(self.nb_layers):
                if l == 0:
                    A_t[l] = X[:,t,:,:,:] # first layer predicts pixels
                    if self.no_R0:
                        R_t[0] = None
                    else:
                        R_t[l],(H_t[l],C_t[l]) = R_layer(A_t[l], None,
                                                         (H_tm1[l],C_tm1[l]))
                else:
                    A_t[l] = A_layer(A_t[l-1])
                    R_t[l], (H_t[l],C_t[l]) = R_layer(A_t[l], None,
                                                      (H_tm1[l],C_tm1[l])

            # Errors from predictions on previous time steps
            if t > 0:
                for l in range(self.nb_layers):
                    E_t[l] = self.E_layer(A_t[l],Ahat_t[l])

            # Decoder: Ahat
            for l in reversed(range(self.nb_layers)):
                Ahat_layer = self.Ahat_layers[l]
                if l == self.nb_layers - 1:
                    Ahat_input = torch.cat((A_t[l],R_t[l]),dim=1) # channel dim
                elif l > 0:
                    target_size = (A_t[l].shape[2],A_t[l].shape[3])
                    Ahat_up = F.interpolate(Ahat_t[l+1],target_size)
                    Ahat_input = torch.cat((A_t[l],R_t[l],Ahat_up),dim=1)
                elif l == 0:
                    target_size = (A_t[l].shape[2],A_t[l].shape[3])
                    Ahat_up = F.interpolate(Ahat_t[l+1],target_size)
                    if self.no_R0 and self.no_skip0:
                        Ahat_input = Ahat_up
                    elif self.no_R0 and not self.no_skip0:
                        Ahat_input = torch.cat((A_t[l],Ahat_up),dim=1)
                    elif not self.no_R0 and self.no_skip0:
                        Ahat_input = torch.cat((R_t[l],Ahat_up),dim=1)
                    else:
                        Ahat_input = torch.cat((A_t[l],R_t[l],Ahat_up),dim=1)
                Ahat_t[l] = Ahat_layer(Ahat_input)

            # Update hidden states
            (H_tm1,C_tm1) = (H_t,C_t)

            # Output
            if self.output == 'pred':
                if t < seq_len-1:
                    outputs.append(Ahat_t[0])
            elif self.output == 'rep':
                if t == seq_len - 1: # only return reps for last time step
                    outputs = R_t
            elif self.output == 'error':
                if t > 0: # first time step doesn't count
                    outputs.append(E_t)

        # Errors and Preds returned as tensors
        if self.output == 'error':
            outputs_t = torch.zeros(seq_len,self.nb_layers)
            for t in range(seq_len-1):
                for l in range(self.nb_layers):
                    outputs_t[t,l] = torch.mean(outputs[t][l])
        elif self.output == 'pred':
            outputs_t = [output.unsqueeze(1) for output in outputs]
            outputs_t = torch.cat(outputs_t,dim=1) # (batch,len,in_channels,H,W)
        # reps returned as list of tensors
        elif self.output == 'rep':
            outputs_t = outputs
        return outputs_t

    def initialize(self,X):
        # input dimensions
        batch_size = X.shape[0]
        height = X.shape[3]
        width = X.shape[4]
        H_0 = []
        C_0 = []
        for l in range(self.nb_layers):
            R_channels = self.R_stack_sizes[l]
            # All hidden states initialized with zeros
            Hl = torch.zeros(batch_size,R_channels,height,width).to(self.device)
            Cl = torch.zeros(batch_size,R_channels,height,width).to(self.device)
            H_0.append(Hl)
            C_0.append(Cl)
            # Update dims
            height = int((height - 2)/2 + 1) # int performs floor
            width = int((width - 2)/2 + 1) # int performs floor
        return (H_0,C_0)
