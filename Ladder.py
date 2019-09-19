# Video ladder network, built from Cricri et al. (2016)

import torch
import torch.nn as nn
import torch.nn.functional as F

from activations import Hardsigmoid, SatLU
from utils import *
from PredNet import ACell, RCell, ECell

# Things to do:
#   -Need to allow different strides and dilations at each layer 
#   -No max pooling? - test with and without?
#       -Need to change all code that computes height and width
#   -Should try ResNet blocks?

# Ahat cell = [Conv,ReLU]
class LAhatCell(nn.Module):
    def __init__(self,R_in_channels,A_in_channels,Ahat_in_channels,
                 out_channels,conv_kernel_size,conv_bias,
                 act='relu',use_BN=False,satlu_act='hardtanh',use_satlu=False,
                 pixel_max=1.0,no_R=False,no_A=False,no_Ahat_lp1=False):
        super(LAhatCell,self).__init__()
        self.R_in_channels = R_in_channels
        self.A_in_channels = A_in_channels
        self.Ahat_in_channels = Ahat_in_channels
        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_bias = conv_bias
        self.act = act
        self.use_BN = use_BN
        self.satlu_act = satlu_act
        self.use_satlu = use_satlu
        self.pixel_max = pixel_max
        self.no_R = no_R # no connection from R (layer 0 with no_R0)
        self.no_A = no_A # no connection from A (layer 0 with no_skip0)
        self.no_Ahat_lp1 = no_Ahat_lp1 # no connection from Ahat_lp1 (top layer)

        conv_stride = 1 # always 1 for simplicity
        conv_pad_ = 0 # padding done manually
        conv_dilation = 1 # always 1 for simplicity
        conv_groups = 1 # always 1 for simplicity
        conv1x1_kernel_size = 1 # used for 1x1 convolutional layers

        # Parameters
        # Standard convolutional layer for Ahat_lp1
        if not no_Ahat_lp1:
            if self.use_BN:
                self.BN = nn.BatchNorm2d(Ahat_in_channels)
            self.conv = nn.Conv2d(Ahat_in_channels,out_channels,
                                  conv_kernel_size,conv_stride,
                                  conv_pad_,conv_dilation,conv_groups,
                                  conv_bias)
        # (1,1) convolutional layer for (Ahat,R)
        if no_R:
            Wr_in_channels = out_channels
        elif no_Ahat_lp1:
            Wr_in_channels = R_in_channels
        else:
            Wr_in_channels = out_channels + R_in_channels
        self.Wr = nn.Conv2d(Wr_in_channels,out_channels,
                            conv1x1_kernel_size,conv_stride,
                            conv_pad_,conv_dilation,conv_groups,
                            conv_bias)
        # (1,1) convolutional layer for (LReLU(Ahat,R),A)
        if not no_A:
            Wa_in_channels = out_channels + A_in_channels
            self.Wa = nn.Conv2d(Wa_in_channels,out_channels,
                                conv1x1_kernel_size,conv_stride,
                                conv_pad_,conv_dilation,conv_groups,
                                conv_bias)
        self.out_act = get_activation(act)
        if use_satlu:
            self.satlu = SatLU(satlu_act,self.pixel_max)

    def forward(self,A_l,R_l,Ahat_lp1):
        # BN + LReLU + Padding + Conv + LReLU
        if not self.no_Ahat_lp1:
            if self.use_BN:
                Ahat_lp1 = self.BN(Ahat_lp1)
            Ahat_lp1 = self.out_act(Ahat_lp1)
            in_height = Ahat_lp1.shape[2]
            in_width = Ahat_lp1.shape[3]
            padding = get_pad_same(in_height,in_width,self.conv_kernel_size)
            Ahat_lp1 = F.pad(Ahat_lp1,padding)
            Ahat_lp1 = self.conv(Ahat_lp1)
            Ahat_lp1 = self.out_act(Ahat_lp1)
        # 1x1 convolution for (Ahat,R)
        if self.no_Ahat_lp1:
            Ahat = R_l
        elif self.no_R:
            Ahat = Ahat_lp1
        else:
            Ahat = torch.cat((Ahat_lp1,R_l),dim=1) # cat on channel dim
        Ahat = self.Wr(Ahat)
        # 1x1 convolution for A
        if not self.no_A:
            Ahat = self.out_act(Ahat)
            Ahat = torch.cat((Ahat,A_l),dim=1)
            Ahat = self.Wa(Ahat)
        if self.use_satlu:
            Ahat = self.satlu(Ahat)
        else:
            Ahat = self.out_act(Ahat)
        return Ahat

class LadderNet(nn.Module):
    def __init__(self,in_channels,stack_sizes,R_stack_sizes,
                 A_kernel_sizes,Ahat_kernel_sizes,R_kernel_sizes,
                 conv_dilation,use_BN,use_satlu,pixel_max,Ahat_act,satlu_act,
                 error_act,LSTM_act,LSTM_c_act,bias=True,
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
        self.conv_dilation = conv_dilation
        self.use_BN = use_BN
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
                         conv_kernel_size,conv_dilation,bias,no_conv,use_BN)
            A_layers.append(cell)
        self.A_layers = nn.ModuleList(A_layers)

        # R cells: convolutional LSTM
        R_layers = []
        for l in range(self.nb_layers):
            if l == 0 and self.no_R0:
                R_layers.append(None)
                continue
            is_last = True # always true - doesn't receive from higher layer
            in_channels = stack_sizes[l]
            out_channels = R_stack_sizes[l]
            kernel_size = R_kernel_sizes[l]
            cell = RCell(in_channels,out_channels,kernel_size,
                         LSTM_act,LSTM_c_act,
                         is_last,self.bias,use_1x1_out,FC,False)
            R_layers.append(cell)
        self.R_layers = nn.ModuleList(R_layers)

        # A_hat cells: BN+LReLU+conv+LReLU+1x1conv+LReLU+1x1conv+LReLU
        Ahat_layers = []
        for l in range(self.nb_layers):
            if l == 0 and no_R0:
                no_R = True
                R_in_channels = None
            else:
                no_R = False
                R_in_channels = R_stack_sizes[l]
            if l == 0 and no_skip0:
                no_A = True
                A_in_channels = None
            else:
                no_A = False
                A_in_channels = stack_sizes[l]
            if l == self.nb_layers-1:
                Ahat_in_channels = None
                no_Ahat_lp1 = True
            else:
                Ahat_in_channels = stack_sizes[l+1]
                no_Ahat_lp1 = False
            out_channels = stack_sizes[l]
            conv_kernel_size = Ahat_kernel_sizes[l]
            use_satlu = self.use_satlu and l == 0 # Lowest layer uses SatLU
            cell = LAhatCell(R_in_channels,A_in_channels,Ahat_in_channels,
                             out_channels,conv_kernel_size,
                             self.bias,act=Ahat_act,use_BN=use_BN,
                             satlu_act=satlu_act,use_satlu=use_satlu,
                             pixel_max=pixel_max,no_R=no_R,no_A=no_A,
                             no_Ahat_lp1=no_Ahat_lp1)
            Ahat_layers.append(cell)
        self.Ahat_layers = nn.ModuleList(Ahat_layers)

        # E cells: subtract, ReLU, cat
        self.E_layer = ECell(error_act) # general: same for all layers

    def forward(self,X):
        # Get initial states
        (H_tm1,C_tm1) = self.initialize(X)

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
            for l in range(self.nb_layers):
                A_layer = self.A_layers[l] # A cell
                R_layer = self.R_layers[l] # R cell
                if l == 0:
                    A_t[l] = X[:,t,:,:,:] # first layer predicts pixels
                    if self.no_R0:
                        R_t[0] = None
                    else:
                        R_t[l],(H_t[l],C_t[l]) = R_layer(A_t[l], None,
                                                         (H_tm1[l],C_tm1[l]))
                else:
                    if self.local_grad:
                        A_t[l] = A_layer(A_t[l-1].detach())
                    else:
                        A_t[l] = A_layer(A_t[l-1])
                    R_t[l], (H_t[l],C_t[l]) = R_layer(A_t[l], None,
                                                      (H_tm1[l],C_tm1[l]))

            # Errors from predictions on previous time steps
            if t > 0:
                for l in range(self.nb_layers):
                    E_t[l] = self.E_layer(A_t[l],Ahat_t[l])

            # Decoder: Ahat
            for l in reversed(range(self.nb_layers)):
                Ahat_layer = self.Ahat_layers[l]
                if l == self.nb_layers - 1:
                    Ahat_up = None
                elif l > 0:
                    target_size = (A_t[l].shape[2],A_t[l].shape[3])
                    Ahat_up = F.interpolate(Ahat_t[l+1],target_size)
                    if self.local_grad:
                        Ahat_up = Ahat_up.detach()
                elif l == 0:
                    target_size = (A_t[l].shape[2],A_t[l].shape[3])
                    Ahat_up = F.interpolate(Ahat_t[l+1],target_size)
                    if self.local_grad:
                        Ahat_up = Ahat_up.detach()
                Ahat_t[l] = Ahat_layer(A_t[l],R_t[l],Ahat_up)

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
