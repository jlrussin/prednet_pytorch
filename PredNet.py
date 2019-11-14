# PredNet architecture

import torch
import torch.nn.functional as F
import torch.nn as nn

from activations import Hardsigmoid, SatLU
from utils import *

# Convolutional LSTM cell used for R cells
class RCell(nn.Module):
    """
    Modified version of 2d convolutional lstm as described in the paper:
    Title: Convolutional LSTM Network: A Machine Learning Approach for
           Precipitation Nowcasting
    Authors: Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong,
             Wang-chun Woo
    arxiv: https://arxiv.org/abs/1506.04214

    Changes are made according to PredNet paper: LSTM is not "fully connected",
    in the sense that i,f, and o do not depend on C.
    """
    def __init__(self, in_channels, hidden_channels, kernel_size,
                 LSTM_act, LSTM_c_act, is_last, bias=True, use_out=True,
                 FC=False, no_ER=False, dropout_p=0.0):
        super(RCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.is_last = is_last # bool
        self.bias = bias
        self.use_out = use_out
        self.FC = FC # use fully connected ConvLSTM
        self.no_ER = no_ER
        self.dropout_p = dropout_p

        # Activations
        self.LSTM_act = get_activation(LSTM_act)
        self.LSTM_c_act = get_activation(LSTM_c_act)

        self.stride = 1 # Stride always 1 for simplicity
        self.dilation = 1 # Dilation always 1 for simplicity
        _pad = 0 # Padding done manually in forward()
        self.groups = 1 # Groups always 1 for simplicity

        # Convolutional layers
        self.Wxi = nn.Conv2d(in_channels,hidden_channels,kernel_size,
                             self.stride,_pad,self.dilation,
                             self.groups,self.bias)
        self.Whi = nn.Conv2d(hidden_channels,hidden_channels,
                             kernel_size,self.stride,_pad,
                             self.dilation,self.groups,self.bias)
        self.Wxf = nn.Conv2d(in_channels,hidden_channels,kernel_size,
                             self.stride,_pad,self.dilation,
                             self.groups,self.bias)
        self.Whf = nn.Conv2d(hidden_channels,hidden_channels,
                             kernel_size,self.stride,_pad,
                             self.dilation,self.groups,self.bias)
        self.Wxc = nn.Conv2d(in_channels,hidden_channels,kernel_size,
                             self.stride,_pad,self.dilation,
                             self.groups,self.bias)
        self.Whc = nn.Conv2d(hidden_channels,hidden_channels,
                             kernel_size,self.stride,_pad,
                             self.dilation,self.groups,self.bias)
        self.Wxo = nn.Conv2d(in_channels,hidden_channels,kernel_size,
                             self.stride,_pad,self.dilation,
                             self.groups,self.bias)
        self.Who = nn.Conv2d(hidden_channels,hidden_channels,
                             kernel_size,self.stride,_pad,
                             self.dilation,self.groups,self.bias)

        # Extra layers for fully connected
        if FC:
            self.Wci = nn.Conv2d(hidden_channels,hidden_channels,kernel_size,
                                 self.stride,_pad,self.dilation,
                                 self.groups,self.bias)
            self.Wcf = nn.Conv2d(hidden_channels,hidden_channels,kernel_size,
                                 self.stride,_pad,self.dilation,
                                 self.groups,self.bias)
            self.Wco = nn.Conv2d(hidden_channels,hidden_channels,kernel_size,
                                 self.stride,_pad,self.dilation,
                                 self.groups,self.bias)
        # 1 x 1 convolution for output
        if use_out:
            self.out = nn.Conv2d(hidden_channels,hidden_channels,1,1,0,1,1)

        # Dropout
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, E, R_lp1, hidden):
        H_tm1, C_tm1 = hidden

        # Upsample R_lp1
        if not self.is_last:
            target_size = (E.shape[2],E.shape[3])
            R_up = F.interpolate(R_lp1,target_size)
            if not self.no_ER:
                x_t = torch.cat((E,R_up),dim=1) # cat on channel dim
            else:
                x_t = R_up
        else:
            x_t = E

        # Dropout on inputs
        x_t = self.dropout(x_t)

        # Manual zero-padding to make H,W same
        in_height = x_t.shape[-2]
        in_width = x_t.shape[-1]
        padding = get_pad_same(in_height,in_width,self.kernel_size)
        x_t_pad = F.pad(x_t,padding)
        H_tm1_pad = F.pad(H_tm1,padding)
        C_tm1_pad = F.pad(C_tm1,padding)

        # No dependence on C for i,f,o?
        if not self.FC:
            i_t = self.LSTM_act(self.Wxi(x_t_pad) + self.Whi(H_tm1_pad))
            f_t = self.LSTM_act(self.Wxf(x_t_pad) + self.Whf(H_tm1_pad))
            C_t = f_t*C_tm1 + i_t*self.LSTM_c_act(self.Wxc(x_t_pad) + \
                                                  self.Whc(H_tm1_pad))
            o_t = self.LSTM_act(self.Wxo(x_t_pad) + self.Who(H_tm1_pad))
            H_t = o_t*self.LSTM_act(C_t)
        else:
            i_t = self.Wxi(x_t_pad) + self.Whi(H_tm1_pad) + self.Wci(C_tm1_pad)
            i_t = self.LSTM_act(i_t)

            f_t = self.Wxf(x_t_pad) + self.Whf(H_tm1_pad) + self.Wcf(C_tm1_pad)
            f_t = self.LSTM_act(f_t)

            C_t = self.Wxc(x_t_pad) + self.Whc(H_tm1_pad)
            C_t = f_t*C_tm1 + i_t*self.LSTM_c_act(C_t)
            C_t_pad = F.pad(C_t,padding)

            o_t = self.Wxo(x_t_pad) + self.Who(H_tm1_pad) + self.Wco(C_t_pad)
            o_t = self.LSTM_act(o_t)

            H_t = o_t*self.LSTM_act(C_t)
        if self.use_out:
            R_t = self.out(H_t)
            R_t = self.LSTM_act(R_t)
        else:
            R_t = H_t

        return R_t, (H_t,C_t)

# A cells = [Conv,ReLU,MaxPool]
class ACell(nn.Module):
    def __init__(self,in_channels,out_channels,
                 conv_kernel_size,conv_dilation,conv_bias,no_conv,
                 use_BN,act_fn='relu'):
        super(ACell,self).__init__()

        # Hyperparameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_dilation = conv_dilation
        self.conv_bias = conv_bias
        self.no_conv = no_conv
        self.use_BN = use_BN
        self.act_fn = act_fn

        if self.use_BN:
            self.BN = nn.BatchNorm2d(in_channels)

        if not no_conv:
            conv_stride = 1 # always 1 for simplicity
            _conv_pad = 0 # padding done manually
            conv_dilation = conv_dilation
            conv_groups = 1 # always 1 for simplicity
            self.conv =  nn.Conv2d(in_channels,out_channels,
                                   conv_kernel_size,conv_stride,
                                   _conv_pad,conv_dilation,conv_groups,
                                   conv_bias)
            self.act = get_activation(act_fn)
        pool_kernel_size = 2 # always 2 for simplicity
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self,E_lm1):
        if self.use_BN:
            E_lm1 = self.BN(E_lm1)
        if not self.no_conv:
            # Manual padding to keep H,W the same
            in_height = E_lm1.shape[2]
            in_width = E_lm1.shape[3]
            padding = get_pad_same(in_height,in_width,self.conv_kernel_size,
                                   self.conv_dilation)
            E_lm1 = F.pad(E_lm1,padding)
            A = self.conv(E_lm1)
            A = self.act(A)
        else:
            A = E_lm1
        A = self.max_pool(A)
        return A

# Ahat cell = [Conv,ReLU]
class AhatCell(nn.Module):
    def __init__(self,in_channels,out_channels,
                 conv_kernel_size,conv_bias,act='relu',
                 satlu_act='hardtanh',use_satlu=False,pixel_max=1.0,
                 use_BN=False):
        super(AhatCell,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_bias = conv_bias
        self.act = act
        self.satlu_act = satlu_act
        self.use_satlu = use_satlu
        self.pixel_max = pixel_max
        self.use_BN = use_BN

        if use_BN:
            self.BN = nn.BatchNorm2d(in_channels)

        conv_stride = 1 # always 1 for simplicity
        conv_pad_ = 0 # padding done manually
        conv_dilation = 1 # always 1 for simplicity
        conv_groups = 1 # always 1 for simplicity

        # Parameters
        self.conv =  nn.Conv2d(in_channels,out_channels,
                               conv_kernel_size,conv_stride,
                               conv_pad_,conv_dilation,conv_groups,
                               conv_bias)
        self.out_act = get_activation(act)
        if use_satlu:
            self.satlu = SatLU(satlu_act,self.pixel_max)

    def forward(self,R_l):
        if self.use_BN:
            R_l = self.BN(R_l)
        # Manual padding to keep dims the same
        in_height = R_l.shape[2]
        in_width = R_l.shape[3]
        padding = get_pad_same(in_height,in_width,self.conv_kernel_size)
        # Compute A_hat
        R_l = F.pad(R_l,padding)
        A_hat = self.conv(R_l)
        A_hat = self.out_act(A_hat)
        if self.use_satlu:
            A_hat = self.satlu(A_hat)
        return A_hat

# E Cell = [subtract,ReLU,Concatenate]
class ECell(nn.Module):
    def __init__(self,error_act):
        super(ECell,self).__init__()
        self.act = get_activation(error_act)
    def forward(self,A,A_hat):
        positive = self.act(A - A_hat)
        negative = self.act(A_hat - A)
        E = torch.cat((positive,negative),dim=1) # cat on channel dim
        return E

# PredNet
class PredNet(nn.Module):
    def __init__(self,in_channels,stack_sizes,R_stack_sizes,
                 A_kernel_sizes,Ahat_kernel_sizes,R_kernel_sizes,
                 use_satlu,pixel_max,Ahat_act,satlu_act,error_act,
                 LSTM_act,LSTM_c_act,bias=True,
                 use_1x1_out=False,FC=False,dropout_p=0.0,send_acts=False,
                 no_ER=False,RAhat=False,no_A_conv=False,higher_satlu=False,
                 local_grad=False,conv_dilation=1,use_BN=False,output='error',
                 device='cpu'):
        super(PredNet,self).__init__()
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
        self.FC = FC # use fully connected ConvLSTM
        self.dropout_p = dropout_p # dropout on inputs to R cells
        self.send_acts = send_acts # send A_t rather than E_t
        self.no_ER = no_ER # no connection between E_l and R_l
        self.RAhat = RAhat # extra connection between R_{l+1} and A_hat_{l}
        self.no_A_conv = no_A_conv
        self.higher_satlu = higher_satlu # higher layers use satlu
        self.local_grad = local_grad # gradients only broadcasted within layers
        self.conv_dilation = conv_dilation # dilation in A cells
        self.use_BN = use_BN
        self.output = output
        self.device = device

        # no convolution in A means stack sizes is fixed
        if no_A_conv:
            if send_acts:
                # A_l receives from A_{l-1} - all A's have same channel dim
                stack_sizes = [in_channels for s in range(len(stack_sizes))]
            else:
                # A_l receives from E_{l-1} - channel dim doubles each layer
                stack_sizes = [in_channels*(2**s) for s in range(len(stack_sizes))]
            self.stack_sizes = stack_sizes

        # Make sure consistent number of layers
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
        # Make sure not doing inconsistent ablations
        msg = "Can't do RAhat and local_grad"
        assert not (RAhat and local_grad), msg

        # R cells: convolutional LSTM
        R_layers = []
        for l in range(self.nb_layers):
            if l == self.nb_layers-1:
                is_last = True
                in_channels = 2*stack_sizes[l]
            else:
                is_last = False
                if not self.no_ER:
                    in_channels = 2*stack_sizes[l] + R_stack_sizes[l+1]
                else:
                    in_channels = R_stack_sizes[l+1]
            out_channels = R_stack_sizes[l]
            kernel_size = R_kernel_sizes[l]
            cell = RCell(in_channels,out_channels,kernel_size,
                         LSTM_act,LSTM_c_act,
                         is_last,self.bias,use_1x1_out,FC,no_ER,dropout_p)
            R_layers.append(cell)
        self.R_layers = nn.ModuleList(R_layers)

        # A cells: conv + ReLU + MaxPool
        A_layers = [None]
        for l in range(1,self.nb_layers): # First A layer is input
            if not self.send_acts:
                in_channels = 2*stack_sizes[l-1] # E layer doubles channels
            else:
                in_channels = stack_sizes[l-1] # input will be A_t, not E_t
            out_channels = stack_sizes[l]
            conv_kernel_size = A_kernel_sizes[l-1]
            cell = ACell(in_channels,out_channels,
                         conv_kernel_size,conv_dilation,bias,no_A_conv,use_BN)
            A_layers.append(cell)
        self.A_layers = nn.ModuleList(A_layers)

        # A_hat cells: conv + ReLU
        Ahat_layers = []
        for l in range(self.nb_layers):
            if RAhat and (l != (self.nb_layers-1)): # Last Ahat is unchanged
                in_channels = R_stack_sizes[l] + R_stack_sizes[l+1]
            else:
                in_channels = R_stack_sizes[l]
            out_channels = stack_sizes[l]
            conv_kernel_size = Ahat_kernel_sizes[l]
            # Use satlu with lowest layer or when no A conv (pixels in [0,1])
            if self.use_satlu and (l==0 or self.higher_satlu):
                cell = AhatCell(in_channels,out_channels,
                                conv_kernel_size,bias,Ahat_act,satlu_act,
                                use_satlu=True,pixel_max=pixel_max,
                                use_BN=use_BN)
            else:
                # relu for l > 0
                cell = AhatCell(in_channels,out_channels,
                                conv_kernel_size,bias,use_BN=use_BN)
            Ahat_layers.append(cell)
        self.Ahat_layers = nn.ModuleList(Ahat_layers)

        # E cells: subtract, ReLU, cat
        self.E_layer = ECell(error_act) # general: same for all layers

    def forward(self,X):
        # Get initial states
        (H_tm1,C_tm1),E_tm1 = self.initialize(X)

        outputs = []

        # Loop through image sequence
        seq_len = X.shape[1]
        for t in range(seq_len):
            A_t = X[:,t,:,:,:] # X dims: (batch,len,channels,height,width)
            # Initialize list of states with consistent indexing
            R_t = [None] * self.nb_layers
            H_t = [None] * self.nb_layers
            C_t = [None] * self.nb_layers
            E_t = [None] * self.nb_layers

            # Update R units starting from the top
            for l in reversed(range(self.nb_layers)):
                R_layer = self.R_layers[l] # cell
                if l == self.nb_layers-1:
                    R_t[l],(H_t[l],C_t[l]) = R_layer(E_tm1[l],None,
                                                     (H_tm1[l],C_tm1[l]))
                else:
                    if not self.local_grad:
                        R_t[l],(H_t[l],C_t[l]) = R_layer(E_tm1[l],
                                                         R_t[l+1],
                                                         (H_tm1[l],C_tm1[l]))
                    else:
                        R_t[l],(H_t[l],C_t[l]) = R_layer(E_tm1[l],
                                                         R_t[l+1].detach(),
                                                         (H_tm1[l],C_tm1[l]))
            if self.output == 'rep':
                if t == seq_len - 1: # only return reps for last time step
                    outputs = R_t

            # Update feedforward path starting from the bottom
            for l in range(self.nb_layers):
                # Compute Ahat
                Ahat_layer = self.Ahat_layers[l]
                if self.RAhat and (l != (self.nb_layers-1)):
                    target_size = (R_t[l].shape[2],R_t[l].shape[3])
                    R_up = F.interpolate(R_t[l+1],target_size)
                    Ahat_input = torch.cat((R_t[l],R_up),dim=1)
                else:
                    Ahat_input = R_t[l]
                Ahat_t = Ahat_layer(Ahat_input)
                if self.output == 'pred':
                    if l == 0 and t > 0:
                        outputs.append(Ahat_t)

                # Compute E
                E_t[l] = self.E_layer(A_t,Ahat_t)

                # Compute A of next layer
                if l < self.nb_layers-1:
                    A_layer = self.A_layers[l+1]
                    if not self.send_acts:
                        if not self.local_grad:
                            A_t = A_layer(E_t[l])
                        else:
                            A_t = A_layer(E_t[l].detach())
                    else:
                        # Send activations rather than errors
                        if not self.local_grad:
                            A_t = A_layer(A_t)
                        else:
                            A_t = A_layer(A_t.detach())

            # Update
            (H_tm1,C_tm1),E_tm1 = (H_t,C_t),E_t
            if self.output == 'error':
                if t > 0:
                    outputs.append(E_t) # First time step doesn't count
        # errors and preds returned as tensors
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
        # get dimensions of E,R for each layer
        H_0 = []
        C_0 = []
        E_0 = []
        for l in range(self.nb_layers):
            channels = self.stack_sizes[l]
            R_channels = self.R_stack_sizes[l]
            # All hidden states initialized with zeros
            Hl = torch.zeros(batch_size,R_channels,height,width).to(self.device)
            Cl = torch.zeros(batch_size,R_channels,height,width).to(self.device)
            El = torch.zeros(batch_size,2*channels,height,width).to(self.device)
            H_0.append(Hl)
            C_0.append(Cl)
            E_0.append(El)
            # Update dims
            height = int((height - 2)/2 + 1) # int performs floor
            width = int((width - 2)/2 + 1) # int performs floor
        return (H_0,C_0), E_0

# Multi-layer convolutional LSTM (passes R instead of E between layers)
class MultiConvLSTM(nn.Module):
    def __init__(self,in_channels,R_stack_sizes,R_kernel_sizes,
                 use_satlu,pixel_max,Ahat_act,satlu_act,error_act,
                 LSTM_act,LSTM_c_act,
                 bias=True,use_1x1_out=False,FC=True,local_grad=False,
                 output='pred',device='cpu'):
        super(MultiConvLSTM,self).__init__()
        self.in_channels = in_channels
        self.R_stack_sizes = R_stack_sizes
        self.R_kernel_sizes = R_kernel_sizes
        self.use_satlu = use_satlu
        self.pixel_max = pixel_max
        self.Ahat_act = Ahat_act
        self.satlu_act = satlu_act
        self.error_act = error_act
        self.LSTM_act = LSTM_act
        self.LSTM_c_act = LSTM_c_act
        self.bias = bias
        self.use_1x1_out = use_1x1_out
        self.FC = FC
        self.local_grad = local_grad
        self.output = output
        self.device = device

        # Make sure a consistent number of layers was given
        self.nb_layers = len(R_stack_sizes)
        msg = "len(R_stack_sizes) must equal len(R_kernel_sizes)"
        assert len(R_kernel_sizes) == self.nb_layers, msg

        # R cells: convolutional LSTM
        R_layers = []
        for l in range(self.nb_layers):
            if l == 0:
                is_last = False
                in_channels = self.in_channels + R_stack_sizes[l+1]
            elif l < self.nb_layers-1:
                is_last = False
                in_channels = R_stack_sizes[l-1] + R_stack_sizes[l+1]
            elif l == self.nb_layers-1:
                is_last = True
                in_channels = R_stack_sizes[l-1]
            out_channels = R_stack_sizes[l]
            kernel_size = R_kernel_sizes[l]
            cell = RCell(in_channels,out_channels,kernel_size,
                         LSTM_act,LSTM_c_act,is_last,self.bias,
                         use_1x1_out,FC)
            R_layers.append(cell)
        self.R_layers = nn.ModuleList(R_layers)

        # Pooling layers: MaxPool
        pool_kernel_size = 2
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

        # A_hat cells: conv + ReLU
        Ahat_layers = []
        for l in range(self.nb_layers):
            in_channels = R_stack_sizes[l]
            kernel_size = R_kernel_sizes[l]
            if l == 0:
                out_channels = self.in_channels # first layer predicts pixels
                if self.use_satlu:
                    use_satlu = True
            else:
                out_channels = R_stack_sizes[l-1]
                use_satlu = False
            cell = AhatCell(in_channels,out_channels,
                            kernel_size,bias,Ahat_act,satlu_act,
                            use_satlu,pixel_max)
            Ahat_layers.append(cell)
        self.Ahat_layers = nn.ModuleList(Ahat_layers)

        # E cells: subtract, ReLU, cat
        self.E_layer = ECell(error_act) # general: same for all layers

    def forward(self,X):

        # Get initial states
        (H_tm1,C_tm1),R_tm1 = self.initialize(X)

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

            # Loop through layers computing R and A
            A_t[0] = X[:,t,:,:,:] # first layer predicts pixels
            for l in range(self.nb_layers):
                R_layer = self.R_layers[l] # cell
                if l == 0:
                    if self.local_grad:
                        R_tm1_lp1 = R_tm1[l+1].detach()
                    else:
                        R_tm1_lp1 = R_tm1[l+1]
                    R_t[l],(H_t[l],C_t[l]) = R_layer(A_t[l],
                                                     R_tm1_lp1,
                                                     (H_tm1[l],C_tm1[l]))
                elif l < self.nb_layers-1:
                    if self.local_grad:
                        R_tm1_lp1 = R_tm1[l+1].detach()
                        A_t[l] = self.max_pool(R_t[l-1].detach())
                    else:
                        R_tm1_lp1 = R_tm1[l+1]
                        A_t[l] = self.max_pool(R_t[l-1])
                    R_t[l],(H_t[l],C_t[l]) = R_layer(A_t[l],
                                                     R_tm1_lp1,
                                                     (H_tm1[l],C_tm1[l]))
                else:
                    if self.local_grad:
                        A_t[l] = self.max_pool(R_t[l-1].detach())
                    else:
                        A_t[l] = self.max_pool(R_t[l-1])
                    R_t[l],(H_t[l],C_t[l]) = R_layer(A_t[l],
                                                     None,
                                                     (H_tm1[l],C_tm1[l]))
            # Compute E for all layers
            if t > 0:
                for l in range(self.nb_layers):
                    E_t[l] = self.E_layer(A_t[l],Ahat_t[l])
            # Compute Ahat
            for l in range(self.nb_layers):
                Ahat_layer = self.Ahat_layers[l]
                Ahat_t[l] = Ahat_layer(R_t[l])
            # Update hidden states
            (H_tm1,C_tm1),R_tm1 = (H_t,C_t),R_t
            # Output pixel-level predictions
            if self.output == 'pred':
                if t < seq_len-1:
                    outputs.append(Ahat_t[0])
            # Output representations
            elif self.output == 'rep':
                if t == seq_len - 1: # only return reps for last time step
                    outputs = R_t
            # Output errors
            elif self.output == 'error':
                if t > 0:
                    outputs.append(E_t) # First time step doesn't count

        # errors and preds returned as tensors
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
        # get dimensions of E,R for each layer
        H_0 = []
        C_0 = []
        R_0 = []
        for l in range(self.nb_layers):
            R_channels = self.R_stack_sizes[l]
            # All hidden states initialized with zeros
            Hl = torch.zeros(batch_size,R_channels,height,width).to(self.device)
            Cl = torch.zeros(batch_size,R_channels,height,width).to(self.device)
            Rl = torch.zeros(batch_size,R_channels,height,width).to(self.device)
            H_0.append(Hl)
            C_0.append(Cl)
            R_0.append(Rl)
            # Update dims
            height = int((height - 2)/2 + 1) # int performs floor
            width = int((width - 2)/2 + 1) # int performs floor
        return (H_0,C_0), R_0
