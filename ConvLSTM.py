# ConvLSTM architecture
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from activation import Hardsigmoid,SatLU

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell. See the paper:
    Title: Convolutional LSTM Network: A Machine Learning Approach for
           Precipitation Nowcasting
    Authors: Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong,
             Wang-chun Woo
    arxiv: https://arxiv.org/abs/1506.04214

    Differs from 'RCell' in PredNet because it has an output layer to make the
    number of channels in the output equal to the number of channels in the
    input.
    """
    def __init__(self, in_channels, hidden_channels, kernel_size,
                 LSTM_act, LSTM_c_act, bias=True, FC=False):
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.FC = FC # use fully connected ConvLSTM

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
        self.out = nn.Conv2d(hidden_channels,in_channels,1,1,0,1,1)
        self.out_sigmoid = nn.Sigmoid()

    def forward(self, X_t, hidden):
        H_tm1, C_tm1 = hidden

        # Manual zero-padding to make H,W same
        in_height = X_t.shape[-2]
        in_width = X_t.shape[-1]
        padding = get_pad_same(in_height,in_width,self.kernel_size)
        X_t_pad = F.pad(X_t,padding)
        H_tm1_pad = F.pad(H_tm1,padding)
        C_tm1_pad = F.pad(C_tm1,padding)

        if not self.FC:
            i_t = self.LSTM_act(self.Wxi(X_t_pad) + self.Whi(H_tm1_pad))
            f_t = self.LSTM_act(self.Wxf(X_t_pad) + self.Whf(H_tm1_pad))
            C_t = f_t*C_tm1 + i_t*self.LSTM_c_act(self.Wxc(X_t_pad) + self.Whc(H_tm1_pad))
            o_t = self.LSTM_act(self.Wxo(X_t_pad) + self.Who(H_tm1_pad))
            H_t = o_t*self.LSTM_act(C_t)
        else:
            i_t = self.Wxi(X_t_pad) + self.Whi(H_tm1_pad) + self.Wci(C_tm1_pad)
            i_t = self.LSTM_act(i_t)

            f_t = self.Wxf(X_t_pad) + self.Whf(H_tm1_pad) + self.Wcf(C_tm1_pad)
            f_t = self.LSTM_act(f_t)

            C_t = self.Wxc(X_t_pad) + self.Whc(H_tm1_pad)
            C_t = f_t*C_tm1 + i_t*self.LSTM_c_act(C_t)
            C_t_pad = F.pad(C_t,padding)

            o_t = self.Wxo(X_t_pad) + self.Who(H_tm1_pad) + self.Wco(C_t_pad)
            o_t = self.LSTM_act(o_t)

            H_t = o_t*self.LSTM_act(C_t)

        # Output layer
        R_t = self.out(H_t)
        R_t = self.out_sigmoid(R_t)


        return R_t, (H_t,C_t)

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size,
                 LSTM_act, LSTM_c_act, bias=True, FC=False):
        super(ConvLSTM,self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.LSTM_act = LSTM_act
        self.LSTM_c_act = LSTM_c_act
        self.bias = bias
        self.FC = FC # use fully connected ConvLSTM

        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size,
                                 LSTM_act, LSTM_c_act, bias=True, FC=False)

    def forward(self,X):
        # Get initial states
        (H_tm1,C_tm1) = self.initialize(X)

        # Loop through image sequence
        preds = []
        seq_len = X.shape[1]
        for t in range(seq_len):
            X_t = X[:,t,:,:,:] # X dims: (batch,len,channels,height,width)

            R_t,(H_t,C_t) = self.cell(X_t,(H_tm1,C_tm1))

            # Update
            (H_tm1,C_tm1) = (H_t,C_t)
            preds.append(R_t.unsqueeze(1))
        preds = torch.cat(preds,dim=1)
        return preds

    def initialize(self,X):
        # input dimensions
        batch_size = X.shape[0]
        out_channels = self.hidden_channels
        height = X.shape[3]
        width = X.shape[4]
        # Hidden states initialized with zeros
        H_0 = torch.zeros(batch_size,hidden_channels,height,width)
        C_0 = torch.zeros(batch_size,hidden_channels,height,width)
        return (H_0,C_0)
