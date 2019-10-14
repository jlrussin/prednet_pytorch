# StackedConvLSTM
# Convolutional LSTMs stacked into feedforward and feedback pathways

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from PredNet import ECell

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size,
                 use_out=True, FC=False):
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.use_out = use_out # Use extra convolutional layer at output
        self.FC = FC # use fully connected ConvLSTM

        # Activations
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        bias = True
        self.stride = 1 # Stride always 1 for simplicity
        self.dilation = 1 # Dilation always 1 for simplicity
        _pad = 0 # Padding done manually in forward()
        self.groups = 1 # Groups always 1 for simplicity

        # Convolutional layers
        self.Wxi = nn.Conv2d(in_channels,hidden_channels,kernel_size,
                             self.stride,_pad,self.dilation,
                             self.groups,bias)
        self.Whi = nn.Conv2d(hidden_channels,hidden_channels,
                             kernel_size,self.stride,_pad,
                             self.dilation,self.groups,bias)
        self.Wxf = nn.Conv2d(in_channels,hidden_channels,kernel_size,
                             self.stride,_pad,self.dilation,
                             self.groups,bias)
        self.Whf = nn.Conv2d(hidden_channels,hidden_channels,
                             kernel_size,self.stride,_pad,
                             self.dilation,self.groups,bias)
        self.Wxc = nn.Conv2d(in_channels,hidden_channels,kernel_size,
                             self.stride,_pad,self.dilation,
                             self.groups,bias)
        self.Whc = nn.Conv2d(hidden_channels,hidden_channels,
                             kernel_size,self.stride,_pad,
                             self.dilation,self.groups,bias)
        self.Wxo = nn.Conv2d(in_channels,hidden_channels,kernel_size,
                             self.stride,_pad,self.dilation,
                             self.groups,bias)
        self.Who = nn.Conv2d(hidden_channels,hidden_channels,
                             kernel_size,self.stride,_pad,
                             self.dilation,self.groups,bias)

        # Extra layers for fully connected
        if FC:
            self.Wci = nn.Conv2d(hidden_channels,hidden_channels,kernel_size,
                                 self.stride,_pad,self.dilation,
                                 self.groups,bias)
            self.Wcf = nn.Conv2d(hidden_channels,hidden_channels,kernel_size,
                                 self.stride,_pad,self.dilation,
                                 self.groups,bias)
            self.Wco = nn.Conv2d(hidden_channels,hidden_channels,kernel_size,
                                 self.stride,_pad,self.dilation,
                                 self.groups,bias)
        # 1 x 1 convolution for output
        if use_out:
            self.out = nn.Conv2d(hidden_channels,hidden_channels,1,1,0,1,1)

    def forward(self, X_t, hidden):
        H_tm1, C_tm1 = hidden

        # Manual zero-padding to make H,W same
        in_height = X_t.shape[-2]
        in_width = X_t.shape[-1]
        padding = get_pad_same(in_height,in_width,self.kernel_size)
        X_t_pad = F.pad(X_t,padding)
        H_tm1_pad = F.pad(H_tm1,padding)
        C_tm1_pad = F.pad(C_tm1,padding)

        # No dependence on C for i,f,o?
        if not self.FC:
            i_t = self.sigmoid(self.Wxi(X_t_pad) + self.Whi(H_tm1_pad))
            f_t = self.sigmoid(self.Wxf(X_t_pad) + self.Whf(H_tm1_pad))
            C_t = f_t*C_tm1 + i_t*self.tanh(self.Wxc(X_t_pad) + \
                                                  self.Whc(H_tm1_pad))
            o_t = self.sigmoid(self.Wxo(X_t_pad) + self.Who(H_tm1_pad))
            H_t = o_t*self.tanh(C_t)
        else:
            i_t = self.Wxi(X_t_pad) + self.Whi(H_tm1_pad) + self.Wci(C_tm1_pad)
            i_t = self.sigmoid(i_t)

            f_t = self.Wxf(X_t_pad) + self.Whf(H_tm1_pad) + self.Wcf(C_tm1_pad)
            f_t = self.sigmoid(f_t)

            C_t = self.Wxc(X_t_pad) + self.Whc(H_tm1_pad)
            C_t = f_t*C_tm1 + i_t*self.tanh(C_t)
            C_t_pad = F.pad(C_t,padding)

            o_t = self.Wxo(X_t_pad) + self.Who(H_tm1_pad) + self.Wco(C_t_pad)
            o_t = self.sigmoid(o_t)

            H_t = o_t*self.tanh(C_t)
        if self.use_out:
            R_t = self.out(H_t)
            R_t = self.tanh(R_t)
        else:
            R_t = H_t

        return R_t, (H_t,C_t)

class StackedConvLSTM(nn.Module):
    def __init__(self,in_channels,stack_sizes,kernel_sizes,use_1x1_out=False,
                 FC=True,local_grad=False,forward_conv=False,
                 output='error',device='cpu'):
        super(StackedConvLSTM,self).__init__()
        self.in_channels = in_channels
        self.stack_sizes = stack_sizes
        self.kernel_sizes = kernel_sizes
        self.use_1x1_out=use_1x1_out
        self.FC = FC # use fully connected ConvLSTM
        self.local_grad = local_grad # gradients only broadcasted within layers
        self.forward_conv = forward_conv
        self.output = output
        self.device = device

        self.nb_layers = len(stack_sizes)

        # Forward layers
        forward_layers = []
        for l in range(self.nb_layers):
            if l == 0:
                in_channels = self.in_channels
            else:
                in_channels = stack_sizes[l-1]
            hidden_channels = stack_sizes[l]
            kernel_size = kernel_sizes[l]
            if self.forward_conv:
                cell = nn.conv2d(in_channels,hidden_channels,kernel_size)
            else:
                cell = ConvLSTMCell(in_channels,hidden_channels,kernel_size,
                                    use_1x1_out,FC)
            forward_layers.append(cell)
        self.forward_layers = nn.ModuleList(forward_layers)

        # MaxPool for forward layers
        self.max_pool = nn.MaxPool2d(2) # forward uses pooling

        # Backward layers
        backward_layers = []
        for l in range(self.nb_layers):
            if l == self.nb_layers - 1:
                in_channels = stack_sizes[l]
            else:
                in_channels = stack_sizes[l] + stack_sizes[l+1]
            hidden_channels = stack_sizes[l]
            kernel_size = kernel_sizes[l]
            cell = ConvLSTMCell(in_channels,hidden_channels,kernel_size,
                            use_1x1_out,FC)
            backward_layers.append(cell)
        self.backward_layers = nn.ModuleList(backward_layers)

        # Conv layers for producing predictions
        conv_layers = []
        for l in range(self.nb_layers):
            in_channels = stack_sizes[l]
            if l == 0:
                out_channels = self.in_channels
            else:
                out_channels = self.stack_sizes[l-1]
            kernel_size = kernel_sizes[l]
            conv = nn.Conv2d(in_channels,out_channels,kernel_size)
            conv_layers.append(conv)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.tanh = nn.Tanh() # Ahat always uses tanh activation

        # E layer for computing errors: [ReLU(A-Ahat);ReLU(Ahat-A)]
        self.E_layer = ECell('relu')

        # Activations
        self.sigmoid = nn.Sigmoid() # layer 0 Ahat activation
        self.tanh = nn.Tanh() # all other Ahat activations

    def forward(self,X):
        # Get initial states
        (H_tm1_f,C_tm1_f,H_tm1_b,C_tm1_b) = self.initialize(X)

        outputs = []
        Ahat_t = [None] * self.nb_layers

        # Loop through image sequence
        seq_len = X.shape[1]
        for t in range(seq_len):
            # Initialize list of states with consistent indexing
            A_t = [None] * self.nb_layers
            R_t_f = [None] * self.nb_layers
            R_t_b = [None] * self.nb_layers
            H_t_f = [None] * self.nb_layers
            C_t_f = [None] * self.nb_layers
            H_t_b = [None] * self.nb_layers
            C_t_b = [None] * self.nb_layers
            E_t = [None] * self.nb_layers

            # Forward path
            for l in range(self.nb_layers):
                # Compute A
                if l == 0:
                    A_t[l] = X[:,t,:,:,:] # (batch,len,channels,height,width)
                else:
                    if self.local_grad:
                        A_t[l] = self.max_pool(R_t_f[l-1].detach())
                    else:
                        A_t[l] = self.max_pool(R_t_f[l-1])
                # Compute R_t_f
                forward_layer = self.forward_layers[l]
                if self.forward_conv:
                    R_t_f[l] = forward_layer(A_t[l])
                    H_t_f[l],C_t_f[l] = None,None
                else:
                    R_t_f[l], (H_t_f[l],C_t_f[l]) = forward_layer(A_t[l],
                                                                  (H_tm1_f[l],
                                                                   C_tm1_f[l]))

            # Compute errors made on previous timestep
            if t > 0:
                for l in range(self.nb_layers):
                    E_t[l] = self.E_layer(A_t[l],Ahat_t[l])

            # Backward path
            for l in reversed(range(self.nb_layers)):
                # Compute R_t_b
                backward_layer = self.backward_layers[l]
                if l == self.nb_layers - 1:
                    R_input = R_t_f[l]
                else:
                    target_size = (R_t_f[l].shape[2],R_t_f[l].shape[3])
                    R_t_b_up = F.interpolate(R_t_b[l+1],target_size)
                    if self.local_grad:
                        R_t_b_up = R_t_b_up.detach()
                    R_input = torch.cat((R_t_f[l],R_t_b_up),dim=1)
                R_t_b[l], (H_t_b[l],C_t_b[l]) = backward_layer(R_input,
                                                               (H_tm1_b[l],
                                                                C_tm1_b[l]))
                # Compute Ahat (prediction about the next time step)
                in_height = R_t_b[l].shape[2]
                in_width = R_t_b[l].shape[3]
                padding = get_pad_same(in_height,in_width,self.kernel_sizes[l])
                R_t_b_l_padded = F.pad(R_t_b[l],padding)
                conv_layer = self.conv_layers[l]
                Ahat_t[l] = conv_layer(R_t_b_l_padded)
                if l == 0:
                    Ahat_t[l] = self.sigmoid(Ahat_t[l])
                else:
                    Ahat_t[l] = self.tanh(Ahat_t[l])

            # Update hidden states
            (H_tm1_f,C_tm1_f) = (H_t_f,C_t_f)
            (H_tm1_b,C_tm1_b) = (H_t_b,C_t_b)

            # Output
            if self.output == 'pred':
                if t < seq_len-1:
                    outputs.append(Ahat_t[0])
            elif self.output == 'rep':
                if t == seq_len - 1: # only return reps for last time step
                    outputs = R_t_b
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
        # get dimensions of E,R for each layer
        H_0 = []
        C_0 = []
        for l in range(self.nb_layers):
            channels = self.stack_sizes[l]
            # All hidden states initialized with zeros
            Hl = torch.zeros(batch_size,channels,height,width).to(self.device)
            Cl = torch.zeros(batch_size,channels,height,width).to(self.device)
            H_0.append(Hl)
            C_0.append(Cl)
            # Update dims
            height = int((height - 2)/2 + 1) # int performs floor
            width = int((width - 2)/2 + 1) # int performs floor
        return (H_0,C_0,H_0,C_0)
