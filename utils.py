import torch.nn as nn
from activations import Hardsigmoid, SatLU

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_activation(activation):
    if activation == 'sigmoid':
        act_fn = nn.Sigmoid()
    elif activation == 'tanh':
        act_fn = nn.Tanh()
    elif activation == 'relu':
        act_fn = nn.ReLU()
    elif activation == 'hardsigmoid':
        act_fn = Hardsigmoid()
    return act_fn

def get_pad_same(in_height,in_width,kernel_size):
    if isinstance(kernel_size,int):
        k_height = kernel_size
        k_width = kernel_size
    elif isinstance(kernel_size,tuple):
        k_height = kernel_size[0]
        k_width = kernel_size[1]
    # Padding so that conv2d gives same dimensions
    pad_width = (k_width - 1) / 2 # stride = 1, dilation = 1
    pad_height = (k_height - 1) / 2 # stride = 1, dilation = 1
    left_pad = int(np.floor(pad_width))
    right_pad = int(np.ceil(pad_width))
    top_pad = int(np.floor(pad_height))
    bottom_pad = int(np.floor(pad_height))
    return (left_pad, right_pad, top_pad, bottom_pad)
