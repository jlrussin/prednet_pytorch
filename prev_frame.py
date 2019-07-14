import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import *

def prev_frame_loss(ccn_data_dir):
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Dataset
    dataset = CCN(ccn_data_dir,8)
    n_seqs = len(dataset)
    batch_size = 1  # batch size is 1 for simple averaging
    dataloader = DataLoader(dataset,batch_size,shuffle=False)

    # Loss
    mse_loss = nn.MSELoss()

    losses = []
    for i,X in enumerate(dataloader):
        p = 100*i/len(dataloader)
        print("Computing average mse loss of previous frame: %.2f%%" % p)
        X = X.to(device)
        X_tm1 = X[:,:-1,:,:,:]
        X_t = X[:,1:,:,:,:]
        loss = mse_loss(X_tm1,X_t)
        losses.append(loss.data.item())
    mean_loss = np.mean(losses)
    print("Previous frame loss for data in %s: %f" % (ccn_data_dir,mean_loss))
