import hickle as hkl
import torch
from torch.utils.data import Dataset

class KITTI(Dataset):
    def __init__(self,X_hkl,sources_hkl,seq_len,norm=True):
        self.X_hkl = X_hkl
        self.sources_hkl = sources_hkl
        self.seq_len = seq_len
        self.norm = norm # normalize pixel values to [0,1]
        # Load source data
        print("Loading sources data from ", sources_hkl)
        self.sources = hkl.load(sources_hkl)
        # Load image data
        print("Loading image data from ", X_hkl)
        X = hkl.load(X_hkl) #(n_images,height,width,in_channels)
        n_images = X.shape[0]
        print("Loaded %d images" % n_images)
        img_seqs = [] # list of tensors
        cur_loc = 0
        while cur_loc < n_images - seq_len + 1:
            cur_source = self.sources[cur_loc]
            end_loc = cur_loc + seq_len - 1
            end_source = self.sources[end_loc]
            if cur_source == end_source:
                img_tensor = torch.tensor(X[cur_loc:end_loc+1],
                                         dtype=torch.float)
                img_tensor = img_tensor.unsqueeze(0)
                if self.norm:
                    img_tensor = img_tensor / 255.
                img_seqs.append(img_tensor)
                cur_loc += seq_len
            else:
                cur_loc += 1
        self.X = torch.cat(img_seqs,dim=0) # (batch,len,height,width,channels)
        self.X = self.X.permute(0,1,4,2,3) # (batch,len,channels,height,width)
        print("Dataset contains %d sequences" % self.X.shape[0])

    def __getitem__(self,index):
        return self.X[index]

    def __len__(self):
        return self.X.shape[0]
