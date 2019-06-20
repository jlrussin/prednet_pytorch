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
        self.X = hkl.load(X_hkl) #(n_images,height,width,in_channels)
        n_images = self.X.shape[0]
        print("Loaded %d images" % n_images)

        # Get starting and ending indices
        self.start_end_idxs = [] # list of (start_loc,end_loc) tuples
        cur_loc = 0
        while cur_loc < n_images - seq_len + 1:
            cur_source = self.sources[cur_loc]
            end_loc = cur_loc + seq_len - 1
            end_source = self.sources[end_loc]
            if cur_source == end_source:
                self.start_end_idxs.append((cur_loc,end_loc))
                cur_loc += seq_len
            else:
                cur_loc += 1
        print("Dataset contains %d sequences" % self.X.shape[0])

    def __getitem__(self,index):
        start,end = self.start_end_idxs[index]
        img_seq = self.X[start:end+1]
        img_tensor = torch.tensor(img_seq,dtype=torch.float)
        img_tensor = img_tensor.permute(0,3,1,2) # (len,channels,height,width)
        if self.norm:
            img_tensor = img_tensor / 255.
        return img_tensor

    def __len__(self):
        return len(self.start_end_idxs)
