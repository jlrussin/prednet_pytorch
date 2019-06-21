import os
import numpy as np
import hickle as hkl
import torch
from torch.utils.data import Dataset
from PIL import Image



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

class CCN(Dataset):
    def __init__(self,X_hkl,labels_hkl,seq_len,
                 norm=True,return_labels=False):
        self.X_hkl = X_hkl
        self.labels_hkl = labels_hkl
        self.seq_len = seq_len
        self.norm = norm # normalize pixel values to [0,1]
        self.return_labels = return_labels # return images, labels
        # Load label data
        print("Loading label data from ", labels_hkl)
        self.labels = hkl.load(labels_hkl)
        # Load image data
        print("Loading image data from ", X_hkl)
        self.X = hkl.load(X_hkl) #(n_seqs,seq_len,height,width,in_channels)
        self.n_seqs = self.X.shape[0]
        print("Loaded %d sequences" % self.n_seqs)

        msg = "Number of image sequences does not match number of labels"
        assert self.X.shape[0] == len(self.labels)

    def __getitem__(self,index):
        img_seq = self.X[index]
        img_tensor = torch.tensor(img_seq,dtype=torch.float)
        img_tensor = img_tensor.permute(0,3,1,2) # (len,channels,height,width)
        if self.norm:
            img_tensor = img_tensor / 255.
        if self.return_labels:
            label = self.labels[index]
            return img_tensor,label
        else:
            return img_tensor

    def __len__(self):
        return self.n_seqs

def ccn_dir_to_hkl(directory,seq_len,val_p,test_p):
    """
    This function assumes:
        -Sorted listdir will be in order
        -Category labels are in split[4]
        -Tick numbers are in split[7]
    """
    print("Loading images from %s" % directory)
    img_seqs = [] # list of lists of images
    img_seq = [] # list of images in one seq
    labels = [] # list of category labels for each seq
    # Loop through files to get images and labels
    for fn in sorted(os.listdir(directory)):
        # Get category label and tick number
        split = fn.split('_')
        label = split[4]
        t = int(split[7])
        # Get image
        img = Image.open(directory + fn)
        arr = np.array(img)
        img_seq.append(arr)
        if t == seq_len-1:
            labels.append(label)
            img_seqs.append(img_seq)
            img_seq = []

    # Split data: train, val, test
    print("Splitting into train, val, test")
    n_seqs = len(img_seqs)
    print("Total number of image sequences: ",n_seqs)
    img_seqs = np.array(img_seqs)
    labels = np.array(labels)

    n_val = int(n_seqs*val_p)
    n_test = int(n_seqs*test_p)
    n_train = n_seqs - n_val - n_test

    indices = np.random.permutation(n_seqs)
    train_ids = indices[0:n_train]
    val_ids = indices[n_train:n_train+n_val]
    test_ids = indices[n_train+n_val:n_seqs]

    train_seqs = img_seqs[train_ids]
    val_seqs = img_seqs[val_ids]
    test_seqs = img_seqs[test_ids]

    train_labels = labels[train_ids].tolist()
    val_labels = labels[val_ids].tolist()
    test_labels = labels[test_ids].tolist()

    print("Number of train sequences: ",n_train)
    print("Number of val sequences: ",n_val)
    print("Number of test sequences: ",n_test)

    # Dump with hickle
    print("Writing train images to X_train.hkl")
    hkl.dump(train_seqs,'X_train.hkl')
    print("Writing train labels to labels_train.hkl")
    hkl.dump(train_labels,'labels_train.hkl')
    print("Writing val images to X_val.hkl")
    hkl.dump(val_seqs,'X_val.hkl')
    print("Writing val labels to labels_val.hkl")
    hkl.dump(val_labels,'labels_val.hkl')
    print("Writing test images to X_test.hkl")
    hkl.dump(test_seqs,'X_test.hkl')
    print("Writing test labels to labels_test.hkl")
    hkl.dump(test_labels,'labels_test.hkl')
