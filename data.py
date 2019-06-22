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
    def __init__(self,img_dir,seq_len,norm=True,return_labels=False):
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.norm = norm # normalize pixel values to [0,1]
        self.return_labels = return_labels # return images, labels
        # Organize filenames into a list of seqs
        self.labels = []
        self.fn_seqs = []
        fn_seq = []
        print("Loading files from %s" % img_dir)
        for fn in sorted(os.listdir(img_dir)):
            fn_seq.append(fn)
            split = fn.split('_')
            if split[4] in ['car','motorcycle']:
                t = int(split[8])
                label = split[4] + '_' + split[5]
            else:
                t = int(split[7])
                label = split[4]
            if t == seq_len-1:
                self.fn_seqs.append(fn_seq)
                self.labels.append(label)
                fn_seq = []
        print("Dataset has %d sequences" % len(self.fn_seqs))

    def __getitem__(self,index):
        fn_seq = self.fn_seqs[index]
        path_seq = [os.path.join(self.img_dir,fn) for fn in fn_seq]
        img_seq = [Image.open(p) for p in path_seq]
        arr_seq = np.stack(img_seq)
        img_tensor = torch.tensor(arr_seq,dtype=torch.float)
        img_tensor = img_tensor.permute(0,3,1,2) # (len,channels,height,width)
        if self.norm:
            img_tensor = img_tensor / 255.
        if self.return_labels:
            label = self.labels[index]
            return img_tensor,label
        else:
            return img_tensor

    def __len__(self):
        return len(self.fn_seqs)

def get_dataloader(args,split,dataloader_kwargs):
    if split == 'train':
        data_path = args.train_data_path
        sources_path = args.train_sources_path
    elif split == 'val':
        data_path = args.val_data_path
        sources_path = args.val_sources_path
    elif split == 'test':
        data_path = args.test_data_path
        sources_path = args.test_sources_path
    if args.dataset == 'KITTI':
        data = KITTI(data_path,sources_path,args.seq_len)
    elif args.dataset == 'CCN':
        data = CCN(data_path,args.seq_len)
    dataloader = DataLoader(data,args.batch_size,
                            shuffle=True,num_workers=1,**dataloader_kwargs)
    return dataloader

def split_ccn(img_dir,seq_len,val_p,test_p):
    """
    This function assumes:
        -Sorted listdir will be in order
        -Category labels are in split[4], except car/motorcycle in 4/5
        -Tick numbers are in split[7], except car/motorcycle in 8
    """
    print("Loading filenmaes from %s" % img_dir)
    fn_seqs = [] # list of lists of filenmanes
    fn_seq = [] # current list of filenames
    for fn in sorted(os.listdir(img_dir)):
        fn_seq.append(fn)
        split = fn.split('_')
        if split[4] in ['car','motorcycle']:
            t = int(split[8])
        else:
            t = int(split[7])
        if t == seq_len-1:
            fn_seqs.append(fn_seq)
            fn_seq = []

    # Split data: train, val, test
    print("Splitting into train, val, test")
    n_seqs = len(fn_seqs)
    print("Total number of sequences: ",n_seqs)

    n_val = int(n_seqs*val_p)
    n_test = int(n_seqs*test_p)
    n_train = n_seqs - n_val - n_test

    train_list = ['train/' for i in range(n_train)]
    val_list = ['val/' for i in range(n_val)]
    test_list = ['test/' for i in range(n_test)]
    all_list = train_list + val_list + test_list
    partition = np.random.permutation(all_list)

    print("Number of train sequences: ",n_train)
    print("Number of val sequences: ",n_val)
    print("Number of test sequences: ",n_test)

    # Make train, val, test directories
    if not os.path.isdir(img_dir + 'train'):
        os.mkdir(img_dir + 'train')
    if not os.path.isdir(img_dir + 'val'):
        os.mkdir(img_dir + 'val')
    if not os.path.isdir(img_dir + 'test'):
        os.mkdir(img_dir + 'test')

    # Move images into train, val, test
    print("Moving images to train, val, test directories")
    for i,fn_seq in enumerate(fn_seqs):
        if i % 1000 == 0:
            print("starting sequence %d" % i)
        for fn in fn_seq:
            new_dir = partition[i]
            old_path = img_dir+fn
            new_path = img_dir+new_dir+fn
            os.rename(old_path,new_path)
    print("Done!")
