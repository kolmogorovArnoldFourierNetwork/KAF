import os
import torch
from torch.utils.data import Dataset
import numpy as np
import gzip
import struct

class CustomEMNIST(Dataset):
    
    def __init__(self, root, split='balanced', train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.split not in ['balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist']:
            raise ValueError("Useless split")

        split_dict = {
            'balanced': 'emnist-balanced',
            'letters': 'emnist-letters',
            'digits': 'emnist-digits',
            'mnist': 'emnist-mnist',
            'byclass': 'emnist-byclass',
            'bymerge': 'emnist-bymerge'
        }
        
        base_filename = split_dict[split]
        images_filename = f"{base_filename}-{'train' if train else 'test'}-images-idx3-ubyte.gz"
        labels_filename = f"{base_filename}-{'train' if train else 'test'}-labels-idx1-ubyte.gz"
        
        self.data = self._read_images(os.path.join(self.root, images_filename))
        self.targets = self._read_labels(os.path.join(self.root, labels_filename))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        
        img = torch.FloatTensor(img).unsqueeze(0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _read_images(self, path):
        with gzip.open(path, 'rb') as f:
            magic, size = struct.unpack('>II', f.read(8))
            nrows, ncols = struct.unpack('>II', f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(size, nrows, ncols)
            data = np.transpose(data, (0, 2, 1))
            return data.astype(np.float32) / 255.0

    def _read_labels(self, path):
        with gzip.open(path, 'rb') as f:
            magic, size = struct.unpack('>II', f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data