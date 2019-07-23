import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class Homes(Dataset):
    def __init__(self, is_train = True, is_valid=False, is_test=False, **kwargs):
        if is_valid:
          self.file_path = kwargs['image_path_val']
        elif is_test:
          self.file_path = kwargs['image_path_test']
        else: # train
          self.file_path = kwargs['image_path']

        self.train = is_train
        self.valid = is_valid
        self.test = is_test

        if self.test:
          self.img_list = [f for f in os.listdir(self.file_path)]
        else:
          self.img_list = [f for f in os.listdir(self.file_path) if 'image' in f]
          self.label_list = [f.replace('image', 'corner') for f in self.img_list]

    def __len__(self):
        return len(self.img_list)

    def _to_tensor(self, array):
        assert (isinstance(array, np.ndarray))
        tensor = torch.from_numpy(array)
        return tensor.float()

    def __getitem__(self, index):
        
        image = np.load(os.path.join(self.file_path, self.img_list[index])).transpose(1,2,0)

        if self.test:
            return self._to_tensor(image).permute(2,0,1), self.img_list[index]

        label = np.load(os.path.join(self.file_path, self.label_list[index]))
        #print(label.shape)
        return self._to_tensor(image).permute(2,0,1), self._to_tensor(label).permute(2,0,1)

def homes(**kwargs):
    return Homes(**kwargs)

homes.njoints = 70  # ugly but works
