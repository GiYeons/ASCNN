import torch.utils.data as data
from torch.utils.data import DataLoader
import torch
import h5py
import random
import numpy as np
from utils import b2float

def Generator(device, file_path='dataset.h5', scale=2, num_workers=3, batch_size=64, shuffle=True):
    dataset = Dataset(file_path=file_path, scale=scale)
    data_load = DataLoader(dataset=dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)

    while(True):
        for x_batch, y_batch in data_load:
            yield x_batch, y_batch

class Dataset(data.Dataset):
    def __init__(self, file_path, scale = 2, transforms = None):
        super(Dataset, self).__init__()

        dataset = h5py.File(file_path, 'r+')
        self.x_set = dataset.get('x_set')
        self.y_set = dataset.get('y_set')

    def augmentation(self, img, type):
        img = torch.from_numpy(img)
        if type == 1:
            return torch.flip(img, dims=[2])
        elif type == 2:
            return torch.rot90(img, k=1, dims=[1, 2])
        elif type == 3:
            return torch.rot90(img, k=2, dims=[1, 2])
        elif type == 4:
            return torch.rot90(img, k=3, dims=[1, 2])
        else:
            return img

    def __getitem__(self, index):
        types = [1,2,3,4,5]
        random.shuffle(types)
        type = types[0]
        augment = self.augmentation
        x = augment(self.x_set[str(index+1)][:], type)
        y = augment(self.y_set[str(index + 1)][:], type)

        return x, y

    def __len__(self):
        return len(self.x_set)
