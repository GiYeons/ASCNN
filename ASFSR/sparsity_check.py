import numpy as np
import torch
from torch.nn import functional as F
from skimage.io import imread, imsave
from skimage.color import rgb2ycbcr
from matlab import imresize
import os

import warnings
warnings.filterwarnings("ignore")


'''
=========== READ ME ===================
Use threshold and dilation to check sparsity of dataset
'''


def cal_sparsity(mask):
    n, c, h, w = mask.shape
    all = n * c * h * w
    sparse = (all - torch.count_nonzero(mask)) / all
    # sparsity = (sparse1+sparse2).item()/2
    sparsity = sparse.item()
    return sparsity

def dilate_mask(mask, dil_ker):
    mask = torch.max_pool2d(mask.float(), kernel_size=(dil_ker,dil_ker), stride= (1,1), padding=(dil_ker//2))
    return mask

def create_mask(img, th):
    blur = F.avg_pool2d(img, kernel_size=(3, 3), stride=1, padding=(3 // 2, 3 // 2), count_include_pad=False)
    loss = torch.abs((img - blur))
    mask = torch.where(loss >= th, 1, 0).float()
    return mask

def sparsity_threshold_relationship(path, scale, dilation=False):

    start = 0.001
    stop = 0.1
    gap = 0.01

    ths = np.arange(start, stop, gap)

    print('thresholds:')
    for th in ths:
        print(th)

    print('Sparsities on dataset:')
    for th in ths:
        spar = avg_sparsity(path, th, scale, dilation=dilation)
        print(round(spar, 2))

def avg_sparsity(path, th, scale, dil_ker, dilation=False):
    avgspar = 0
    images = os.listdir(path)

    for image in images:
        # open image
        img = imread(path+image)
        img = np.float32(img)/255
        img = imresize(img, scalar_scale=1/scale)

        # convert to torch tensor
        if len(img.shape) < 3:
            img = np.expand_dims(img, axis=0)
        else:
            img = rgb2ycbcr(img)[:,:,0:1]/255
            img = np.moveaxis(img, -1, 0)

        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).float()
        #print(img.shape)

        # get mask
        mask = create_mask(img, th)
        if dilation == True:
            mask = dilate_mask(mask, dil_ker)


        if image == 'butterfly.bmp':

            imsave('mask.png', np.uint8(mask.numpy()[0][0]*255))

        #print(mask.shape)
        avgspar += cal_sparsity(mask)

    avgspar = avgspar/len(images)
    return avgspar

path = '/home/wstation/Set14/'

scale = 2
dilation = True
dil_ker = 3
#ths = [0.0022, 0.0068, 0.016, 0.038]
#ths = [0.016, 0.038, 0.068, 0.114]
#ths = [0.002, 0.016, 0.03, 0.044]
# ths = [0.03, 0.05, 0.07, 0.09]
# ths = [0.03, 0.06, 0.09, 0.12]
# ths = [0.03, 0.07, 0.11, 0.15]
#
# for th in ths:
#     spar = avg_sparsity(path, th, scale, dil_ker, dilation=dilation)
#     print(round(spar, 2), end='\n')

#sparsity_threshold_relationship(path, scale, dilation=True)

scales = [2, 3, 4]
root = '/home/wstation/'
sets = ['Set14/', 'Set5/', 'BSDS100/', 'DIV2K_val/']
th = 0.04

for scale in scales:
    avg = 0
    for set in sets:
        path = root+set
        avg += avg_sparsity(path, th, scale, dil_ker, dilation=dilation)

    print(avg/len(sets))
