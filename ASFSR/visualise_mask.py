import numpy as np
import torch
from torch.nn import functional as F
from skimage.io import imread, imsave
from skimage.color import rgb2ycbcr
from matlab import imresize
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore")

def cal_sparsity(mask):
    n, c, h, w = mask.shape
    all = n * c * h * w
    sparse = (all - torch.count_nonzero(mask)) / all
    # sparsity = (sparse1+sparse2).item()/2
    sparsity = sparse.item()
    return sparsity

def dilate_mask(mask, ker):
    mask = torch.max_pool2d(mask.float(), kernel_size=(ker,ker), stride= (1,1), padding=(ker//2))
    return mask

def create_mask(img, th):
    blur = F.avg_pool2d(img, kernel_size=(3, 3), stride=1, padding=(3 // 2, 3 // 2), count_include_pad=False)
    loss = torch.abs((img - blur))

    hf_map = (loss.numpy()[0][0])
    # plt.imsave('hf_map.png', hf_map)
    plt.imshow(hf_map)
    plt.colorbar()
    plt.axis('off')
    plt.show()

    # plt.colorbar()

    mask = torch.where(loss >= th, 1, 0).float()
    return mask


def show_mask(path, th, scale, dil_ker, dilation=False):
    img = imread(path)
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

    dil = ''
    # get mask
    mask = create_mask(img, th)
    if dilation == True:
        mask = dilate_mask(mask, dil_ker)
        dil = 'dil'+str(dil_ker)

    save_as = 'see_masks/th_'+str(th)+ '_' + dil + '.bmp'

    print(round(cal_sparsity(mask), 2))
    imsave(save_as, np.uint8(mask.numpy()[0][0]*255))


path = '/home/wstation/Set5/butterfly.bmp'

scale = 2
# ths = [0.0022, 0.0068, 0.016, 0.038]
#ths = [0.016, 0.038, 0.068, 0.114]
# ths = [0.002, 0.016, 0.03, 0.044]

ths = [0.11]
dilation = False
dil_ker = 3
for th in ths:
    show_mask(path, th, scale, dil_ker, dilation=dilation)

