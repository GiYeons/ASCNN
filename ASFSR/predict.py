import os
from matlab import imresize, convertDouble2Byte as d2int
from skimage.io import imread, imsave
from skimage.color import rgb2ycbcr
import numpy as np
import torch
from utils import PSNR, rgb2y_uint8, SSIM

# import the network we want to predict for
from ESPCN.baseline import Net

def val_psnr(model, th, dilker, dilation, val_path, scale):
    images = sorted(os.listdir(val_path))
    avg_psnr = 0
    avg_ssim = 0
    for image in images:

        # read image
        # convert it to grayscale
        # image format: (height, width, channel)
        img = imread(val_path + image)
        try:
            h, w, c = img.shape
            img = rgb2ycbcr(img)[:,:,0:1]
        except:
            h, w = img.shape
            img = np.expand_dims(img, axis = -1)

        # convert to float and normalize (0~1)
        img = np.float64(img)/255

        # make image divisible by scale
        # we cut the last ends to achieve this
        orig = img[0:h - (h % scale), 0:w - (w % scale)]

        # low resolution
        lr = imresize(orig, scalar_scale=1 / scale, method='bicubic')

        # change image format to : (batch, channel, height, width)
        # convert to pytorch tensor and float32
        lr_data= np.moveaxis(lr, -1, 0)
        lr_data = np.expand_dims(lr_data, axis=0)
        lr_data= torch.from_numpy(lr_data).float().to(device)

        # predict
        sr = model(lr_data, th, dilker, dilation)

        #convert back to numpy (cpu)
        # image format: (height, width)
        sr = sr[0].cuda().data.cpu().numpy()
        sr = sr[0]

        gray_orig = orig[:,:,0]
        gray_sr = sr

        avg_ssim +=SSIM(gray_sr, gray_orig, scale)
        avg_psnr += PSNR(gray_orig, gray_sr, boundary=scale)


    # print(round(avg_ssim/len(images), 4), end=' ')
    return avg_psnr/len(images), avg_ssim/len(images)

device = torch.device('cuda:2')


sets = ['Set5/', 'Set14/', 'BSDS100/', 'Urban100/', 'DIV2K_val/']
dir_n = 'ESPCN/checkpoint/scale_x4/'
# paths = ['sp_02_dilk7', 'sp_04_dilk7', 'sp_06_dilk7', 'sp_08_dilk7']
# paths = ['sp_02_dilk3', 'sp_04_dilk3', 'sp_06_dilk3', 'sp_08_dilk3']
# paths = ['r8_th_01_dilk3', 'r8_th_04_dilk3', 'r8_th_07_dilk3', 'r8_th_10_dilk3']
# paths = ['th_01_dilk3', 'th_04_dilk3', 'th_07_dilk3', 'th_10_dilk3']
# paths = ['th_04_dilk3']
# paths = ['sp_02', 'sp_04', 'sp_06', 'sp_08']

# ths = [0.01, 0.025, 0.048, 0.088]
# ths = [0.0022, 0.0068, 0.016, 0.038]
# ths = [0.021, 0.05, 0.084, 0.13]
# ths = [0.03, 0.05, 0.07, 0.09]

paths = ['baseline']

# ths = [0.01, 0.04, 0.07, 0.10]
# ths = [0]
ths = [0.04]
dilation= True
dilker = 3

scale = 4
r = 4
model = Net(scale, r=r).float().to(device)


'''
the code below (from line 102) is to test for different models quickly at the same time
try to run without this code
just use: 
ckpnt = torch.load(path) # load weight path
model.load_state_dict(ckpnt) # load the weights to model
avg_psnr, avg_ssim = val_psnr(model, th, dilker, dilation, val_path, scale) # evaluate model with validation path
'''

for set in sets:
    for idx, path in enumerate(paths):
        path = dir_n + path + '.pth'
        ckpnt = torch.load(path)
        model.load_state_dict(ckpnt)
        val_path = '/home/wstation/' + set
        th = ths[idx]

        result1, result2 = val_psnr(model, th, dilker, dilation, val_path, scale)
        print(round(result1, 2), end='/')
        print(round(result2, 4), end='')
    print(end='\t')
