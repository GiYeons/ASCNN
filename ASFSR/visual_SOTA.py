import os
from matlab import imresize, convertDouble2Byte as d2int
from skimage.io import imread, imsave
from skimage.color import rgb2ycbcr, ycbcr2rgb, yuv2rgb
import numpy as np
import torch
from utils import PSNR, rgb2y_uint8, SSIM
from ASFSR.baseline import Net


def val_psnr(model, th, dilker, dilation, im_path, scale, gray=True):
    img = imread(im_path)
    h, w, c = img.shape

    img = np.float64(img)/255

    hr = img[0:h - (h % scale), 0:w - (w % scale)]
    lr = imresize(hr, scalar_scale=1 / scale, method='bicubic')
    bic = imresize(lr, scalar_scale=scale, method='bicubic')

    lr = rgb2ycbcr(lr)[:,:,0:1]/255
    orig = rgb2ycbcr(hr)[:,:,0]/255


    lr_data= np.moveaxis(lr, -1, 0)

    lr_data = np.expand_dims(lr_data, axis=0)
    lr_data= torch.from_numpy(lr_data).float().to(device)

    sr = model(lr_data, th, dilker, dilation)
    sr = sr[0][0].cuda().data.cpu().numpy()

    sr_ssim =SSIM(sr, orig, scale)
    sr_psnr = PSNR(orig, sr, boundary=scale)

    bic_ssim = SSIM(rgb2ycbcr(bic)[:,:,0]/255, orig, scale)
    bic_psnr = PSNR(orig, rgb2ycbcr(bic)[:,:,0]/255, boundary=scale)


    tmp = rgb2ycbcr(bic)
    tmp = np.float32(tmp)
    tmp[:,:,0] = sr*255
    sr = ycbcr2rgb(tmp)

    return d2int(hr), d2int(bic), d2int(sr), round(bic_psnr, 2), round(bic_ssim, 4), round(sr_psnr, 2), round(sr_ssim, 4)

device = torch.device('cuda:2')


dir_n = 'ASFSR/checkpoint/scale_x2/'

# paths = ['th_01_dilk3', 'th_04_dilk3', 'th_07_dilk3', 'th_10_dilk3']
ths = [0.04]
paths = ['baseline']

# ths = [0.01, 0.04, 0.07, 0.10]

dilation= True
dilker = 3

scale = 2
r = 4
model = Net(scale, r=r).float().to(device)


im_path = '/home/wstation/DIV2K_val/0823.png'
im_name =  im_path.split('/')[-1]
model_name = dir_n.split('/')[0]
save_as = 'visual/'

for idx, path in enumerate(paths):
    path = dir_n + path + '.pth'
    ckpnt = torch.load(path)
    model.load_state_dict(ckpnt)
    th = ths[idx]

    hr, bic, sr, bic_psnr, bic_ssim, sr_psnr, sr_ssim = val_psnr(model, th, dilker, dilation, im_path, scale)

    imsave(save_as + 'hr_' + im_name, hr)
    imsave(save_as + 'bic_psnr_'+str(bic_psnr) +'_ssim_'+str(bic_ssim) +'_'+ im_name, bic)
    imsave(save_as+ model_name + '_sr_psnr_' +str(sr_psnr) +'_ssim_'+str(sr_ssim) +'_'+ im_name, sr)
