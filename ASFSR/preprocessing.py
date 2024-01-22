import numpy as np
import h5py
import os
from skimage.io import imread
from matlab import imresize, convertDouble2Byte as d2int
from utils import rgb2y_uint8
from skimage.transform import rotate
from utils import d2bytes, b2double
from skimage.color import rgb2ycbcr


'''
==============READ ME==========
we have to create datasets for training
we preprocess the images and store them as _.hy file
'''


def make_data(scale, lr_size, x_set, y_set, path, sub_cnt=0, stride=None, rgb=False):
    hr_size = (lr_size * scale)
    if stride == None: stride = hr_size    # 왜??

    # data augmentation by downscaling (we want to make more data because 91 images is too small)
    downscales = [None, 0.98, 0.96, 0.94, 0.92, 0.9,
                  0.88, 0.86, 0.84, 0.82, 0.8,
                  0.78, 0.76, 0.74, 0.72, 0.7,
                  0.68, 0.66, 0.64, 0.62, 0.6,
                  0.58, 0.56, 0.54, 0.52, 0.5]  # 다운스케일에 사용하는 계수들

    images = sorted(os.listdir(path))  # path내의 모든 파일과 디렉토리의 리스트 리턴

    img_cnt = 0
    for image_n in images:
        image = imread(path + image_n)
        if rgb == False: image = rgb2ycbcr(image)[:,:,0:1]/255     # y만 뽑은거임
        img_cnt += 1

        # increase the sides of the image
        reflect = lr_size // 2  # LR이 32x32면 16을 reflect(이만큼만 해도 (0,0) 픽셀이 중심이 될수있음 )
        image = np.pad(image, ((reflect, reflect+stride+scale), (reflect, reflect+stride+scale), (0, 0)), 'reflect')    #어떤 서브이미지를 뽑더라도 모든 픽셀이 중심이 될수있도록
        imshape = image.shape
        h = imshape[0]
        w = imshape[1]

        if h <hr_size or w < hr_size:   # HR 사이즈보다 작으면 추가로 패딩해주네(원본이미지가 너무 작은 경우를 고려한듯 )
            image = np.pad(image, ((0, hr_size), (0, hr_size), (0,0)), 'reflect')


        for downscale in downscales:
            img =(image)

            if downscale != None:
                img = imresize(img, scalar_scale=downscale, method='bicubic')

            imshape = img.shape
            h = imshape[0]
            w = imshape[1]

            #modcrop : make image divisible by scale
            hr_img = img[0:h - (h % scale), 0:w - (w % scale)]  # HR이 64x64인데 x3으로 다운스케일이 안되니까 mod만큼 자른거임

            lr_img = imresize(hr_img, scalar_scale=1 / scale, method='bicubic') # 스케일만큼 다운스케일

            #remove borders
            hr_img = hr_img[scale:-scale, scale:-scale]
            lr_img = lr_img[1:-1, 1:-1]


            #conver to uint8
            hr_img = d2int(hr_img)
            lr_img = d2int(lr_img)  # normalize 한걸 저장할때는 다시 되돌린 것임

            imshape = hr_img.shape
            h = imshape[0]
            w = imshape[1]

            if h >= hr_size and w >= hr_size:
                for height in range(0, h - (hr_size), stride):
                    for width in range(0, w - (hr_size), stride):
                        #create hr sub_images
                        left = width
                        right = left + hr_size
                        top = height
                        bottom = top + hr_size
                        hr_subim = hr_img[top:bottom, left:right]
                        # bic_subim = np.float32(bic_img[top:bottom, left:right])

                        # create lr sub_images
                        left = width // scale
                        right = left + lr_size
                        top = height // scale
                        bottom = top + lr_size
                        lr_subim = lr_img[top:bottom, left:right]

                        # store data sets with name as sub_cnt
                        sub_cnt += 1
                        x_set.create_dataset(str(sub_cnt), data=np.moveaxis(lr_subim, -1, 0))
                        y_set.create_dataset(str(sub_cnt), data=np.moveaxis(hr_subim, -1, 0))

                        percent = int(img_cnt / len(images) * 100)
                        print('\r...preprocessing', str(percent) + '% ', '|image no.',img_cnt, '|LR shape:', lr_subim.shape, '|HR shape:', hr_subim.shape, '|subimage no.',sub_cnt, end='')
    return sub_cnt

# path of images
data_path = '/home/wstation/Set91/'

# path to save training data
save_as = 'dataset/LR_HR/scale_x3/'
os.makedirs(save_as, exist_ok=True) # check if folder exist
save_as += 'dataset_cpy1.h5' # save data as

scale = 3
lr_size = 32

if scale ==2:
    stride = 8
elif scale == 3:
    stride = 6
else:
    stride = 4

'''
we create two groups:
x_set (low resolution)
y_set (high resolution)
we put the data in order inside each group
'''

with h5py.File(save_as, 'w') as dataset:
    x_set = dataset.create_group('x_set')
    y_set = dataset.create_group('y_set')
    make_data(scale, lr_size, x_set, y_set, data_path, stride=stride)
dataset.close()