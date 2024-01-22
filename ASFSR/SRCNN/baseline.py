from torch import nn
import torch
import math
from torch.nn import functional as F

import warnings
warnings.filterwarnings("ignore")

class Net(nn.Module):
    def __init__(self, scale, im_c=1, fn=32, dfn = 64, r= None):
        super(Net, self).__init__()
        self.scale = scale

        self.first_part = nn.Conv2d(in_channels=im_c, out_channels=dfn, kernel_size=(9, 9), stride=(1,1), padding=(9//2))
        self.mid_part = nn.Conv2d(in_channels=dfn, out_channels=fn, kernel_size=(5, 5), stride=(1,1), padding=(5//2))
        self.last_part = nn.Conv2d(in_channels=fn, out_channels=im_c, kernel_size=(5, 5), stride=(1,1), padding=(5//2))

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):    # 선언된 레이어에 따라 다르게 초기화를 할 때 if문 쓰는것
                torch.nn.init.normal_(m.weight.data, mean=0.0,std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
                # .data: 텐서 가져와줌


    def forward(self, inx, th=None, dilker=None, dilation=False, cut=None):

        x = F.interpolate(inx, scale_factor=self.scale, mode='bicubic', align_corners=False)
        if cut!=None:
            x = x[:,:,cut:-cut, cut:-cut]


        x = self.relu(self.first_part(x))
        x = self.relu(self.mid_part(x))
        x = self.last_part(x)

        return x