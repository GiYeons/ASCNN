from torch import nn
import torch
import math
from torch.nn import functional as F

import warnings
warnings.filterwarnings("ignore")

class Psconv_block(nn.Module):
    def __init__(self, scale, cin, cout, ker):
        super(Psconv_block, self).__init__()
        cout = cout*(scale**2)

        self.conv = nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=(ker, ker), stride=(1,1), padding=(ker//2))
        self.upsample = nn.PixelShuffle(scale)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, mean=0.0,std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)


    def forward(self, inx):
        x= self.conv(inx)
        y = self.upsample(x)
        return y


class Net(nn.Module):
    def __init__(self, scale, im_c=1, fn=32, dfn = 64, r=None):
        super(Net, self).__init__()
        self.scale = scale

        self.first_part = nn.Conv2d(in_channels=im_c, out_channels=dfn, kernel_size=(5, 5), stride=(1,1), padding=(5//2))
        self.mid_part = nn.Conv2d(in_channels=dfn, out_channels=fn, kernel_size=(3, 3), stride=(1,1), padding=(3//2))
        self.last_part = Psconv_block(scale=scale, cin=fn, cout=im_c, ker=3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, mean=0.0,std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        self.tanh = nn.Tanh()

    def forward(self, inx, th=None, dilker=None, dilation=False):

        x = self.tanh(self.first_part(inx))
        x = self.tanh(self.mid_part(x))
        x = self.last_part(x)
        return x