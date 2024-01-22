from torch import nn
import torch
import math
from torch.nn import functional as F


import warnings
warnings.filterwarnings("ignore")

class Last_conv_block(nn.Module):
    def __init__(self, cin, cout, ker, r):
        super(Last_conv_block, self).__init__()
        self.ker = ker

        self.high_par = nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=(ker, ker), stride=(1,1), padding=(ker//2))

        self.low_par1 = nn.Conv2d(in_channels=cin, out_channels=cin // r, kernel_size=(1, 1), stride=(1, 1))
        self.low_par2 =  nn.Conv2d(in_channels=cin//r, out_channels=cout, kernel_size=(ker, ker), stride=(1,1), padding=(ker//2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, mean=0.0,std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)


    def forward(self, inx, mask, inv_mask):
        x1 = self.high_par(inx) * mask

        x2 = self.low_par1(inx)
        x2 = self.low_par2(x2) * inv_mask

        y = x1+x2
        return y

class Conv_block(nn.Module):
    def __init__(self, cin, cout, ker, r):
        super(Conv_block, self).__init__()
        self.ker = ker

        self.high_par = nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=(ker, ker), stride=(1,1), padding=(ker//2))

        self.low_par1 =  nn.Conv2d(in_channels=cin, out_channels=cout//r, kernel_size=(ker, ker), stride=(1,1), padding=(ker//2))
        self.low_par2 = nn.Conv2d(in_channels=cout//r, out_channels=cout, kernel_size=(1, 1), stride=(1,1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, mean=0.0,std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

    def forward(self, inx, mask, inv_mask):
        x1 = self.high_par(inx) * mask

        x2 = self.low_par1(inx) * inv_mask
        x2 = self.low_par2(x2) * inv_mask

        y = x1+x2
        return y


class Net(nn.Module):
    def __init__(self, scale, im_c=1, fn=32, dfn = 64, r= 8):
        super(Net, self).__init__()
        self.scale = scale

        self.first_part = Conv_block(cin=im_c, cout=dfn, ker=9, r=r)
        self.mid_part = Conv_block(cin=dfn, cout=fn, ker=5, r=r)

        self.last_part = Last_conv_block(cin=fn, cout=im_c, ker=5, r=r)

        self.relu = nn.ReLU(inplace=True)

    def dilate_mask(self, mask, dilker):
        mask = F.max_pool2d(mask.float(), kernel_size=(dilker, dilker), stride=(1, 1), padding=(dilker // 2))
        return mask

    def upsample_mask(self, mask):
        mask = mask.repeat(1, self.scale**2, 1, 1)
        mask = F.pixel_shuffle(mask, self.scale)
        inv_mask = torch.where(mask == 1, 0, 1)
        return mask, inv_mask

    def create_mask(self, inx, th, dilker, dilation=True):
        blur = F.avg_pool2d(inx, kernel_size=(3, 3), stride=1, padding=(3 // 2, 3 // 2), count_include_pad=False)
        loss = torch.abs((inx - blur))
        mask = torch.where(loss >= th, 1, 0).float()

        if dilation == True:
            mask = self.dilate_mask(mask, dilker)
        inv_mask = torch.where(mask==1, 0, 1)

        return mask, inv_mask

    def forward(self, inx, th, dilker=3, dilation=True, cut=None):
        mask, inv_mask = self.create_mask(inx, th, dilker, dilation)
        mask, inv_mask = self.upsample_mask(mask)

        x = F.interpolate(inx, scale_factor=self.scale, mode='bicubic', align_corners=False)
        if cut!=None:
            x = x[:,:,cut:-cut, cut:-cut]
            mask = mask[:,:,cut:-cut, cut:-cut]
            inv_mask = inv_mask[:,:,cut:-cut, cut:-cut]

        x = self.relu(self.first_part(x, mask, inv_mask))
        x = self.relu(self.mid_part(x, mask, inv_mask))
        x = self.last_part(x, mask, inv_mask)

        return x