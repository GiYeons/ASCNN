from torch import nn
import torch
import math
from torch.nn import functional as F

import warnings
warnings.filterwarnings("ignore")   # 경고메세지 끄기

class Tconv_block(nn.Module):
    def __init__(self, scale, cin, cout, ker, r):
        super(Tconv_block, self).__init__()
        self.scale = scale
        self.ker = ker

        self.high_par = nn.ConvTranspose2d(in_channels=cin, out_channels=cout, kernel_size=(ker, ker),stride=(scale, scale), padding=(ker // 2), output_padding=scale - 1)
        self.low_par1 = nn.Conv2d(in_channels=cin, out_channels=cin // r, kernel_size=(1, 1), stride=(1, 1))
        self.low_par2 = nn.ConvTranspose2d(in_channels=cin // r, out_channels=cout, kernel_size=(ker, ker),stride=(scale, scale), padding=(ker // 2), output_padding=scale - 1)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                nn.init.zeros_(m.bias.data)
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, mean=0.0,std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        #===========
        self.ones = nn.Parameter(data = torch.ones(size=(cin, 1, 1, 1)).float(), requires_grad=False)
        # ===========

    def expand_hw(self, x):
        b, c, h, w = x.shape
        x = F.conv_transpose2d(x, self.ones[0:c], stride=(self.scale, self.scale), output_padding=self.scale-1, groups=c)
        return x

    def tconv_to_conv_par(self, par):
        par = torch.rot90(par, 2, [2, 3])
        par = par.transpose(0, 1)
        return par

    def eval_forward(self, inx, mask_idx, inv_mask_idx):
        high_par = self.tconv_to_conv_par(self.high_par.weight)
        low_par1 = self.low_par1.weight
        low_par2 = self.tconv_to_conv_par(self.low_par2.weight)

        x = self.expand_hw(inx)
        b, c, h, w = x.shape
        cout, cin, ker, ker = high_par.data.shape

        x = F.pad(x, pad=(ker // 2, ker // 2, ker // 2, ker // 2))
        x = x.unfold(2, ker, 1).unfold(3, ker, 1)
        x = x.transpose(0, 1)
        x = x.contiguous().view(cin, -1, ker, ker)
        x = x.transpose(0, 1)

        x_out = x.new(b * h * w, cout, 1, 1)

        x_out[mask_idx, :, :, :] = F.conv2d(x[mask_idx], high_par)
        x_out[inv_mask_idx, :, :, :] = F.conv2d(F.conv2d(x,low_par1)[inv_mask_idx], low_par2)
        x = x_out

        x = x.view(b, h * w, cout)
        x = x.transpose(1, 2)

        y = F.fold(x, (h, w), (1, 1))
        return y

    def forward(self, inx, mask, inv_mask, eval=False):
        if eval == True:
            return self.eval_forward(inx, mask_idx= mask, inv_mask_idx= inv_mask)

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


    def eval_forward(self, inx, mask_idx, inv_mask_idx):
        b, c, h, w = inx.shape
        high_par = self.high_par.weight
        low_par1 = self.low_par1.weight
        low_par2 = self.low_par2.weight
        cout, cin, ker, ker = high_par.data.shape

        x = F.pad(inx, pad=(ker//2, ker//2, ker//2, ker//2))    # 원래 있어야 했던 패딩
        x = x.unfold(2, ker, 1).unfold(3, ker, 1)           #이미지를 펴는 과정
        x = x.transpose(0, 1)
        x = x.contiguous().view(cin, -1, ker, ker)
        x = x.transpose(0, 1)

        x_out = x.new(b * h * w, cout, 1, 1)        # 결과의 크기

        x_out[mask_idx, :, :, :] = F.conv2d(x[mask_idx], high_par)
        x_out[inv_mask_idx, :, :, :] = F.conv2d(F.conv2d(x[inv_mask_idx], low_par1), low_par2)
        x = x_out

        x = x.view(b, h * w, cout)
        x = x.transpose(1, 2)

        y = F.fold(x, (h, w), (1, 1))
        return y

    def forward(self, inx, mask, inv_mask, eval=False):
        if eval == True:
            return self.eval_forward(inx, mask_idx= mask, inv_mask_idx= inv_mask)

        x1 = self.high_par(inx) * mask

        x2 = self.low_par1(inx) * inv_mask
        x2 = self.low_par2(x2) * inv_mask

        y = x1+x2
        return y


class Net(nn.Module):
    def __init__(self, scale, im_c=1, fn=16, dfn = 32, r= 4):
        super(Net, self).__init__()
        self.scale = scale

        self.first_part = Conv_block(cin = im_c, cout=dfn, ker=5, r=r)
        self.reduction = Conv_block(cin=dfn, cout=fn, ker=1, r=r)

        self.mid_part1 = Conv_block(cin = fn, cout=fn, ker=3, r=r)
        self.mid_part2 = Conv_block(cin = fn, cout=fn, ker=3, r=r)
        self.mid_part3 = Conv_block(cin = fn, cout=fn, ker=3, r=r)
        self.mid_part4 = Conv_block(cin = fn, cout=fn, ker=3, r=r)

        self.expansion = Conv_block(cin=fn, cout=dfn, ker=1, r=r)
        self.last_part = Tconv_block(scale=scale, cin=dfn, cout=im_c, ker=9, r=r)

        self.relu = nn.ReLU(inplace=True)


    def dilate_mask(self, mask, dilker):
        mask = F.max_pool2d(mask.float(), kernel_size=(dilker, dilker), stride=(1, 1), padding=(dilker // 2))
        return mask

    def create_mask(self, inx, th, dilker, dilation=True):
        # count_include_pad: 패딩 연산에 미포함
        blur = F.avg_pool2d(inx, kernel_size=(3, 3), stride=1, padding=(3 // 2, 3 // 2), count_include_pad=False)
        loss = torch.abs((inx - blur))
        mask = torch.where(loss >= th, 1, 0).float()

        if dilation == True:
            mask = self.dilate_mask(mask, dilker)
        inv_mask = torch.where(mask==1, 0, 1)   #inverse

        return mask, inv_mask

    def upsample_mask(self, mask):
        mask = mask.repeat(1, self.scale**2, 1, 1)
        mask = F.pixel_shuffle(mask, self.scale)
        inv_mask = torch.where(mask == 1, 0, 1)
        return mask, inv_mask

    def get_mask_index(self, mask):
        mask = torch.flatten(mask, start_dim=0)
        inv_mask = torch.where(mask == 1, 0, 1)

        mask_idx = torch.nonzero(mask)[:, 0]
        inv_mask_idx = torch.nonzero(inv_mask)[:, 0]

        return mask_idx, inv_mask_idx

    def eval_forward(self, inx, th, dilker, dilation=True):
        mask, inv_mask = self.create_mask(inx, th, dilker, dilation)
        mask_idx, inv_mask_idx = self.get_mask_index(mask)

        x = self.relu(self.first_part(inx, mask_idx, inv_mask_idx, eval=True))
        x = self.relu(self.reduction(x, mask_idx, inv_mask_idx, eval=True))

        x = self.relu(self.mid_part1(x, mask_idx, inv_mask_idx, eval=True))
        x = self.relu(self.mid_part2(x, mask_idx, inv_mask_idx, eval=True))
        x = self.relu(self.mid_part3(x, mask_idx, inv_mask_idx, eval=True))
        x = self.relu(self.mid_part4(x, mask_idx, inv_mask_idx, eval=True))

        x = self.relu(self.expansion(x, mask_idx, inv_mask_idx, eval=True))

        mask, inv_mask = self.upsample_mask(mask)
        mask_idx, inv_mask_idx = self.get_mask_index(mask)
        y = self.last_part(x, mask_idx, inv_mask_idx, eval=True)

        return y


    def forward(self, inx, th, dilker=3, dilation=True, eval=False):
        if eval == True:
            return self.eval_forward(inx, th, dilker, dilation)

        mask, inv_mask = self.create_mask(inx, th, dilker, dilation)

        x = self.relu(self.first_part(inx, mask, inv_mask, eval=False))
        x = self.relu(self.reduction(x, mask, inv_mask, eval=False))

        x = self.relu(self.mid_part1(x, mask, inv_mask, eval=False))
        x = self.relu(self.mid_part2(x, mask, inv_mask, eval=False))
        x = self.relu(self.mid_part3(x, mask, inv_mask, eval=False))
        x = self.relu(self.mid_part4(x, mask, inv_mask, eval=False))

        x = self.relu(self.expansion(x, mask, inv_mask, eval=False))

        mask, inv_mask = self.upsample_mask(mask)
        y = self.last_part(x, mask, inv_mask, eval=False)

        return y