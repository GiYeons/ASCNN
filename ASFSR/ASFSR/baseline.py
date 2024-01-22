from torch import nn
import torch
import math
from torch.nn import functional as F

import warnings
warnings.filterwarnings("ignore")


class Net(nn.Module):
    def __init__(self, scale, im_c=1, fn=16, dfn = 32, r= 4):
        super(Net, self).__init__()
        self.scale = scale


        self.first_part = nn.Conv2d(in_channels=im_c, out_channels=dfn, kernel_size=(5, 5), stride=(1, 1), padding=(5 // 2))

        self.reduction = nn.Conv2d(in_channels=dfn, out_channels=fn, kernel_size=(1, 1), stride=(1, 1))

        self.mid_part1 = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=(3, 3), stride=(1, 1), padding=(3 // 2))
        self.mid_part2 = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=(3, 3), stride=(1, 1), padding=(3 // 2))
        self.mid_part3 = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=(3, 3), stride=(1, 1), padding=(3 // 2))
        self.mid_part4 = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=(3, 3), stride=(1, 1), padding=(3 // 2))

        self.expansion = nn.Conv2d(in_channels=fn, out_channels=dfn, kernel_size=(1, 1), stride=(1, 1))

        self.last_part =nn.ConvTranspose2d(in_channels=dfn, out_channels=im_c, kernel_size=(9, 9), stride=(scale, scale), padding=(9 // 2, 9//2), output_padding=scale - 1)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, inx, th=None, dilker=None, dilation=False, eval=False):   # inx는 input임

        x = self.relu(self.first_part(inx))
        x = self.relu(self.reduction(x))

        x = self.relu(self.mid_part1(x))
        x = self.relu(self.mid_part2(x))
        x = self.relu(self.mid_part3(x))
        x = self.relu(self.mid_part4(x))

        x = self.relu(self.expansion(x))

        y = self.last_part(x)

        return y