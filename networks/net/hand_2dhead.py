import torch
from torch import nn
import torch.nn.functional as F

class hand_2dhead(nn.Module):
    def __init__(
            self,
            joint_nb=21,
            stacks=2,
            blocks=1,
            in_channels=256
    ):
        super(hand_2dhead, self).__init__()

        # hand head
        self.joint_nb = joint_nb
        self.channels = in_channels
        self.blocks = blocks
        self.stacks = stacks
        block = Bottleneck
        self.features = self.channels // block.expansion

        hg, res1, res2, fc1, fc1_,fc2, fc2_ = [], [], [], [], [], [], []
        hm, hm_ = [],[]
        seg, seg_ = [], []
        for i in range(self.stacks):
            hg.append(Hourglass(block, self.blocks, self.features, 4))
            res1.append(self.make_residual(block, self.channels, self.features, self.blocks))
            res2.append(self.make_residual(block, self.channels, self.features, self.blocks))
            fc1.append((self.basic_block(self.channels, self.channels)))
            fc2.append((self.basic_block(self.channels, self.channels)))
            hm.append(nn.Conv2d(self.channels, self.joint_nb, kernel_size=1, bias=True))
            seg.append(nn.Conv2d(self.channels, 1, kernel_size=1, bias=True))
            if i < self.stacks - 1:
                fc1_.append(nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=True))
                fc2_.append(nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=True))
                hm_.append(nn.Conv2d(self.joint_nb, self.channels, kernel_size=1, bias=True))
                seg_.append(nn.Conv2d(1, self.channels, kernel_size=1, bias=True))

        self.hg = nn.ModuleList(hg)
        self.res1 = nn.ModuleList(res1)
        self.res2 = nn.ModuleList(res2)
        self.fc1 = nn.ModuleList(fc1)
        self.fc1_ = nn.ModuleList(fc1_)
        self.fc2 = nn.ModuleList(fc2)
        self.fc2_ = nn.ModuleList(fc2_)
        self.hm = nn.ModuleList(hm)
        self.hm_ = nn.ModuleList(hm_)
        self.seg = nn.ModuleList(seg)
        self.seg_ = nn.ModuleList(seg_)

    def basic_block(self, in_planes, out_planes):
        bn = nn.BatchNorm2d(in_planes)
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        return nn.Sequential(conv, bn, nn.LeakyReLU(inplace=True))



    def make_residual(self, block, inplanes, planes, blocks, stride=1):
        skip = None
        if stride != 1 or inplanes != planes * block.expansion:
            skip = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=True))
        layers = []
        layers.append(block(inplanes, planes, stride, skip))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)


    def forward(self, x):
        hm, seg, encoding = [], [], []
        for i in range(self.stacks):
            y1, y2, _ = self.hg[i](x)
            y1 = self.res1[i](y1)
            y1 = self.fc1[i](y1)
            pred_hm = self.hm[i](y1)
            hm.append(pred_hm)

            y2 = self.res2[i](y2)
            y2 = self.fc2[i](y2)
            pred_seg = self.seg[i](y2)
            seg.append(pred_seg)


            if i < self.stacks - 1:
                fc1_ = self.fc1_[i](y1)
                hm_ = self.hm_[i](pred_hm)
                fc2_ = self.fc2_[i](y2)
                seg_ = self.seg_[i](pred_seg)
                x = x + fc1_ + fc2_ + hm_ + seg_
                encoding.append(x)
            else:
                encoding.append(x + y1 + y2)
        return hm, seg, encoding


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, skip=None, groups=1):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True, groups=groups)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True, groups=groups)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01,inplace=True)  # negative_slope=0.01
        self.skip = skip
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.leakyrelu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.leakyrelu(out)
        out = self.conv3(out)

        if self.skip is not None:
            residual = self.skip(x)

        out += residual

        return out

class Hourglass(nn.Module):
    def __init__(
            self,
            block,
            num_blocks,
            planes,
            depth=4
    ):

        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            # channel changes: planes*block.expansion->planes->2*planes
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res_ = []
                if j == 1:
                    res_.append(self._make_residual(block, num_blocks, planes))
                else:
                    res_.append(self._make_residual(block, num_blocks, planes))
                    res_.append(self._make_residual(block, num_blocks, planes))

                res.append(nn.ModuleList(res_))
            if i == 0:
                res_ = []
                res_.append(self._make_residual(block, num_blocks, planes))
                res_.append(self._make_residual(block, num_blocks, planes))
                res.append(nn.ModuleList(res_))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up11 = self.hg[n - 1][0][0](x)  # skip branches
        up12 = self.hg[n - 1][0][1](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1][0](low1)

        if n > 1:
            low21,low22,temp = self._hour_glass_forward(n - 1, low1)
        else:
            temp = low1
            low21 = self.hg[n - 1][3][0](low1)
            low22 = self.hg[n - 1][3][1](low1)
        low31 = self.hg[n - 1][2][0](low21)
        low32 = self.hg[n - 1][2][1](low22)

        up21 = F.interpolate(low31, scale_factor=2)
        up22 = F.interpolate(low32, scale_factor=2)
        out1 = up11 + up21
        out2 = up12 + up22
        return out1, out2, temp

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


if __name__ == "__main__":
    from termcolor import cprint
    from utils.misc import param_count
    feat = torch.randn(1, 256, 64, 64).float()
    handhead = hand_2dhead()
    cprint('params hand_2dhead: {:.3f}M'.format(
            param_count(handhead)), 'blue')
    print(handhead.hg)
    """
    pred_hm ([1, 21, 64, 64])
    pred_seg ([1, 1, 64, 64])
    pred_encoding ([1, 256, 64, 64])
    """


