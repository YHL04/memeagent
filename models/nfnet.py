

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


nfnet_params = {
    'FTEST': {
        'width': [256, 512, 768, 768], 'depth': [1, 1, 1, 1], 'drop_rate': 0.2},
    'F0': {
        'width': [256, 512, 1536, 1536], 'depth': [1, 2, 6, 3], 'drop_rate': 0.2},
    'F1': {
        'width': [256, 512, 1536, 1536], 'depth': [2, 4, 12, 6], 'drop_rate': 0.3},
    'F2': {
        'width': [256, 512, 1536, 1536], 'depth': [3, 6, 18, 9], 'drop_rate': 0.4},
    'F3': {
        'width': [256, 512, 1536, 1536], 'depth': [4, 8, 24, 12], 'drop_rate': 0.4},
    'F4': {
        'width': [256, 512, 1536, 1536], 'depth': [5, 10, 30, 15], 'drop_rate': 0.5},
    'F5': {
        'width': [256, 512, 1536, 1536], 'depth': [6, 12, 36, 18], 'drop_rate': 0.5},
    'F6': {
        'width': [256, 512, 1536, 1536], 'depth': [7, 14, 42, 21], 'drop_rate': 0.5},
    'F7': {
        'width': [256, 512, 1536, 1536], 'depth': [8, 16, 48, 24], 'drop_rate': 0.5},
}


class NFNet(nn.Module):
    """
    TODO:
        Review for bugs and understand design choices
    """

    def __init__(self, out_dim, variant="FTEST", alpha=0.2, se_ratio=0.5):
        super(NFNet, self).__init__()

        block_params = nfnet_params[variant]
        dropout_rate = block_params['drop_rate']

        # Create NF blocks
        blocks = []
        expected_std = 1.0
        in_c = block_params['width'][0] // 2

        block_args = zip(
            block_params['width'],
            block_params['depth'],
            [0.5] * 4,
            [128] * 4,
            [1, 2, 2, 2]
        )

        for(block_width, stage_depth, expand_ratio, group_size, stride) in block_args:
            for block_index in range(stage_depth):
                beta = 1. / expected_std
                out_c = block_width

                blocks.append(NFBlock(
                    in_c=in_c,
                    out_c=out_c,
                    stride=stride if block_index == 0 else 1,
                    alpha=alpha,
                    beta=beta,
                    se_ratio=se_ratio,
                    group_size=group_size
                ))

                in_c = out_c

        self.stem = Stem()
        self.body = nn.Sequential(*blocks)
        self.final_conv = WSConv2D(in_c, 2 * in_c, kernel_size=1)

        # Omit dropout in the MEME paper
        self.linear = nn.Linear(2 * in_c, out_dim)
        nn.init.normal_(self.linear.weight, 0, 0.01)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = F.gelu(self.final_conv(x))

        x = torch.mean(x, dim=(2, 3))
        x = self.linear(x)

        return x


class Stem(nn.Module):

    def __init__(self):
        super(Stem, self).__init__()

        self.conv0 = WSConv2D(4, 16, kernel_size=3, stride=2)
        self.conv1 = WSConv2D(16, 32, kernel_size=3, stride=1)
        self.conv2 = WSConv2D(32, 64, kernel_size=3, stride=1)
        self.conv3 = WSConv2D(64, 128, kernel_size=3, stride=2)

    def forward(self, x):
        x = F.gelu(self.conv0(x))
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = self.conv3(x)

        return x


class WSConv2D(nn.Conv2d):
    def __init__(self, in_c, out_c, kernel_size,
                 stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros'):
        super(WSConv2D, self).__init__(in_c, out_c, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        nn.init.xavier_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(out_c, 1, 1, 1))

        self.register_buffer('eps', torch.tensor(1e-4, requires_grad=False), persistent=False)
        self.register_buffer('fan_in', torch.tensor(np.prod(self.weight.shape[1:]), requires_grad=False).type_as(self.weight), persistent=False)

    def standardize_weights(self):
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
        scale = torch.rsqrt(torch.maximum(var * self.fan_in, self.eps))
        return (self.weight - mean) * scale * self.gain

    def forward(self, x):
        return F.conv2d(
            input=x,
            weight=self.standardize_weights(),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


class NFBlock(nn.Module):

    def __init__(self, in_c, out_c,
                 expansion=0.5, se_ratio=0.5,
                 stride=1, beta=1.0, alpha=0.2,
                 group_size=1
                 ):
        super(NFBlock, self).__init__()

        self.in_c = in_c
        self.out_c = out_c
        self.expansion = expansion
        self.se_ratio = se_ratio
        self.beta = beta
        self.alpha = alpha
        self.group_size = group_size

        width = int(out_c * expansion)
        self.groups = width // group_size
        self.width = group_size * self.groups
        self.stride = stride

        self.conv0 = nn.Conv2d(in_c, width, kernel_size=1)
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=self.groups)
        self.conv1b = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, groups=self.groups)
        self.conv2 = nn.Conv2d(width, out_c, kernel_size=1)

        self.use_projection = self.stride > 1 or self.in_c != self.out_c
        if self.use_projection:
            if stride > 1:
                self.shortcut_avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.conv_shortcut = WSConv2D(self.in_c, self.out_c, kernel_size=1)

        self.squeeze_excite = SqueezeExcite(out_c, out_c, se_ratio=se_ratio)
        self.skip_gain = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        x = F.gelu(x) * self.beta

        if self.stride > 1:
            _x = self.shortcut_avg_pool(x)
            _x = self.conv_shortcut(_x)
        elif self.use_projection:
            _x = self.conv_shortcut(x)
        else:
            _x = x

        x = F.gelu(self.conv0(x))
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv1b(x))
        x = self.conv2(x)

        x = (self.squeeze_excite(x) * 2) * x

        if _x.size(-1) != x.size(-1):
            _x = F.pad(_x, (0, 1))

        if _x.size(-2) != x.size(-2):
            _x = F.pad(_x, (0, 0, 0, 1))

        return x * self.alpha * self.skip_gain + _x


class SqueezeExcite(nn.Module):

    def __init__(self, in_c, out_c, se_ratio=0.5):
        super(SqueezeExcite, self).__init__()

        self.in_c = in_c
        self.out_c = out_c
        self.se_ratio = se_ratio

        self.hidden_c = max(1, int(self.in_c * self.se_ratio))

        self.linear1 = nn.Linear(self.in_c, self.hidden_c)
        self.linear2 = nn.Linear(self.hidden_c, self.out_c)

    def forward(self, x):
        _x = x

        x = torch.mean(x, (2, 3))
        x = F.gelu(self.linear1(x))
        x = F.sigmoid(self.linear2(x))

        b, c, _, _ = _x.size()
        return x.view(b, c, 1, 1).expand_as(_x)


if __name__ == "__main__":
    model = NFNet()

