import math
import torch
from torch import nn
import torch.nn.functional as nn_func


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=4, stride=1, bias=False, width=False, cuda=True):
        # assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = nn.Conv1d(in_planes, out_planes * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        device = 'cpu'
        if cuda:
            device = 'cuda'
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1, device=device),
                                     requires_grad=True)
        query_index = torch.arange(kernel_size, device=device).unsqueeze(0)
        key_index = torch.arange(kernel_size, device=device).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)  # B, H, C, W
        else:
            x = x.permute(0, 3, 1, 2)  # B, W, C, H
        n, w, c, h = x.shape
        x = x.contiguous().view(n * w, c, h)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(n * w, self.groups, self.group_planes * 2, h),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(n * w, 3, self.groups, h, h).sum(dim=1)
        # (N, groups, H, H, W)
        similarity = nn_func.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(n * w, self.out_planes * 2, h)
        output = self.bn_output(stacked_output).view(n, w, self.out_planes, 2, h).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride=1, downsample=None, groups=1, cuda=True):
        super(AxialBlock, self).__init__()
        assert (len(kernel_size) == 2)
        self.conv_down = conv1x1(inplanes, outplanes)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.height_block = AxialAttention(outplanes, outplanes, kernel_size[0], groups=groups, cuda=cuda)
        self.width_block = AxialAttention(outplanes, outplanes, kernel_size[1], groups=groups, stride=stride,
                                          width=True, cuda=cuda)
        self.conv_up = conv1x1(outplanes, outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.height_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def make_axial_attention_layer(inplanes, outplanes, blocks, kernel_size, stride=1, groups=8, cuda=True):
    downsample = None
    if stride != 1 or inplanes != outplanes:
        downsample = nn.Sequential(
            conv1x1(inplanes, outplanes),
            nn.BatchNorm2d(outplanes),
        )

    layers = [AxialBlock(inplanes, outplanes, kernel_size, stride, downsample, groups=groups, cuda=cuda)]
    inplanes = outplanes
    if stride != 1:
        kernel_size[0] = kernel_size[0] // 2
        kernel_size[1] = kernel_size[1] // 2

    for _ in range(1, blocks):
        layers.append(AxialBlock(inplanes, outplanes, kernel_size, groups=groups, cuda=cuda))

    return nn.Sequential(*layers)
