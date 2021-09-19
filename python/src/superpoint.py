import math

import torch
from torch import nn
import torch.nn.functional as F

from src.netutils import restore_prob_map


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=8, stride=1, bias=False, width=False, cuda=True):
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
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1, device=device), requires_grad=True)
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
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
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
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

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
        self.hight_block = AxialAttention(inplanes, inplanes, kernel_size[0], groups=groups, cuda=cuda)
        self.width_block = AxialAttention(inplanes, inplanes, kernel_size[1], groups=groups, stride=stride, width=True,
                                          cuda=cuda)
        self.conv_up = conv1x1(inplanes, outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.hight_block(x)
        out = self.width_block(out)

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
            conv1x1(inplanes, outplanes, stride),
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


class Encoder(nn.Module):
    def __init__(self, image_channels=1):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class Detector(nn.Module):
    def __init__(self, cuda):
        super(Detector, self).__init__()
        kernel_size = [60, 80]  # depends on image size 240/4 320/4

        self.layer1 = make_axial_attention_layer(64, 128, 1, kernel_size, cuda=cuda)
        self.layer2 = make_axial_attention_layer(128, 256, 2, kernel_size, stride=2, cuda=cuda)

        # like a ResNet block
        self.conv1 = nn.Conv2d(256, 65, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(65)
        self.conv2 = nn.Conv2d(65, 65, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(65)
        self.relu = nn.ReLU()

        self.identity_downsample = nn.Sequential(
            nn.Conv2d(64, 65, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(65))

    def forward(self, x):
        identity = self.identity_downsample(x)
        out = self.layer1(x)
        out = self.layer2(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class Descriptor(nn.Module):
    def __init__(self, cuda):
        super(Descriptor, self).__init__()
        kernel_size = [60, 80]  # depends on image size 240/4 320/4

        self.layer1 = make_axial_attention_layer(64, 128, 1, kernel_size, cuda=cuda)
        self.layer2 = make_axial_attention_layer(128, 256, 2, kernel_size, stride=2, cuda=cuda)
        self.out_layer = make_axial_attention_layer(256, 128, 2, kernel_size, stride=1, cuda=cuda)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.out_layer(x)
        return x


class SuperPoint(nn.Module):
    def __init__(self, settings):
        super(SuperPoint, self).__init__()
        self.settings = settings
        self.is_descriptor_enabled = True  # used to disable descriptor head when training MagicPoint

        self.encoder = Encoder()
        self.detector = Detector(settings.cuda)
        self.descriptor = Descriptor(settings.cuda)

    def disable_descriptor(self):
        self.is_descriptor_enabled = False
        params = self.descriptor.parameters(recurse=True)
        for param in params:
            param.requires_grad = False

    def enable_descriptor(self):
        self.is_descriptor_enabled = True
        params = self.descriptor.parameters(recurse=True)
        for param in params:
            param.requires_grad = True

    def initialize_descriptor(self):
        for layer in self.descriptor.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, image):
        # this function can be called for warped image during MagicPoint training
        # so should it should be disabled
        if len(image.shape) <= 2:
            return torch.empty((1,)), torch.empty((1,)), torch.empty((1,))

        img_h, img_w = image.shape[-2:]  # get dims before image will be changed
        if self.settings.cuda:
            image = image.cuda()

        image = self.encoder(image)
        prob = self.detector(image)
        if self.is_descriptor_enabled:
            desc = self.descriptor(image)
        else:
            shape = prob.shape
            desc = torch.zeros((shape[0], 256, shape[2], shape[3]))
            if self.settings.cuda:
                desc = desc.cuda()

        softmax_result = torch.exp(prob)
        softmax_result = softmax_result / (torch.sum(softmax_result, dim=1, keepdim=True) + .00001)

        prob_map = restore_prob_map(softmax_result, img_h, img_w, self.settings.cell)
        return prob_map, desc, prob
