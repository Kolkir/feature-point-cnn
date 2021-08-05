import torch
from torch import nn
from python.netutils import restore_prob_map


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNetBackbone(nn.Module):
    def __init__(self, image_channels=1):
        super(ResNetBackbone, self).__init__()

        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_residual_blocks=2, in_channels=64, intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_residual_blocks=2, in_channels=64, intermediate_channels=128, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def make_layers(self, num_residual_blocks, in_channels,  intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(intermediate_channels))
        layers.append(ResNetBlock(in_channels, intermediate_channels, identity_downsample, stride))

        for i in range(num_residual_blocks - 1):
            layers.append(ResNetBlock(intermediate_channels, intermediate_channels))

        return nn.Sequential(*layers)


class SuperPointBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res_channels, settings):
        super(SuperPointBlock, self).__init__()
        # bias=False - because conv followed by BatchNorm
        self.conv1 = nn.Conv2d(in_channels, out_channels, settings.detdesc_kernel_size_a, settings.detdesc_stride,
                               settings.detdesc_padding_a, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, res_channels, settings.detdesc_kernel_size_b, settings.detdesc_stride,
                               settings.detdesc_padding_b, bias=False)
        self.bn2 = nn.BatchNorm2d(res_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


class SuperPoint(nn.Module):
    def __init__(self, settings):
        super(SuperPoint, self).__init__()
        self.settings = settings
        self.is_descriptor_enabled = True  # used to disable descriptor head when training MagicPoint

        self.encoder = ResNetBackbone()
        # self.detector = SuperPointBlock(*self.settings.detector_dims, self.settings)
        # self.descriptor = SuperPointBlock(*self.settings.descriptor_dims, self.settings)

        self.detector = self.encoder.make_layers(num_residual_blocks=2, in_channels=128, intermediate_channels=65, stride=1)
        self.descriptor = self.encoder.make_layers(num_residual_blocks=2, in_channels=128, intermediate_channels=128, stride=1)

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
