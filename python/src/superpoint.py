import torch
from torch import nn

from src.attention_blocks import make_axial_attention_layer
from src.netutils import restore_prob_map
from src.resnet_blocks import make_resnet_layers


class Encoder(nn.Module):
    def __init__(self, cuda, image_channels=1):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = make_resnet_layers(num_residual_blocks=2, in_channels=64, intermediate_channels=64, stride=1)
        self.layer2 = make_resnet_layers(num_residual_blocks=2, in_channels=64, intermediate_channels=128, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.layer = make_resnet_layers(num_residual_blocks=2, in_channels=128, intermediate_channels=65, stride=1)

    def forward(self, x):
        out = self.layer(x)
        return out, x


class Descriptor(nn.Module):
    def __init__(self, cuda, use_attention=False):
        super(Descriptor, self).__init__()

        if use_attention:
            kernel_size = [30, 40]  # depends on image size 240/4 320/4
            self.layer = make_axial_attention_layer(128, 256, 2, kernel_size, stride=1, cuda=cuda)
            self.out_layer = make_resnet_layers(num_residual_blocks=2, in_channels=256, intermediate_channels=128,
                                                stride=1)
        else:
            self.layer = make_resnet_layers(num_residual_blocks=1, in_channels=128, intermediate_channels=256,
                                            stride=1)
            self.out_layer = nn.Sequential(
                nn.Dropout2d(0.25),
                nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            )

    def forward(self, image, feature_embeddings):
        out = self.layer(image)
        out = self.out_layer(out)
        return out


class SuperPoint(nn.Module):
    def __init__(self, settings):
        super(SuperPoint, self).__init__()
        self.settings = settings
        self.is_descriptor_enabled = True  # used to disable descriptor head when training MagicPoint

        self.encoder = Encoder(settings.cuda)
        self.detector = Detector()
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
        prob, embeddings = self.detector(image)
        if self.is_descriptor_enabled:
            desc = self.descriptor(image, embeddings)
        else:
            shape = prob.shape
            desc = torch.zeros((shape[0], 128, shape[2], shape[3]))
            if self.settings.cuda:
                desc = desc.cuda()

        softmax_result = torch.exp(prob)
        softmax_result = softmax_result / (torch.sum(softmax_result, dim=1, keepdim=True) + .00001)

        prob_map = restore_prob_map(softmax_result, img_h, img_w, self.settings.cell)
        return prob_map, desc, prob
