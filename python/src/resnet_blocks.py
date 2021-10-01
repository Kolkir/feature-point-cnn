from torch import nn


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


def make_resnet_layers(num_residual_blocks, in_channels, intermediate_channels, stride):
    layers = []

    identity_downsample = nn.Sequential(
        nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(intermediate_channels))
    layers.append(ResNetBlock(in_channels, intermediate_channels, identity_downsample, stride))

    for i in range(num_residual_blocks - 1):
        layers.append(ResNetBlock(intermediate_channels, intermediate_channels))

    return nn.Sequential(*layers)