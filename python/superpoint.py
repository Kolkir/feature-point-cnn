import torch
from torch import nn, norm, unsqueeze, quantization


def create_encoder_block(in_channels: int, out_channels: int, settings):
    conv_a = nn.Conv2d(in_channels, out_channels, settings.encoder_kernel_size, settings.encoder_stride,
                       settings.encoder_padding)
    conv_b = nn.Conv2d(out_channels, out_channels, settings.encoder_kernel_size, settings.encoder_stride,
                       settings.encoder_padding)
    return conv_a, conv_b


def create_detdesc_block(in_channels: int, out_channels: int, res_channels: int, settings):
    conv_a = nn.Conv2d(in_channels, out_channels, settings.detdesc_kernel_size_a, settings.detdesc_stride,
                       settings.detdesc_padding_a)
    conv_b = nn.Conv2d(out_channels, res_channels, settings.detdesc_kernel_size_b, settings.detdesc_stride,
                       settings.detdesc_padding_b)
    return conv_a, conv_b


class SuperPoint(nn.Module):
    def __init__(self, settings):
        super(SuperPoint, self).__init__()
        self.settings = settings
        self.is_descriptor_enabled = True  # used to disable descriptor head when training MagicPoint

        if self.settings.do_quantization:
            self.quant = quantization.QuantStub()
            self.dequant = quantization.DeQuantStub()

        # self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Create encoder
        self.encoder_conv = nn.ModuleDict()
        for i, dim in enumerate(self.settings.encoder_dims):
            conv_a, conv_b = create_encoder_block(*dim, self.settings)
            self.encoder_conv['encoder_conv{0}_a'.format(i)] = conv_a
            self.encoder_conv['encoder_relu{0}_a'.format(i)] = nn.ReLU(inplace=True)
            self.encoder_conv['encoder_conv{0}_b'.format(i)] = conv_b
            self.encoder_conv['encoder_relu{0}_b'.format(i)] = nn.ReLU(inplace=True)

        # Create detector hea
        self.detector_conv = nn.ModuleDict()
        self.detector_conv['detector_conv_a'], self.detector_conv['detector_conv_b'] = create_detdesc_block(
            *self.settings.detector_dims, self.settings)
        self.detector_conv['detector_relu'] = nn.ReLU(inplace=True)

        # Create descriptor head
        self.descriptor_conv = nn.ModuleDict()
        self.descriptor_conv['descriptor_conv_a'], self.descriptor_conv['descriptor_conv_b'] = create_detdesc_block(
            *self.settings.descriptor_dims, self.settings)
        self.descriptor_conv['descriptor_relu'] = nn.ReLU(inplace=True)

    def disable_descriptor(self):
        self.is_descriptor_enabled = False

    def enable_descriptor(self):
        self.is_descriptor_enabled = True

    def detector_parameters(self):
        encoder_params = list(self.encoder_conv.parameters(recurse=True))
        detector_params = list(self.detector_conv.parameters(recurse=True))
        return encoder_params + detector_params

    def forward(self, x):
        """ Forward pass
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          point: Output point pytorch tensor shaped N x d1 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # this function can be called for warped image during MagicPoint training
        # so should it should be disabled
        if len(x.shape) <= 2:
            return torch.empty((1,)), torch.empty((1,))

        if self.settings.cuda:
            x = x.cuda()

        if self.settings.do_quantization:
            x = self.quant(x)
        x = self.encoder_forward_pass(x)
        point = self.detdesc_forward_pass(x, self.detector_conv, 'detector')
        desc = self.descriptor_forward_pass(point, x)

        if self.settings.do_quantization:
            point = self.dequant(point)
            desc = self.dequant(desc)

        dn = norm(desc, p=2, dim=1)
        desc = desc.div(unsqueeze(dn, 1))  # normalize
        return point, desc

    def descriptor_forward_pass(self, point, x):
        if self.is_descriptor_enabled:
            desc = self.detdesc_forward_pass(x, self.descriptor_conv, 'descriptor')
        else:
            shape = point.shape
            desc = torch.zeros((shape[0], 256, shape[2], shape[3]))
            if self.settings.cuda:
                desc = desc.cuda()
        return desc

    def detdesc_forward_pass(self, x, head, prefix):
        relu = head[prefix + '_relu']
        res_a = relu(head[prefix + '_conv_a'](x))
        out = head[prefix + '_conv_b'](res_a)
        return out

    def encoder_forward_pass(self, x):
        last_step = len(self.settings.encoder_dims) - 1
        for i, dim in enumerate(self.settings.encoder_dims):
            relu_a = self.encoder_conv['encoder_relu{0}_a'.format(i)]
            x = relu_a(self.encoder_conv['encoder_conv{0}_a'.format(i)](x))
            relu_b = self.encoder_conv['encoder_relu{0}_b'.format(i)]
            x = relu_b(self.encoder_conv['encoder_conv{0}_b'.format(i)](x))
            if i != last_step:
                x = self.pool(x)
        return x
