import torch
from torch import nn, norm, unsqueeze, quantization

from python.netutils import restore_prob_map


def create_encoder_block(in_channels: int, out_channels: int, settings):
    conv_a = nn.Conv2d(in_channels, out_channels, settings.encoder_kernel_size, settings.encoder_stride,
                       settings.encoder_padding)
    batch_norm_a = nn.BatchNorm2d(out_channels)
    conv_b = nn.Conv2d(out_channels, out_channels, settings.encoder_kernel_size, settings.encoder_stride,
                       settings.encoder_padding)
    batch_norm_b = nn.BatchNorm2d(out_channels)
    return conv_a, batch_norm_a, conv_b, batch_norm_b


def create_detdesc_block(in_channels: int, out_channels: int, res_channels: int, settings):
    conv_a = nn.Conv2d(in_channels, out_channels, settings.detdesc_kernel_size_a, settings.detdesc_stride,
                       settings.detdesc_padding_a)
    batch_norm_a = nn.BatchNorm2d(out_channels)
    conv_b = nn.Conv2d(out_channels, res_channels, settings.detdesc_kernel_size_b, settings.detdesc_stride,
                       settings.detdesc_padding_b)
    batch_norm_b = nn.BatchNorm2d(res_channels)
    return conv_a, batch_norm_a, conv_b, batch_norm_b


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
            conv_a, bn_a, conv_b, bn_b = create_encoder_block(*dim, self.settings)
            self.encoder_conv['encoder_conv{0}_a'.format(i)] = conv_a
            self.encoder_conv['encoder_bn{0}_a'.format(i)] = bn_a
            self.encoder_conv['encoder_relu{0}_a'.format(i)] = nn.ReLU(inplace=True)
            self.encoder_conv['encoder_conv{0}_b'.format(i)] = conv_b
            self.encoder_conv['encoder_bn{0}_b'.format(i)] = bn_b
            self.encoder_conv['encoder_relu{0}_b'.format(i)] = nn.ReLU(inplace=True)

        # Create detector hea
        self.detector_conv = nn.ModuleDict()
        self.detector_conv['detector_conv_a'], self.detector_conv['detector_bn_a'], self.detector_conv[
            'detector_conv_b'], self.detector_conv['detector_bn_b'] = create_detdesc_block(
            *self.settings.detector_dims, self.settings)
        self.detector_conv['detector_relu'] = nn.ReLU(inplace=True)

        # Create descriptor head
        self.descriptor_conv = nn.ModuleDict()
        self.descriptor_conv['descriptor_conv_a'], self.descriptor_conv['descriptor_bn_a'], self.descriptor_conv[
            'descriptor_conv_b'], self.descriptor_conv['descriptor_bn_b'] = create_detdesc_block(
            *self.settings.descriptor_dims, self.settings)
        self.descriptor_conv['descriptor_relu'] = nn.ReLU(inplace=True)

    def disable_descriptor(self):
        self.is_descriptor_enabled = False
        params = self.descriptor_conv.parameters(recurse=True)
        for param in params:
            param.requires_grad = False

    def enable_descriptor(self):
        self.is_descriptor_enabled = True
        params = self.descriptor_conv.parameters(recurse=True)
        for param in params:
            param.requires_grad = True

    def initialize_descriptor(self):
        for layer in self.descriptor_conv.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, image):
        """ Forward pass
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          point: Output point pytorch tensor shaped N x 1 x H x W.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # this function can be called for warped image during MagicPoint training
        # so should it should be disabled
        if len(image.shape) <= 2:
            return torch.empty((1,)), torch.empty((1,)), torch.empty((1,))

        img_h, img_w = image.shape[-2:]
        if self.settings.cuda:
            image = image.cuda()

        if self.settings.do_quantization:
            image = self.quant(image)

        image = self.encoder_forward_pass(image)
        prob = self.detdesc_forward_pass(image, self.detector_conv, 'detector')
        desc = self.descriptor_forward_pass(prob, image)

        if self.settings.do_quantization:
            prob = self.dequant(prob)
            desc = self.dequant(desc)

        dn = norm(desc, p=2, dim=1)
        desc = desc.div(unsqueeze(dn, 1))  # normalize

        softmax_result = torch.exp(prob)
        softmax_result = softmax_result / (torch.sum(softmax_result, dim=1, keepdim=True) + .00001)

        prob_map = restore_prob_map(softmax_result, img_h, img_w, self.settings.cell)
        return prob_map, desc, prob

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

        x = head[prefix + '_conv_a'](x)
        x = head[prefix + '_bn_a'](x)
        x = relu(x)

        x = head[prefix + '_conv_b'](x)
        x = head[prefix + '_bn_b'](x)
        return x

    def encoder_forward_pass(self, x):
        last_step = len(self.settings.encoder_dims) - 1
        for i, dim in enumerate(self.settings.encoder_dims):

            x = self.encoder_conv['encoder_conv{0}_a'.format(i)](x)
            x = self.encoder_conv['encoder_bn{0}_a'.format(i)](x)
            x = self.encoder_conv['encoder_relu{0}_a'.format(i)](x)

            x = self.encoder_conv['encoder_conv{0}_b'.format(i)](x)
            x = self.encoder_conv['encoder_bn{0}_b'.format(i)](x)
            x = self.encoder_conv['encoder_relu{0}_b'.format(i)](x)

            if i != last_step:
                x = self.pool(x)
        return x
