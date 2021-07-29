from collections import OrderedDict

import torch
from torch import nn, norm, unsqueeze, quantization
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from python.netutils import restore_prob_map


def create_encoder_block(in_channels: int, out_channels: int, settings):
    # bias=False - because conv followed by BatchNorm
    conv_a = nn.Conv2d(in_channels, out_channels, settings.encoder_kernel_size, settings.encoder_stride,
                       settings.encoder_padding, bias=False)
    batch_norm_a = nn.BatchNorm2d(out_channels)
    conv_b = nn.Conv2d(out_channels, out_channels, settings.encoder_kernel_size, settings.encoder_stride,
                       settings.encoder_padding, bias=False)
    batch_norm_b = nn.BatchNorm2d(out_channels)
    return conv_a, batch_norm_a, conv_b, batch_norm_b


def create_detdesc_block(in_channels: int, out_channels: int, res_channels: int, settings):
    # bias=False - because conv followed by BatchNorm
    conv_a = nn.Conv2d(in_channels, out_channels, settings.detdesc_kernel_size_a, settings.detdesc_stride,
                       settings.detdesc_padding_a, bias=False)
    batch_norm_a = nn.BatchNorm2d(out_channels)
    conv_b = nn.Conv2d(out_channels, res_channels, settings.detdesc_kernel_size_b, settings.detdesc_stride,
                       settings.detdesc_padding_b, bias=False)
    batch_norm_b = nn.BatchNorm2d(res_channels)
    return conv_a, batch_norm_a, conv_b, batch_norm_b


class SuperPoint(nn.Module):
    def __init__(self, settings):
        super(SuperPoint, self).__init__()
        self.settings = settings
        self.is_descriptor_enabled = True  # used to disable descriptor head when training MagicPoint
        self.grad_checkpointing = True

        if self.settings.do_quantization:
            self.quant = quantization.QuantStub()
            self.dequant = quantization.DeQuantStub()

        # Create encoder
        self.encoder_dict1 = OrderedDict()
        self.encoder_dict2 = OrderedDict()

        def fill_encoder_block(module_dict, index, dimensions, add_pool):
            conv_a, bn_a, conv_b, bn_b = create_encoder_block(*dimensions, self.settings)
            module_dict['encoder_conv{0}_a'.format(index)] = conv_a
            module_dict['encoder_relu{0}_a'.format(index)] = nn.ReLU(inplace=True)
            module_dict['encoder_bn{0}_a'.format(index)] = bn_a
            module_dict['encoder_conv{0}_b'.format(index)] = conv_b
            module_dict['encoder_relu{0}_b'.format(index)] = nn.ReLU(inplace=True)
            module_dict['encoder_bn{0}_b'.format(index)] = bn_b
            if add_pool:
                module_dict['encoder_pool{0}'.format(index)] = nn.MaxPool2d(kernel_size=2, stride=2)

        fill_encoder_block(self.encoder_dict1, 0, self.settings.encoder_dims[0], add_pool=True)
        last_step = len(self.settings.encoder_dims) - 1
        for i, dim in enumerate(self.settings.encoder_dims):
            if i != 0:
                fill_encoder_block(self.encoder_dict2, i, dim, add_pool=(i != last_step))

        self.encoder1 = nn.Sequential(self.encoder_dict1)
        self.encoder2 = nn.Sequential(self.encoder_dict2)

        # Create detector head
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
        if self.grad_checkpointing:
            prob = grad_checkpoint(self.detector_forward_pass, image)
            desc = grad_checkpoint(self.descriptor_forward_pass, prob, image)
        else:
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

    def detector_forward_pass(self, image):
        prob = self.detdesc_forward_pass(image, self.detector_conv, 'detector')
        return prob

    def descriptor_forward_pass(self, prob, x):
        if self.is_descriptor_enabled:
            desc = self.detdesc_forward_pass(x, self.descriptor_conv, 'descriptor')
        else:
            shape = prob.shape
            desc = torch.zeros((shape[0], 256, shape[2], shape[3]))
            if self.settings.cuda:
                desc = desc.cuda()
        return desc

    def detdesc_forward_pass(self, x, head, prefix):
        x = head[prefix + '_conv_a'](x)
        x = head[prefix + '_relu'](x)
        x = head[prefix + '_bn_a'](x)

        x = head[prefix + '_conv_b'](x)
        x = head[prefix + '_bn_b'](x)
        return x

    def encoder_forward_pass(self, x):
        x = self.encoder1(x)
        if self.grad_checkpointing:
            x = grad_checkpoint(self.encoder2, x)
        else:
            x = self.encoder2(x)
        return x
