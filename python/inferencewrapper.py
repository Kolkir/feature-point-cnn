from saveutils import load_checkpoint_for_inference
from superpoint import SuperPoint
import torch
import torchsummary

import torch.quantization
import numpy as np
from netutils import get_points, get_descriptors


class InferenceWrapper(object):
    def __init__(self, weights_path, settings):

        self.name = 'SuperPoint'
        self.settings = settings

        self.net = SuperPoint(self.settings)
        load_checkpoint_for_inference(weights_path, self.net)

        if settings.cuda:
            self.net = self.net.cuda()
            print('Model moved to GPU')

        if settings.do_quantization:
            # x86
            # self.net.qconfig = torch.quantization.get_default_qconfig('fbgemm')

            # If you want to deploy in ARM On
            self.net.qconfig = torch.quantization.get_default_qconfig('qnnpack')

            model_fp32_fused = torch.quantization.fuse_modules(self.net, [
                ['encoder_conv.encoder_conv0_a', 'encoder_conv.encoder_relu0_a'],
                ['encoder_conv.encoder_conv0_b', 'encoder_conv.encoder_relu0_b'],
                ['encoder_conv.encoder_conv1_a', 'encoder_conv.encoder_relu1_a'],
                ['encoder_conv.encoder_conv1_b', 'encoder_conv.encoder_relu1_b'],
                ['encoder_conv.encoder_conv2_a', 'encoder_conv.encoder_relu2_a'],
                ['encoder_conv.encoder_conv2_b', 'encoder_conv.encoder_relu2_b'],
                ['encoder_conv.encoder_conv3_a', 'encoder_conv.encoder_relu3_a'],
                ['encoder_conv.encoder_conv3_b', 'encoder_conv.encoder_relu3_b'],
                ['descriptor_conv.descriptor_conv_a', 'descriptor_conv.descriptor_relu'],
                ['detector_conv.detector_conv_a', 'detector_conv.detector_relu']
                ])
            model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)
            self.net = torch.quantization.convert(model_fp32_prepared)
        else:
            torchsummary.summary(self.net, (1, 240, 320), device='cuda' if settings.cuda else 'cpu')

    def run(self, img):
        """ Process a image to extract points and descriptors.
        Input
          img - HxW float32 input image in range [0,1].
        Output
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          """
        input_tensor = self.prepare_input(img)
        img_h, img_w = img.shape[1], img.shape[2]

        outs = self.net.forward(input_tensor)
        pointness_map, descriptors_map = outs[0], outs[1]

        points = get_points(pointness_map, img_h, img_w, self.settings)
        descriptors = get_descriptors(points, descriptors_map, img_h, img_w, self.settings)

        return points, descriptors

    def prepare_input(self, img):
        if not torch.is_tensor(img):
            assert img.ndim == 2, 'Image must be grayscale.'
            assert img.dtype == np.float32, 'Image must be float32.'
            img_h, img_w = img.shape[0], img.shape[1]
            input_tensor = img.copy()
            input_tensor = input_tensor.reshape(1, img_h, img_w)
            input_tensor = torch.from_numpy(input_tensor)
        else:
            input_tensor = img
            img_h, img_w = img.shape[1], img.shape[2]
        input_tensor = torch.autograd.Variable(input_tensor).view(1, 1, img_h, img_w)
        if self.settings.cuda:
            input_tensor = input_tensor.cuda()
        return input_tensor

    def trace(self, img, out_file_name):
        # trace script
        input_tensor = self.prepare_input(img)
        traced_net = torch.jit.trace(self.net, input_tensor)
        traced_net.save(out_file_name + "_script.pt")

        # just weights for cpp
        state_dict = {('.'.join(k.split('.')[1:])): v for k, v in self.net.state_dict().items()}
        torch.save(state_dict, out_file_name + "_params.pt", _use_new_zipfile_serialization=True)
