from superpoint import SuperPoint
import torch
import torchsummary
from weightsloader import load_weights_legacy
import torch.quantization
from netutils import get_points, get_descriptors


class InferenceWrapper(object):
    def __init__(self, weights_path, settings):

        self.name = 'SuperPoint'
        self.settings = settings

        self.net = SuperPoint(self.settings)
        miss_keys, _ = self.net.load_state_dict(load_weights_legacy(weights_path))
        if miss_keys:
            print('Can not load network some keys are missing:')
            print(miss_keys)
            exit(-1)

        # model must be set to eval mode for static quantization logic to work
        self.net.eval()

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
            torchsummary.summary(self.net, (1, 640, 480), device='cpu')

        if settings.cuda:
            self.net = self.net.cuda()
            print('Model moved to GPU')

    def run(self, img):
        """ Process a numpy image to extract points and descriptors.
        Input
          img - HxW numpy float32 input image in range [0,1].
        Output
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          """
        img_h, img_w = img.shape[0], img.shape[1]
        input_tensor = self.prepare_input(img)
        outs = self.net.forward(input_tensor)
        pointness_map, descriptors_map = outs[0], outs[1]

        points = get_points(pointness_map, img_h, img_w, self.settings)
        descriptors = get_descriptors(points, descriptors_map, img_h, img_w, self.settings)

        return points, descriptors

    def prepare_input(self, img):
        assert img.ndim == 2, 'Image must be grayscale.'
        assert img.dtype == np.float32, 'Image must be float32.'
        input_tensor = img.copy()
        img_h, img_w = img.shape[0], img.shape[1]
        input_tensor = input_tensor.reshape(1, img_h, img_w)
        input_tensor = torch.from_numpy(input_tensor)
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
