from superpoint import SuperPoint
import torch
import torchsummary
import numpy as np
from nms import corners_nms
from weightsloader import load_weights_legacy
import torch.quantization


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

    def get_points(self, pointness_map, img_h, img_w):
        pointness_map = pointness_map.data.cpu().numpy().squeeze()
        softmax_result = np.exp(pointness_map)
        softmax_result = softmax_result / (np.sum(softmax_result, axis=0) + .00001)
        # removing dustbin dimension
        no_dustbin = softmax_result[:-1, :, :]
        # reshape to get full resolution
        no_dustbin = no_dustbin.transpose(1, 2, 0)
        img_h_cells = int(img_h / self.settings.cell)
        img_w_cells = int(img_w / self.settings.cell)
        confidence_map = np.reshape(no_dustbin, [img_h_cells, img_w_cells, self.settings.cell, self.settings.cell])
        confidence_map = np.transpose(confidence_map, [0, 2, 1, 3])
        confidence_map = np.reshape(confidence_map,
                                    [img_h_cells * self.settings.cell, img_w_cells * self.settings.cell])
        # threshold confidence level
        xs, ys = np.where(confidence_map >= self.settings.confidence_thresh)
        # if we didn't find any features
        if len(xs) == 0:
            return np.zeros((3, 0))

        # get points coordinates
        points = np.zeros((3, len(xs)))
        points[0, :] = ys
        points[1, :] = xs
        points[2, :] = confidence_map[xs, ys]
        # NMS
        points = corners_nms(points, img_h, img_w, dist_thresh=self.settings.nms_dist)
        # sort by confidence(why do we need this? nms returns sorted values)
        indices = np.argsort(points[2, :])
        points = points[:, indices[::-1]]
        # remove points along border.
        border_width = self.settings.border_remove
        horizontal_remove_idx = np.logical_or(points[0, :] < border_width, points[0, :] >= (img_w - border_width))
        vertical_remove_idx = np.logical_or(points[1, :] < border_width, points[1, :] >= (img_h - border_width))
        total_remove_idx = np.logical_or(horizontal_remove_idx, vertical_remove_idx)
        points = points[:, ~total_remove_idx]
        return points

    def get_descriptors(self, points, descriptors_map, img_h, img_w):
        if points.shape[1] == 0:
            return np.zeros((descriptors_map.shape[1], 0))
        # interpolate into descriptor map using 2D point locations
        sample_points = torch.from_numpy(points[:2, :].copy())
        sample_points[0, :] = (sample_points[0, :] / (float(img_w) / 2.)) - 1.
        sample_points[1, :] = (sample_points[1, :] / (float(img_h) / 2.)) - 1.
        sample_points = sample_points.transpose(0, 1).contiguous()
        sample_points = sample_points.view(1, 1, -1, 2)
        sample_points = sample_points.float()
        if self.settings.cuda:
            sample_points = sample_points.cuda()
        desc = torch.nn.functional.grid_sample(descriptors_map, sample_points, align_corners=True)
        desc = desc.data.cpu().numpy().reshape(descriptors_map.shape[1], -1)
        desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return desc

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

        points = self.get_points(pointness_map, img_h, img_w)
        descriptors = self.get_descriptors(points, descriptors_map, img_h, img_w)

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
