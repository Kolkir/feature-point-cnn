from src.homographies import homography_adaptation
from src.saveutils import load_checkpoint_for_inference
from src.superpoint import SuperPoint
import torch
import torchsummary

import torch.quantization
import numpy as np
from src.netutils import get_points, get_descriptors


class InferenceWrapper(object):
    def __init__(self, weights_path, settings):

        self.name = 'SuperPoint'
        self.settings = settings

        self.net = SuperPoint(self.settings)
        if not load_checkpoint_for_inference(weights_path, self.net):
            exit(1)

        if settings.cuda:
            self.net = self.net.cuda()
            print('Model moved to GPU')

        self.net.eval()
        # torchsummary.summary(self.net, (1, 240, 320), device='cuda' if settings.cuda else 'cpu')

    def run(self, img):
        with torch.no_grad():
            """ Process a image to extract points and descriptors.
            Input
              img - HxW float32 input image in range [0,1].
            Output
              corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
              desc - 256xN numpy array of corresponding unit normalized descriptors.
              """
            input_tensor = self.prepare_input(img)
            img_h, img_w = input_tensor.shape[2], input_tensor.shape[3]

            point_prob_map, descriptors_map, _ = self.net(input_tensor)

            points = get_points(point_prob_map, img_h, img_w, self.settings)
            descriptors = get_descriptors(points, descriptors_map, img_h, img_w, self.settings)

            return points, descriptors

    def run_with_homography_adaptation(self, img, config):
        with torch.no_grad():
            """ Process a image to extract points and descriptors.
            Input
              img - NxCxHxW float32 input image in range [0,1].
            Output
              corners - Nx3xN numpy array with corners [x_i, y_i, confidence_i]^T.
              """
            input_tensor = self.prepare_input(img)
            img_h, img_w = input_tensor.shape[2], input_tensor.shape[3]

            batch_prob_map = homography_adaptation(input_tensor, self.net, config)

            prob_maps = torch.unbind(batch_prob_map)

            points_list = []
            for prob_map in prob_maps:
                points = get_points(prob_map.unsqueeze(0), img_h, img_w, self.settings)
                points_list.append(points)

            return points_list

    def prepare_input(self, img):
        if not torch.is_tensor(img):
            assert img.ndim == 3
            assert img.dtype == np.float32, 'Image must be float32.'
            assert img.shape[2] == 3, 'Image must be rgb.'
            input_tensor = img.copy()
            input_tensor = input_tensor.transpose((2, 0, 1))
            input_tensor = torch.from_numpy(input_tensor)
            input_tensor = input_tensor.unsqueeze(0)
        else:
            input_tensor = img
        return input_tensor

    def trace(self, img, out_file_name):
        # trace script
        input_tensor = self.prepare_input(img)
        traced_net = torch.jit.trace(self.net, input_tensor)
        traced_net.save(out_file_name + "_script.pt")

        # just weights for cpp
        state_dict = {('.'.join(k.split('.')[1:])): v for k, v in self.net.state_dict().items()}
        torch.save(state_dict, out_file_name + "_params.pt", _use_new_zipfile_serialization=True)
