import re
from collections import OrderedDict
import torch


def load_weights_legacy(weights_path):
    state_dict = torch.load(weights_path)
    r = re.compile(r'conv(.)(\w)\.(.+)')
    new_dict = OrderedDict()
    for key, value in state_dict.items():
        match = r.match(key)
        if match:
            index = match.group(1)
            letter = match.group(2)
            params_type = match.group(3)
            if 'P' in index:
                new_key = 'detector_conv.detector_conv_' + letter + '.' + params_type
            elif 'D' in index:
                new_key = 'descriptor_conv.descriptor_conv_' + letter + '.' + params_type
            else:
                index = int(index)
                new_key = 'encoder_conv.encoder_conv' + str(index - 1) + '_' + letter + '.' + params_type
            new_dict[new_key] = value
        else:
            print('Invalid state dict key: {0}'.format(key))
            exit(-1)
    return new_dict