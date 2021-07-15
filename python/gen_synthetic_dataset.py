import argparse
from pathlib import Path
from tqdm import tqdm
import synthetic_shapes
import numpy as np
import cv2


def main():
    config = {
        'split_sizes': {'training': 25000, 'validation': 200, 'test': 500},
        'image_size': [960, 1280],
        'generate_background': {
            'min_kernel_size': 150, 'max_kernel_size': 500,
            'min_rad_ratio': 0.02, 'max_rad_ratio': 0.031},
        'primitives': {
            'draw_stripes': {'transform_params': (0.1, 0.1)},
            'draw_multiple_polygons': {'kernel_boundaries': (50, 100)}
        },
        'preprocessing': {
            'resize': [240, 320],
            'blur_size': 11,
        }
    }
    drawing_primitives = [
        'draw_lines',
        'draw_polygon',
        'draw_multiple_polygons',
        'draw_ellipses',
        'draw_star',
        'draw_checkerboard',
        'draw_stripes',
        'draw_cube',
        'gaussian_noise'
    ]

    parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
    parser.add_argument('path', type=str, help='Directory for a new dataset')
    opt = parser.parse_args()
    print(opt)

    Path(opt.path).mkdir(parents=True, exist_ok=True)

    index = 0
    for split, size in config['split_sizes'].items():
        im_dir, pts_dir = [Path(opt.path, i, split) for i in ['images', 'points']]
        im_dir.mkdir(parents=True, exist_ok=True)
        pts_dir.mkdir(parents=True, exist_ok=True)

        for primitive in drawing_primitives:
            print(primitive)
            for i in tqdm(range(size), desc=split, leave=False):
                image = synthetic_shapes.generate_background(config['image_size'], **config['generate_background'])
                points = np.array(getattr(synthetic_shapes, primitive)(
                    image, **config['primitives'].get(primitive, {})))
                points = np.flip(points, 1)  # reverse convention with opencv

                b = config['preprocessing']['blur_size']
                image = cv2.GaussianBlur(image, (b, b), 0)
                points = (points * np.array(config['preprocessing']['resize'], float)
                          / np.array(config['image_size'], float))
                image = cv2.resize(image, tuple(config['preprocessing']['resize'][::-1]),
                                   interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(str(Path(im_dir, '{}.png'.format(index))), image)
                np.save(str(Path(pts_dir, '{}.npy'.format(index))), points)
                index += 1
            print('\ndone')


if __name__ == '__main__':
    main()
