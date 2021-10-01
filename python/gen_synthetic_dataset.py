import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import src.synthetic_shapes as synthetic_shapes
import numpy as np
import cv2
import time


def main():
    config = {
        'split_sizes': {'train': 3000, 'test': 500},
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
    total_len = 0
    for split, size in config['split_sizes'].items():
        total_len += size * len(drawing_primitives)

    with multiprocessing.Manager() as manager:
        with ProcessPoolExecutor() as executor:
            futures = []
            progress_counter_lock = manager.Lock()
            progress_counter = manager.Value('progress_counter', 0)
            for split, size in config['split_sizes'].items():
                out_dir = Path(opt.path, split)
                out_dir.mkdir(parents=True, exist_ok=True)
                for primitive in drawing_primitives:
                    futures.append(
                        executor.submit(generate_primitive, config, out_dir, primitive, size, split, progress_counter,
                                        progress_counter_lock))

            pbar = tqdm(range(total_len))
            current_progress_counter = 0
            while current_progress_counter < (total_len - 1):
                time.sleep(0.05)
                with progress_counter_lock:
                    pbar.update(progress_counter.value - current_progress_counter)
                    current_progress_counter = progress_counter.value

            for future in futures:
                future.result()


def generate_primitive(config, out_dir, primitive, size, split, progress_counter, progress_counter_lock):
    print(f'{split} - {primitive} generation started ...', flush=True)
    index = 0
    for i in range(size):
        image = synthetic_shapes.generate_background(config['image_size'], **config['generate_background'])
        points = np.array(getattr(synthetic_shapes, primitive)(
            image, **config['primitives'].get(primitive, {})))
        points = np.flip(points, 1)  # reverse coordinates ordering for opencv - become [y,x]

        b = config['preprocessing']['blur_size']
        image = cv2.GaussianBlur(image, (b, b), 0)
        points = (points * np.array(config['preprocessing']['resize'], float)
                  / np.array(config['image_size'], float))
        image = cv2.resize(image, tuple(config['preprocessing']['resize'][::-1]),
                           interpolation=cv2.INTER_LINEAR)

        filename = str(Path(out_dir, f'{primitive}_{index}.npz'))
        points = np.flip(points, 1)  # restore coordinates ordering - become [x,y]
        points = np.transpose(points)
        # add confidence information
        points = np.vstack([points, np.ones([1, points.shape[1]])])

        # add missed channel dimension
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32) / 255.

        np.savez_compressed(filename, image=image, points=points)
        index += 1
        with progress_counter_lock:
            progress_counter.value += 1
    print(f'{split} - {primitive} generation finished', flush=True)


if __name__ == '__main__':
    main()
