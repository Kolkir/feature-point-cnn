import argparse

from preprocess_coco import preprocess_coco
from settings import SuperPointSettings
from inferencewrapper import InferenceWrapper
from trainwrapper import TrainWrapper
from camera import Camera
import numpy as np
import cv2
import time


def main():
    settings = SuperPointSettings()

    parser = argparse.ArgumentParser(description='PyTorch SuperPoint network.')
    parser.add_argument('--H', type=int, default=480,
                        help='Input image height.')
    parser.add_argument('--W', type=int, default=640,
                        help='Input image width')
    parser.add_argument('--nms_dist', type=int, default=settings.nms_dist,
                        help='Non Maximum Suppression (NMS) distance.')
    parser.add_argument('--conf_thresh', type=float, default=settings.confidence_thresh,
                        help='Detector confidence threshold.')
    parser.add_argument('--nn_thresh', type=float, default=settings.nn_thresh,
                        help='Descriptor matching threshold).')
    parser.add_argument('--cuda', action='store_true',
                        help='Use cuda GPU to speed up network processing speed')

    subparsers = parser.add_subparsers(dest='run_mode', required=True)
    inference_parser = subparsers.add_parser('inference')
    inference_parser.add_argument('--weights_path', type=str, default='superpoint.pth',
                                  help='Path to pretrained weights file.', required=True)
    inference_parser.add_argument('--camid', type=int, default=0,
                                  help='OpenCV webcam video capture ID, usually 0 or 1.')
    inference_parser.add_argument('--quantization', action='store_true',
                                  help='Enable model quantization for the CPU inference')
    inference_parser.add_argument('--out_file_name', type=str, default='superpoint',
                                  help='Filename prefix for the output pytorch script model.')

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('-p', '--checkpoint_path', type=str, default='checkpoints',
                              help='Path where training checkpoints will be saved.')
    train_parser.add_argument('-b', '--batch_size', type=int, default=32,
                              help='Training batch size')

    train_group = train_parser.add_mutually_exclusive_group()
    # train_group.required = True
    train_group.add_argument('-s', '--synthetic_path', type=str,
                             help='Path to the synthetic shapes dataset.')
    train_coco_group = train_group.add_argument_group()
    train_coco_group.add_argument('-c', '--coco_path', type=str,
                                  help='Path to the coco dataset.')
    train_coco_group.add_argument('-g', '--generate_points', action='store_true',
                                  help='Generate points for the COCO dataset.')
    train_coco_group.add_argument('--magic_point_weights', type=str, default='magicpoint.pth',
                                  help='Path to pretrained MagicPoint weights file.')

    opt = parser.parse_args()
    print(opt)

    settings.read_options(opt)

    if opt.run_mode == "inference":
        run_inference(opt, settings)

    elif opt.run_mode == "train":
        if opt.synthetic_path:
            print('Start MagicPoint training...')
            train_net = TrainWrapper(checkpoint_path=opt.checkpoint_path,
                                     settings=settings)
            train_net.train_magic_point(opt.synthetic_path)
            print('MagicPoint training finished')
        elif opt.coco_path and opt.generate_points:
            print('Pre-processing COCO dataset...')
            preprocess_coco(opt.coco_path, opt.magic_point_weights, settings)
            print('Pre-processing finished')
        elif opt.coco_path:
            print('Start SuperPoint training...')
            train_net = TrainWrapper(checkpoint_path=opt.checkpoint_path,
                                     settings=settings)
            train_net.train_super_point(opt.coco_path)
            print('SuperPoint training finished')
    else:
        print('Invalid run mode')


def run_inference(opt, settings):
    camera = Camera(opt.camid, opt.W, opt.H)
    print('Loading pre-trained network...')
    net = InferenceWrapper(weights_path=opt.weights_path, settings=settings)
    print('Successfully loaded pre-trained network.')
    win_name = 'SuperPoint features'
    cv2.namedWindow(win_name)
    prev_frame_time = 0
    while True:
        frame, ret = camera.get_frame()
        if ret:
            points, descriptors = net.run(frame)

            res_img = (np.dstack((frame, frame, frame)) * 255.).astype('uint8')
            for point in points.T:
                point_int = (int(round(point[0])), int(round(point[1])))
                cv2.circle(res_img, point_int, 1, (0, 255, 0), -1, lineType=16)

            # draw FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = 'FPS: ' + str(int(fps))
            cv2.putText(res_img, fps, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2, cv2.LINE_AA)

            cv2.imshow(win_name, res_img)
            key = cv2.waitKey(delay=1)
            if key == ord('q'):
                print('Quitting, \'q\' pressed.')
                break
            if key == ord('s'):
                net.trace(frame, opt.out_file_name)
                print('Model saved, \'s\' pressed.')
        else:
            break
    camera.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
