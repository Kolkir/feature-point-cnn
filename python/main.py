import argparse

from src.preprocess_coco import preprocess_coco
from src.inference import run_inference
from src.settings import SuperPointSettings
from src.trainwrapper import TrainWrapper


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
    train_parser.add_argument('-d', '--batch_size_divider', type=int, default=1,
                              help='In the case of low GPU memory divides batch size to use gradient accumulation')
    train_parser.add_argument('--magic_point', action='store_true',
                              help='Restrict training to MagicPoint only')

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
        elif opt.coco_path and not opt.magic_point:
            print('Start SuperPoint training...')
            train_net = TrainWrapper(checkpoint_path=opt.checkpoint_path,
                                     settings=settings)
            train_net.train_super_point(opt.coco_path, opt.magic_point_weights)
            print('SuperPoint training finished')
        elif opt.coco_path and opt.magic_point:
            print('Start MagicPoint training with COCO...')
            train_net = TrainWrapper(checkpoint_path=opt.checkpoint_path,
                                     settings=settings)
            train_net.train_magic_point(opt.coco_path, use_coco=True)
            print('SuperPoint training finished')
    else:
        print('Invalid run mode')


if __name__ == '__main__':
    main()
