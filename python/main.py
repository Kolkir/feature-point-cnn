import argparse
from settings import SuperPointSettings
from inferencewrapper import InferenceWrapper
from camera import Camera
import numpy as np
import cv2


def main():
    settings = SuperPointSettings()

    parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
    parser.add_argument('--run_mode', type=str, default='inference',
                        help='Run mode inference or training.')
    parser.add_argument('--weights_path', type=str, default='superpoint.pth',
                        help='Path to pretrained weights file.')
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
    parser.add_argument('--camid', type=int, default=0,
                        help='OpenCV webcam video capture ID, usually 0 or 1.')
    parser.add_argument('--cuda', action='store_true',
                        help='Use cuda GPU to speed up network processing speed')
    parser.add_argument('--out_file_name', type=str, default='superpoint',
                        help='Path to output pytorch script model.')
    opt = parser.parse_args()
    print(opt)

    settings.read_options(opt)

    if opt.run_mode == "inference":
        camera = Camera(opt.camid, opt.W, opt.H)
        print('Loading pre-trained network...')
        net = InferenceWrapper(weights_path=opt.weights_path, settings=settings)
        print('Successfully loaded pre-trained network.')

        win_name = 'SuperPoint features'
        cv2.namedWindow(win_name)

        while True:
            frame, ret = camera.get_frame()
            if ret:
                points, descriptors = net.run(frame)

                res_img = (np.dstack((frame, frame, frame)) * 255.).astype('uint8')
                for point in points.T:
                    point_int = (int(round(point[0])), int(round(point[1])))
                    cv2.circle(res_img, point_int, 1, (0, 255, 0), -1, lineType=16)
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

    elif opt.run_mode == "train":
        print('==> Start network training...')
    else:
        print('Invalid run mode')


if __name__ == '__main__':
    main()
