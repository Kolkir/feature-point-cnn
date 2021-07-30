import time

import cv2
import numpy as np

from python.camera import Camera
from python.inferencewrapper import InferenceWrapper


def run_inference(opt, settings):
    camera = Camera(opt.camid, opt.W, opt.H)
    print('Loading pre-trained network...')
    net = InferenceWrapper(weights_path=opt.weights_path, settings=settings)
    print('Successfully loaded pre-trained network.')
    win_name = 'SuperPoint features'
    cv2.namedWindow(win_name)
    prev_frame_time = 0

    stop_img = None
    stop_features = None

    while True:
        frame, ret = camera.get_frame()
        frame = cv2.blur(frame, (5, 5))
        if ret:
            new_img = (np.dstack((frame, frame, frame)) * 255.).astype('uint8')
            features = get_features(frame, net)
            if stop_features is None:
                draw_features(features, new_img)
                stop_img = new_img
            else:
                correspondences, indices,  = get_best_correspondences(stop_features, features)
                draw_features(correspondences, new_img, indices)

            # combine images
            res_img = np.hstack((stop_img, new_img))

            # draw FPS
            new_frame_time = time.perf_counter()
            time_diff = new_frame_time - prev_frame_time
            prev_frame_time = new_frame_time

            draw_fps(time_diff, res_img)

            cv2.imshow(win_name, res_img)
            key = cv2.waitKey(delay=1)
            if key == ord('q'):
                print('Quitting, \'q\' pressed.')
                break
            if key == ord('s'):
                n = 20  # top 20 features
                stop_features = features[0:n, :]
                stop_img = (np.dstack((frame, frame, frame)) * 255.).astype('uint8')
                draw_features(stop_features, stop_img)
            if key == ord('t'):
                net.trace(frame, opt.out_file_name)
                print('Model saved, \'t\' pressed.')
        else:
            break
    camera.close()
    cv2.destroyAllWindows()


def get_correspondences(stop_features, features, dist_thresh=0.1):
    correspondence = []
    for stop_feature in stop_features:
        for feature in features:
            a = stop_feature[3:]
            b = feature[3:]
            dist = np.linalg.norm(a - b, ord=2)
            if dist <= dist_thresh:
                correspondence.append(feature)
    return correspondence


def get_best_correspondences(stop_features, features):
    correspondences = []
    indices = []
    for stop_index, stop_feature in enumerate(stop_features):
        min_dist = 100.
        min_index = -1
        for new_index, feature in enumerate(features):
            a = stop_feature[3:]
            b = feature[3:]
            dist = np.linalg.norm(a - b, ord=2)
            if dist <= min_dist:
                min_dist = dist
                min_index = new_index
        if min_index >= 0:
            correspondences.append(features[min_index])
            indices.append(stop_index)
    return correspondences, indices


def get_features(frame, net):
    points, descriptors, _ = net.run(frame)
    points = points.T
    descriptors = descriptors.T
    points = np.hstack((points, descriptors))
    points = points[points[:, 2].argsort()[::-1]]  # sort by increasing the confidence level
    return points


def draw_fps(time_diff, image):
    fps = 1. / time_diff
    fps_str = 'FPS: ' + str(int(fps))
    cv2.putText(image, fps_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2, cv2.LINE_AA)


def draw_features(features, image, indices=None):
    if indices is None:
        indices = range(len(features))
    for i, point in zip(indices, features):
        point_int = (int(round(point[0])), int(round(point[1])))
        cv2.circle(image, point_int, 2, (0, 255, 0), -1, lineType=16)
        cv2.putText(image, str(i), point_int, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
