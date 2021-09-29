import time

import cv2
import numpy as np
import scipy.spatial.distance

from src.camera import Camera
from src.inferencewrapper import InferenceWrapper


def run_inference(opt, settings):
    camera = Camera(opt.camid)
    print('Loading pre-trained network...')
    net = InferenceWrapper(weights_path=opt.weights_path, settings=settings)
    print('Successfully loaded pre-trained network.')
    win_name = 'SuperPoint features'
    cv2.namedWindow(win_name)
    prev_frame_time = 0

    stop_img = None
    stop_features = None
    do_blur = False

    while True:
        frame, ret = camera.get_frame()
        if do_blur:
            frame = cv2.blur(frame, (3, 3))
        if ret:
            img_size = (opt.W, opt.H)
            new_img = (np.dstack((frame, frame, frame)) * 255.).astype('uint8')
            query_img = cv2.resize(frame, img_size, interpolation=cv2.INTER_AREA)
            features = get_features(query_img, net)
            if stop_features is None:
                draw_features(features, new_img, img_size)
                stop_img = new_img
            else:
                correspondences, indices, = get_best_correspondences2(stop_features, features)
                draw_features(correspondences, new_img, img_size, indices)

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
                stop_features = features
                stop_img = (np.dstack((frame, frame, frame)) * 255.).astype('uint8')
                draw_features(stop_features, stop_img, img_size)
            if key == ord('b'):
                do_blur = not do_blur
            if key == ord('t'):
                net.trace(frame, opt.out_file_name)
                print('Model saved, \'t\' pressed.')
        else:
            break
    camera.close()
    cv2.destroyAllWindows()


def get_best_correspondences2(stop_features, features):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(queryDescriptors=features[:, 3:].astype(np.float32),
                       trainDescriptors=stop_features[:, 3:].astype(np.float32))
    matches_idx = np.array([m.queryIdx for m in matches])

    correspondences = np.array([features[idx] for idx in matches_idx])
    indices = np.array([m.trainIdx for m in matches])
    return correspondences, indices


def get_best_correspondences(stop_features, features):
    correspondences = []
    indices = []
    for stop_index, stop_feature in enumerate(stop_features):
        min_dist = 100
        min_index = -1
        a = stop_feature[3:]
        a = a / np.linalg.norm(a)
        for new_index, feature in enumerate(features):
            b = feature[3:]
            b = b / np.linalg.norm(b)
            dist = (a - b) ** 2
            dist = np.mean(dist)
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


def draw_features(features, image, img_size, indices=None):
    if indices is None:
        indices = range(len(features))
    sx = image.shape[1] / img_size[0]
    sy = image.shape[0] / img_size[1]
    for i, point in zip(indices, features):
        point_int = (int(round(point[0] * sx)), int(round(point[1] * sy)))
        cv2.circle(image, point_int, 2, (0, 255, 0), -1, lineType=16)
        cv2.putText(image, str(i), point_int, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)

