import sys
import cv2
from src.synthetic_dataset import SyntheticDataset
from src.netutils import get_points, make_prob_map_from_labels
from src.settings import SuperPointSettings


def draw_points(image, points, color):
    for point in points:
        point_int = (int(round(point[0])), int(round(point[1])))
        cv2.circle(image, point_int, 2, color, -1, lineType=16)


def test_synthetic(path):
    settings = SuperPointSettings()
    dataset = SyntheticDataset(path, settings, 'training')
    for item in dataset:
        image, point_labels = item
        show_data('Original', image, point_labels, (255, 255, 255), settings)
        key = cv2.waitKey(delay=0)
        if key == ord('q'):
            break


def show_data(name, image, point_labels, color, settings):
    img_h, img_w = image.shape[1:]
    prob_map = make_prob_map_from_labels(point_labels.numpy(), img_h, img_w, settings.cell)
    points = get_points(prob_map, img_h, img_w, settings)
    points = points.T
    # Draw result
    original_img = image.permute(1, 2, 0).data.cpu()
    original_img = original_img.numpy()
    original_img = cv2.UMat(original_img)
    draw_points(original_img, points, color=color)
    cv2.imshow(name, original_img)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
        test_synthetic(path)
