import cv2
from threading import Thread


class Camera(object):
    def __init__(self, index, img_w, img_h):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap:
            print('Failed to open camera {0}'.format(index))
            exit(-1)

        self.resize_width = img_w
        self.resize_height = img_h

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.status = False
        self.frame = None

    def update(self):
        while True:
            if self.cap.isOpened():
                (self.status, self.frame) = self.cap.read()

    def get_frame(self):
        ret, frame = self.status, self.frame
        if ret is False:
            print('Camera failed to capture a frame')
            return None, False

        frame = cv2.resize(frame, (self.resize_width, self.resize_height),
                           interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype('float32') / 255.0

        return frame, True

    def close(self):
        self.cap.release()
