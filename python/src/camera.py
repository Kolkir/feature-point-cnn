import cv2
from threading import Thread


class Camera(object):
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            print('Failed to open camera {0}'.format(index))
            exit(-1)

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.status = False
        self.frame = None

    def update(self):
        while True:
            if self.cap.isOpened():
                (self.status, self.frame) = self.cap.read()
            else:
                break

    def get_frame(self):
        ret, frame = self.status, self.frame
        if ret is False:
            print('Camera failed to capture a frame')
            return None, False

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = frame.astype('float32') / 255.0

        return frame, True

    def close(self):
        self.cap.release()
        self.thread.join()
