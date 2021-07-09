class SuperPointSettings:
    def __init__(self):
        self.do_quantization = False # Makes sense for CPU inference
        self.encoder_kernel_size = (3, 3)
        self.encoder_stride = (1, 1)
        self.encoder_padding = (1, 1)

        self.detdesc_kernel_size_a = (3, 3)
        self.detdesc_kernel_size_b = (1, 1)
        self.detdesc_stride = (1, 1)
        self.detdesc_padding_a = (1, 1)
        self.detdesc_padding_b = (0, 0)

        self.encoder_dims = [(1, 64),
                             (64, 64),
                             (64, 128),
                             (128, 128)]

        self.detector_dims = (128, 256, 65)
        self.descriptor_dims = (128, 256, 256)  # original value for last layer was 256

        self.do_quantization = False
        self.cuda = False
        self.nms_dist = 4
        self.confidence_thresh = 0.015
        self.nn_thresh = 0.7  # L2 descriptor distance for good match.
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.

        # training params
        self.train_image_size = (240, 320)
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 5

    def read_options(self, opt):
        self.cuda = opt.cuda
        self.nms_dist = opt.nms_dist
        self.confidence_thresh = opt.conf_thresh
        self.nn_thresh = opt.nn_thresh
        if  opt.run_mode == 'inference':
            self.do_quantization = opt.quantization
        self.batch_size = opt.batch_size
