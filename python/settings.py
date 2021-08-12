class SuperPointSettings:
    def __init__(self):
        self.cuda = False
        self.nms_dist = 4
        self.confidence_thresh = 0.015
        self.nn_thresh = 0.7  # L2 descriptor distance for good match.
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.

        # training params
        self.train_image_size = (240, 320)
        self.batch_size = 32
        self.batch_size_divider = 1  # Used for gradient accumulation
        self.learning_rate = 0.001
        self.epochs = 100
        self.use_amp = True
        self.data_loader_num_workers = 8

    def read_options(self, opt):
        self.cuda = opt.cuda
        self.nms_dist = opt.nms_dist
        self.confidence_thresh = opt.conf_thresh
        self.nn_thresh = opt.nn_thresh
        if opt.run_mode != 'inference':
            self.batch_size = opt.batch_size
            self.batch_size_divider = opt.batch_size_divider
