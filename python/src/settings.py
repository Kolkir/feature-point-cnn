class SuperPointSettings:
    def __init__(self):
        self.cuda = False
        self.nms_dist = 4
        self.confidence_thresh = 0.015
        self.nn_thresh = 0.7  # L2 descriptor distance for good match.
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.

        # loss params
        self.lambda_loss = 0.01  # original 0.0001
        self.lambda_d = 250  # 250 in the original version when the descriptor len=256
        self.positive_margin = 1
        self.negative_margin = 0.2

        # training params
        self.train_image_size = (240, 320)
        self.batch_size = 32
        self.batch_size_divider = 1  # Used for gradient accumulation

        self.learning_rate = 0.001
        self.optimizer_beta1 = 0.9
        self.optimizer_beta2 = 0.999
        self.optimizer_eps = 1.0e-8
        self.optimizer_weight_decay = 0.01
        self.scheduler_sin_range = 500
        self.epochs = 100

        self.use_amp = True
        self.data_loader_num_workers = 4
        self.write_statistics = True

    def read_options(self, opt):
        self.cuda = opt.cuda
        self.nms_dist = opt.nms_dist
        self.confidence_thresh = opt.conf_thresh
        self.nn_thresh = opt.nn_thresh
        self.write_statistics = opt.write_statistics
        if opt.run_mode != 'inference':
            self.batch_size = opt.batch_size
            self.batch_size_divider = opt.batch_size_divider
