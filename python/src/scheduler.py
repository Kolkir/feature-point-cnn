
class Scheduler:
    def __init__(self, optimizer, lr, gamma, update_steps, plateau_steps):
        self.optimizer = optimizer
        self.lr = lr
        self.metric_value = -1
        self.gamma = gamma
        self.update_steps = update_steps
        self.plateau_steps = plateau_steps
        self.current_step = 0

    def assign_learning_rate(self, new_lr):
        print(f'New learning rate {new_lr}')
        self.lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def step(self, update_step, metric_value):
        if update_step % self.update_steps == 0:
            if self.metric_value < 0:
                self.metric_value = metric_value.clone()
            elif metric_value > self.metric_value and self.current_step < self.plateau_steps:
                self.current_step += 1
            elif metric_value > self.metric_value and self.current_step >= self.plateau_steps:
                self.current_step = 0
                self.assign_learning_rate(self.lr * self.gamma)
                self.metric_value = metric_value.clone()
            else:
                self.metric_value = metric_value.clone()

