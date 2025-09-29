import torch
import numpy as np

class LearningRateSchedulerLinear:
    
    def __init__(self, optimizer, start_factor, end_factor, n_epochs):
        self.optimizer = optimizer
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.n_epochs = n_epochs
        
        self.val_loss_min = np.Inf
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=self.start_factor, 
                                                        end_factor=self.end_factor, total_iters=self.n_epochs, 
                                                        last_epoch=-1)
        self.prev_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]

    def __call__(self, val_loss):
        self.scheduler.step()

        # Output the new learning rate if it is updated
        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = param_group['lr']
            if new_lr != self.prev_lr[i]:
                print(f'Learning rate updated ({self.prev_lr[i]:.6f} --> {new_lr:.6f})')
                self.prev_lr[i] = new_lr
        
        # Output when the validation loss is reduced
        if val_loss < self.val_loss_min:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            self.val_loss_min = val_loss

def r2_score(y_t, y_hat):
    ss_res = np.sum((y_t - y_hat) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    return 1 - ss_res / ss_tot

def mean_squared_error(y_t, y_hat, squared=True):
    mse = np.mean((y_t - y_hat) ** 2)
    if squared:
        return mse
    else:
        return np.sqrt(mse)