import torch.optim as optim
import math
from net import *
import util
import numpy as np
class Trainer():
    def __init__(self, model, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, cl=True):
        self.scaler = scaler
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.clip = clip
        self.step = step_size
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl

    def train(self, input, real_val, idx=None):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, idx=idx)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        if self.iter%self.step==0 and self.task_level<=self.seq_out_len:
            self.task_level +=1
        if self.cl:
            loss = self.loss(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0)
        else:
            loss = self.loss(predict, real, 0.0)

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()
        # mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        self.iter += 1
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse



class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)

        # for param in self.params:
        #     grad_norm += math.pow(param.grad.data.norm(), 2)
        #
        # grad_norm = math.sqrt(grad_norm)
        # if grad_norm > 0:
        #     shrinkage = self.max_grad_norm / grad_norm
        # else:
        #     shrinkage = 1.
        #
        # for param in self.params:
        #     if shrinkage < 1:
        #         param.grad.data.mul_(shrinkage)
        self.optimizer.step()
        return  grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()


def create_data_loader_from_arrays(X, y, train_ratio=0.6, val_ratio=0.2, device='cpu', normalize=2):
    """
    Create data loaders from numpy arrays using existing DataLoaderS logic.
    
    Args:
        X (np.ndarray): Input sequences, shape (samples, seq_len, num_nodes)
        y (np.ndarray): Target sequences, shape (samples, horizon, num_nodes)
        train_ratio (float): Ratio of data for training
        val_ratio (float): Ratio of data for validation
        device (str): Device to use
        normalize (int): Normalization method (0, 1, or 2)
        
    Returns:
        object: Data loader object compatible with existing training code
    """
    from util import DataLoaderS
    
    # Convert to format expected by DataLoaderS (samples, features)
    # For now, just use the first node and first horizon as a simple case
    # This can be extended for multi-node, multi-horizon cases
    
    # Flatten to 2D: (samples, features)
    X_flat = X.reshape(X.shape[0], -1)  # (samples, seq_len * num_nodes)
    y_flat = y[:, 0, 0]  # Just first horizon, first node for simplicity
    
    # Combine for DataLoaderS format
    data = np.column_stack([X_flat, y_flat.reshape(-1, 1)])
    
    # Save to temporary file for DataLoaderS
    import tempfile
    import os
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    np.savetxt(temp_file.name, data, delimiter=',')
    temp_file.close()
    
    # Create DataLoaderS
    horizon = y.shape[1]
    window = X.shape[1]
    
    try:
        data_loader = DataLoaderS(temp_file.name, train_ratio, val_ratio, 
                                torch.device(device), horizon, window, normalize)
    finally:
        # Clean up temp file
        os.unlink(temp_file.name)
    
    return data_loader
