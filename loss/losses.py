import torch.nn as nn
import torch

class Regloss(nn.Module):
    def __init__(self):
        super(Regloss, self).__init__()
    
    def forward(self, y_true, y_pred):
        no_object_loss = torch.pow((1 - y_true[:, 0]) * y_pred[:, 0],2).mean()
        object_loss = torch.pow((y_true[:, 0]) * (y_pred[:, 0] - 1),2).mean()

        reg_loss = (y_true[:, 0] * (torch.pow(y_true[:, 1:5] - y_pred[:, 1:5],2).sum(1))).mean()    
        
        loss = no_object_loss + object_loss + reg_loss
        return loss
