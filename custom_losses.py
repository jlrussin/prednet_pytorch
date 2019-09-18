import torch
import torch.nn as nn

# Custom error loss function for PredNet
class ELoss(nn.Module):
    def __init__(self,layer_lambdas):
        super(ELoss,self).__init__()
        nb_layers = len(layer_lambdas)
        layer_lambdas = torch.tensor(layer_lambdas)
        self.layer_lambdas = layer_lambdas.view(1,nb_layers)
    def forward(self,errors):
        weighted_errors = errors*self.layer_lambdas
        total_error = torch.sum(weighted_errors)
        return total_error

def get_loss_fn(loss,layer_lambdas):
    if loss == 'E':
        loss_fn = ELoss(layer_lambdas)
    elif loss == 'MSE':
        loss_fn = nn.MSELoss()
    elif loss == 'L1':
        loss_fn = nn.L1Loss()
    elif loss == 'BCE':
        loss_fn = nn.BCELoss()
    return loss_fn
