# metrics.py
import torch

def mse(pred, true):
    return torch.mean(torch.abs(pred - true)**2)

def evm(pred, true):
    power_true = torch.mean(torch.abs(true)**2)
    error_power = torch.mean(torch.abs(pred - true)**2)
    return torch.sqrt(error_power/power_true)*100
