import torch
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

"""
Calculate different error metrics: mae, mse, mape
"""
def validation_error(true_val, pred_val, err_type = 'mse'):
    if err_type == 'mae':
        err = torch.mean(torch.abs(true_val - pred_val))
        return err
    
    if err_type == 'mse': # Mean squared error 
        err = torch.sqrt(torch.mean((true_val - pred_val)**2))
        return err
    
    if err_type == 'mape': # Mean absolute percentage error (MAPE)
        err = mean_absolute_percentage_error(true_val, pred_val)*100
        return err

"""
Calculate c_l (technically c_n) from validation data
"""
def validation_cl(input_y, validation_x, y_mean, y_scale):
    bound_idx = torch.hstack((torch.ones(1), torch.argwhere(torch.diff(torch.sign(torch.diff(validation_x[:, -2]))) != 0.).flatten()+1)).int()
    bound_idx = bound_idx[torch.argwhere(torch.diff(bound_idx) > 3)]

    ct = 0 
    cl = []
    while ct+2 < bound_idx.shape[0]: 
        int_cpu = torch.trapz(x = (validation_x[bound_idx[ct]:bound_idx[ct+1]+1, -2] + 1)/2, 
                              y = (input_y[bound_idx[ct]:bound_idx[ct+1]+1] + y_mean)/y_scale)
        int_cpl = torch.trapz(x = (validation_x[bound_idx[ct+1]:bound_idx[ct+2]+1, -2] + 1)/2, 
                              y = (input_y[bound_idx[ct+1]:bound_idx[ct+2]+1] + y_mean)/y_scale )
        cl.append(int_cpu.item() + int_cpl.item())
        ct += 2
    return np.array(cl)


def validation_coeffs(input_y, validation_x, y_mean, y_scale):
    bound_idx = torch.hstack((torch.ones(1), torch.argwhere(torch.diff(torch.sign(torch.diff(validation_x[:, -2]))) != 0.).flatten()+1)).int()
    bound_idx = bound_idx[torch.argwhere(torch.diff(bound_idx) > 3)]

    ct = 0 
    loss = []
    while ct+2 < bound_idx.shape[0]: 
        int_cpu = torch.trapz(x = (validation_x[bound_idx[ct]:bound_idx[ct+1]+1, -2] + 1)/2, 
                              y = (input_y[bound_idx[ct]:bound_idx[ct+1]+1] + y_mean)/y_scale)
        int_cpl = torch.trapz(x = (validation_x[bound_idx[ct+1]:bound_idx[ct+2]+1, -2] + 1)/2, 
                              y = (input_y[bound_idx[ct+1]:bound_idx[ct+2]+1] + y_mean)/y_scale )
        int_cpmu = torch.trapz(x = (validation_x[bound_idx[ct]:bound_idx[ct+1]+1, -2] + 1)/2, 
                              y = ((validation_x[bound_idx[ct]:bound_idx[ct+1]+1, -2] + 1)/2) * (input_y[bound_idx[ct]:bound_idx[ct+1]+1] + y_mean)/y_scale)
        int_cpml = torch.trapz(x = (validation_x[bound_idx[ct+1]:bound_idx[ct+2]+1, -2] + 1)/2, 
                              y = ((validation_x[bound_idx[ct+1]:bound_idx[ct+2]+1, -2] + 1)/2) * (input_y[bound_idx[ct+1]:bound_idx[ct+2]+1] + y_mean)/y_scale )
        cn = np.abs(int_cpu.item()) + np.abs(int_cpl.item())
        cm = np.abs(int_cpmu.item()) + np.abs(int_cpml.item()) # not exactly that but close enough
        loss.append(cn + cm)
        ct += 2
    return np.array(loss)

"""
[UNUSED] Cp_crit loss
"""
import math
def test_loss(batch_x, batch_pred, batch_cpcrit, scale=1):
    supersonic_idx = torch.logical_and(batch_pred.mean <= batch_cpcrit, batch_x[:, -3]/10>=0.6) #batch_pred.mean < batch_cpcrit
    loss = scale*torch.mean(torch.abs(batch_cpcrit[supersonic_idx]-batch_pred.mean[supersonic_idx])**2)
    if math.isnan(loss.item()):
        loss = torch.Tensor([0])
    return loss