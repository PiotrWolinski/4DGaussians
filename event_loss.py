import torch
from torchvision.utils import save_image
import math
import random
import os
import torch.nn.functional as F

def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.
    :param x: float or ndarray the input linear value in range 0-255
    :param threshold: float threshold 0-255 the threshold for transisition from linear to log mapping
    """
    # converting x into np.float32.
    if x.dtype is not torch.float64:
        x = x.double()
    f = (1./threshold) * math.log(threshold)
    y = torch.where(x <= threshold, x*f, torch.log(x))

    return y.float()

def event_loss_call(image_start, image_end, event_data, timestamps):
    '''
    simulate the generation of event stream and calculate the event loss
    '''
    start, end = timestamps

    # Accumulate events between timestamps
    event_clone = event_data.clone()
    accumulated_events = event_clone[start:end, :, :].sum(dim=0)

    # Calculate thresholds
    start_value = lin_log(torch.sum(image_start, dim=0))
    end_value = lin_log(torch.sum(image_end, dim=0))     

    # Get differences between two rendered images at consecutive timestamps
    thres_pos = (end_value - start_value) / 0.95 # Assume higher contrast sensitivity due to good noise calibration
    thres_neg = (end_value - start_value) / 0.95 
    #print(f"[Iteration {iteration}] thres_pos: {thres_pos}, thres_neg: {thres_neg}")
        
    pos = accumulated_events >= 0
    neg = accumulated_events <= 0

    loss_pos = torch.mean(((thres_pos * pos) - ((accumulated_events + 0.5) * pos)) ** 2)
    loss_neg = torch.mean(((thres_neg * neg) - ((accumulated_events - 0.5) * neg)) ** 2)
        
    event_loss = loss_pos + loss_neg
    
    return event_loss
