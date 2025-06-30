import torch
from torchvision.utils import save_image
import math
import random
import os

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


def event_loss_call(image_start, image_end, event_data, timestamps, resolution_h, resolution_w, iteration, img_i):
    '''
    simulate the generation of event stream and calculate the event loss
    '''

    start, end = timestamps
    # Ensure event_data is 1D
    if event_data.dim() != 1:
        raise ValueError(f"event_data must be a 1D tensor, but got shape {event_data.shape}")
    loss = []
    
     # Ensure timestamps are close together
    if abs(end - start) > 2:
        print(f"Skipping iteration {iteration} due to large timestamp difference: start={start}, end={end}")
        return torch.tensor(0, dtype=torch.float64)

    # Ensure timestamps are within bounds
    if start >= event_data.size(0) or end > event_data.size(0):
        print(f"Skipping iteration {iteration} due to out-of-bounds indices: start={start}, end={end}")
        return torch.tensor(0, dtype=torch.float64)
        
    # Accumulate events between timestamps
    event_clone = event_data.clone()
    accumulated_events = event_clone[start:end].sum()
    print(f"[Iteration {iteration}] Accumulated events: {accumulated_events}")    

    # Calculate difference between rendered images
    image_diff = torch.abs(image_end - image_start)
    print(f"[Iteration {iteration}] Image difference shape: {image_diff.shape}")
        
        # Calculate thresholds
    start_value = lin_log(event_data[start] * 255)
    end_value = lin_log(event_data[end - 1] * 255)
    print(f"[Iteration {iteration}] lin_log(start_value): {start_value}, lin_log(end_value): {end_value}")
        

    thres_pos = (end_value - start_value) / 0.3
    thres_neg = (end_value - start_value) / 0.2
    print(f"[Iteration {iteration}] thres_pos: {thres_pos}, thres_neg: {thres_neg}")
        
            
        #for j in range(start + 1, end):
         #   event_cur += event_clone[j, :]
        
    pos = accumulated_events >= 0
    neg = accumulated_events <= 0
    zero = accumulated_events == 0

    loss_pos = torch.mean(((thres_pos * pos) - ((accumulated_events + 0.5) * pos)) ** 2)
    loss_neg = torch.mean(((thres_neg * neg) - ((accumulated_events - 0.5) * neg)) ** 2)
        
    event_loss = loss_pos + loss_neg
    print(f"[Iteration {iteration}] Event Loss: {event_loss}")

    
    return event_loss