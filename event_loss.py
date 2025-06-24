# event_loss.py
import torch
import math
import random

def lin_log(x, threshold=20):
    if x.dtype is not torch.float64:
        x = x.double()
    f = (1./threshold) * math.log(threshold)
    y = torch.where(x <= threshold, x*f, torch.log(x))
    return y.float()

def event_loss_call(all_rgb_sub_frames, event_data, combination, resolution_h, resolution_w, iteration, img_i):
    loss = []
    # To avoid high variance, we can sample a fixed number of pairs
    chose = random.sample(combination, min(len(combination), 10))
    for its in range(len(chose)):
        start_idx = chose[its][0]
        end_idx = chose[its][1]

        start_rgb = all_rgb_sub_frames[start_idx]
        end_rgb = all_rgb_sub_frames[end_idx]
        
        # Convert RGB to Grayscale internally
        start_gray = start_rgb[0, :, :] * 0.299 + start_rgb[1, :, :] * 0.587 + start_rgb[2, :, :] * 0.114
        end_gray = end_rgb[0, :, :] * 0.299 + end_rgb[1, :, :] * 0.587 + end_rgb[2, :, :] * 0.114

        log_intensity_start = lin_log(start_gray * 255)
        log_intensity_end = lin_log(end_gray * 255)
        
        thres_pos = (log_intensity_end - log_intensity_start) / 0.3
        thres_neg = (log_intensity_end - log_intensity_start) / 0.2
        
        event_clone = event_data.clone()
        event_cur = event_clone[start_idx].view(resolution_h, resolution_w)
        for j in range(start_idx + 1, end_idx):
            event_cur += event_clone[j].view(resolution_h, resolution_w)
        
        pos_mask = event_cur >= 0
        neg_mask = event_cur <= 0

        loss_pos = torch.mean(((thres_pos * pos_mask) - ((event_cur + 0.5) * pos_mask)) ** 2)
        loss_neg = torch.mean(((thres_neg * neg_mask) - ((event_cur - 0.5) * neg_mask)) ** 2)

        loss.append(loss_pos + loss_neg)

    event_loss = torch.mean(torch.stack(loss, dim=0), dim=0)
    return event_loss