import h5py
import numpy as np
import torch
import os
import sys
import json
from tqdm import tqdm
import cv2


EVENT_CAMERA_PARAMS = {
    "cam15": {
        "K": np.array([
            [444.11582755, 0, 327.73888769],  # [fx, 0, cx]
            [0, 445.28439246, 221.25515107],  # [0, fy, cy]
            [0, 0, 1.0]
        ]),
        "D": np.array([-0.3353364, 0.11296593, 0.00115195, -0.00364223]) # [k1, k2, p1, p2]
    },
    "cam16": {
        "K": np.array([
            [427.20830098, 0, 325.60321272],  # [fx, 0, cx]
            [0, 432.00697177, 240.44006261],  # [0, fy, cy]
            [0, 0, 1.0]
        ]),
        "D": np.array([-0.30277843, 0.08794749, -0.00079008, 0.00287706]) # [k1, k2, p1, p2]
    }
}

SOURCE_DATA_DIR = 'data/final_dataset/falling_bag' 

# A list of tuples: (path_to_h5_file, corresponding_cam_id_string)
EVENT_SOURCES = [
    ('/home/debmalya/Documents/4DGaussians_events/4DGaussians/data/h5_data/1642_paper_bag_data.h5', 'cam16'),
    #('/home/debmalya/Documents/4DGaussians_events/4DGaussians/data/h5_data/1642_paper_bag_data.h5', 'cam16'),
]
OUTPUT_DIR = '/home/debmalya/Documents/4DGaussians_events/4DGaussians/data/temp_pt'

H, W = 480, 640 # Image resolution
TIMESTAMP_PATH = '/home/debmalya/Documents/4DGaussians_events/4DGaussians/data/h5_data/1642_time_paper_bag.txt'

TIME_RESOLUTION = 2_000 # Set event aggregation window to 2 ms
H, W = 480, 640 # Image resolution


def get_timestamps_from_txt(txt_path):
    """Reads the start and end timestamps from the timestamps.txt file."""
    with open(txt_path, 'r') as f:
        timestamps = [int(line.strip()) for line in f.readlines()]
    return timestamps[0], timestamps[-1]  # Return start and end timestamps
    
    


def create_event_tensor(h5_path, height, width):

    with h5py.File(h5_path, 'r') as f:
        t = np.array(f['t'][:], dtype=int)  # Convert timestamps to NumPy array
        x = np.array(f['x'][:], dtype=int)  # Convert x-coordinates to NumPy array
        y = np.array(f['y'][:], dtype=int)  # Convert y-coordinates to NumPy array
        p = np.array(f['p'][:], dtype=int)  # Convert polarities to NumPy array
        print(f"Length of x: {len(x)}, Sample: {x[:5]}")
        print(f"Length of y: {len(y)}, Sample: {y[:5]}")
        print(f"Length of p: {len(p)}, Sample: {p[:5]}")

        events = {
            'x': x,
            'y': y,
            't': t,
            'p': p
        }

    cam_params = EVENT_CAMERA_PARAMS[cam_id]
    K = cam_params["K"]
    D = cam_params["D"]

    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, None, (width, height), cv2.CV_32FC1)

    x_int = events['x'].astype(int)
    y_int = events['y'].astype(int)

    valid_mask = (x_int < width) & (y_int < height) & (x_int >= 0) & (y_int >= 0)

    x_undistorted = mapx[y_int[valid_mask], x_int[valid_mask]]
    y_undistorted = mapy[y_int[valid_mask], x_int[valid_mask]]

    undistorted_events = {
        'x': x_undistorted,
        'y': y_undistorted,
        't': events['t'][valid_mask],
        'p': events['p'][valid_mask]
    }

    events = undistorted_events
    # Discretize time into intervals
    min_time = events['t'].min()
    max_time = events['t'].max()
    time_bins = np.arange(min_time, max_time, TIME_RESOLUTION)

    event_tensor = np.zeros((len(time_bins), height, width), dtype=np.int8)

    for start_idx in range(len(event_tensor)-1):
        t_start, t_end = time_bins[start_idx], time_bins[start_idx + 1]

        # Find events within the current time interval
        interval_mask = (events['t'] >= t_start) & (events['t'] < t_end)
        x_interval = events['x'][interval_mask].astype(int)
        y_interval = events['y'][interval_mask].astype(int)
        p_interval = events['p'][interval_mask].astype(int)

        # Map polarities to [-1, 1]
        polarities = np.where(p_interval == 0, -1, 1)

        # Accumulate polarities into the tensor
        for x, y, polarity in zip(x_interval, y_interval, polarities):
            event_tensor[start_idx, y, x] += polarity
    
    event_tensor = torch.tensor(event_tensor, dtype=torch.int8)
    
    print(f"Maximum value in the event tensor: {torch.max(event_tensor)}")
    

    #Debugging: Print a sample of the event accumulator after processing the frame
    #print(f"Event accumulator after frame {i}: {[row[10:] for row in event_accumulator[10:]]}")


    # Debugging: Print the final event tensor shape and sample values
    print(f"Final event tensor shape: {event_tensor.shape}")
    print(f"Sample values from event tensor: {event_tensor[:5][:5]}")
    return event_tensor


if __name__ == "__main__":
    timestamps = get_timestamps_from_txt(TIMESTAMP_PATH)


    for h5_file_path, cam_id in EVENT_SOURCES:
        
        print(f"\nProcessing events for {cam_id}...")
        event_tensor = create_event_tensor(h5_file_path, H, W)


        output_filename = f"events_{cam_id}.pt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        torch.save(event_tensor, output_path)
        
        print(f"\n--- Success for {cam_id}! ---")
        print(f"Final event tensor shape: {event_tensor.shape}")
        print(f"Saved to: {output_path}")