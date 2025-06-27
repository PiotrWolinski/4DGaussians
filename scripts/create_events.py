import h5py
import numpy as np
import torch
import os
import sys
import json
from tqdm import tqdm

SOURCE_DATA_DIR = 'data/final_dataset/falling_bag' 

# A list of tuples: (path_to_h5_file, corresponding_cam_id_string)
EVENT_SOURCES = [
    ('data/h5_data/1642_paper_bag_data.h5', 'cam16'),
    #('data/h5_data/1642_paper_bag_data.h5', 'cam16'),
]
OUTPUT_DIR = 'data/temp_pt'

TIMESTAMP_PATH = 'data/h5_data/'

H, W = 480, 752 # Image resolution


def get_timestamps_from_json(txt_path):
    """Reads the 'time' field for all frames from the transforms.json file."""
    with open(txt_path, 'r') as f:
        timestamps = [int(line.strip()) for line in f.readlines()]
    timestamps = np.array(timestamps)
    return timestamps
    
    


def create_event_tensor(h5_path, frame_boundaries, height, width):
    
    num_frames = len(frame_boundaries) + 1
    event_accumulator = [[0 for _ in range(width)] for _ in range(height)]
    with h5py.File(h5_path, 'r') as f:
        t = list(map(int, f['t'][:]))  # Cast timestamps to int
        x = list(map(int, f['x'][:]))  # Cast x-coordinates to int
        y = list(map(int, f['y'][:]))  # Cast y-coordinates to int
        p = list(map(int, f['p'][:]))  # Cast polarities to int
        print(f"Length of x: {len(x)}, Sample: {x[:5]}")
        print(f"Length of y: {len(y)}, Sample: {y[:5]}")
        print(f"Length of p: {len(p)}, Sample: {p[:5]}")

        events = {
            'x': x,
            'y': y,
            't': t,
            'p': p
        }

    all_boundaries = [0.0] + frame_boundaries
    print ("all_boundaries:", all_boundaries)
    for i in tqdm(range(len(all_boundaries) - 1), desc=f"Processing {os.path.basename(h5_path)}"):
        t_start, t_end = all_boundaries[i], all_boundaries[i+1]

        # Debugging: Print the time boundaries for the current frame
        print(f"Frame {i}: t_start={t_start}, t_end={t_end}")

        time_slice_indices = [idx for idx, time in enumerate(events['t']) if t_start <= time < t_end]
        print(f"Number of events in time slice: {len(time_slice_indices)}")
        if not time_slice_indices:
            continue

      

        time_slice_events = {key: [events[key][idx] for idx in time_slice_indices] for key in events}

        x_s = list(map(int, time_slice_events['x']))  # Cast x-coordinates to int
        y_s = list(map(int, time_slice_events['y']))  # Cast y-coordinates to int
        p_s = list(map(int, time_slice_events['p']))  # Cast polarities to int
        # Debugging: Print polarities
        print(f"Polarities for frame {i}: {p_s[:10]}")

        polarities = [-1 if p == 0 else 1 for p in p_s]
        polarities = list(map(int, polarities))
        print(f"Mapped polarities for frame {i}: {polarities[:10]}")

        valid_indices = [
            idx for idx in range(len(x_s))
            if 0 <= x_s[idx] < width and 0 <= y_s[idx] < height
        ]
        x_s = [x_s[idx] for idx in valid_indices]
        y_s = [y_s[idx] for idx in valid_indices]
        polarities = [polarities[idx] for idx in valid_indices]

        # Debugging: Print valid indices
        print(f"x_s valid indices: {x_s[:10]}")
        print(f"y_s valid indices: {y_s[:10]}")

        
        for idx in range(len(x_s)):
            x_idx = x_s[idx]
            y_idx = y_s[idx]
            polarity = polarities[idx]
            event_accumulator[y_idx][x_idx] += polarity

        # Debugging: Print a sample of the event accumulator after processing the frame
        print(f"Event accumulator after frame {i}: {[row[10:] for row in event_accumulator[10:]]}")
    
    # Remove zeros from the event accumulator
    event_accumulator = [[value for value in row if value != 0] for row in event_accumulator if any(value != 0 for value in row)]
    
    max_length = max(len(row) for row in event_accumulator)
    event_accumulator = [row + [0] * (max_length - len(row)) for row in event_accumulator]

    event_tensor = torch.tensor(event_accumulator, dtype=torch.int32)

    # Debugging: Print the final event tensor shape and sample values
    print(f"Final event tensor shape: {event_tensor.shape}")
    print(f"Sample values from event tensor: {event_tensor[:5][:5]}")
    return event_tensor


if __name__ == "__main__":
    timestamps_path = os.path.join(TIMESTAMP_PATH, '1642_time_paper_bag.txt')
    if not os.path.exists(timestamps_path):
        print(f"FATAL: timestamps.txt not found in {EVENT_SOURCES}. Please run prepare_poses.sh first.")
        sys.exit(1)

    # We get the global timeline from the master JSON file
    frame_boundaries = get_timestamps_from_json(timestamps_path)
    
    for h5_file_path, cam_id in EVENT_SOURCES:
        
        print(f"\nProcessing events for {cam_id}...")
        event_tensor = create_event_tensor(h5_file_path, frame_boundaries, H, W)


        output_filename = f"events_{cam_id}.pt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        torch.save(event_tensor, output_path)
        
        print(f"\n--- Success for {cam_id}! ---")
        print(f"Final event tensor shape: {event_tensor.shape}")
        print(f"Saved to: {output_path}")