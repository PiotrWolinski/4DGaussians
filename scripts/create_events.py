# In scripts/create_events.py
import h5py
import numpy as np
import torch
import os
import sys
import json
from tqdm import tqdm

# ==============================================================================
# --- Configuration ---
# ==============================================================================
# Path to the directory containing your source data (cam folders, etc.)
SOURCE_DATA_DIR = 'data/final_datasets/falling_bag' 

# A list of tuples: (path_to_h5_file, corresponding_cam_id_string)
EVENT_SOURCES = [
    ('data/h5_data/1634_paper_bag_data.h5', 'cam15'),
    ('data/h5_data/1642_paper_bag_data.h5', 'cam16'),
]

H, W = 480, 752 # Image resolution
# ==============================================================================

def get_timestamps_from_json(json_path):
    """Reads the 'time' field for all frames from the transforms.json file."""
    with open(json_path, 'r') as f:
        meta = json.load(f)
    # Sort frames by their original index to ensure chronological order
    all_frames = sorted(meta['frames'], key=lambda x: (x['cam_id'], x['original_frame_idx']))
    timestamps = [frame['time'] for frame in all_frames]
    
    # Create boundaries between frames
    time_boundaries = []
    for i in range(len(timestamps) - 1):
        midpoint = (timestamps[i] + timestamps[i+1]) / 2.0
        time_boundaries.append(midpoint)
    if len(timestamps) > 1:
        last_interval = timestamps[-1] - timestamps[-2]
        time_boundaries.append(timestamps[-1] + last_interval / 2.0)
    
    return time_boundaries, len(all_frames)

def create_event_tensor(h5_path, frame_boundaries, height, width):
    # ... (This function remains exactly the same as the last version) ...
    num_frames = len(frame_boundaries) + 1
    event_accumulator = np.zeros((num_frames - 1, height, width), dtype=np.int16)
    with h5py.File(h5_path, 'r') as f:
        t = f['t'][:]; x = f['x'][:]; y = f['y'][:]; p = f['p'][:]
        events = np.core.records.fromarrays([x, y, t, p], names='x, y, t, p')
        if np.max(events['t']) > 1e9: events['t'] /= 1e6
    all_boundaries = [0.0] + frame_boundaries
    for i in tqdm(range(num_frames - 1), desc=f"Processing {os.path.basename(h5_path)}"):
        t_start, t_end = all_boundaries[i], all_boundaries[i+1]
        time_slice_indices = np.where((events['t'] >= t_start) & (events['t'] < t_end))[0]
        if not time_slice_indices.size: continue
        time_slice_events = events[time_slice_indices]
        x_s, y_s, p_s = time_slice_events['x'], time_slice_events['y'], time_slice_events['p']
        polarities = np.where(p_s == 0, -1, 1).astype(np.int16)
        np.add.at(event_accumulator[i], (y_s.astype(int), x_s.astype(int)), polarities)
    return torch.from_numpy(event_accumulator)

if __name__ == "__main__":
    json_path = os.path.join(SOURCE_DATA_DIR, 'transforms.json')
    if not os.path.exists(json_path):
        print(f"FATAL: transforms.json not found in {SOURCE_DATA_DIR}. Please run prepare_poses.sh first.")
        sys.exit(1)

    # We get the global timeline from the master JSON file
    frame_boundaries, num_total_frames = get_timestamps_from_json(json_path)
    
    for h5_file_path, cam_id in EVENT_SOURCES:
        # NOTE: This script assumes the event data in each H5 file covers the *entire* duration
        # of the scene, and we are just binning it according to the global timeline.
        print(f"\nProcessing events for {cam_id}...")
        event_tensor = create_event_tensor(h5_file_path, frame_boundaries, H, W)

        # The output tensor should only contain slices for the specific event camera.
        # This requires a more complex mapping.
        # For now, we make a simplifying assumption: each event file corresponds
        # to a sequence of frames for that camera.
        # A more robust solution would filter the json frames by cam_id.

        output_filename = f"events_{cam_id}.pt"
        output_path = os.path.join(SOURCE_DATA_DIR, output_filename)
        torch.save(event_tensor, output_path)
        
        print(f"\n--- Success for {cam_id}! ---")
        print(f"Final event tensor shape: {event_tensor.shape}")
        print(f"Saved to: {output_path}")