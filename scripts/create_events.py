import h5py
import numpy as np
import torch
import os
import glob
from tqdm import tqdm



# 1. Path to your unified dataset directory (the one with the 'images' folder)
DATASET_PATH = 'data/multipleview/unified_test4' # IMPORTANT: Change this

# 2. Path to your raw event data file
H5_FILE_PATH = 'data/h5_data/data_test.h5' # IMPORTANT: Change this

# 3. Image resolution (Height, Width)
H, W = 480, 640 # IMPORTANT: Change this to match your image dimensions

# ==============================================================================

def get_sorted_frame_paths(dataset_path):
    """
    Scans the 'images' subdirectory and returns a sorted list of all image file paths.
    The sorting ensures chronological order based on the filename convention.
    """
    images_dir = os.path.join(dataset_path, 'images')
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"The 'images' directory was not found in {dataset_path}")
    
    image_extensions = ['.jpg', '.jpeg', '.png']
    all_frames = []
    for ext in image_extensions:
        all_frames.extend(glob.glob(os.path.join(images_dir, f'*{ext}')))
        
    # Sort based on the filename (e.g., cam01_frame_00001, cam01_frame_00002, ...)
    # This assumes your filenames provide a natural chronological order.
    all_frames.sort()
    
    if not all_frames:
        raise ValueError("No image frames found in the 'images' directory.")
        
    return all_frames

def get_frame_timestamps_from_capture_rate(num_frames, capture_rate_hz=10.0):
    """
    Generates placeholder timestamps assuming a constant frame rate.
    
    NOTE: This is an APPROXIMATION. For best results, use real timestamps from
    your capture hardware. The timestamps are in seconds.
    """
    print(f"Warning: Using estimated timestamps based on a capture rate of {capture_rate_hz} Hz.")
    print("For best quality, provide real hardware-synchronized timestamps.")
    
    frame_duration = 1.0 / capture_rate_hz
    # We need the time boundaries between frames.
    # Boundary 'i' is the midpoint time between frame 'i' and frame 'i+1'.
    time_boundaries = [(i + 0.5) * frame_duration for i in range(num_frames)]
    
    return time_boundaries


def create_event_tensor(h5_path, frame_boundaries, height, width):
    """
    Processes a .h5 event file and creates a frame-synchronous event tensor.
    
    Args:
        h5_path (str): Path to the .h5 file.
        frame_boundaries (list): A list of timestamps representing the time boundary
                                 AFTER each frame.
        height (int): The height of the event sensor.
        width (int): The width of the event sensor.

    Returns:
        torch.Tensor: A tensor of shape [NUM_FRAMES - 1, H, W]
    """
    num_frames = len(frame_boundaries) + 1
    
    # The output tensor has one slice for each interval BETWEEN frames.
    event_accumulator = np.zeros((num_frames - 1, height, width), dtype=np.int16)
    
    print(f"Loading event data from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        # 1. Load the four separate 1D arrays
        t = f['t'][:]
        x = f['x'][:]
        y = f['y'][:]
        p = f['p'][:]
        
        # 2. Create the structured NumPy array that the rest of the script expects
        #    This combines the separate arrays into the familiar (x, y, t, p) format.
        events = np.core.records.fromarrays([x, y, t, p], names='x, y, t, p')
        
        

        # Convert timestamps to seconds if they are in microseconds
        if np.max(events['t']) > 1e9: # A simple heuristic to detect microseconds
             print("Timestamps appear to be in microseconds, converting to seconds.")
             events['t'] = events['t'] / 1e6

    print("Binning events into frame intervals...")
    # Add a zero at the beginning to represent the start time
    all_boundaries = [0.0] + frame_boundaries

    for i in tqdm(range(num_frames - 1), desc="Processing Frame Intervals"):
        t_start = all_boundaries[i]
        t_end = all_boundaries[i+1]
        
        # Find all events that occurred within this time slice
        time_slice_indices = np.where((events['t'] >= t_start) & (events['t'] < t_end))[0]
        
        if len(time_slice_indices) == 0:
            continue
            
        time_slice_events = events[time_slice_indices]
        
        # Get coordinates and polarities
        x_slice = time_slice_events['x']
        y_slice = time_slice_events['y']
        p_slice = time_slice_events['p']
        
        # Convert polarities {0, 1} to {-1, 1}
        polarities = np.where(p_slice == 0, -1, 1).astype(np.int16)
        
        # Use numpy.add.at for efficient, un-buffered accumulation at indices
        # This is much faster than a Python loop for large numbers of events
        np.add.at(event_accumulator[i], (y_slice.astype(int), x_slice.astype(int)), polarities)
        event_tensor = torch.from_numpy(event_accumulator)
        
    return event_tensor


if __name__ == "__main__":
    # 1. Get a sorted list of all frame files
    sorted_frames = get_sorted_frame_paths(DATASET_PATH)
    num_total_frames = len(sorted_frames)
    print(f"Found {num_total_frames} total image frames.")

    # 2. Get the timestamps for each frame.
    #    Replace this with your actual timestamp loading logic if you have it!
    #    This function generates approximate timestamps based on a 30Hz camera rate.
    frame_boundaries = get_frame_timestamps_from_capture_rate(num_total_frames, capture_rate_hz=30.0)

    # 3. Create the event tensor
    event_tensor = create_event_tensor(H5_FILE_PATH, frame_boundaries, H, W)

    # 4. Save the final tensor to the dataset directory
    output_path = os.path.join(DATASET_PATH, 'events.pt')
    torch.save(event_tensor, output_path)
    
    print("\n--- Success! ---")
    print(f"Final event tensor shape: {event_tensor.shape}")
    print(f"Saved to: {output_path}")