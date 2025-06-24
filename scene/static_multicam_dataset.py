import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from utils.graphics_utils import fov2focal, focal2fov

class StaticMulticamDataset(Dataset):
    def __init__(self, datadir, split='train', llffhold=8):
        self.datadir = datadir
        self.split = split
        self.llffhold = llffhold
        self.transform = transforms.ToTensor()
        
        json_path = os.path.join(self.datadir, "transforms.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Could not find transforms.json in {datadir}")

        with open(json_path, 'r') as f:
            self.meta = json.load(f)

        # Load intrinsics
        self.width = self.meta['w']
        self.height = self.meta['h']
        self.fov_x = self.meta.get('camera_angle_x')
        self.focal_length = fov2focal(self.fov_x, self.width)
        self.fov_y = focal2fov(self.focal_length, self.height)

        # Load camera poses into a dictionary for easy lookup
        self.poses = {}
        for cam_data in self.meta['cameras']:
            cam_id = cam_data['id']
            c2w = np.array(cam_data['transform_matrix'])
            w2c = np.linalg.inv(c2w)
            self.poses[cam_id] = (w2c[:3, :3], w2c[:3, 3]) # Store (R, T)

        # Load and SPLIT frame information
        all_frames = self.meta['frames']
        self.frames = []
        for i, frame in enumerate(all_frames):
            original_frame_idx = frame.get("original_frame_idx", i)
            is_test_frame = (original_frame_idx % self.llffhold == 0)
            
            if self.split == 'train' and not is_test_frame:
                frame['global_index'] = i # Store original index for event map
                self.frames.append(frame)
            elif self.split == 'test' and is_test_frame:
                frame['global_index'] = i
                self.frames.append(frame)

        print(f"Loaded {len(self.frames)} frames for split '{self.split}'.")

        self.frames.sort(key=lambda x: x['time'])
        
        # Normalize time based on the max time of the *entire* dataset
        max_time = all_frames[-1]['time'] if all_frames else 1.0
        for frame in self.frames:
            frame['normalized_time'] = frame['time'] / max_time if max_time > 0 else 0.0

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]

        frame_info = self.frames[index]
        
        image_path = os.path.join(self.datadir, frame_info['file_path'])
        img = Image.open(image_path)
        img_tensor = self.transform(img)

        cam_id = frame_info['cam_id']
        pose = self.poses[cam_id]
        time = frame_info['normalized_time']
        global_idx = frame_info['global_index']

        return img_tensor, pose, time, global_idx