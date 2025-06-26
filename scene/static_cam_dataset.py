import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scene.colmap_loader import qvec2rotmat
from scene.neural_3D_dataset_NDC import get_spiral

from utils.graphics_utils import fov2focal, focal2fov
from torchvision import transforms as T

class StaticMultiCamDataset(Dataset):
    def __init__(self, datadir, split, llffhold=8):
        
        self.datadir = datadir
        self.split = split
        self.llffhold = llffhold

        # Load transforms.json
        json_path = os.path.join(self.datadir, "transforms.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Could not find transforms.json in {datadir}")

        with open(json_path, 'r') as f:
            self.transforms_data = json.load(f)

        # Extract camera intrinsics
        self.focal = [self.transforms_data["fl_x"], self.transforms_data["fl_y"]]
        height = self.transforms_data["h"]
        width = self.transforms_data["w"]
        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[1], width)

        self.transform = T.ToTensor()
        self.image_paths, self.image_poses, self.image_times = self.load_images_path()
        
    
    def load_images_path(self):
        image_paths = []
        image_poses = []
        image_times = []
        camera_extrinsics = {camera["id"]: np.array(camera["transform_matrix"]) for camera in self.transforms_data["cameras"]}

        for idx, frame in enumerate(self.transforms_data["frames"]):
            cam_id = frame["cam_id"]

            transform_matrix = camera_extrinsics[cam_id]
            R = transform_matrix[:3, :3]
            T = transform_matrix[:3, 3]

            image_path = os.path.join(self.datadir, frame["file_path"])
            if not os.path.exists(image_path):
                print(f"Warning: File not found: {image_path}")
                continue

            
            image_paths.append(image_path)
            image_poses.append((R, T))
            image_times.append(float(idx / len(self.transforms_data["frames"])))

        if self.split == "train":
                image_paths = [image_paths[i] for i in range(len(image_paths)) if i % self.llffhold != 0]
                image_poses = [image_poses[i] for i in range(len(image_poses)) if i % self.llffhold != 0]
                image_times = [image_times[i] for i in range(len(image_times)) if i % self.llffhold != 0]
        elif self.split == "test":
                image_paths = [image_paths[i] for i in range(len(image_paths)) if i % self.llffhold == 0]
                image_poses = [image_poses[i] for i in range(len(image_poses)) if i % self.llffhold == 0]
                image_times = [image_times[i] for i in range(len(image_times)) if i % self.llffhold == 0]

        print(f"Loaded {len(image_paths)} image paths.")
        return image_paths, image_poses, image_times
    
    def get_video_cam_infos(self, datadir):
        from scene.dataset_readers import CameraInfo
        poses_arr = np.load(os.path.join(datadir, "transforms.json"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        near_fars = poses_arr[:, -2:]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        N_views = 300
        val_poses = get_spiral(poses, near_fars, N_views=N_views)

        cameras = []
        len_poses = len(val_poses)
        times = [i/len_poses for i in range(len_poses)]
        image = Image.open(self.image_paths[0])
        image = self.transform(image)

        for idx, p in enumerate(val_poses):
            image_path = None
            image_name = f"{idx}"
            time = times[idx]
            pose = np.eye(4)
            pose[:3,:] = p[:3,:]
            R = pose[:3,:3]
            R = - R
            R[:,0] = -R[:,0]
            T = -pose[:3,3].dot(R)
            FovX = self.FovX
            FovY = self.FovY
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))
        return cameras
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.transform(img)
        return img, self.image_poses[index], self.image_times[index]
    def load_pose(self,index):
        return self.image_poses[index]

        