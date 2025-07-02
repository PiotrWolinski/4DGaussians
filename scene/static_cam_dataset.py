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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov

class StaticMultiCamDataset(Dataset):
    def __init__(self, datadir, split, llffhold=10):
        
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
        self.len_cams = len(self.transforms_data["cameras"])
        self.sequence_length = int(len(self.transforms_data["frames"]) / self.len_cams)
        self.possible_times = np.linspace(0, 1, self.sequence_length)

        self.transform = T.ToTensor()
        self.image_paths, self.image_poses, self.image_times = self.load_images_path()


        event_cam_data_path = os.path.join(self.datadir, "transforms_event.json")
        with open(event_cam_data_path, "r") as f:
            self.event_transforms = json.load(f)

        self.event_tensors, self.event_poses = self.load_event_cameras(event_cam_data_path)

        # Get number of even frames in the given tensors
        self.event_boundaries = [tensor.shape[0] for tensor in self.event_tensors]
    

    def load_images_path(self):
        image_paths = []
        image_poses = []
        image_times = []
        camera_extrinsics = {camera["id"]: np.array(camera["transform_matrix"]) for camera in self.transforms_data["cameras"]}

        # TODO: This is an assumption that all the cameras hold the same number of frames
        # In general this should be true
        frames_count = sum(1 if frame["cam_id"] == "cam01" else 0 for frame in self.transforms_data["frames"])

        per_cam_idx = 0
        prev_cam_id = "cam01"

        for frame in self.transforms_data["frames"]:
            cam_id = frame["cam_id"]

            if cam_id != prev_cam_id:
                per_cam_idx = 0

            transform_matrix = camera_extrinsics[cam_id]
            R = transform_matrix[:3, :3]
            T = transform_matrix[:3, 3]

            image_path = os.path.join(self.datadir, frame["file_path"])
            if not os.path.exists(image_path):
                print(f"Warning: File not found: {image_path}")
                continue

            image_paths.append(image_path)
            image_poses.append((R, T))
            image_times.append(float(per_cam_idx / (frames_count-1)))

            if prev_cam_id == cam_id:
                per_cam_idx += 1
            else:
                per_cam_idx = 0

            prev_cam_id = cam_id

        if self.split == "train":
                image_paths = [image_paths[i] for i in range(len(image_paths)) if i % self.llffhold != 0]
                image_poses = [image_poses[i] for i in range(len(image_poses)) if i % self.llffhold != 0]
                image_times = [image_times[i] for i in range(len(image_times)) if i % self.llffhold != 0]
        elif self.split == "test":
                image_paths = [image_paths[i] for i in range(len(image_paths)) if i % self.llffhold == 0]
                image_poses = [image_poses[i] for i in range(len(image_poses)) if i % self.llffhold == 0]
                image_times = [image_times[i] for i in range(len(image_times)) if i % self.llffhold == 0]

        print(f"Loaded {len(image_paths)} image paths.")
        return image_paths, image_poses, np.array(image_times)
    

    def load_event_cameras(self, event_data_path: str):
        event_cams_data = None
        with open(event_data_path, 'r') as f:
            event_cams_data = json.load(f)

        cam_poses = {cam["id"]: np.array(cam["transform_matrix"]) for cam in event_cams_data["cameras"]}
        parsed_poses = []
        tensors = []

        for cam_id in cam_poses.keys():
            transform_matrix = cam_poses[cam_id]
            R = transform_matrix[:3, :3]
            T = transform_matrix[:3, 3]

            parsed_poses.append((R, T))
            tensors.append(torch.load(os.path.join(self.datadir, f"events_{cam_id}.pt"), map_location="cuda"))

        return tensors, parsed_poses
    
    def get_event_camera(self, index: int, time: float):
        height = self.event_transforms["h"]
        width = self.event_transforms["w"]
        focal = [self.event_transforms["fl_x"], self.event_transforms["fl_y"]]
        image, _, _ = self[index]
        w2c = self.event_poses[index]
        R,T = w2c
        FovX = focal2fov(focal[0], height)
        FovY = focal2fov(focal[0], width)
    
        # Image is not important there so just take "noise" from the dataset
        return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"ev_{index}",uid=index,data_device=torch.device("cuda"),time=time,
                              mask=None)

 
    def get_next_timestamp_id(self, current_timestamp: float) -> int | None:
        """Returns idx of the next timestamp from the dataset for the given camera.
        
        Returns None if such timestamp does not exist (current is the last one from the dataset.)
        """

        # Discard if this is the last timestamp
        if current_timestamp == 1.0:
            return None
        
        current_idx = np.ceil(current_timestamp * self.sequence_length)

        return int(current_idx + 1)
        

    def timestamp_to_event_idx(self, timestamp: float, event_boundary: int) -> int:
        """Converts normalized camera timestamps into indices in the event tensors."""
        event_timestamp = int(event_boundary * timestamp)

        if event_timestamp >= event_boundary:
            event_timestamp -= 1

        return event_timestamp


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
