import os
import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.dataset_readers import CameraInfo
from scene.neural_3D_dataset_NDC import get_spiral, average_poses
# <<< FIX 1: Change the import alias to avoid collision >>>
from torchvision import transforms

class multipleview_dataset(Dataset):
    def __init__(self, datadir, split, llffhold=8):
        self.datadir = datadir
        self.split = split
        self.llffhold = llffhold
        self.transform = transforms.ToTensor()
        
        try:
            poses_arr = np.load(os.path.join(self.datadir, "poses_bounds_multipleview.npy"))
        except FileNotFoundError:
            poses_arr = np.load(os.path.join(self.datadir, "poses_bounds.npy"))
            
        self.poses = poses_arr[:, :-2].reshape([-1, 3, 5])
        self.bds = poses_arr[:, -2:]

        H, W, self.focal = self.poses[0, :, -1]
        self.height = int(H)
        self.width = int(W)
        self.FovY = focal2fov(self.focal, self.height)
        self.FovX = focal2fov(self.focal, self.width)

        self.camera_poses = []
        for i in range(self.poses.shape[0]):
            c2w = self.poses[i, :3, :4]

            c2w_4x4 = np.eye(4)
            
            c2w_4x4[:3, :4] = c2w
        
            c2w_ = c2w_4x4.copy()
            c2w_[:, 1] *= -1 # flip y
            c2w_[:, 2] *= -1 # flip z

            
            w2c = np.linalg.inv(c2w_)
            
            R = w2c[:3, :3]
            T = w2c[:3, 3]
            self.camera_poses.append((R, T))

        self.image_paths, self.image_poses, self.image_times = self.load_images_path()

        if self.split == "test":
            self.video_cam_infos = self.get_video_cam_infos()

    def load_images_path(self):
        all_image_paths = []
        all_image_poses = []
        all_image_times = []

        cam_dirs = sorted(glob.glob(os.path.join(self.datadir, "cam*")))
        cam_dirs = [d for d in cam_dirs if os.path.isdir(d)]

        for cam_idx, cam_dir in enumerate(cam_dirs):
            print(f"Processing images from folder: {cam_dir}")
            
            image_extensions = ['.jpg', '.jpeg', '.png']
            frame_paths = sorted([os.path.join(cam_dir, f) for f in os.listdir(cam_dir) 
                                  if os.path.splitext(f)[1].lower() in image_extensions])
            
            num_frames = len(frame_paths)
            if num_frames == 0:
                print(f"Warning: No images found in {cam_dir}")
                continue

            for frame_idx, frame_path in enumerate(frame_paths):
                is_test_frame = (frame_idx % self.llffhold == 0)
                if self.split == 'train' and is_test_frame:
                    continue
                if self.split == 'test' and not is_test_frame:
                    continue

                all_image_paths.append(frame_path)
                all_image_poses.append(self.camera_poses[cam_idx])
                all_image_times.append(frame_idx / (num_frames - 1.0) if num_frames > 1 else 0.5)

        print(f"Loaded {len(all_image_paths)} image paths for '{self.split}' split.")
        return all_image_paths, all_image_poses, all_image_times

    def get_video_cam_infos(self):
        poses_for_spiral = self.poses[:, :3, :4]
        
        near_fars_placeholder = np.array([[0.1, 10.0]] * len(poses_for_spiral))
        
        val_poses = get_spiral(poses_for_spiral, near_fars_placeholder, N_views=120)
        
        cameras = []
        num_poses = len(val_poses)
        
        # Check if image_paths is not empty before accessing
        if not self.image_paths:
             print("Warning: Cannot generate video path because no test images were loaded.")
             return []

        sample_image = Image.open(self.image_paths[0])
        w, h = sample_image.size
        
        for idx, p in enumerate(val_poses):
            c2w = p 
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, 3] 
            
            cam_info = CameraInfo(
                uid=idx, R=R, T=T, 
                FovY=self.FovY, FovX=self.FovX,
                image=self.transform(sample_image),
                image_path=None, image_name=f"video_{idx}",
                width=w, height=h,
                time=idx / (num_poses - 1.0), mask=None
            )
            cameras.append(cam_info)
        return cameras

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            # This is what the DataLoader worker likely expects anyway.
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError("Slicing with a step is not supported.")
            # We will process and return the first item of the requested slice.
            index = start
        img = Image.open(self.image_paths[index])
        img = self.transform(img)
        return img, self.image_poses[index], self.image_times[index]

    def load_pose(self, index):
        return self.image_poses[index]