from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type = dataset_type

    def __getitem__(self, index):
        if isinstance(index, slice):
            # If we get a slice, process and return the first item of the slice.
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError("Slicing with a step is not supported.")
            index = start
        # This is the primary path for your new dataset format.
        # "MultipleView" is also added to catch the case if the loader identifies it differently.
        if self.dataset_type in ["LLFF_Multicam", "dynerf", "MultipleView"]:
            # 1. Unpack the tuple returned by multipleview_dataset
            image, (R, T), time = self.dataset[index]
            
            # 2. Access the focal length as a scalar attribute
            focal_length = self.dataset.focal 
            
            # 3. Calculate Field of View from the scalar focal length
            FovX = focal2fov(focal_length, image.shape[2]) # image.shape[2] is width
            FovY = focal2fov(focal_length, image.shape[1]) # image.shape[1] is height
            
            # 4. Set other metadata for the Camera object
            mask = None
            image_name = f"{index}"
            uid = index
            colmap_id = index
            
        elif self.dataset_type == "PanopticSports":
            # This case remains as is
            return self.dataset[index]
            
        else:
            # This is the fallback for original Colmap and Blender loaders
            # It expects a CameraInfo object, not a tuple.
            caminfo = self.dataset[index]
            image = caminfo.image
            R = caminfo.R
            T = caminfo.T
            FovX = caminfo.FovX
            FovY = caminfo.FovY
            time = caminfo.time
            mask = caminfo.mask
            image_name = caminfo.image_name
            uid = caminfo.uid
            colmap_id = uid 

        # This check was moved from the `else` block to apply to ALL cases.
        # It ensures 'time' is always a tensor, preventing the TypeError.
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, dtype=torch.float32)

        # Finally, create the Camera object with the correctly extracted information.
        return Camera(colmap_id=colmap_id, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, gt_alpha_mask=None,
                      image_name=image_name, uid=uid, data_device=torch.device("cuda"), time=time,
                      mask=mask)
       
    def __len__(self):
        return len(self.dataset)
