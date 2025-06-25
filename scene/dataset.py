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
            return [self[i] for i in range(*index.indices(len(self)))]

        # All our supported data loaders now return a consistent tuple format.
        # Unpack the tuple from the underlying dataset (e.g., StaticMulticamDataset)
        image, (R, T), time, global_idx = self.dataset[index]
        
        # Get the FoV from the dataset object's attributes
        FovX = self.dataset.fov_x
        FovY = self.dataset.fov_y
        
        # Other metadata for the Camera object
        mask = None
        image_name = f"{global_idx}" # Use the global index for a unique name
        uid = global_idx
        colmap_id = global_idx # Use the global index for event map lookup

        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, dtype=torch.float32)

        # Finally, create the Camera object with the correctly extracted information.
        return Camera(colmap_id=colmap_id, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, gt_alpha_mask=None,
                      image_name=image_name, uid=uid, data_device=torch.device("cuda"), time=time,
                      mask=mask)
       
    def __len__(self):
        return len(self.dataset)
