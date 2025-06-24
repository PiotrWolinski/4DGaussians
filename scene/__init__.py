import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
from scene.dataset_readers import add_points

class Scene:
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}

        # --- CORRECTED, PRIORITIZED DATA LOADER LOGIC ---
        if os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print(f"Found transforms.json in {args.source_path}, using Static_Multicam loader.")
            scene_info = sceneLoadTypeCallbacks["Static_Multicam"](args.source_path, args.llffhold)
            dataset_type = "Static_Multicam"
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming old Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.extension)
            dataset_type = "Blender"
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            print(f"Found sparse folder in {args.source_path}, using Colmap loader.")
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold)
            dataset_type = "Colmap"
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            print(f"Found poses_bounds.npy in {args.source_path}, using D-NeRF/LLFF loader.")
            # Note: This might now be redundant if your poses_bounds files are handled by Static_Multicam
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
            dataset_type = "dynerf"
        elif os.path.exists(os.path.join(args.source_path,"dataset.json")):
            print(f"Found dataset.json in {args.source_path}, using Nerfies loader.")
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, False, args.eval)
            dataset_type = "nerfies"
        elif os.path.exists(os.path.join(args.source_path,"train_meta.json")):
            print(f"Found train_meta.json in {args.source_path}, using PanopticSports loader.")
            scene_info = sceneLoadTypeCallbacks["PanopticSports"](args.source_path)
            dataset_type = "PanopticSports"
        else:
            assert False, "Could not recognize scene type!"
        # --- END OF CORRECTED LOGIC ---
        
        self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)

        if not self.loaded_iter:
            if scene_info.ply_path and os.path.exists(scene_info.ply_path):
                 with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            # The JSON camera saving is for the viewer and can be complex with dataset objects.
            # It's safer to handle this separately if viewer support is needed.
            # json_cams = []
            # ...

        if scene_info.point_cloud is not None:
            xyz_max = scene_info.point_cloud.points.max(axis=0)
            xyz_min = scene_info.point_cloud.points.min(axis=0)
            if args.add_points:
                print("add points.")
                scene_info = scene_info._replace(point_cloud=add_points(scene_info.point_cloud, xyz_max=xyz_max, xyz_min=xyz_min))
            self.gaussians._deformation.deformation_net.set_aabb(xyz_max,xyz_min)
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", f"iteration_{self.loaded_iter}", "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path, "point_cloud", f"iteration_{self.loaded_iter}"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)
        
    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera
        
    def getVideoCameras(self, scale=1.0):
        return self.video_camera