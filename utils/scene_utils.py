import torch
import os
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']

import numpy as np

import copy
@torch.no_grad()
def render_training_image(scene, gaussians, viewpoints, render_func, pipe, background, stage, iteration, time_now, dataset_type):
    def render(gaussians, viewpoint, path, scaling, cam_type):
        render_pkg = render_func(viewpoint, gaussians, pipe, background, stage=stage, cam_type=cam_type)
        label1 = f"stage:{stage},iter:{iteration}"
        times =  time_now/60
        if times < 1:
            end = "min"
        else:
            end = "mins"
        label2 = "time:%.2f" % times + end
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        if dataset_type == "PanopticSports":
            gt_np = viewpoint['image'].permute(1,2,0).cpu().numpy()
        else:
            gt_np = viewpoint.original_image.permute(1,2,0).cpu().numpy()
        
        image_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        depth_np = depth.permute(1, 2, 0).cpu().numpy()
        depth_np /= (depth_np.max() + 1e-8) # Add epsilon to avoid division by zero
        depth_np = np.repeat(depth_np, 3, axis=2)

        # <<< FIX: ADAPTIVE CHANNEL CONVERSION FOR VISUALIZATION >>>
        # Check if ground truth is grayscale (1 channel) and render is RGB (3 channels)
        if gt_np.shape[2] == 1 and image_np.shape[2] == 3:
            # Convert the grayscale GT to a 3-channel image by repeating the channel
            gt_np = np.repeat(gt_np, 3, axis=2)
        # <<< END OF FIX >>>

        image_np = np.concatenate((gt_np, image_np, depth_np), axis=1)
        image_with_labels = Image.fromarray((np.clip(image_np,0,1) * 255).astype('uint8'))  
        draw1 = ImageDraw.Draw(image_with_labels)
        try:
            # Try to load a system font.
            font = ImageFont.truetype('./utils/TIMES.TTF', size=40)
        except IOError:
            # If the specific font is not found, use the default PIL font.
            print("Warning: TIMES.TTF not found. Using default font for debug images.")
            font = ImageFont.load_default()

        text_color = (255, 0, 0)  
        label1_position = (10, 10)
        # Adjust label2 position based on default font if needed
        label2_width = draw1.textlength(label2, font=font)
        label2_position = (image_with_labels.width - 20 - label2_width, 10)
        
        draw1.text(label1_position, label1, fill=text_color, font=font)
        draw1.text(label2_position, label2, fill=text_color, font=font)
        
        image_with_labels.save(path)

    render_base_path = os.path.join(scene.model_path, f"{stage}_render")
    point_cloud_path = os.path.join(render_base_path,"pointclouds")
    image_path = os.path.join(render_base_path,"images")
    if not os.path.exists(os.path.join(scene.model_path, f"{stage}_render")):
        os.makedirs(render_base_path)
    if not os.path.exists(point_cloud_path):
        os.makedirs(point_cloud_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    
    for idx in range(len(viewpoints)):
        image_save_path = os.path.join(image_path,f"{iteration}_{idx}.jpg")
        render(gaussians,viewpoints[idx],image_save_path,scaling = 1,cam_type=dataset_type)
    
    # This part of the function seems incomplete, but the fix above addresses the crash.
    # pc_mask = gaussians.get_opacity
    # pc_mask = pc_mask > 0.1

def visualize_and_save_point_cloud(point_cloud, R, T, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    R = R.T
    T = -R.dot(T)
    transformed_point_cloud = np.dot(R, point_cloud) + T.reshape(-1, 1)
    ax.scatter(transformed_point_cloud[0], transformed_point_cloud[1], transformed_point_cloud[2], c='g', marker='o')
    ax.axis("off")
    plt.savefig(filename)