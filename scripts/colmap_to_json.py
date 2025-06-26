import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path to import colmap_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat

def c2w_from_colmap(R, t):
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1:3] *= -1  # OpenCV to OpenGL coordinate system conversion
    return c2w

def main(colmap_dir, source_dir, output_dir, total_cams):
    print(f"Reading COLMAP data from: {colmap_dir}")
    images_bin = read_extrinsics_binary(os.path.join(colmap_dir, "0", "images.bin"))
    cameras_bin = read_intrinsics_binary(os.path.join(colmap_dir, "0", "cameras.bin"))

    cam_intr = list(cameras_bin.values())[0]
    fl_x, fl_y, cx, cy = cam_intr.params
    w, h = cam_intr.width, cam_intr.height

    cam_poses = {}
    for img_data in images_bin.values():
        cam_id_str = os.path.splitext(img_data.name)[0].split('_')[0]
        R = qvec2rotmat(img_data.qvec)
        t = img_data.tvec
        c2w = c2w_from_colmap(R, t)
        cam_poses[cam_id_str] = c2w.tolist()

    train_frames, test_frames = [], []
    llffhold = 8

    for i in range(1, total_cams + 1):
        cam_id_str = f"cam{i:02d}"
        cam_dir = os.path.join(source_dir, cam_id_str)
        if not os.path.isdir(cam_dir): continue

        frame_files = sorted([f for f in os.listdir(cam_dir) if f.endswith(('.jpg', '.png'))])
        num_frames = len(frame_files)
        
        for frame_idx, frame_file in enumerate(frame_files):
            new_filename = f"{cam_id_str}_{frame_file}"
            file_path_for_json = f"./images/{new_filename}"
            
            frame_data = {
                "file_path": os.path.splitext(file_path_for_json)[0],
                "transform_matrix": cam_poses[cam_id_str],
                "time": frame_idx / (num_frames - 1.0) if num_frames > 1 else 0.5,
            }

            if frame_idx % llffhold == 0:
                test_frames.append(frame_data)
            else:
                train_frames.append(frame_data)

    base_json = {
        "fl_x": fl_x, "fl_y": fl_y, "cx": cx, "cy": cy, "w": w, "h": h,
        "camera_angle_x": np.arctan(w / (2 * fl_x)) * 2,
        "camera_angle_y": np.arctan(h / (2 * fl_y)) * 2,
    }

    with open(os.path.join(output_dir, "transforms_train.json"), "w") as f:
        json.dump({**base_json, "frames": train_frames}, f, indent=4)
    with open(os.path.join(output_dir, "transforms_test.json"), "w") as f:
        json.dump({**base_json, "frames": test_frames}, f, indent=4)
    print("Successfully created transforms_train.json and transforms_test.json")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))