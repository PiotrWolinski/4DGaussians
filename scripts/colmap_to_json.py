import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path to import colmap_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat, read_intrinsics_text, read_extrinsics_text


def c2w_from_colmap(R, t):
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1:3] *= -1  # OpenCV to OpenGL coordinate system conversion
    return c2w


def main(colmap_dir, source_dir, output_dir, total_cams):
    print(f"Reading COLMAP data from: {colmap_dir}")
    images_bin = read_extrinsics_binary(os.path.join(colmap_dir, "images.bin"))
    cameras_bin = read_intrinsics_binary(os.path.join(colmap_dir, "cameras.bin"))

    cam_intr = list(cameras_bin.values())[0]
    fl_x, fl_y, cx, cy, *_ = cam_intr.params
    w, h = cam_intr.width, cam_intr.height
    
    avg_fx = 0
    avg_fy = 0

    avg_cx = 0
    avg_cy = 0

    for cam in cameras_bin.values():
        avg_fx += cam.params[0]
        avg_fy += cam.params[1]
        avg_cx += cam.params[2]
        avg_cy += cam.params[3]

    avg_fx /= len(cameras_bin)
    avg_fy /= len(cameras_bin)
    avg_cx /= len(cameras_bin)
    avg_cy /= len(cameras_bin)


    cam_poses = {}
    for img_data in images_bin.values():
        cam_id_str = os.path.splitext(img_data.name)[0].split('_')[2]
        R = qvec2rotmat(img_data.qvec)
        t = img_data.tvec
        c2w = c2w_from_colmap(R, t)
        cam_poses[cam_id_str] = c2w.tolist()

    train_frames, test_frames = [], []

    cam_data = []

    for cam_id in cam_poses.keys():
        cam_data.append({
            "id": cam_id,
            "transform_matrix": cam_poses[cam_id]
        })

    frames_combined = []
    for i in range(1, total_cams + 1):
        cam_id_str = f"cam{i:02d}"
        cam_dir = os.path.join(source_dir, cam_id_str)
        if not os.path.isdir(cam_dir): continue

        frame_files = sorted([f for f in os.listdir(cam_dir) if f.endswith(('.jpg', '.png'))])
        num_frames = len(frame_files)
        
        for frame_idx, frame_file in enumerate(frame_files):
            new_filename = f"{cam_id_str}/{frame_file}"
            
            frame_data = {
                "file_path": new_filename,
                "cam_id": cam_id_str,
                "time": frame_idx / (num_frames - 1.0) if num_frames > 1 else 0.5,
                "original_frame_id": frame_idx
            }

            frames_combined.append(frame_data)

    base_json = {
        "fl_x": avg_fx, "fl_y": avg_fy, "cx": avg_cx, "cy": avg_cy, "w": w, "h": h
    }

    with open(os.path.join(output_dir, "transforms.json"), "w") as f:
        json.dump({**base_json, "cameras": cam_data, "frames": frames_combined}, f, indent=4)

    # Read event camera data
    event_cameras_extrinsics = read_extrinsics_text(os.path.join(colmap_dir, "event_cameras_extrinsics.txt"))
    event_cameras_intrinsics = read_intrinsics_text(os.path.join(colmap_dir, "event_cameras_intrinsics.txt"))

    avg_fx = 0
    avg_fy = 0

    avg_cx = 0
    avg_cy = 0

    for cam in event_cameras_intrinsics.values():
        avg_fx += cam.params[0]
        avg_fy += cam.params[1]
        avg_cx += cam.params[2]
        avg_cy += cam.params[3]

    avg_fx /= len(event_cameras_intrinsics)
    avg_fy /= len(event_cameras_intrinsics)
    avg_cx /= len(event_cameras_intrinsics)
    avg_cy /= len(event_cameras_intrinsics)

    event_intr = list(event_cameras_intrinsics.values())[0]
    w, h = event_intr.width, event_intr.height

    event_base_json = {
        "fl_x": avg_fx, "fl_y": avg_fy, "cx": avg_cx, "cy": avg_cy, "w": w, "h": h
    }

    event_cam_poses = {}
    for img_data in event_cameras_extrinsics.values():
        cam_id_str = os.path.splitext(img_data.name)[0].split('_')[2]
        R = qvec2rotmat(img_data.qvec)
        t = img_data.tvec
        c2w = c2w_from_colmap(R, t)
        event_cam_poses[cam_id_str] = c2w.tolist()


    event_cam_data = []

    for cam_id in event_cam_poses.keys():
        event_cam_data.append({
            "id": cam_id,
            "transform_matrix": event_cam_poses[cam_id]
        })
    

    with open(os.path.join(output_dir, "transforms_event.json"), "w") as f:
        json.dump({**event_base_json, "cameras": event_cam_data}, f, indent=4)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))