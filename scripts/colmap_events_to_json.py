import os
import sys
import json
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scene.colmap_loader import read_intrinsics_text, read_extrinsics_text, qvec2rotmat


def c2w_from_colmap(R, t):
    """
    Convert COLMAP camera extrinsics (rotation and translation) to camera-to-world matrix.
    """
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1:3] *= -1  # OpenCV to OpenGL coordinate system conversion
    return c2w


def create_event_transforms(colmap_dir, output_dir):
    print(f"Reading event camera data from: {colmap_dir}")

    # Read intrinsics and extrinsics from the provided .txt files
    event_cameras_intrinsics = read_intrinsics_text(os.path.join(colmap_dir, "event_cameras_intrinsics.txt"))
    event_cameras_extrinsics = read_extrinsics_text(os.path.join(colmap_dir, "event_cameras_extrinsics.txt"))

    # Calculate average intrinsics
    avg_fx, avg_fy, avg_cx, avg_cy = 0, 0, 0, 0
    for cam in event_cameras_intrinsics.values():
        avg_fx += cam.params[0]
        avg_fy += cam.params[1]
        avg_cx += cam.params[2]
        avg_cy += cam.params[3]

    avg_fx /= len(event_cameras_intrinsics)
    avg_fy /= len(event_cameras_intrinsics)
    avg_cx /= len(event_cameras_intrinsics)
    avg_cy /= len(event_cameras_intrinsics)

    # Get width and height from the first camera
    event_intr = list(event_cameras_intrinsics.values())[0]
    w, h = event_intr.width, event_intr.height

    # Prepare base JSON data for intrinsics
    event_base_json = {
        "fl_x": avg_fx,
        "fl_y": avg_fy,
        "cx": avg_cx,
        "cy": avg_cy,
        "w": w,
        "h": h
    }

    # Process extrinsics to create camera poses
    event_cam_poses = {}
    for img_data in event_cameras_extrinsics.values():
        cam_id_str = os.path.splitext(img_data.name)[0].split('_')[2]
        R = qvec2rotmat(img_data.qvec)
        t = img_data.tvec
        c2w = c2w_from_colmap(R, t)
        event_cam_poses[cam_id_str] = c2w.tolist()

    # Prepare camera data for JSON
    event_cam_data = []
    for cam_id in event_cam_poses.keys():
        event_cam_data.append({
            "id": cam_id,
            "transform_matrix": event_cam_poses[cam_id]
        })

    # Write the transforms_events.json file
    with open(os.path.join(output_dir, "transforms_events.json"), "w") as f:
        json.dump({**event_base_json, "cameras": event_cam_data}, f, indent=4)

    print(f"--- Success! Created transforms_events.json in {output_dir} ---")


if __name__ == "__main__":
    colmap_dir = sys.argv[1]  # Path to the directory containing .txt files
    output_dir = sys.argv[2]  # Path to the output directory

    create_event_transforms(colmap_dir, output_dir)