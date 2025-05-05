import os
import sys
import shutil

from pathlib import Path

folder_path = sys.argv[1]

colmap_path = "./colmap_tmp"
images_path = os.path.join(colmap_path, "images")
os.makedirs(images_path, exist_ok=True)

dir1=os.path.join("data",folder_path)
for camera_id in sorted(os.listdir(dir1)):
    camera_dir = os.path.join(dir1, camera_id)
    if Path(camera_dir).is_dir():
        for file_name in os.listdir(camera_dir):
            if file_name.startswith("frame_00001"):
                src_path = os.path.join(camera_dir, file_name)
                dst_dir = os.path.join(images_path, camera_id)
                os.mkdir(dst_dir)
                dst_path = os.path.join(dst_dir, f"frame_00001.jpg")
                shutil.copyfile(src_path, dst_path) 

print("Done copying inital frames for 3D structure initialization.")
