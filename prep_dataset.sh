#!/bin/bash

# Stop on first error
set -e

# --- Configuration ---
# The name of your dataset directory inside data/multipleview/
DATASET_NAME="camera"

# The name of the final unified dataset directory that will be created
UNIFIED_DATASET_NAME="unified_${DATASET_NAME}"

# Path to the source and destination directories
SOURCE_DIR="data/multipleview/${DATASET_NAME}"
OUTPUT_DIR="data/${UNIFIED_DATASET_NAME}"
COLMAP_WORKDIR="colmap_tmp"

# Camera configuration
NUM_GRAYSCALE_CAMS=3
NUM_EVENT_CAMS=0
TOTAL_CAMS=$((NUM_GRAYSCALE_CAMS + NUM_EVENT_CAMS))

# --- Main Script ---

echo "--- Starting Dataset Preparation for [${DATASET_NAME}] ---"

# 1. Setup and Cleanup
echo ">> Step 1: Cleaning up and creating directories..."
rm -rf "${COLMAP_WORKDIR}"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${COLMAP_WORKDIR}/images"
mkdir -p "${OUTPUT_DIR}/images"

# 2. Extract Initial Frames for COLMAP
# We take the first frame from each camera to build the initial 3D model.
echo ">> Step 2: Extracting initial frames for COLMAP..."
for i in $(seq 1 ${TOTAL_CAMS}); do
  CAM_ID=$(printf "cam%02d" ${i})
  CAM_DIR="${SOURCE_DIR}/${CAM_ID}"
  
  if [ -d "${CAM_DIR}" ]; then
    # Find the first frame (assuming .jpg or .png)
    FIRST_FRAME=$(find "${CAM_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) | sort | head -n 1)
    if [ -f "${FIRST_FRAME}" ]; then
      # Copy and rename for COLMAP to avoid name conflicts
      FILENAME=$(basename "${FIRST_FRAME}")
      EXTENSION="${FILENAME##*.}"
      cp "${FIRST_FRAME}" "${COLMAP_WORKDIR}/images/${CAM_ID}_initial.${EXTENSION}"
    else
      echo "Warning: No image files found in ${CAM_DIR}"
    fi
  else
    echo "Warning: Camera directory not found: ${CAM_DIR}"
  fi
done

# 3. Run COLMAP to get poses
echo ">> Step 3: Running COLMAP to generate camera poses..."
DB_PATH="${COLMAP_WORKDIR}/database.db"
IMG_PATH="${COLMAP_WORKDIR}/images"
SPARSE_PATH="${COLMAP_WORKDIR}/sparse"
mkdir -p "${SPARSE_PATH}"

colmap feature_extractor --database_path "${DB_PATH}" --image_path "${IMG_PATH}"
colmap exhaustive_matcher --database_path "${DB_PATH}"
colmap mapper --database_path "${DB_PATH}" --image_path "${IMG_PATH}" --output_path "${SPARSE_PATH}"

# Check if COLMAP was successful
if [ ! -d "${SPARSE_PATH}/0" ]; then
    echo "FATAL: COLMAP reconstruction failed. No model was generated in ${SPARSE_PATH}."
    exit 1
fi

# 4. Generate JSON from COLMAP output using a helper Python script
# This is the step where a Python script is almost unavoidable because parsing binary files
# and constructing complex JSON is very difficult and error-prone in pure Bash.
echo ">> Step 4: Converting COLMAP model to transforms.json..."

# We will create a small, dedicated Python script for this conversion.
# Create the python helper script on the fly
cat << 'EOF' > colmap_to_json.py
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
EOF

# Now run the helper script
python colmap_to_json.py "${SPARSE_PATH}" "${SOURCE_DIR}" "${OUTPUT_DIR}" "${TOTAL_CAMS}"
rm colmap_to_json.py # Clean up the helper script

# 5. Copy all files to the unified directory
echo ">> Step 5: Populating final unified dataset directory..."
# Copy images
for i in $(seq 1 ${TOTAL_CAMS}); do
  CAM_ID=$(printf "cam%02d" ${i})
  CAM_DIR="${SOURCE_DIR}/${CAM_ID}"
  if [ -d "${CAM_DIR}" ]; then
    for FRAME_FILE in $(find "${CAM_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \)); do
      FILENAME=$(basename "${FRAME_FILE}")
      cp "${FRAME_FILE}" "${OUTPUT_DIR}/images/${CAM_ID}_${FILENAME}"
    done
  fi
done

# Copy point cloud
cp "${SPARSE_PATH}/0/points3D.ply" "${OUTPUT_DIR}/points3D.ply"

# Copy event files
echo "Copying event files..."
for i in $(seq 1 ${NUM_EVENT_CAMS}); do
    EVENT_CAM_IDX=$((NUM_GRAYSCALE_CAMS + i))
    EVENT_CAM_ID=$(printf "cam%02d" ${EVENT_CAM_IDX})
    # Assuming event files are in the root of the source directory
    EVENT_FILE="events_${EVENT_CAM_ID}.pt"
    if [ -f "${SOURCE_DIR}/${EVENT_FILE}" ]; then
        cp "${SOURCE_DIR}/${EVENT_FILE}" "${OUTPUT_DIR}/${EVENT_FILE}"
    else
        echo "Warning: Event file not found: ${SOURCE_DIR}/${EVENT_FILE}"
    fi
done

# 6. Cleanup
echo ">> Step 6: Cleaning up temporary files..."
rm -rf "${COLMAP_WORKDIR}"

echo "--- Dataset preparation complete! ---"
echo "Unified dataset created at: ${OUTPUT_DIR}"