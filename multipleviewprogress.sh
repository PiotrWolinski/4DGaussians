workdir=$1

zeros="00"

python scripts/extractimages.py multipleview/$workdir
cp ./data/multipleview/$workdir/database.db ./colmap_tmp/database.db


# Creating images list for feature extraction
mkdir ./colmap_tmp/images_lists
for i in $(seq 1 14); do
  temp_num="$zeros$i"
  cam_id=${temp_num:(-2)}
  echo "frame_00001_cam${cam_id}.jpg" > ./colmap_tmp/images_lists/cam${cam_id}.txt
done


# Extract frames from each camera individually to use its specific parameters
for i in $(seq 1 14); do
  temp_num="$zeros$i"
  cam_id=${temp_num:(-2)}
  echo "Parsing cam${cam_id}"
  colmap feature_extractor \
    --database_path ./colmap_tmp/database.db \
    --image_path ./colmap_tmp/images \
    --image_list_path ./colmap_tmp/images_lists/cam${cam_id}.txt \
    --ImageReader.existing_camera_id $i \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.default_focal_length_factor 0.63 \
    --SiftExtraction.max_image_size 4096 \
    --SiftExtraction.max_num_features 16384 \
    --SiftExtraction.estimate_affine_shape 1 \
    --SiftExtraction.domain_size_pooling 1
done

# Match extracted features between different views
colmap exhaustive_matcher \
  --database_path ./colmap_tmp/database.db \
  --SiftMatching.guided_matching 1

# Copy pose priors obtained from kalibr
mkdir ./colmap_tmp/sparse
cp -r ./data/multipleview/$workdir/sparse_manual/ ./colmap_tmp/sparse_manual/

# Convert kalibr calibration into binary format
mkdir ./colmap_tmp/sparse_manual_bin
colmap model_converter --input_path ./colmap_tmp/sparse_manual/ --output_path ./colmap_tmp/sparse_manual_bin/ --output_type BIN

# Triangulate 3D points from manual reconstruction
colmap point_triangulator \
  --database_path ./colmap_tmp/database.db \
  --image_path ./colmap_tmp/images \
  --input_path ./colmap_tmp/sparse_manual_bin \
  --output_path ./colmap_tmp/sparse

colmap mapper \
    --database_path ./colmap_tmp/database.db \
    --input_path ./colmap_tmp/sparse \
    --image_path ./colmap_tmp/images \
    --output_path ./colmap_tmp/sparse \
    --Mapper.ba_local_max_num_iterations=100 \
    --Mapper.ba_global_max_num_iterations=200 \
    --Mapper.init_num_trials=500  \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_extra_params 0 \
    --Mapper.fix_existing_images 1 \

mkdir ./colmap_tmp/sparse/0
mv ./colmap_tmp/sparse/cameras.bin ./colmap_tmp/sparse/0/
mv ./colmap_tmp/sparse/images.bin ./colmap_tmp/sparse/0/
mv ./colmap_tmp/sparse/points3D.bin ./colmap_tmp/sparse/0/

mkdir ./data/multipleview/$workdir/sparse_
cp -r ./colmap_tmp/sparse/0/* ./data/multipleview/$workdir/sparse_

mkdir ./colmap_tmp/dense
colmap image_undistorter \
  --image_path ./colmap_tmp/images \
  --input_path ./colmap_tmp/sparse/0 \
  --output_path ./colmap_tmp/dense \
  --output_type COLMAP
colmap patch_match_stereo \
  --workspace_path ./colmap_tmp/dense \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency=true \
  --PatchMatchStereo.window_radius=7 \
  --PatchMatchStereo.num_iterations=10

colmap stereo_fusion \
  --workspace_path ./colmap_tmp/dense \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path ./colmap_tmp/dense/fused.ply

python scripts/downsample_point.py ./colmap_tmp/dense/fused.ply ./data/multipleview/$workdir/points3D_multipleview.ply

# git clone https://github.com/Fyusion/LLFF.git
pip install scikit-image
python LLFF/imgs2poses.py ./colmap_tmp/

cp ./colmap_tmp/poses_bounds.npy ./data/multipleview/$workdir/poses_bounds_multipleview.npy

# rm -rf ./colmap_tmp
# rm -rf ./LLFF
