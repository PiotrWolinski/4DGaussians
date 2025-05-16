workdir=$1

python scripts/extractimages.py multipleview/$workdir
cp ./data/multipleview/$workdir/database.db ./colmap_tmp/database.db

zeros="00"

# Extract frames from each camera individually to use its specific parameters
for i in $(seq 1 14);
do
    temp_num="$zeros$i"
    cam_id=${temp_num:(-2)}
    echo "Parsing cam${cam_id}"
    colmap feature_extractor \
        --database_path ./colmap_tmp/database.db \
        --image_path ./colmap_tmp/images/cam${cam_id} \
        --ImageReader.existing_camera_id $i \
        --ImageReader.camera_model OPENCV \
        --SiftExtraction.max_image_size 4096 \
        --SiftExtraction.max_num_features 16384 \
        --SiftExtraction.estimate_affine_shape 1 \
        --SiftExtraction.domain_size_pooling 1
done

colmap exhaustive_matcher \
    --database_path ./colmap_tmp/database.db \
    --SiftMatching.guided_matching 1 \

mkdir ./colmap_tmp/sparse
colmap mapper \
    --database_path ./colmap_tmp/database.db \
    --image_path ./colmap_tmp/images \
    --output_path ./colmap_tmp/sparse \
    --Mapper.ba_local_max_num_iterations=100 \
    --Mapper.ba_global_max_num_iterations=200 \
    --Mapper.init_num_trials=500  \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_extra_params 0
    # --Mapper.ba_global_function_tolerance=0.000001 \
    # --Mapper.min_num_matches=8 \

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
    # --PatchMatchStereo.filter_geom_consistency_max_cost=2 \
    # --PatchMatchStereo.filter_min_ncc=0.05 \
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

