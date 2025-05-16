workdir=$1

rm -rf ./colmap_tmp
rm -rf ./data/multipleview/$workdir/sparse_
rm ./data/multipleview/$workdir/points3D_multipleview.ply
rm ./data/multipleview/$workdir/poses_bounds_multipleview.npy