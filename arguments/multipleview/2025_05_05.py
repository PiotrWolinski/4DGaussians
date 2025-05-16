ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 1]
    },
    multires = [1],
    defor_depth = 1,
    net_width = 192,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.005,
    l1_time_planes =  0.00015,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    empty_voxel=False,
    render_process=True,
    static_mlp=False
)

OptimizationParams = dict(
    dataloader=True,
    iterations = 20_000,
    batch_size=1,
    coarse_iterations = 15_000,
    densify_until_iter = 12_000,
    opacity_reset_interval = 8000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    # pruning_interval = 2000
)

# ModelParams = dict(
#     extension=".jpg",
#     add_points=True
# )

# ModelHiddenParams = dict(
#     kplanes_config = {
#      'grid_dimensions': 2,
#      'input_coordinate_dim': 4,
#      'output_coordinate_dim': 16,
#      'resolution': [64, 64, 64, 150]
#     },
#     multires = [1], # [1,2,4]
#     defor_depth = 1,
#     net_width = 128,
#     plane_tv_weight = 0.0002,
#     time_smoothness_weight = 0.005,
#     l1_time_planes =  0.001,
#     render_process=True
# )
# OptimizationParams = dict(
#     # dataloader=True,
#     iterations = 500, # 14_000
#     batch_size=1,
#     coarse_iterations = 2_000, # 3_000
#     densify_until_iter = 10_000,
#     opacity_reset_interval = 300000,
#     grid_lr_init = 0.5, # 0.0016
#     grid_lr_final = 16,
#     opacity_threshold_coarse = 0.01,
#     opacity_threshold_fine_init = 0.005,
#     opacity_threshold_fine_after = 0.005,
#     pruning_interval = 2000
# )