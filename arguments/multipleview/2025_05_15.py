ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 10]
    },
    multires = [1,2],
    defor_depth = 2,
    net_width = 192,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.02,
    l1_time_planes =  0.0001,
    weight_decay_iteration=0,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    render_process=True,
    static_mlp=False
)

OptimizationParams = dict(
    dataloader=False,
    iterations = 20_000,
    batch_size=2,
    coarse_iterations = 15_000,
    densify_until_iter = 12_000,
    opacity_reset_interval = 8000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    # pruning_interval = 2000
)