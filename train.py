#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import random
import os, sys
import torch
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import glob
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.scene_utils import render_training_image
from time import time
import copy

try:
    from event_loss import event_loss_call
    EVENT_LOSS_FOUND = True
except ImportError:
    EVENT_LOSS_FOUND = False

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def scene_reconstruction(
    dataset,
    opt,
    hyper,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    gaussians,
    scene,
    stage,
    tb_writer,
    train_iter,
    timer,
):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            return
        if stage in checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

    bg_color = [0, 0, 0] 
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    final_iter = train_iter

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    
    grayscale_train_dataset = scene.getTrainCameras()
    grayscale_cams_list = [grayscale_train_dataset[i] for i in range(len(grayscale_train_dataset))]
    test_dataset = scene.getTestCameras()
    video_dataset = scene.getVideoCameras()

    event_maps = {}
    event_cam_ids = []
    if dataset.use_event:
        print("--------------- Loading Event Data ---------------")
        event_files = glob.glob(os.path.join(dataset.source_path, "events_cam*.pt"))
        if not event_files:
            print(f"\n[WARNING] No event files (events_cam*.pt) found. Disabling event loss.")
            dataset.use_event = False
        else:
            for event_file_path in event_files:
                cam_id = os.path.basename(event_file_path).split('.')[0].replace('events_', '')
                print(f"Loading event data for {cam_id}...")
                event_maps[cam_id] = torch.load(event_file_path).to(device="cuda")
                event_cam_ids.append(cam_id)
    
    batch_size = opt.batch_size
    print("data loading done")
    
    if opt.dataloader:
        train_loader = DataLoader(grayscale_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=list)
        train_loader_iter = iter(train_loader)
    
    for iteration in range(first_iter, final_iter + 1):
        if network_gui.conn is None: network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    viewpoint = video_dataset[iteration % len(video_dataset)]
                    custom_cam.time = viewpoint.time
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage, cam_type=scene.dataset_type)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive): break
            except Exception as e:
                print(f"GUI Error: {e}")
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0: gaussians.oneupSHdegree()

        # --- Data Sampling ---
        try:
            viewpoint_cams = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            viewpoint_cams = next(train_loader_iter)

        event_loss_data = None
        if dataset.use_event and stage == "fine" and grayscale_cams_list and event_cam_ids:
            teacher_cam = random.choice(grayscale_cams_list)
            rand_event_cam_id = random.choice(event_cam_ids)
            event_map = event_maps[rand_event_cam_id]
            num_event_intervals = event_map.shape[0]
            if num_event_intervals > 5:
                start_event_idx = random.randint(0, num_event_intervals - 5)
                gt_event_bins = event_map[start_event_idx : start_event_idx + 5]
                start_time_normalized = start_event_idx / float(num_event_intervals)
                event_loss_data = {
                    "teacher_cam": teacher_cam,
                    "start_time": start_time_normalized,
                    "gt_event_bins": gt_event_bins
                }
        
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        # --- Rendering and Loss Calculation ---
        images, gt_images, radii_list, visibility_filter_list, vps_list = [], [], [], [], []

        # 1. Render for Image Reconstruction Loss
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type)
            images.append(render_pkg["render"].unsqueeze(0))
            gt_images.append(viewpoint_cam.original_image.cuda().unsqueeze(0))
            radii_list.append(render_pkg["radii"])
            visibility_filter_list.append(render_pkg["visibility_filter"])
            vps_list.append(render_pkg["viewspace_points"])

        # 2. Render for Event Loss (if active)
        if event_loss_data is not None:
            sub_frame_imgs_for_event_loss = []
            time_interval = 1.0 / 30.0 # Approximate frame time
            for i in range(5):
                cam = copy.deepcopy(event_loss_data["teacher_cam"])
                time_offset = i * (time_interval / 5.0)
                cam.time = torch.tensor(event_loss_data["start_time"] + time_offset)
                render_pkg_sub = render(cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type)
                sub_frame_imgs_for_event_loss.append(render_pkg_sub["render"])
                vps_list.append(render_pkg_sub["viewspace_points"]) # Accumulate for gradients

        # --- Loss Aggregation ---
        image_tensor = torch.cat(images, 0)
        gt_image_tensor = torch.cat(gt_images, 0)
        
        
        Ll1 = l1_loss(image_tensor, gt_image_tensor)
        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        loss = (1.0 - opt.lambda_dssim) * Ll1
        if opt.lambda_dssim != 0:
            loss += opt.lambda_dssim * (1.0 - ssim(image_tensor, gt_image_tensor.repeat(1,3,1,1)))

        total_event_loss = torch.tensor(0.0, device="cuda")
        if event_loss_data is not None:
            combination = [[i, j] for i in range(4) for j in range(i + 1, 5)]
            event_loss = event_loss_call(
                sub_frame_imgs_for_event_loss, event_loss_data["gt_event_bins"], combination,
                event_loss_data["teacher_cam"].image_height, event_loss_data["teacher_cam"].image_width,
                iteration, event_loss_data["teacher_cam"].uid
            )
            total_event_loss = event_loss * dataset.lambda_event
            loss += total_event_loss

        if stage == "fine" and hyper.time_smoothness_weight != 0:
            tv_loss = gaussians.compute_regulation(
                hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight
            )
            loss += tv_loss

        loss.backward()
        if stage == "coarse" and iteration % 10 == 0 and iteration > 0:
            print("\n--- COARSE STAGE GRADIENT CHECK ---")
            xyz_grad = gaussians._xyz.grad
            if xyz_grad is not None:
                print(f"XYZ Grads: norm={xyz_grad.norm().item():.6f}, max={xyz_grad.abs().max().item():.6f}")
            else:
                print("XYZ Grads: NONE")

            scaling_grad = gaussians._scaling.grad
            if scaling_grad is not None:
                print(f"Scaling Grads: norm={scaling_grad.norm().item():.6f}, max={scaling_grad.abs().max().item():.6f}")
            else:
                print("Scaling Grads: NONE")

            opacity_grad = gaussians._opacity.grad
            if opacity_grad is not None:
                print(f"Opacity Grads: norm={opacity_grad.norm().item():.6f}, max={opacity_grad.abs().max().item():.6f}")
            else:
                print("Opacity Grads: NONE")
        # --- Gradient and Densification Stat Accumulation ---
        viewspace_point_tensor_grad = torch.zeros_like(vps_list[0])
        for grad_tensor in vps_list:
            if grad_tensor.grad is not None:
                viewspace_point_tensor_grad += grad_tensor.grad
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_.item() + 0.6 * ema_psnr_for_log
            total_point = gaussians.get_xyz.shape[0]

            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}", "PSNR": f"{psnr_:.{2}f}", "Points": f"{total_point}"}
                if dataset.use_event and total_event_loss.item() > 0:
                    postfix["EventLoss"] = f"{total_event_loss.item():.{5}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == final_iter:
                progress_bar.close()

            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type, total_event_loss.item())
            
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, stage)
                
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) or \
                   (iteration < 3000 and iteration % 50 == 49) or \
                   (iteration < 60000 and iteration % 100 == 99):
                    test_cam_to_render = test_dataset[iteration % len(test_dataset)]
                    train_cam_to_render = grayscale_train_dataset[iteration % len(grayscale_train_dataset)]
                    render_training_image(scene, gaussians, [test_cam_to_render], render, pipe, background, stage + "test", iteration, timer.get_elapsed_time(), scene.dataset_type)
                    render_training_image(scene, gaussians, [train_cam_to_render], render, pipe, background, stage + "train", iteration, timer.get_elapsed_time(), scene.dataset_type)
            timer.start()

            # --- Densification ---
            if iteration < opt.densify_until_iter:
                # Use the visibility filter from the supervised grayscale views
                visibility_filter = torch.stack(visibility_filter_list).any(dim=0)
                if len(visibility_filter.shape) > 1: visibility_filter = visibility_filter.squeeze()
                
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], 
                    torch.stack(radii_list).max(dim=0).values.squeeze()[visibility_filter]
                )
                
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
                
                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration * (opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after) / opt.densify_until_iter
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration * (opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after) / opt.densify_until_iter
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                
                if iteration % opt.densification_interval == 0 and opt.add_point:
                    gaussians.grow(5, 5, scene.model_path, iteration, stage)
                
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
            
            if iteration < final_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt_{stage}_{iteration}.pth")

def training(
    dataset,
    hyper,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    expname,
):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians)
    timer.start()
    scene_reconstruction(
        dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
        checkpoint_iterations, checkpoint, debug_from, gaussians, scene,
        "coarse", tb_writer, opt.coarse_iterations, timer
    )
    scene_reconstruction(
        dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
        checkpoint_iterations, checkpoint, debug_from, gaussians, scene,
        "fine", tb_writer, opt.iterations, timer
    )

def prepare_output_and_logger(expname):
    if not args.model_path:
        unique_str = expname if expname else str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss, 
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    stage,
    dataset_type,
    event_loss=0.0,
):
    if tb_writer:
        tb_writer.add_scalar(f"{stage}/train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar(f"{stage}/train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar(f"{stage}/iter_time", elapsed, iteration)
        if event_loss > 0:
            tb_writer.add_scalar(f'{stage}/train_loss_patches/event_loss', event_loss, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        test_cameras = scene.getTestCameras()
        train_cameras = scene.getTrainCameras()
        # To avoid OOM, sample a smaller number of validation views
        num_val_samples = 5
        test_indices = np.linspace(0, len(test_cameras) - 1, num_val_samples, dtype=int)
        train_indices = np.linspace(0, len(train_cameras) - 1, num_val_samples, dtype=int)

        validation_configs = (
            {"name": "test", "cameras": [test_cameras[i] for i in test_indices]},
            {"name": "train", "cameras": [train_cameras[i] for i in train_indices]},
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test, psnr_test = 0.0, 0.0
                for idx, viewpoint in enumerate(tqdm(config["cameras"], desc=f"Evaluating {config['name']}")):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, stage=stage, cam_type=dataset_type)
                    image_rgb = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    image_for_metric = image_rgb[0:1, :, :] * 0.299 + \
                                       image_rgb[1:2, :, :] * 0.587 + \
                                       image_rgb[2:3, :, :] * 0.114
                    
                    if tb_writer and (idx < 5):
                        try:
                            tb_writer.add_images(f"{stage}/{config['name']}_view_{viewpoint.image_name}/render", image_rgb[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                gt_log = gt_image.repeat(3, 1, 1) if gt_image.shape[0] == 1 else gt_image
                                tb_writer.add_images(f"{stage}/{config['name']}_view_{viewpoint.image_name}/ground_truth", gt_log[None], global_step=iteration)
                        except Exception as e:
                            print(f"Tensorboard logging error: {e}")
                            
                    l1_test += l1_loss(image_for_metric, gt_image).mean().double()
                    psnr_test += psnr(image_for_metric, gt_image).mean().double()

                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.4f} PSNR {psnr_test:.2f}")
                
                if tb_writer:
                    tb_writer.add_scalar(f"{stage}/{config['name']}/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{stage}/{config['name']}/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            if stage == "fine": # Only log dynamic properties in fine stage
                tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum() / scene.gaussians.get_xyz.shape[0], iteration)
                tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1) / 100, iteration, max_bins=500)
        torch.cuda.empty_cache()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000, 7000, 14000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[14000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    
    args = parser.parse_args(sys.argv[1:])
    
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    if args.coarse_iterations not in args.save_iterations:
        args.save_iterations.append(args.coarse_iterations)

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args),
        hp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.expname,
    )

    print("\nTraining complete.")