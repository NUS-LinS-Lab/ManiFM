import torch
import torch.nn as nn
import os
import sys
sys.path.append(".")
sys.path.append("..")
from dataset import RigidBodyDataset, ClothDataset, MPMDataset
from torch.utils.data import DataLoader, ConcatDataset
from networks import ManiFM
import numpy as np
import random
from omegaconf import OmegaConf
from safetensors.torch import load_model
from pathlib import Path
import trimesh
import viser
from utils.pred_utils import (
    load_hand_pointcloud_and_normals,
    load_object_point_cloud_and_normal_rigid,
    load_object_point_cloud_and_normal_clothes,
    load_object_point_cloud_and_normal_mpm,
    vis_pc_heatmap,
    vis_vector,
    vis_pc_heatmap_trimesh_pc,
    vis_pc_trimesh,
    point_cloud_nms,
    pointcloud_motion_to_wrench,
)
# from utils.utils import from_wrench_to_contact_force

def set_seed(num=666):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)


def load_dataset_list(cfg):
    '''
    remember to add new args if update the config file'''
    dataset_list = []
    info_list = []
    if "rigid_body" in cfg.dir.data_dir.keys():
        if OmegaConf.is_list(cfg.dir.data_dir.rigid_body):
            try:
                if len(cfg.dir.data_dir.rigid_body) > 0:
                    rigid_body_dataset = RigidBodyDataset(pkl_file_directory=cfg.dir.data_dir.rigid_body, load_ratio=cfg.dir.load_ratio, generate_new_wrench=True, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, num_object=cfg.dir.num_object, num_palm_pose=cfg.dir.num_palm_pose, num_motion=cfg.dir.num_motion, use_scale=cfg.dir.use_scale,  remove_pc_num=cfg.dir.remove_pc_num, remove_pc_prob=cfg.dir.remove_pc_prob, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
                    dataset_list.append(rigid_body_dataset)
                    info_list.append(f"[info] Rigid body dataset size: {len(rigid_body_dataset)}")
            except:
                info_list.append(f"Missing rigid_body dataset")
        if OmegaConf.is_dict(cfg.dir.data_dir.rigid_body):
            try:
                if len(cfg.dir.data_dir.rigid_body.force_closure.path) > 0:
                    force_closure_dataset = RigidBodyDataset(pkl_file_directory=cfg.dir.data_dir.rigid_body.force_closure.path, load_ratio=cfg.dir.load_ratio, generate_new_wrench=False, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, num_object=cfg.dir.data_dir.rigid_body.force_closure.num_object, num_palm_pose=cfg.dir.data_dir.rigid_body.force_closure.num_palm_pose, num_motion=cfg.dir.data_dir.rigid_body.force_closure.num_motion, use_scale=cfg.dir.use_scale,  remove_pc_num=cfg.dir.remove_pc_num, remove_pc_prob=cfg.dir.remove_pc_prob, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
                    dataset_list.append(force_closure_dataset)
                    info_list.append(f"[info] Rigid body/Force closure dataset size: {len(force_closure_dataset)}")
            except:
                info_list.append(f"Missing force_closure dataset")
                
            try:
                if len(cfg.dir.data_dir.rigid_body.leap_hand.path) > 0:
                    leap_hand_dataset = RigidBodyDataset(pkl_file_directory=cfg.dir.data_dir.rigid_body.leap_hand.path, load_ratio=cfg.dir.load_ratio, generate_new_wrench=True, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, num_object=cfg.dir.data_dir.rigid_body.leap_hand.num_object, num_palm_pose=cfg.dir.data_dir.rigid_body.leap_hand.num_palm_pose, num_motion=cfg.dir.data_dir.rigid_body.leap_hand.num_motion, use_scale=cfg.dir.use_scale, remove_pc_num=cfg.dir.remove_pc_num, remove_pc_prob=cfg.dir.remove_pc_prob, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
                    dataset_list.append(leap_hand_dataset)
                    info_list.append(f"[info] Rigid body/Leap Hand dataset size: {len(leap_hand_dataset)}")
            except:
                info_list.append(f"Missing leap_hand dataset")
            
            try:
                if len(cfg.dir.data_dir.rigid_body.kinova3f_hand.path) > 0:
                    kinova3f_hand_dataset = RigidBodyDataset(pkl_file_directory=cfg.dir.data_dir.rigid_body.kinova3f_hand.path, load_ratio=cfg.dir.load_ratio, generate_new_wrench=True, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, num_object=cfg.dir.data_dir.rigid_body.kinova3f_hand.num_object, num_palm_pose=cfg.dir.data_dir.rigid_body.kinova3f_hand.num_palm_pose, num_motion=cfg.dir.data_dir.rigid_body.kinova3f_hand.num_motion, use_scale=cfg.dir.use_scale,  remove_pc_num=cfg.dir.remove_pc_num, remove_pc_prob=cfg.dir.remove_pc_prob, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
                    dataset_list.append(kinova3f_hand_dataset)
                    info_list.append(f"[info] Rigid body/Kinova3f hand dataset size: {len(kinova3f_hand_dataset)}")
            except:
                info_list.append(f"Missing kinova3f_hand dataset")
                
            try:
                if len(cfg.dir.data_dir.rigid_body.panda_hand.path) > 0:
                    panda_hand_dataset = RigidBodyDataset(pkl_file_directory=cfg.dir.data_dir.rigid_body.panda_hand.path, load_ratio=cfg.dir.load_ratio, generate_new_wrench=True, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, num_object=cfg.dir.data_dir.rigid_body.panda_hand.num_object, num_palm_pose=cfg.dir.data_dir.rigid_body.panda_hand.num_palm_pose, num_motion=cfg.dir.data_dir.rigid_body.panda_hand.num_motion, use_scale=cfg.dir.use_scale, remove_pc_num=cfg.dir.remove_pc_num, remove_pc_prob=cfg.dir.remove_pc_prob, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
                    dataset_list.append(panda_hand_dataset)
                    info_list.append(f"[info] Rigid body/Panda hand dataset size: {len(panda_hand_dataset)}")
            except:
                info_list.append(f"Missing panda_hand dataset")
    
    # try:
    if len(cfg.dir.data_dir.cloth) > 0:
        cloth_dataset = ClothDataset(cloth_file_directory=cfg.dir.data_dir.cloth, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, use_scale=cfg.dir.use_scale, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
        dataset_list.append(cloth_dataset)
        info_list.append(f"[info] Cloth dataset size: {len(cloth_dataset)}")
    # except:
    #     info_list.append(f"Missing cloth dataset")
        
    # try:
    if len(cfg.dir.data_dir.mpm) > 0:
        mpm_dataset = MPMDataset(pkl_file_directory=cfg.dir.data_dir.mpm, load_ratio=1.0, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, use_scale=cfg.dir.use_scale, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
        dataset_list.append(mpm_dataset)
        info_list.append(f"[info] Mpm dataset size: {len(mpm_dataset)}")
    # except:
    #     info_list.append(f"Missing mpm dataset")

    try:
        if len(cfg.dir.data_dir.rope) > 0:
            rope_dataset = MPMDataset(pkl_file_directory=cfg.dir.data_dir.rope, load_ratio=1.0, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, use_scale=cfg.dir.use_scale, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
            dataset_list.append(rope_dataset)
            info_list.append(f"[info] Rope dataset size: {len(rope_dataset)}")
    except:
        info_list.append(f"Missing Rope dataset")
    return dataset_list, info_list

if __name__ == "__main__":
    config_file_path = "configs/vis.json"
    cfg = OmegaConf.load(config_file_path)
    set_seed(cfg.train.seed)
    dataset_list, info_list = load_dataset_list(cfg)
    print(info_list)
    test_dataset = ConcatDataset(dataset_list)

    device = "cuda"
    model = ManiFM(cfg.model, device)
    
    gpu_number = 0
    checkpoint_path = "checkpoints/200k_204epc_model.safetensors"
    pred_time = 16
    
    device = torch.device(f"cuda:{gpu_number}")
    model = ManiFM(cfg.model, device).cuda(gpu_number)
    load_model(model, str(checkpoint_path))
    model.eval()

    server = viser.ViserServer(port=8018)
    data_idx_slider = server.add_gui_slider("data_idx", 0, len(test_dataset)-1, step=1, initial_value=0)

    @data_idx_slider.on_update
    def _(_) -> None:
        """Callback for when a file is uploaded."""
        data_idx = data_idx_slider.value
        data = test_dataset[data_idx]
        with torch.no_grad():
            predicted_contact_points_heatmap, contact_forces = model.infer(
                data["input_hand_point_cloud"].unsqueeze(0).repeat(pred_time, 1, 1,).to(device), 
                data["input_object_point_cloud"].unsqueeze(0).repeat(pred_time, 1, 1,).to(device))
        
            pred_contact_points_heatmap = predicted_contact_points_heatmap.detach().cpu().numpy()  # shape=(b, 2048, 1)
            pred_contact_force_map = contact_forces.detach().cpu().numpy()  # shape=(b, 2048, 3)

        data_point_cloud = data["input_object_point_cloud"][:, :3].detach().cpu().numpy() # shape=(2048, 3)
        
        to_show_cnt = 0
        success_pred_dict = {}

        for i in range(pred_time):
            normalized_hmap = (pred_contact_points_heatmap[i, :, 0] - pred_contact_points_heatmap[i, :, 0].min()) / (pred_contact_points_heatmap[i, :, 0].max() - pred_contact_points_heatmap[i, :, 0].min())

            filter_radius = 0.3
            
            filtered_points, keep_idx = point_cloud_nms(
                data_point_cloud,
                pred_contact_points_heatmap[i, :, 0],
                filter_radius,
            )
            
            if len(keep_idx) == 0 or len(keep_idx) > 4 or np.average(normalized_hmap) > 1.0 or np.percentile(normalized_hmap, 50) > 1.0: 
                # if too uncertain (too much deviation) then we skip 
                print(len(keep_idx), np.average(normalized_hmap))
                continue


            vis_pc_heatmap(
                server,
                data_point_cloud,
                pred_contact_points_heatmap[i, :, 0],
                strid=f"pc_hmap_pred_{i}",
                trans=[0.0, (to_show_cnt + 1) * 2.0, 0.0],
                radius=0.03,
            )
            force_mesh_list = []
            for idx, pid in enumerate(keep_idx):
                if pid != 2048:
                    contact_point = data_point_cloud[pid]
                    force_vector = pred_contact_force_map[i, pid]
                    force_mesh_list.append(vis_vector(contact_point, force_vector, length=0.3, cyliner_r=0.03, color=[255, 255, 20, 255]))
            server.add_mesh_trimesh(
                f"pred_force_mesh_{i}",
                trimesh.Scene(force_mesh_list)
                .dump(concatenate=True)
                .apply_translation([0.0, (to_show_cnt + 1) * 2.0, 0.0]),
            )
            to_show_cnt += 1

        server.add_point_cloud(
            "/begin_points",
            points=data_point_cloud,
            point_size=0.02,
            point_shape="circle",
            colors=(159, 196, 255),
        )
        motion_vector_mesh_list = []
        for p1, m in zip(data_point_cloud, data["input_object_point_cloud"][:, 7:10].detach().cpu().numpy()):
            motion_vector_mesh_list.append(
                vis_vector(p1, m, length=np.linalg.norm(m), color=[255, 100, 255, 245], cyliner_r=0.002)
            )
        server.add_mesh_trimesh(
            "motion_vector_mesh_list",
            trimesh.Scene(motion_vector_mesh_list).dump(concatenate=True),
        )
    input()