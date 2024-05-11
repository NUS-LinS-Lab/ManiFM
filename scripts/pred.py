import torch
import torch.nn as nn
import os
import sys
sys.path.append(".")
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
    point_cloud_nms,
    pointcloud_motion_to_wrench,
)
from utils.data_utils import from_wrench_to_contact_force

def set_seed(num=666):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)


demo_configs = {
    "microwave": {
        "object_type": "rigid",
    },
    "rope": {
        "object_type": "mpm",
    },
    "Tshirt": {
        "object_type": "clothes",
    },
    "plasticine": {
        "object_type": "mpm",
    },
}

if __name__ == "__main__":
    config_file_path = "configs/test.json"
    cfg = OmegaConf.load(config_file_path)
    set_seed(cfg.train.seed)
    
    demo_name = "rope_mpm"
    gpu_number = 0
    checkpoint_path = "checkpoints/200k_204epc_model.safetensors"
    pred_time = 10
    
    # if demo_name not in demo_configs:
    #     raise ValueError(f"Unknown demo name: {demo_name}")
    
    pp_file_path = Path(__file__).parent.parent
    dataset_directory_path = Path.joinpath(pp_file_path, "dataset")
    test_demo_data = np.load(os.path.join(dataset_directory_path, f"test_demo_data/{demo_name}.pkl"), allow_pickle=True)
    object_type = test_demo_data["object_type"]
    
    
    if object_type == "rigid":
        input_object_point_cloud, offset_, norm_offset_ = (
            load_object_point_cloud_and_normal_rigid(
                test_demo_data["down_sampled_begin_points"],
                test_demo_data["down_sampled_begin_normals"],
                point_motion=test_demo_data["point_motion"],
                scale=test_demo_data["object_scale"],
            )
        )
    elif object_type == "clothes":
        input_object_point_cloud, offset_, norm_offset_ = (
            load_object_point_cloud_and_normal_clothes(
                test_demo_data["down_sampled_begin_points"],
                test_demo_data["down_sampled_begin_normals"],
                point_motion=test_demo_data["point_motion"],
                scale=test_demo_data["object_scale"],
            )
        )
    elif object_type == "mpm":
        input_object_point_cloud, offset_, norm_offset_ = (
            load_object_point_cloud_and_normal_mpm(
                test_demo_data["down_sampled_begin_points"],
                test_demo_data["down_sampled_begin_normals"],
                point_motion=test_demo_data["point_motion"],
                scale=test_demo_data["object_scale"],
            )
        )
        
        
    device = torch.device(f"cuda:{gpu_number}")
    model = ManiFM(cfg.model, device).cuda(gpu_number)
    load_model(model, str(checkpoint_path))
    model.eval()

    with torch.no_grad():
        """return"""
        input_hand_point_cloud = load_hand_pointcloud_and_normals(test_demo_data["robot_name"])
        predicted_contact_points_heatmap, contact_forces = model.infer(
            input_hand_point_cloud.unsqueeze(0).repeat(pred_time, 1, 1,).cuda(gpu_number),
            input_object_point_cloud.unsqueeze(0).repeat(pred_time, 1, 1,).cuda(gpu_number),
        )
        pred_contact_points_heatmap = predicted_contact_points_heatmap.detach().cpu().numpy()  # shape=(b, 2048, 1)
        pred_contact_force_map = contact_forces.detach().cpu().numpy()  # shape=(b, 2048, 3)

    data_point_cloud = input_object_point_cloud[:, :3].detach().cpu().numpy() # shape=(2048, 3)
    server = viser.ViserServer(port=8024)

    to_show_cnt = 0
    for i in range(pred_time):
        normalized_hmap = (pred_contact_points_heatmap[i, :, 0] - pred_contact_points_heatmap[i, :, 0].min()) / (pred_contact_points_heatmap[i, :, 0].max() - pred_contact_points_heatmap[i, :, 0].min())
        
        if "Hand" in test_demo_data["robot_name"]:
            filter_radius = 0.02 / test_demo_data["object_scale"]
        elif "Arm" in test_demo_data["robot_name"]:
            filter_radius = 0.2
            
        filtered_points, keep_idx = point_cloud_nms(
            data_point_cloud,
            pred_contact_points_heatmap[i, :, 0],
            filter_radius,
        )
        
        if len(keep_idx) == 0 or len(keep_idx) > 4 or np.average(normalized_hmap) > 1.0 or np.percentile(normalized_hmap, 50) > 1.0: 
            # if too uncertain (too much deviation) then we skip 
            print(len(keep_idx), np.average(normalized_hmap))
            
            continue
        
        if object_type == "rigid": 
            # if the object is rigid, we can optimize the force and see whether the contact points and forces can produce the target point motion
            try: 
                wrench = pointcloud_motion_to_wrench(test_demo_data["down_sampled_begin_points"], test_demo_data["point_motion"])
                prob, f_global_array = from_wrench_to_contact_force(
                    test_demo_data["down_sampled_begin_points"], 
                    test_demo_data["down_sampled_begin_normals"],
                    keep_idx, 0.1 * wrench / (np.linalg.norm(wrench) + 1e-8) )
                if prob > 0.1:  # fail to solve 
                    continue                    
                else:
                    force_mesh_list = []
                    for idx, pid in enumerate(keep_idx):
                        if pid != 2048:
                            contact_point = data_point_cloud[pid]
                            force_vector = f_global_array[idx]
                            force_mesh_list.append(vis_vector(contact_point, force_vector, length=0.3, cyliner_r=0.03, color=[255, 255, 20, 255]))
                    server.add_mesh_trimesh(
                        f"pred_force_mesh_{i}",
                        trimesh.Scene(force_mesh_list)
                        .dump(concatenate=True)
                        .apply_translation([0.0, (to_show_cnt + 1) * 2.0, 0.0]),
                    )
                    vis_pc_heatmap(
                        server,
                        data_point_cloud,
                        pred_contact_points_heatmap[i, :, 0],
                        strid=f"pc_hmap_pred_{i}",
                        trans=[0.0, (to_show_cnt + 1) * 2.0, 0.0],
                        radius=0.03,
                    )
                    to_show_cnt += 1
            except Exception as e:
                print(f"Fail to solve the force for {i}-th prediction, exception: {e}")
                continue
        else: 
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
        points=test_demo_data["down_sampled_begin_points"],
        point_size=0.02,
        point_shape="circle",
        colors=[255, 0, 0],
    )
    server.add_point_cloud(
        "/end_points",
        points=test_demo_data["down_sampled_begin_points"]+test_demo_data["point_motion"],
        point_size=0.02,
        point_shape="circle",
        colors=[0, 255, 0],
    )

    begin_normals_vector_mesh_list = []
    for p, n in zip(test_demo_data["down_sampled_begin_points"], test_demo_data["down_sampled_begin_normals"]):
        begin_normals_vector_mesh_list.append(
            vis_vector(p, n, length=0.1, color=[255, 255, 100, 245])
        )
    server.add_mesh_trimesh(
        "begin_normals_vector_mesh_list",
        trimesh.Scene(begin_normals_vector_mesh_list).dump(concatenate=True),
    )

    motion_vector_mesh_list = []
    for p1, p2 in zip(test_demo_data["down_sampled_begin_points"], test_demo_data["down_sampled_begin_points"]+test_demo_data["point_motion"]):
        motion_vector_mesh_list.append(
            vis_vector(p1, p2 - p1, length=np.linalg.norm(p2 - p1), color=[255, 100, 255, 245], cyliner_r=0.002)
        )
    server.add_mesh_trimesh(
        "motion_vector_mesh_list",
        trimesh.Scene(motion_vector_mesh_list).dump(concatenate=True),
    )

    a = input()

    