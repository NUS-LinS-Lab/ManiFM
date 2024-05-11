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
    point_cloud_nms_til_num_contact,
    pointcloud_motion_to_wrench,
    vis_hand
)
from utils.opt_utils import (
    select_top_contacts,
    generate_icp_poses,
    define_finger_orders,
    select_best_candidate
)
from optimization import (
    LeapHand,
    LeapHandNb,
    AllegroHand,
    AllegroHandNb,
    PointCloud
)
import pybullet as p


def set_seed(num=666):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)


def build_hand_model(root_dir, hand_name):
    if hand_name == "LeapHand":
        hand_nb = LeapHandNb(
            hand_urdf_path=os.path.join(root_dir, 'optimization/assets/hands/leap_hand/robot_nb.urdf'), 
            hand_mesh_path=os.path.join(root_dir, 'optimization/assets/hands/leap_hand'))

        physics_client_id = p.connect(p.DIRECT)
        p.setGravity(0, 0, 0)
        hand = LeapHand(
                hand_urdf_path=os.path.join(root_dir, 'optimization/assets/hands/leap_hand/robot.urdf'), 
                hand_mesh_path=os.path.join(root_dir, 'optimization/assets/hands/leap_hand'),
                physics_client_id=physics_client_id
            )
    elif hand_name == "AllegroHand":
        hand_nb = AllegroHandNb(
            hand_urdf_path=os.path.join(root_dir, 'optimization/assets/hands/allegro_hand_right/allegro_hand_right_nb.urdf'), 
            hand_mesh_path=os.path.join(root_dir, 'optimization/assets/hands/allegro_hand_right'))

        physics_client_id = p.connect(p.DIRECT)
        p.setGravity(0, 0, 0)
        hand = AllegroHand(
                hand_urdf_path=os.path.join(root_dir, "optimization/assets/hands/allegro_hand_right/allegro_hand_right_pb.urdf"), 
                hand_mesh_path=os.path.join(root_dir, "optimization/assets/hands/allegro_hand_right"),
                physics_client_id=physics_client_id
            )
    return hand, hand_nb


def build_object_model(hand_nb, points, normals):
    pc = PointCloud(points, -normals)
    hand_nb.set_pc(pc)
    return pc


demo_configs = {
    "object": {
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
        "opt": {
            "object_type": "rigid",
        },
    },
    "hand": "AllegroHand", # LeapHand
}

if __name__ == "__main__":
    config_file_path = "configs/test.json"
    cfg = OmegaConf.load(config_file_path)
    set_seed(cfg.train.seed)
    
    obj_name = "opt"
    gpu_number = 0
    checkpoint_path = "checkpoints/200k_204epc_model.safetensors"
    pred_time = 1
    
    if obj_name not in demo_configs["object"]:
        raise ValueError(f"Unknown object name: {obj_name}")
    
    pp_file_path = Path(__file__).parent.parent
    dataset_directory_path = Path.joinpath(pp_file_path, "dataset")
    test_demo_data = np.load(os.path.join(dataset_directory_path, f"test_demo_data/{obj_name}.pkl"), allow_pickle=True)
    
     # the input_hand_point_cloud in the demo file is for LeapHand, and we can reload input_hand_point_cloud for other hands such as AllegroHand here.
    input_hand_point_cloud = load_hand_pointcloud_and_normals(demo_configs["hand"])
    
    if demo_configs["object"][obj_name]["object_type"] == "rigid":
        input_object_point_cloud, offset_, norm_offset_ = (
            load_object_point_cloud_and_normal_rigid(
                test_demo_data["down_sampled_begin_points"],
                test_demo_data["down_sampled_begin_normals"],
                point_motion=test_demo_data["point_motion"],
            )
        )
    elif demo_configs["object"][obj_name]["object_type"] == "clothes":
        input_object_point_cloud, offset_, norm_offset_ = (
            load_object_point_cloud_and_normal_clothes(
                test_demo_data["down_sampled_begin_points"],
                test_demo_data["down_sampled_begin_normals"],
                point_motion=test_demo_data["point_motion"],
            )
        )
    elif demo_configs["object"][obj_name]["object_type"] == "mpm":
        test_demo_data = {
            "down_sampled_begin_points": test_demo_data[0]["init_state"],
            "down_sampled_begin_normals": test_demo_data[0]["init_state_normal"],
            "point_motion": test_demo_data[0]["target_state"]-test_demo_data[0]["init_state"],
        }
        input_object_point_cloud, offset_, norm_offset_ = (
            load_object_point_cloud_and_normal_mpm(
                test_demo_data["down_sampled_begin_points"],
                test_demo_data["down_sampled_begin_normals"],
                point_motion=test_demo_data["point_motion"],
            )
        )
        
        
    device = torch.device(f"cuda:{gpu_number}")
    model = ManiFM(cfg.model, device).cuda(gpu_number)
    load_model(model, str(checkpoint_path))
    model.eval()

    with torch.no_grad():
        """return"""
        predicted_contact_points_heatmap, contact_forces = model.infer(
            input_hand_point_cloud.unsqueeze(0).repeat(pred_time, 1, 1,).cuda(gpu_number),
            input_object_point_cloud.unsqueeze(0).repeat(pred_time, 1, 1,).cuda(gpu_number),
        )
        pred_contact_points_heatmap = predicted_contact_points_heatmap.detach().cpu().numpy()  # shape=(b, 2048, 1)
        pred_contact_force_map = contact_forces.detach().cpu().numpy()  # shape=(b, 2048, 3)

    data_point_cloud = input_object_point_cloud[:, :3].detach().cpu().numpy() # shape=(2048, 3)
    server = viser.ViserServer(port=8048)

    to_show_cnt = 0
    for i in range(pred_time):
        filtered_points, keep_idx = point_cloud_nms_til_num_contact(
            data_point_cloud,
            pred_contact_points_heatmap[i, :, 0],
            radius=0.02 / norm_offset_,
            least_group_size=0,
            min_heatmap_value=0.0,
            num_contacts=3,
            max_iter=10
        )
        vis_pc_heatmap(
            server,
            test_demo_data["down_sampled_begin_points"],
            pred_contact_points_heatmap[i, :, 0],
            strid=f"pc_hmap_pred_{i}",
            trans=[0.0, (to_show_cnt + 1) * 2.0, 0.0],
            radius=0.03,
        )
        if demo_configs["object"][obj_name]["object_type"] == "rigid": 
            # if the object is rigid, we can optimize the hand pose for the target point motion
            try: 
                chosen_idx = select_top_contacts(keep_idx, pred_contact_points_heatmap[i, :, 0])
                contact_finger_names_list = define_finger_orders(chosen_idx)
                hand, hand_nb = build_hand_model(pp_file_path, demo_configs["hand"])
                pc = build_object_model(hand_nb, test_demo_data["down_sampled_begin_points"], test_demo_data["down_sampled_begin_normals"])
                candidates = generate_icp_poses(hand, hand_nb, pc, chosen_idx, contact_finger_names_list, pred_contact_points_heatmap[i, :, 0])

                init_pose, finger_names = select_best_candidate(candidates)
                hand_nb.setJoints(init_pose)
                wrench = pointcloud_motion_to_wrench(test_demo_data["down_sampled_begin_points"], test_demo_data["point_motion"])
                info = hand_nb.contact_wrench_jointpos_opt(finger_names, wrench=0.1 * wrench / (np.linalg.norm(wrench) + 1e-8), mu=1, step_num=100, step=0.1)

                if info['flag'] != "success":  # fail to solve
                    
                    vis_hand(
                        server,
                        hand,
                        hand_nb,
                        info["pose_list"][0],
                        strid=f"init_hand_mesh_{i}",
                        trans=[0.0, (to_show_cnt + 1) * 2.0, 0.0]
                    )
                    vis_hand(
                        server,
                        hand,
                        hand_nb,
                        info["pose_list"][-1],
                        strid=f"pred_hand_mesh_{i}",
                        trans=[0.0, (to_show_cnt + 1) * 2.0, 0.0]
                    )
                    to_show_cnt += 1
                else:
                    force_mesh_list = []
                    for idx, pid in enumerate(chosen_idx):
                        if pid != 2048:
                            contact_point = info["finger_force"][finger_names[idx]][0]
                            force_vector = info["finger_force"][finger_names[idx]][1]
                            force_mesh_list.append(vis_vector(contact_point, force_vector, length=0.2, cyliner_r=0.01, color=[255, 255, 20, 255]))
                    server.add_mesh_trimesh(
                        f"pred_force_mesh_{i}",
                        trimesh.Scene(force_mesh_list)
                        .dump(concatenate=True)
                        .apply_translation([0.0, (to_show_cnt + 1) * 2.0, 0.0]),
                    )
                    
                    vis_hand(
                        server,
                        hand,
                        hand_nb,
                        info["pose_list"][-1],
                        strid=f"pred_hand_mesh_{i}",
                        trans=[0.0, (to_show_cnt + 1) * 2.0, 0.0]
                    )

                    # also show hand at the inital object position
                    vis_hand(
                        server,
                        hand,
                        hand_nb,
                        info["pose_list"][-1],
                        strid=f"pred_hand_mesh_on_object_{i}",
                        trans=[0.0, 0.0, 0.0]
                    )                    
                    to_show_cnt += 1
            except Exception as e:
                print(f"Fail to solve the force for {i}-th prediction, exception: {e}")
                continue
        else: # TODO: optimize and visualize the hand pose for deformable objects.
            print("optimization for deformable objects is not available now.")


    server.add_point_cloud(
        "/begin_points",
        points=test_demo_data["down_sampled_begin_points"],
        point_size=0.005,
        point_shape="circle",
        colors=[0, 0, 255],
    )
    server.add_point_cloud(
        "/end_points",
        points=test_demo_data["down_sampled_begin_points"]+test_demo_data["point_motion"],
        point_size=0.005,
        point_shape="circle",
        colors=[0, 255, 0],
    )

    begin_normals_vector_mesh_list = []
    for p, n in zip(test_demo_data["down_sampled_begin_points"], test_demo_data["down_sampled_begin_normals"]):
        begin_normals_vector_mesh_list.append(
            vis_vector(p, n, length=0.05, color=[255, 255, 100, 245], cyliner_r=0.001)
        )
    server.add_mesh_trimesh(
        "begin_normals_vector_mesh_list",
        trimesh.Scene(begin_normals_vector_mesh_list).dump(concatenate=True),
    )

    motion_vector_mesh_list = []
    for p1, p2 in zip(test_demo_data["down_sampled_begin_points"], test_demo_data["down_sampled_begin_points"]+test_demo_data["point_motion"]):
        motion_vector_mesh_list.append(
            vis_vector(p1, p2 - p1, length=np.linalg.norm(p2 - p1), color=[255, 100, 255, 245], cyliner_r=0.001)
        )
    server.add_mesh_trimesh(
        "motion_vector_mesh_list",
        trimesh.Scene(motion_vector_mesh_list).dump(concatenate=True),
    )


    a = input()
    