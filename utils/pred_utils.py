import os 
import numpy as np
import cvxpy as cp
import trimesh 
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation as R
import matplotlib
import matplotlib.cm as cm
import warnings
import torch 
from pathlib import Path 
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
colormap_pred = cm.get_cmap('plasma')


############################
###### Load for pred #######
############################

def load_hand_pointcloud_and_normals(handname="LeapHand"):
    ''' load the hand point cloud and normal '''
    pp_file_path = Path(__file__).parent.parent
    dataset_directory_path = Path.joinpath(pp_file_path, "dataset")
    hand_point_cloud = np.load(os.path.join(dataset_directory_path, f"robot_hand_config/{handname}/point_cloud.npy"))
    hand_point_normal = np.load(os.path.join(dataset_directory_path, f"robot_hand_config/{handname}/point_normal.npy"))
    hand_point_cloud_input = torch.from_numpy(np.concatenate([hand_point_cloud, hand_point_normal], axis=1)).float()
    return hand_point_cloud_input

def wrench_to_pointcloud_motion(object_pointcloud, wrench, mass=1, rot_inertia=np.eye(3), start_v=np.array([0.0, 0.0, 0.0]), start_w=np.array([0.0, 0.0, 0.0]), timestep=0.5):
    '''
    change the timestep here to change the motion magnitude'''
    # Extract force and torque from wrench
    force = wrench[:3]
    torque = wrench[3:]

    # Compute linear and angular accelerations
    linear_acc = force / mass
    angular_acc = np.linalg.inv(rot_inertia).dot(torque)

    # Calculate displacements
    translation = start_v * timestep + 0.5 * linear_acc * timestep * timestep
    rotation = R.from_rotvec(start_w * timestep + 0.5 * angular_acc * timestep * timestep)
    rotation_matrix = rotation.as_matrix()
    moved_pointcloud = (rotation_matrix.dot(object_pointcloud.T)).T + translation

    pointcloud_motion = moved_pointcloud - object_pointcloud

    return moved_pointcloud, pointcloud_motion

def pointcloud_motion_to_wrench(object_pointcloud, pointcloud_motion, mass=1, rot_inertia=np.eye(3), start_v=np.array([0.0, 0.0, 0.0]), start_w=np.array([0.0, 0.0, 0.0]), timestep=0.5):
    moved_pointcloud = object_pointcloud + pointcloud_motion
    mass_center = np.mean(object_pointcloud, axis=0)
    moved_mass_center = np.mean(moved_pointcloud, axis=0)

    translation = moved_mass_center - mass_center
    linear_acc = 2 * (translation - start_v * timestep) / (timestep * timestep)

    rotated_pointcloud = moved_pointcloud - translation
    rotation_matrix = (np.linalg.inv(object_pointcloud.T @ object_pointcloud) @ object_pointcloud.T @ rotated_pointcloud).T
    rotation = R.from_matrix(rotation_matrix)
    angular_acc = 2 * (rotation.as_rotvec() - start_v * timestep) / (timestep * timestep)

    force = mass * linear_acc
    torque = rot_inertia @ angular_acc

    wrench = np.zeros(6)
    wrench[:3] = force
    wrench[3:] = torque

    return wrench

''' load the object point cloud and normal '''
def load_object_point_cloud_and_normal_rigid(point_cloud, point_normal, wrench=None, point_motion=None, part_seg_indicator=torch.ones((2048,1)), f_coef=0.5, scale=None, num_points=2048) -> torch.Tensor:
    '''
    Args:
    - point_cloud: The point cloud. np, shape [N, 3].
    - point_normal: The point normal. np, shape [N, 3].
    '''
    assert (wrench is not None or point_motion is not None), "Either wrench or point_motion should be provided"
    # normalize the point cloud
    point_cloud_torch = torch.from_numpy(point_cloud).float().clone()
    normals_torch = torch.from_numpy(point_normal).float().clone()
    
    offset = torch.mean(point_cloud_torch, dim=0)
    
    point_cloud_torch -= offset
    norm_offset = 1 * torch.max(torch.abs(point_cloud_torch))
    
    point_cloud_torch /= norm_offset
    
    motion_mask = torch.ones(num_points, 1)
    if wrench is not None:
        moved_pointcloud, pointcloud_motion = wrench_to_pointcloud_motion(point_cloud_torch.cpu().numpy(), wrench)
        point_motion_torch = torch.from_numpy(pointcloud_motion).float()
    elif point_motion is not None:
        point_motion_torch = torch.from_numpy(point_motion).float()
    
    point_motion = point_motion_torch * motion_mask
    
    heatmap = torch.zeros(point_motion_torch.shape[0], 1)
    heatmap_mask = torch.zeros(point_motion_torch.shape[0], 1)
            
    input_object_point_cloud = torch.concat(
        [
            point_cloud_torch,
            normals_torch,
            part_seg_indicator,
            point_motion_torch, # 3
            heatmap,   # 1
            heatmap_mask,   # 1
        ]
        , dim=1
    ) # num_points * 32

    physical_properties = torch.zeros(num_points, 6)    # 6 physical properties
    physical_property_mask = torch.zeros(num_points, 6)    # 6 physical properties
    physical_properties[:, 0] = f_coef
    physical_property_mask[:, 0] = 1
            
    if scale is None:
        scale = norm_offset
    input_object_point_cloud = torch.cat([input_object_point_cloud, torch.tensor(float(scale)).repeat(num_points, 1)], dim=1)  
    input_object_point_cloud = torch.concat(
        [
            input_object_point_cloud,
            physical_properties,
            physical_property_mask,
        ], dim=1)
    return input_object_point_cloud, offset, norm_offset


def load_object_point_cloud_and_normal_clothes(point_cloud, point_normal, point_motion, scale=None, part_seg_indicator=torch.ones((2048,1)), heatmap=None, num_points=2048, f_coef=0.5, k_stiff_stretching=5.0e2, k_stiff_bending=1.0e-2) -> torch.Tensor:
    '''
    Args:
    - point_cloud: The point cloud. np, shape [N, 3].
    - point_normal: The point normal. np, shape [N, 3].
    - point_motion: The point motion. np, shape [N, 3].
    
    pay attention to the point_motion
    '''
    
    # normalize the point cloud
    point_cloud_torch = torch.from_numpy(point_cloud).float()
    normals_torch = torch.from_numpy(point_normal).float()
    
    offset = torch.mean(point_cloud_torch, dim=0)
    point_cloud_torch -= offset
    norm_offset = 1 * torch.max(torch.abs(point_cloud_torch))
    point_cloud_torch /= norm_offset
    
    motion_mask = torch.ones(num_points, 1)
    point_motion_torch = torch.from_numpy(point_motion).float()
    point_motion = point_motion_torch * motion_mask
    
    if heatmap is None:
        heatmap = torch.zeros(point_motion_torch.shape[0], 1)
        heatmap_mask = torch.zeros(point_motion_torch.shape[0], 1)
    else: 
        heatmap = torch.from_numpy(heatmap).float()
        heatmap_mask = torch.ones(point_motion_torch.shape[0], 1)
        
            
    input_object_point_cloud = torch.concat(
        [
            point_cloud_torch,  # 0:3
            normals_torch,  # 3:6
            part_seg_indicator,  # 6:7
            point_motion_torch, # 7:10
            heatmap,   # 1
            heatmap_mask,   # 1
        ]
        , dim=1
    ) # num_points * 32

    physical_properties = torch.zeros(num_points, 6)    # 6 physical properties
    physical_property_mask = torch.zeros(num_points, 6)    # 6 physical properties
    physical_properties[:, 0] = f_coef
    physical_property_mask[:, 0] = 1
    physical_properties[:, 3] = k_stiff_stretching
    physical_property_mask[:, 3] = 1
    physical_properties[:, 4] = k_stiff_bending
    physical_property_mask[:, 4] = 1
            
    if scale is None:
        scale = norm_offset
    input_object_point_cloud = torch.cat([input_object_point_cloud, torch.tensor(float(scale)).repeat(num_points, 1)], dim=1)  
    input_object_point_cloud = torch.concat(
        [
            input_object_point_cloud,
            physical_properties,
            physical_property_mask,
        ], dim=1)
    
    return input_object_point_cloud, offset.detach().cpu().numpy(), norm_offset.detach().cpu().numpy()  

def load_object_point_cloud_and_normal_mpm(point_cloud, point_normal, point_motion, scale=None, part_seg_indicator=torch.ones((2048,1)), heatmap=None, num_points=2048, E=3.0, nu=0.2) -> torch.Tensor:
    '''
    Args:
    - point_cloud: The point cloud. np, shape [N, 3].
    - point_normal: The point normal. np, shape [N, 3].
    - point_motion: The point motion. np, shape [N, 3].
    
    pay attention to the point_motion
    '''
    
    # normalize the point cloud
    point_cloud_torch = torch.from_numpy(point_cloud).float()
    normals_torch = torch.from_numpy(point_normal).float()
    
    offset = torch.mean(point_cloud_torch, dim=0)
    point_cloud_torch -= offset
    norm_offset = 1 * torch.max(torch.abs(point_cloud_torch))
    point_cloud_torch /= norm_offset
    
    motion_mask = torch.ones(num_points, 1)
    point_motion_torch = torch.from_numpy(point_motion).float()
    point_motion = point_motion_torch * motion_mask
    
    if heatmap is None:
        heatmap = torch.zeros(point_motion_torch.shape[0], 1)
        heatmap_mask = torch.zeros(point_motion_torch.shape[0], 1)
    else: 
        heatmap = torch.from_numpy(heatmap).float()
        heatmap_mask = torch.ones(point_motion_torch.shape[0], 1)
        
            
    input_object_point_cloud = torch.concat(
        [
            point_cloud_torch,  # 0:3
            normals_torch,  # 3:6
            part_seg_indicator,  # 7:11
            point_motion_torch, # 11:14
            heatmap,   # 4
            heatmap_mask,   # 1
        ]
        , dim=1
    ) # num_points * 32

    physical_properties = torch.zeros(num_points, 6)    # 6 physical properties
    physical_property_mask = torch.zeros(num_points, 6)    # 6 physical properties
    physical_properties[:, 1] = 1/E
    physical_property_mask[:, 1] = 1
    physical_properties[:, 2] = nu
    physical_property_mask[:, 2] = 1
            
    if scale is None:
        scale = norm_offset
    input_object_point_cloud = torch.cat([input_object_point_cloud, torch.tensor(float(scale)).repeat(num_points, 1)], dim=1)  
    input_object_point_cloud = torch.concat(
        [
            input_object_point_cloud,
            physical_properties,
            physical_property_mask,
        ], dim=1)
    
    return input_object_point_cloud, offset.detach().cpu().numpy(), norm_offset.detach().cpu().numpy()  

def point_cloud_nms(pc, hmap, radius=0.2, heatmap_threshold_percentile=90, least_group_size=10, min_heatmap_value=0.25):
    
    '''
    Perform non-maximum suppression on the point cloud based on the heatmap.
    
    Args:
    - pc: np, shape=(n, 3)
    - hmap: np, shape=(n,)
    - radius: float, the radius to search the neighbors
    - heatmap_threshold_percentile: float, the percentile to filter the heatmap
    '''
    normalized_hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())
    
    heatmap_threshold = np.percentile(normalized_hmap, heatmap_threshold_percentile)
    selected_indices = np.where(normalized_hmap > heatmap_threshold)[0]
    selected_points = pc[selected_indices]
    selected_heatmap_values = normalized_hmap[selected_indices]

    tree = KDTree(selected_points)
    indices = tree.query_radius(selected_points, r=radius)

    keep_mask = np.zeros(len(selected_points), dtype=bool)

    for i, ind in enumerate(indices):
        if selected_heatmap_values[i] >= selected_heatmap_values[ind].max() and len(ind) >= least_group_size and selected_heatmap_values[i] > min_heatmap_value:
            if selected_heatmap_values[ind].mean() > 0.05:
                keep_mask[i] = True

    filtered_points = selected_points[keep_mask]
    filtered_points = np.unique(filtered_points, axis=0)
    keep_idx_in_sel = np.where(keep_mask)[0]
    keep_idx_in_ori = selected_indices[keep_idx_in_sel]

    return filtered_points, keep_idx_in_ori


def point_cloud_nms_til_num_contact(pc, hmap, radius=0.2, heatmap_threshold_percentile=90, least_group_size=10, min_heatmap_value=0.25, num_contacts=0, max_iter=3):
    
    '''
    Iteratively perform non-maximum suppression til given number of contact points.
    
    Args:
    - pc: np, shape=(n, 3)
    - hmap: np, shape=(n,)
    - radius: float, the radius to search the neighbors
    - heatmap_threshold_percentile: float, the percentile to filter the heatmap
    - num_contacts: int, the number of contact points
    '''
    for _ in range(max_iter):
        # print(pc.shape, hmap.shape, radius, heatmap_threshold_percentile, least_group_size, min_heatmap_value)
        filtered_points, keep_idx_in_ori = point_cloud_nms(pc, hmap, radius, heatmap_threshold_percentile, least_group_size, min_heatmap_value)
        if keep_idx_in_ori.shape[0] >= num_contacts:
            break
        radius /= 2

    return filtered_points, keep_idx_in_ori


############################
###### Visusalization ######
############################
def normalize(x):
    '''
    Normalize the input vector. If the magnitude of the vector is zero, a small value is added to prevent division by zero.

    Parameters:
    - x (np.ndarray): Input vector to be normalized.

    Returns:
    - np.ndarray: Normalized vector.
    '''
    if len(x.shape) == 1:
        mag = np.linalg.norm(x)
        if mag == 0:
            mag = mag + 1e-10
        return x / mag
    else: 
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        return x / norms
    
def sample_transform_w_normals(new_palm_center, new_face_vector, sample_roll, ori_face_vector=np.array([1.0, 0.0, 0.0])):
    '''
    Compute the transformation matrix from the original palm pose to a new palm pose.
    
    Parameters:
    - new_palm_center (np.ndarray): The point of the palm center [x, y, z].
    - new_face_vector (np.ndarray): The direction vector representing the new palm facing direction.
    - sample_roll (float): The roll angle in range [0, 2*pi).
    - ori_face_vector (np.ndarray): The original direction vector representing the palm facing direction. Default is [1.0, 0.0, 0.0].

    Returns:
    - rst_transform (np.ndarray): A 4x4 transformation matrix.
    '''

    rot_axis = np.cross(ori_face_vector, normalize(new_face_vector))
    rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-16)
    rot_ang = np.arccos(np.clip(np.dot(ori_face_vector, new_face_vector), -1.0, 1.0))

    if rot_ang > 3.1415 or rot_ang < -3.1415: 
        rot_axis = np.array([1.0, 0.0, 0.0]) if not np.isclose(ori_face_vector, np.array([1.0, 0.0, 0.0])).all() else np.array([0.0, 1.0, 0.0])
    
    rot = R.from_rotvec(rot_ang * rot_axis).as_matrix()
    roll_rot = R.from_rotvec(sample_roll * new_face_vector).as_matrix()

    final_rot = roll_rot @ rot
    rst_transform = np.eye(4)
    rst_transform[:3, :3] = final_rot
    rst_transform[:3, 3] = new_palm_center
    return rst_transform

def vis_vector(start_point, vector, length=0.1, cyliner_r=0.003, color=[255, 255, 100, 245], no_arrow=False):
    '''
    start_points: np.ndarray, shape=(3,)
    vectors: np.ndarray, shape=(3,)
    length: cylinder length 
    '''
    normalized_vector = normalize(vector)
    end_point = start_point + length * normalized_vector

    # create a mesh for the force
    force_cylinder = trimesh.creation.cylinder(radius=cyliner_r, 
                                               segment=np.array([start_point, end_point]))
    
    # create a mesh for the arrowhead
    cone_transform = sample_transform_w_normals(end_point, normalized_vector, 0, ori_face_vector=np.array([0.0, 0.0, 1.0]))
    arrowhead_cone = trimesh.creation.cone(radius=2*cyliner_r, 
                                           height=4*cyliner_r, 
                                           transform=cone_transform)
    # combine the two meshes into one
    if not no_arrow:
        force_mesh = force_cylinder + arrowhead_cone 
    else: 
        force_mesh = force_cylinder
    force_mesh.visual.face_colors = color

    return force_mesh

def vis_pc_heatmap(server, pc, hmap, strid="pc_hmap_pred", trans=[0.0, 0.0, 0.0], pc_moved=None, radius=0.03) -> None:
    """
    Draw the predicted heatmap and point cloud.
    
    Args:
    - pc: The point cloud. np, shape [N, 3].
    - hmap: The heatmap. np, shape [N].
    """
    normalized_hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())
    hmap_colored = colormap_pred(normalized_hmap * 0.4 + 0.6)  
    hmap_colored = colormap_pred(normalized_hmap)  
    hmap_rgb = hmap_colored[:, :3] 
    hmap_rgb_uint8 = (hmap_rgb * 255).astype('uint8')
        
    server.add_point_cloud(
        "/"+strid,
        points=pc + np.array(trans),
        point_size=radius,
        point_shape='circle',
        colors=hmap_rgb_uint8,
    )
    if pc_moved is not None:
        server.add_point_cloud(
        "/"+strid+"_moved",
        points=pc_moved+np.array(trans),
        point_size=radius,
        point_shape='sparkle',
        colors=(200, 100, 200),
    )
      
def vis_pc_trimesh(pc, colors, radius=0.02):
    pc_trimesh_list = []
    for i in range(pc.shape[0]):
        single_sphere = trimesh.creation.uv_sphere(radius=radius, count=(6,6)).apply_translation(pc[i])
        single_sphere.visual.face_colors = colors[i]
        pc_trimesh_list.append(single_sphere)
    return trimesh.Scene(pc_trimesh_list)
  
def vis_pc_heatmap_trimesh_pc(pc, hmap, radius=0.02):
    
    normalized_hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())
    hmap_colored = colormap_pred(normalized_hmap * 0.4 + 0.6)  
    hmap_colored = colormap_pred(normalized_hmap)  
    hmap_rgb = hmap_colored[:, :3] 
    hmap_rgb_uint8 = (hmap_rgb * 255).astype('uint8')
    pc_trimesh = vis_pc_trimesh(pc, hmap_rgb_uint8, radius=radius)
    return pc_trimesh


    
def vis_hand(server, hand, hand_nb, pose, strid="pre_hand_mesh", trans=[0.0, 0.0, 0.0]) -> None:
    '''
    Draw the hand.
    '''
    joint_pos, init_T = hand_nb.nb_jointpos_to_pb_jointpos_basepose(pose, return_type="mat")
    init_T[:3, 3] += np.array(trans)
    hand_mesh = hand.get_hand_trimesh(joint_pos, torch.Tensor(init_T))
    server.add_mesh_trimesh(strid, 
                            trimesh.Scene(hand_mesh).dump(concatenate=True))
        
        
        
