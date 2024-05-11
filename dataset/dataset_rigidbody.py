import os
import sys
sys.path.append(".")
import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from utils.data_utils import from_wrench_to_contact_force, pc_add_noise_torch, normal_add_noise_torch, replace_noisy_points, generate_random_wrenches
import random
import numpy as np

CPPAD = 2048
HANDS = ["LeapHand", "PandaHand", "Kinova3FHand", "MovoArm"]

file_path = __file__
dataset_directory_path = os.path.dirname(file_path)

class RigidBodyDataset(Dataset):
    """
    Load "*.pkl" files. The format is as follows:
    [
        {   'hand': str,  # LeapHand
            'object_path': str,
            'scale': array([float]),
            'f_coef': float,
            'palm_pose': list of list, shape=(N, 7),    # N palm poses
            'joint_pos': [array(), array()], # list of N np.array for joint pos, each array has shape (16,) for 16 joints
            'contact_point_id': list of list, shape=(N, 4), # N contact point ids for 4 fingers
            'wrenches': [array(), array()], # list of N np.array for wrenches, each array has shape (M, 6) for M wrenches
            'point_cloud': np.array, shape=(2048, 3),
            'point_normal': np.array, shape=(2048, 3),
            'point_to_region_id': np.array, shape=(2048,),
            TODO: other cameara keys ...    
        },
        ... # number of objects of dicts
    ]
    """
    def __init__(
        self, 
        pkl_file_directory, 
        use_scale=False, 
        gaussian_heatmap_max_value=10.0,
        generate_new_wrench=True,
        load_ratio=1.0, # load ratio for every OBJECT
        only_one_object=False,
        use_gaussian_map_ratio=0.0,
        augment_part_seg_indicator_ratio=0.0,
        load_target_wrench=False,
        physcial_property_num=6,
        use_region=True,
        use_physics=True,
        num_object=None,
        num_palm_pose=None,
        num_motion=10,
        remove_pc_num=0,  # if 0 , not remove any point cloud; otherwise, default is 512 
        remove_pc_prob=0.5,
        noisy_upper_level=0.05,
        noisy_normal_upper_level=np.pi/6,
        random_flip_normal_upper_level=0.2
    ):
        super().__init__()
        
        # indexed by hand index
        self.hands = []
        self.hands_graph = []
        self.object_paths = []
        self.scales = []
        self.f_coefs = []
        self.point_clouds = []
        self.point_normals = []
        self.point_to_region_ids = []
        self.hand_point_clouds = []

        '''
        per grasp point cloud and point normal list 
        '''
        self.point_clouds_per_grasp = []
        self.point_normals_per_grasp = []
        self.point_to_region_ids_per_grasp = []
        self.gaussian_heatmaps_per_grasp = []
        
        # indexed by palm index
        self.palm_poses = []
        self.joint_poses = []
        self.contact_point_ids = []
        self.gaussian_heatmaps = []

        self.target_wrenches = []
        self.forces = []

        self.hand_indices = []
        self.palm_indices = []

        self.use_scale = use_scale

        self.generate_new_wrench = generate_new_wrench

        self.max_num_contact_points = 4
        self.gaussian_heatmap_max_value = gaussian_heatmap_max_value
        
        self.use_gaussian_map_ratio = use_gaussian_map_ratio
        self.augment_part_seg_indicator_ratio = augment_part_seg_indicator_ratio
        self.load_target_wrench = load_target_wrench
        self.physcial_property_num = physcial_property_num
        self.use_region = use_region
        self.use_physics = use_physics
        self.remove_pc_num = remove_pc_num
        self.remove_pc_prob = remove_pc_prob
        self.noisy_upper_level = noisy_upper_level
        self.noisy_normal_upper_level = noisy_normal_upper_level
        self.random_flip_normal_upper_level = random_flip_normal_upper_level
        
        hand_pc = {}
        hand_pn = {}
        for hand in HANDS:
            hand_point_cloud = np.load(os.path.join(dataset_directory_path, f"robot_hand_config/{hand}/point_cloud.npy"))
            hand_point_normal = np.load(os.path.join(dataset_directory_path, f"robot_hand_config/{hand}/point_normal.npy"))
            hand_pc[hand] = hand_point_cloud
            hand_pn[hand] = hand_point_normal

        for dirname in pkl_file_directory:
            for n, filename in enumerate(tqdm(os.listdir(dirname))):
                filepath = os.path.join(dirname, filename)
                with open(filepath, 'rb') as f:
                    file_data = pickle.load(f)
                for m, item in enumerate(file_data):    # object num for iteration
                    if num_object is not None and m >= num_object:
                        break
                    hand_index = len(self.hands)
                    self.hands.append(item['hand'])

                    self.hand_point_clouds.append(torch.from_numpy(np.concatenate([hand_pc[item['hand']], hand_pn[item['hand']]], axis=1)).float())

                    obj_path = item['object_path']
                    self.object_paths.append(obj_path)
                    self.f_coefs.append(torch.tensor(item['f_coef'], dtype=torch.float))

                    point_cloud = torch.tensor(item['point_cloud'], dtype=torch.float)
                    sampled_normals_torch = torch.from_numpy(item['point_normal'].copy()).to(dtype=torch.float)
                    
                    offset = torch.mean(point_cloud, dim=0)
                    point_cloud -= offset
                    norm_offset = 1 * torch.max(torch.abs(point_cloud))
                    point_cloud /= norm_offset
                    
                    '''
                    add noise 
                    '''
                    noise_sigma = self.noisy_upper_level * random.random()
                    noise_normal_sigma = self.noisy_normal_upper_level * random.random()
                    noisy_points_torch = pc_add_noise_torch(point_cloud.clone(), sigma=noise_sigma)
                    noisy_normals_torch = normal_add_noise_torch(sampled_normals_torch, sigma=noise_normal_sigma)
                    flip_normal_ratio = self.random_flip_normal_upper_level * random.random()
                    num_normals_to_flip = int(flip_normal_ratio * noisy_normals_torch.shape[0])
                    indices_to_flip = torch.randperm(noisy_normals_torch.shape[0])[:num_normals_to_flip]
                    noisy_normals_torch[indices_to_flip] *= -1

                    self.point_clouds.append(noisy_points_torch)
                    self.scales.append(norm_offset)
                    self.point_normals.append(noisy_normals_torch)
                    point_to_region_id_ori = torch.tensor(item['point_to_region_id'], dtype=torch.long)
                    self.point_to_region_ids.append(point_to_region_id_ori.clone())
                    
                    for i in range(len(item['palm_pose'])): # palm num for iteration
                        if num_palm_pose is not None and i >= num_palm_pose:
                            break
                        palm_index = len(self.palm_poses)
                        self.palm_poses.append(torch.tensor(item['palm_pose'][i], dtype=torch.float))
                        self.joint_poses.append(torch.tensor(item['joint_pos'][i], dtype=torch.float))
                        contact_point = -torch.ones(self.max_num_contact_points, dtype=torch.long)
                        contact_point_data = torch.tensor(item['contact_point_id'][i], dtype=torch.long)
                        contact_point[:contact_point_data.shape[0]] = contact_point_data
                        
                        gaussian_heatmap_ori = torch.tensor(item['heatmap'][i][:, :self.max_num_contact_points], dtype=torch.float)  # shape=(n, 4)
                        
                        '''
                        remove some points and padding with random sampled points to keep the same number of points
                        '''
                        if random.random() < self.remove_pc_prob:
                            ''' try to remove '''
                            valid_contact_point_idx = contact_point[contact_point != -1]
                            contact_points_xyz = noisy_points_torch[valid_contact_point_idx]
                            noisy_points_torch_updated, nearest_idxs, random_idxs_to_pad = replace_noisy_points(noisy_points_torch.clone(), contact_points_xyz.clone(), remove_points_num=self.remove_pc_num)

                            
                            ''' also need to check whether the remove points are from the contact points'''
                            if torch.all(~torch.isin(valid_contact_point_idx, nearest_idxs)).item():
                                
                                noisy_normals_torch_updated = noisy_normals_torch.clone()
                                noisy_normals_torch_updated[nearest_idxs] = noisy_normals_torch[random_idxs_to_pad]
                                point_to_region_id_updated = point_to_region_id_ori.clone()
                                point_to_region_id_updated[nearest_idxs] = point_to_region_id_ori[random_idxs_to_pad]
                                gaussian_heatmap_updated = gaussian_heatmap_ori.clone()
                                gaussian_heatmap_updated[nearest_idxs] = gaussian_heatmap_ori[random_idxs_to_pad]
                                self.point_clouds_per_grasp.append(noisy_points_torch_updated)
                                self.point_normals_per_grasp.append(noisy_normals_torch_updated)
                                self.point_to_region_ids_per_grasp.append(point_to_region_id_updated)
                                self.gaussian_heatmaps_per_grasp.append(gaussian_heatmap_updated)
                                    
                            else: 
                                
                                self.point_clouds_per_grasp.append(noisy_points_torch)
                                self.point_normals_per_grasp.append(noisy_normals_torch)
                                self.point_to_region_ids_per_grasp.append(point_to_region_id_ori)
                                self.gaussian_heatmaps_per_grasp.append(gaussian_heatmap_ori)
                        else: 
                            self.point_clouds_per_grasp.append(noisy_points_torch)
                            self.point_normals_per_grasp.append(noisy_normals_torch)
                            self.point_to_region_ids_per_grasp.append(point_to_region_id_ori)
                            self.gaussian_heatmaps_per_grasp.append(gaussian_heatmap_ori)
                            
                        
                        contact_point[contact_point == -1] = CPPAD
                        self.contact_point_ids.append(contact_point)
                        self.gaussian_heatmaps.append(gaussian_heatmap_ori)
                        
                        if self.generate_new_wrench:
                            for j in range(num_motion):
                                target_wrench, forces = self.generate_wrench_and_force(point_cloud.numpy(), item['point_normal'], contact_point, item['f_coef'])
                                self.target_wrenches.append(target_wrench.float())
                                self.forces.append(forces.float())
                                self.hand_indices.append(hand_index)
                                self.palm_indices.append(palm_index)
                        else:
                            for j in range(item['wrenches'][i].shape[0]):
                                prob_value, f_value = from_wrench_to_contact_force(item['point_cloud'], item['point_normal'], item['contact_point_id'][i], item['wrenches'][i][j], mu=item['f_coef'])
                                if  prob_value < 0.1:
                                    self.target_wrenches.append(torch.tensor(item['wrenches'][i][j], dtype=torch.float))
                                    contact_force = f_value.reshape(-1, 3)
                                    force = torch.zeros(self.max_num_contact_points, 3)
                                    force[contact_point != CPPAD] = torch.from_numpy(contact_force).float()
                                    self.forces.append(force)
                                else:
                                    print("error")
                                self.hand_indices.append(hand_index)
                                self.palm_indices.append(palm_index)

                        if i > load_ratio * len(item['palm_pose']):
                            break
                    if only_one_object: # for tiny dataset
                        break

    def __len__(self):
        return len(self.target_wrenches)

    def generate_wrench_and_force(self, pointcloud, normal, contact_point_id, f_coef):
        # in case sometimes the contact point id is CPPAD
        force = torch.zeros(self.max_num_contact_points, 3)
        contact_idx = contact_point_id[contact_point_id != CPPAD].numpy()
        # inward normal ! !!  ! ! ! !  !
        target_wrench, contact_force = generate_random_wrenches(pointcloud[contact_idx], -normal[contact_idx], f_coef, sample_num=1, return_force=True)
        force[contact_point_id != CPPAD] = torch.from_numpy(contact_force[0]).float()

        return torch.from_numpy(target_wrench[0]).float(), force

    def wrench_to_pointcloud_motion(self, object_pointcloud, wrench, mass=1, rot_inertia=np.eye(3), start_v=np.array([0.0, 0.0, 0.0]), start_w=np.array([0.0, 0.0, 0.0]), timestep=0.5):
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
        rotation = Rotation.from_rotvec(start_w * timestep + 0.5 * angular_acc * timestep * timestep)
        rotation_matrix = rotation.as_matrix()
        moved_pointcloud = (rotation_matrix.dot(object_pointcloud.T)).T + translation

        pointcloud_motion = moved_pointcloud - object_pointcloud

        return moved_pointcloud, pointcloud_motion

    def get_motion_from_pc_and_wrench(self, pcd, normal, wrench):
        if isinstance(pcd, torch.Tensor):
            pcd = pcd.cpu().numpy()
            normal = normal.cpu().numpy()
            wrench = wrench.cpu().numpy()
        
        _, motion = self.wrench_to_pointcloud_motion(pcd, wrench)
        motion = torch.from_numpy(motion).float()

        return motion
    
    def __getitem__(self, idx):
        hand_idx = self.hand_indices[idx]
        palm_idx = self.palm_indices[idx]

        hand = self.hands[hand_idx]
        obj_path = self.object_paths[hand_idx]
        f_coef = self.f_coefs[hand_idx]
        
        point_cloud = self.point_clouds[hand_idx]
        scale = self.scales[hand_idx]
        point_normal = self.point_normals[hand_idx]
        point_to_region_id = self.point_to_region_ids[hand_idx]
        
        point_cloud_per_grasp = self.point_clouds_per_grasp[palm_idx]
        point_normal_per_grasp = self.point_normals_per_grasp[palm_idx]
        point_to_region_id_per_grasp = self.point_to_region_ids_per_grasp[palm_idx]
        gaussian_heatmap_per_grasp = self.gaussian_heatmaps_per_grasp[palm_idx]
                            
        hand_point_cloud = self.hand_point_clouds[hand_idx]
        palm_pose = self.palm_poses[palm_idx]
        joint_pos = self.joint_poses[palm_idx]
        contact_point_id  = self.contact_point_ids[palm_idx]
        gaussian_heatmap = self.gaussian_heatmaps[palm_idx]
        target_wrench = self.target_wrenches[idx]
        force = self.forces[idx]

        motion = self.get_motion_from_pc_and_wrench(point_cloud, point_normal, target_wrench)
        # motion_mask. 
        # for rigid body object, all 1
        # for deformable objects, 1 for keypoints, and 0 for non-keypoints. And we also need to multiply mask on motion to set those motion to zero for non-keypoints
        # TODO: add keypoint-wise mask for deformable objects
        motion_mask = torch.ones(motion.shape[0], 1)
        motion = motion * motion_mask

        # heatmap mask. TODO: see when to use heatmap in input. Currently we do not use it and set heatmap mask to zero
        heatmap_mask = torch.zeros(gaussian_heatmap_per_grasp.shape[0], 1)
        if random.random() < self.use_gaussian_map_ratio:
            heatmap_mask = torch.ones(gaussian_heatmap_per_grasp.shape[0], 1)
            
        
        num_points = point_cloud.shape[0]
        part_seg_indicator = torch.zeros(num_points, self.max_num_contact_points) # self.max_num_contact_points is K
        # contact point id is a k-dim vector with CPPAD and point index
        for i in range(contact_point_id.shape[0]):
            if contact_point_id[i] != CPPAD:
                part_seg_indicator[:, i][point_to_region_id_per_grasp == point_to_region_id_per_grasp[contact_point_id[i]]] = 1

        if random.random() < self.augment_part_seg_indicator_ratio:
            part_seg_indicator = torch.ones(num_points, 1)            
        else:
            part_seg_indicator = part_seg_indicator.max(dim=1, keepdim=True)[0]

        if not self.use_region:
            part_seg_indicator = torch.ones(num_points, 1)
        
        physical_properties = torch.zeros(num_points, self.physcial_property_num)    # 6 physical properties
        physical_property_mask = torch.zeros(num_points, self.physcial_property_num)    # 6 physical properties
        if self.use_physics:
            physical_properties[:, 0] = f_coef
            physical_property_mask[:, 0] = 1

        input_object_point_cloud = torch.concat(
            [
                point_cloud_per_grasp, # 3, 0-3
                point_normal_per_grasp, # 3, 3-6
                part_seg_indicator,  # 1, 6-7
                motion, # 3, 7-10
                gaussian_heatmap_per_grasp.max(dim=1, keepdim=True)[0] * heatmap_mask,   # 1, 10-11
                heatmap_mask,   # 1, 11-12
            ]
            , dim=1
        ) # num_points * 12

        if self.use_scale:
            input_object_point_cloud = torch.cat([input_object_point_cloud, torch.tensor(float(scale)).repeat(num_points, 1)], dim=1)   # 12-13

        input_object_point_cloud = torch.concat(
            [
                input_object_point_cloud,
                physical_properties, # 13-19
                physical_property_mask,  # 19-25
            ], dim=1)
        
        input_hand_point_cloud = hand_point_cloud

        contact_positions = torch.zeros(self.max_num_contact_points, 3) # self.max_num_contact_points is K
        # contact point id is a k-dim vector with CPPAD and point index
        for i in range(contact_point_id.shape[0]):
            if contact_point_id[i] != CPPAD:
                contact_positions[i] = point_cloud_per_grasp[contact_point_id[i]]

        force_map = torch.zeros(gaussian_heatmap_per_grasp.shape[0], 3)
        for j in range(self.max_num_contact_points):
            if contact_point_id[j] < CPPAD:
                force_map[gaussian_heatmap_per_grasp[:, j]>0, :] = force[j, :]
                force_map[gaussian_heatmap_per_grasp[:, j]>0, :] *= (gaussian_heatmap_per_grasp[gaussian_heatmap_per_grasp[:, j]>0, j] / torch.max(gaussian_heatmap_per_grasp[:, j])).unsqueeze(-1).repeat(1,3) # normalize the force
        force_map /= torch.linalg.norm(force_map, dim=1).max()

        # return {
        #     "input_object_point_cloud": input_object_point_cloud,
        #     "input_hand_point_cloud": input_hand_point_cloud,
        #     "hand": hand,
        #     "point_cloud": point_cloud_per_grasp,
        #     "point_normal": point_normal_per_grasp,
        #     "point_to_region_id": point_to_region_id_per_grasp,
        #     "contact_point_id": contact_point_id,
        #     "part_seg_indicator": part_seg_indicator,
        #     "heatmaps": gaussian_heatmap_per_grasp,
        #     "mix_heatmap": gaussian_heatmap_per_grasp.max(dim=1)[0],    # here we use mean rather than sum to avoid situations that more than one contact points in a region
        #     "force": force,
        #     "target_wrench": target_wrench if self.load_target_wrench else torch.zeros(6),
        #     "forcemap": force_map,
            
        #     "scale": scale,
        #     "f_coef": f_coef,
        #     "palm_pose": palm_pose,
        #     "joint_pos": joint_pos,
        #     "obj_path": obj_path,
        # }

        output_dict = {
            "input_object_point_cloud": input_object_point_cloud,
            "input_hand_point_cloud": input_hand_point_cloud,
            "hand": hand,
            "point_cloud": point_cloud_per_grasp,
            "point_normal": point_normal_per_grasp,
            "point_to_region_id": point_to_region_id_per_grasp,
            "contact_point_id": contact_point_id,
            "part_seg_indicator": part_seg_indicator,
            "heatmaps": gaussian_heatmap_per_grasp,
            "mix_heatmap": gaussian_heatmap_per_grasp.max(dim=1)[0],    # here we use mean rather than sum to avoid situations that more than one contact points in a region
            "force": force,
            "target_wrench": target_wrench if self.load_target_wrench else torch.zeros(6),
            "forcemap": force_map,
            
            "scale": scale,
            "f_coef": f_coef,
            "palm_pose": palm_pose,
            "joint_pos": joint_pos,
            "obj_path": obj_path,
        }

        for k,v in output_dict.items():
            if isinstance(v, np.ndarray):
                print(k)
                assert False

        return {
            "input_object_point_cloud": input_object_point_cloud,
            "input_hand_point_cloud": input_hand_point_cloud,
            "hand": hand,
            "point_cloud": point_cloud_per_grasp,
            "point_normal": point_normal_per_grasp,
            "point_to_region_id": point_to_region_id_per_grasp,
            "contact_point_id": contact_point_id,
            "part_seg_indicator": part_seg_indicator,
            "heatmaps": gaussian_heatmap_per_grasp,
            "mix_heatmap": gaussian_heatmap_per_grasp.max(dim=1)[0],    # here we use mean rather than sum to avoid situations that more than one contact points in a region
            "force": force,
            "target_wrench": target_wrench if self.load_target_wrench else torch.zeros(6),
            "forcemap": force_map,
            
            # "scale": scale,
            # "f_coef": f_coef,
            # "palm_pose": palm_pose,
            # "joint_pos": joint_pos,
            "obj_path": obj_path,
        }
    