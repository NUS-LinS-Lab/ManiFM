import os
import sys
sys.path.append(".")
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.data_utils import pc_add_noise_torch, normal_add_noise_torch
import random
import numpy as np

CPPAD = 2048

file_path = __file__
dataset_directory_path = os.path.dirname(file_path)

class ClothDataset(Dataset):
    '''
    deformable.npz
    T: number of timestep
    N: number of points in point clouds
    M: number forward steps
    K: number of key points
    {   
        density, <class 'numpy.ndarray'>, shape ()
        init_state, <class 'numpy.ndarray'>, shape (T, N, 3)
        init_state_normal, <class 'numpy.ndarray'>, shape (T, N, 3)
        attached_point, <class 'numpy.ndarray'>, shape (T, M, 2)
        attached_point_target, <class 'numpy.ndarray'>, shape (T, M, 2, 3)
        target_state, <class 'numpy.ndarray'>, shape (T, M, K, 3)
        target_state_normal, <class 'numpy.ndarray'>, shape (T, M, K, 3)
        response_matrix, <class 'numpy.ndarray'>, shape (T, K, N, 3, 3)
        all_target_state, <class 'numpy.ndarray'>, shape (T, M, N, 3)
        frictional_coeff, <class 'numpy.ndarray'>, shape ()
        k_stiff_stretching, <class 'numpy.ndarray'>, shape ()
        k_stiff_bending, <class 'numpy.ndarray'>, shape ()
        keypoints, <class 'numpy.ndarray'>, shape (K,)
        heatmap, <class 'numpy.ndarray'>, shape (T, M, N, 4)
    }
    '''
    def __init__(
        self, 
        cloth_file_directory, 
        use_scale=False, 
        gaussian_heatmap_max_value=10.0,
        generate_new_wrench=False, # unify the input for ConcatDataset
        load_ratio=1.0, # unify the input for ConcatDataset
        only_one_object=False, # unify the input for ConcatDataset
        use_gaussian_map_ratio=0.0,
        augment_part_seg_indicator_ratio=0.0,
        physcial_property_num=6,
        use_region=True,
        use_physics=True,
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
        self.point_to_region_ids = []
        self.hand_point_clouds = []

        # indexed by point cloud index
        self.point_clouds = []
        self.point_normals = []

        # indexed by f coef index
        self.f_coefs = []
        self.k_stiff_stretchings = []
        self.k_stiff_bendings = []

        self.contact_point_ids = []
        self.gaussian_heatmaps = []
        self.motions = []
        self.contact_motions = []

        self.hand_indices = []
        self.point_cloud_indices = []
        self.f_coef_indices = []

        self.use_scale = use_scale
        

        self.max_num_contact_points = 4
        self.gaussian_heatmap_max_value = gaussian_heatmap_max_value

        self.use_gaussian_map_ratio = use_gaussian_map_ratio
        self.augment_part_seg_indicator_ratio = augment_part_seg_indicator_ratio
        self.physcial_property_num = physcial_property_num
        self.use_region = use_region
        self.use_physics = use_physics
        self.noisy_upper_level = noisy_upper_level
        self.noisy_normal_upper_level = noisy_normal_upper_level
        self.random_flip_normal_upper_level = random_flip_normal_upper_level
        
        # MovoArm is the only robot we use for cloth
        hand_point_cloud = np.load(os.path.join(dataset_directory_path, f"robot_hand_config/MovoArm/point_cloud.npy"))
        hand_point_normal = np.load(os.path.join(dataset_directory_path, f"robot_hand_config/MovoArm/point_normal.npy"))
        
        # all the data below is unavailable in npz file and is added now
        hand_index = len(self.hands_graph)
        self.hands.append("movo")
        
        obj_path = ""
        self.object_paths.append(obj_path)
        self.point_to_region_ids.append(torch.zeros(size=(2048,), dtype=torch.long))

        self.hand_point_clouds.append(torch.from_numpy(np.concatenate([hand_point_cloud, hand_point_normal], axis=1)).float())
            
        for cloth_dir in cloth_file_directory:
            for n, npz_file in enumerate(tqdm(os.listdir(cloth_dir))):
                if npz_file.endswith('heatmap.npz'):
                    filepath = os.path.join(cloth_dir, npz_file)
                    data = np.load(filepath)
                    data = dict(data)
                    f_coef_index = len(self.f_coefs)
                    obj_path = filepath # ""
                    self.object_paths.append(obj_path)
                    self.f_coefs.append(torch.tensor(data['frictional_coeff'], dtype=torch.float))
                    try:
                        self.k_stiff_stretchings.append(torch.tensor(data['k_stiff_stretching'], dtype=torch.float))
                    except:
                        self.k_stiff_stretchings.append(torch.tensor(data['kp'], dtype=torch.float))
                    try:
                        self.k_stiff_bendings.append(torch.tensor(data['k_stiff_bending'], dtype=torch.float))
                    except:
                        self.k_stiff_bendings.append(torch.tensor(data['kd'], dtype=torch.float))
                    for t in range(data["target_state"].shape[0]):
                        init_state = data['init_state'][t]
                        mass_pos = np.mean(init_state, axis=0)
                        init_state -= mass_pos
                        norm_denom = 1 * np.max(np.abs(init_state))
                        init_state /= norm_denom
                        noise_sigma = self.noisy_upper_level * random.random()
                        noise_normal_sigma = self.noisy_normal_upper_level * random.random()
                        noisy_points_torch = pc_add_noise_torch(torch.from_numpy(init_state.copy()).to(dtype=torch.float), sigma=noise_sigma)
                        sampled_normals_torch = torch.from_numpy(data['init_state_normal'][t].copy()).to(dtype=torch.float)
                        noisy_normals_torch = normal_add_noise_torch(sampled_normals_torch, sigma=noise_normal_sigma)
                        flip_normal_ratio = self.random_flip_normal_upper_level * random.random()
                        num_normals_to_flip = int(flip_normal_ratio * noisy_normals_torch.shape[0])
                        indices_to_flip = torch.randperm(noisy_normals_torch.shape[0])[:num_normals_to_flip]
                        noisy_normals_torch[indices_to_flip] *= -1

                        self.point_normals.append(noisy_normals_torch)
                        point_cloud_index = len(self.point_clouds)
                        self.point_clouds.append(noisy_points_torch)
                        self.scales.append(norm_denom)

                        self.point_normals.append(noisy_normals_torch)
                      
                        for m in range(data["target_state"].shape[1]):
                            contact_point = torch.tensor(data['attached_point'][t][m], dtype=torch.long)
                            contact_point[contact_point == -1] = CPPAD
                            self.contact_point_ids.append(contact_point)
                            self.gaussian_heatmaps.append(torch.tensor(data['heatmap'][t][m][:, :self.max_num_contact_points], dtype=torch.float))
                            try:
                                target_state = (data['all_target_state'][t][m] - mass_pos) / norm_denom
                            except:
                                target_state = (data['target_state'][t][m] - mass_pos) / norm_denom
                            motion = target_state - init_state
                            self.motions.append(torch.tensor(motion, dtype=torch.float))

                            target_attached_points = (data['attached_point_target'][t][m] - mass_pos) / norm_denom
                            attach_point_ids = data['attached_point'][t][m].astype(np.int64)
                            attach_points = init_state[attach_point_ids[attach_point_ids!=CPPAD]]
                            target_attached_points = target_attached_points[attach_point_ids[:2]!=CPPAD]
                            contact_motion = torch.zeros(self.max_num_contact_points, 3)
                            contact_motion[attach_point_ids!=CPPAD] = torch.tensor(target_attached_points - attach_points, dtype=torch.float)
                            self.contact_motions.append(contact_motion)

                            self.hand_indices.append(hand_index)
                            self.point_cloud_indices.append(point_cloud_index)
                            self.f_coef_indices.append(f_coef_index)

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        hand_idx = self.hand_indices[idx]
        point_cloud_idx = self.point_cloud_indices[idx]
        f_coef_idx = self.f_coef_indices[idx]

        hand = self.hands[hand_idx]
        obj_path = self.object_paths[f_coef_idx]
        point_to_region_id = self.point_to_region_ids[hand_idx]
        hand_point_cloud = self.hand_point_clouds[hand_idx]

        point_cloud = self.point_clouds[point_cloud_idx]
        scale = self.scales[point_cloud_idx]
        point_normal = self.point_normals[point_cloud_idx]

        f_coef = self.f_coefs[f_coef_idx]
        k_stiff_stretching = self.k_stiff_stretchings[f_coef_idx]
        k_stiff_bending = self.k_stiff_bendings[f_coef_idx]

        contact_point_id  = self.contact_point_ids[idx]
        gaussian_heatmap = self.gaussian_heatmaps[idx]
        motion = self.motions[idx]

        # motion_mask. 
        # for rigid body object, all 1
        # for deformable objects, 1 for keypoints, and 0 for non-keypoints. And we also need to multiply mask on motion to set those motion to zero for non-keypoints
        # TODO: add keypoint-wise mask for deformable objects
        motion_mask = torch.ones(motion.shape[0], 1)
        motion = motion * motion_mask

        contact_motion = self.contact_motions[idx]

        # heatmap mask. TODO: see when to use heatmap in input. Currently we do not use it and set heatmap mask to zero
        heatmap_mask = torch.zeros(gaussian_heatmap.shape[0], 1)
        if random.random() < self.use_gaussian_map_ratio:
            heatmap_mask = torch.ones(gaussian_heatmap.shape[0], 1)
        
        num_points = point_cloud.shape[0]
        part_seg_indicator = torch.ones(num_points, 1) # self.max_num_contact_points is K
        
        if not self.use_region:
            part_seg_indicator = torch.ones(num_points, 1)

        physical_properties = torch.zeros(num_points, self.physcial_property_num)    # 6 physical properties
        physical_property_mask = torch.zeros(num_points, self.physcial_property_num)    # 6 physical properties
        if self.use_physics:
            physical_properties[:, 0] = f_coef  # 13-14
            physical_property_mask[:, 0] = 1
            physical_properties[:, 3] = k_stiff_stretching # 16-17
            physical_property_mask[:, 3] = 1
            physical_properties[:, 4] = k_stiff_bending  # 17-18
            physical_property_mask[:, 4] = 1

        input_object_point_cloud = torch.concat(
            [
                point_cloud, # 0-3
                point_normal, # 3-6
                part_seg_indicator,  # 6-7
                motion, # 7-10
                gaussian_heatmap.max(dim=1, keepdim=True)[0] * heatmap_mask,   # 10-11
                heatmap_mask,   # 11-12
            ]
            , dim=1
        ) # num_points * 32

        if self.use_scale:  # 12-13
            input_object_point_cloud = torch.cat([input_object_point_cloud, torch.tensor(float(scale)).repeat(num_points, 1)], dim=1)  
               
        input_object_point_cloud = torch.concat(
            [
                input_object_point_cloud,
                physical_properties,
                physical_property_mask,
            ], dim=1)

        input_hand_point_cloud = hand_point_cloud

        contact_positions = torch.zeros(self.max_num_contact_points, 3) # self.max_num_contact_points is K
        # contact point id is a k-dim vector with CPPAD and point index
        for i in range(contact_point_id.shape[0]):
            if contact_point_id[i] != CPPAD:
                contact_positions[i] = point_cloud[contact_point_id[i]]

        force_map = torch.zeros(gaussian_heatmap.shape[0], 3)
        for j in range(self.max_num_contact_points):
            if contact_point_id[j] < CPPAD:
                force_map[gaussian_heatmap[:, j]>0, :] = contact_motion[j, :]
                force_map[gaussian_heatmap[:, j]>0, :] *= (gaussian_heatmap[gaussian_heatmap[:, j]>0, j] / torch.max(gaussian_heatmap[:, j])).unsqueeze(-1).repeat(1,3) # normalize the force
        force_map /= torch.linalg.norm(force_map, dim=1).max()

        # return {
        #     "input_object_point_cloud": input_object_point_cloud,
        #     "input_hand_point_cloud": input_hand_point_cloud,
        #     "hand": hand, #
        #     "point_cloud": point_cloud, #
        #     "point_normal": point_normal, #
        #     "point_to_region_id": point_to_region_id, # 
        #     "contact_point_id": contact_point_id, #
        #     "part_seg_indicator": part_seg_indicator,
        #     "heatmaps": gaussian_heatmap, #
        #     "mix_heatmap": gaussian_heatmap.max(dim=1)[0], #    # here we use mean rather than sum to avoid situations that more than one contact points in a region
        #     "force": contact_motion,
        #     "target_wrench": torch.zeros(6), # no target wrench for cloth
        #     "forcemap": force_map,
            
        #     "scale": scale, #
        #     "f_coef": f_coef, #
        #     "palm_pose": torch.zeros(6), # no gt palm pose for cloth
        #     "joint_pos": torch.zeros(16), # no gt joint pos for cloth
        #     "obj_path": obj_path, #
        # }

        output_dict = {
            "input_object_point_cloud": input_object_point_cloud,
            "input_hand_point_cloud": input_hand_point_cloud,
            "hand": hand, #
            "point_cloud": point_cloud, #
            "point_normal": point_normal, #
            "point_to_region_id": point_to_region_id, # 
            "contact_point_id": contact_point_id, #
            "part_seg_indicator": part_seg_indicator,
            "heatmaps": gaussian_heatmap, #
            "mix_heatmap": gaussian_heatmap.max(dim=1)[0], #    # here we use mean rather than sum to avoid situations that more than one contact points in a region
            "force": contact_motion,
            "target_wrench": torch.zeros(6), # no target wrench for cloth
            "forcemap": force_map,
            
            "scale": scale, #
            "f_coef": f_coef, #
            "palm_pose": torch.zeros(6), # no gt palm pose for cloth
            "joint_pos": torch.zeros(16), # no gt joint pos for cloth
            "obj_path": obj_path, #
        }

        for k,v in output_dict.items():
            if isinstance(v, np.ndarray):
                print(k)
                assert False

        return {
            "input_object_point_cloud": input_object_point_cloud,
            "input_hand_point_cloud": input_hand_point_cloud,
            "hand": hand, #
            "point_cloud": point_cloud, #
            "point_normal": point_normal, #
            "point_to_region_id": point_to_region_id, # 
            "contact_point_id": contact_point_id, #
            "part_seg_indicator": part_seg_indicator,
            "heatmaps": gaussian_heatmap, #
            "mix_heatmap": gaussian_heatmap.max(dim=1)[0], #    # here we use mean rather than sum to avoid situations that more than one contact points in a region
            "force": contact_motion,
            "target_wrench": torch.zeros(6), # no target wrench for cloth
            "forcemap": force_map,
            
            # "scale": scale, #
            # "f_coef": f_coef, #
            # "palm_pose": torch.zeros(6), # no gt palm pose for cloth
            # "joint_pos": torch.zeros(16), # no gt joint pos for cloth
            "obj_path": obj_path, #
        }

