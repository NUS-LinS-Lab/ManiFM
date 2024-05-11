import os
import sys
sys.path.append(".")
import pickle
import torch
from torch.utils.data import Dataset
from utils.data_utils import pc_add_noise_torch, normal_add_noise_torch
from tqdm import tqdm
import random
import numpy as np

CPPAD = 2048
HANDS = ["LeapHand", "PandaHand", "Kinova3FHand", "MovoArm"]

file_path = __file__
dataset_directory_path = os.path.dirname(file_path)

class MPMDataset(Dataset):
    """
    Load "*.pkl" files. The format is as follows:
    init state <class'numpy .ndarray'>
    (2848，3)
    target state <class 'numpy.ndarray'>
    (2048，3)
    keypoints <class 'numpy.ndarray'>
    (10,)
    target state normal <class 'numpy.ndarray'>
    (2848，3)
    init state normal <class 'numpy.ndarray'>
    (2048，3)
    response matrix <class 'numpy.ndarray'>(10，2048，3，3)
    action <class 'numpy.ndarray'>
    (4，3)
    num attached points <class 'int'>
    4
    E <class 'int'>
    80
    nu <class"float'>
    attached point target <class 'list'>[[5, 69, 122, 254, 285, 442, 482, 486, 524, 597, 628, 622, 701, 752, 819, 843, 944, 985, 995, 1013, 1021, 1078, 1079,1098, 1505,1568, 1611, 1698, 1789, 1798, 1832, 1833, 1834], [22, 117, 176, 385, 615, 660, 764, 906, 987, 1036,1090, 1139, 1292, 1349, 1366, 1398, 1416, 1428, 1788, 1784, 1785, 1888, 1821, 1823, 1856, 18571, [84, 158, 504, 6341304, 1434, 1802, 2015, 2044], [78, 189, 230, 396, 437, 506, 605, 686, 717, 725, 728, 811, 937, 1373,758，1231，1458，1595，1694，1738， 1749， 1752，1763， 1767， 1903, 1907]]
    """
    def __init__(
        self, 
        pkl_file_directory, 
        use_scale=False, 
        gaussian_heatmap_max_value=10.0,
        generate_new_wrench=False,
        load_ratio=1.0,
        only_one_object=False,
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
        
        self.hands = []
        self.hands_graph = []
        self.object_paths = []
        self.scales = []
        self.Es = []
        self.nus = []
        self.point_clouds = []
        self.point_normals = []
        self.point_to_region_ids = []
        self.hand_point_clouds = []

        
        self.contact_point_ids = []
        self.gaussian_heatmaps = []
        self.num_attached_points = []

        self.forces = []
        self.motions = []
        self.keypoints = []

        self.use_scale = use_scale

        self.max_num_contact_points = 4
        self.gaussian_heatmap_max_value = gaussian_heatmap_max_value

        self.load_ratio = load_ratio
        self.use_gaussian_map_ratio = use_gaussian_map_ratio
        self.augment_part_seg_indicator_ratio = augment_part_seg_indicator_ratio
        self.physcial_property_num = physcial_property_num
        self.use_region = use_region
        self.use_physics = use_physics
        
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
                if random.random() > self.load_ratio:   # only load load_ratio data
                    continue    
                filepath = os.path.join(dirname, filename)
                with open(filepath, 'rb') as f:
                    file_data = pickle.load(f)
                for m, item in enumerate(file_data):    # object num for iteration
                    item["hand"] = "LeapHand"
                    self.hands.append(item['hand'])
                    
                    self.hand_point_clouds.append(torch.from_numpy(np.concatenate([hand_pc[item['hand']], hand_pn[item['hand']]], axis=1)).float())

                    self.object_paths.append("")                    
                    self.Es.append(torch.tensor(item['E'], dtype=torch.float))
                    self.nus.append(torch.tensor(item['nu'], dtype=torch.float))
                    normalized_init_state = torch.tensor(item['init_state'], dtype=torch.float)
                    offset = torch.mean(normalized_init_state, dim=0)
                    normalized_init_state -= offset
                    norm_offset = 1 * torch.max(torch.abs(normalized_init_state))
                    normalized_init_state /= norm_offset

                    normalize_target_state = torch.tensor(item['target_state'], dtype=torch.float)
                    normalize_target_state -= offset
                    normalize_target_state /= norm_offset

                    noise_sigma = self.noisy_upper_level * random.random()
                    noise_normal_sigma = self.noisy_normal_upper_level * random.random()
                    noisy_points_torch = pc_add_noise_torch(normalized_init_state.clone(), sigma=noise_sigma)
                    sampled_normals_torch = torch.from_numpy(item['init_state_normal'].copy()).to(dtype=torch.float)
                    noisy_normals_torch = normal_add_noise_torch(sampled_normals_torch, sigma=noise_normal_sigma)
                    flip_normal_ratio = self.random_flip_normal_upper_level * random.random()
                    num_normals_to_flip = int(flip_normal_ratio * noisy_normals_torch.shape[0])
                    indices_to_flip = torch.randperm(noisy_normals_torch.shape[0])[:num_normals_to_flip]
                    noisy_normals_torch[indices_to_flip] *= -1
                    
                    self.point_clouds.append(noisy_points_torch)
                    self.point_normals.append(noisy_normals_torch)
                    self.scales.append(norm_offset)
                    self.point_to_region_ids.append(torch.zeros(2048, dtype=torch.long)) # no prior segmentation for mpm

                    # contact_point is randomly selected for each contact region, this is OK because we only use this for exclude the 2048 padding
                    self.num_attached_points.append(item["num_attached_points"])
                    contact_point = torch.ones(self.max_num_contact_points, dtype=torch.long) * CPPAD
                    for i in range(item["num_attached_points"]):
                        if len(item["attached_point_target"][i]) == 0:
                            continue

                        contact_point[i] = random.choice(item["attached_point_target"][i])
                    self.contact_point_ids.append(contact_point)
                    
                    self.gaussian_heatmaps.append(torch.tensor(item['heatmap'][:, :self.max_num_contact_points], dtype=torch.float))
                    self.forces.append(torch.tensor(item['action']/norm_offset, dtype=torch.float))
                    self.motions.append(normalize_target_state - normalized_init_state)

    def __len__(self):
        return len(self.gaussian_heatmaps)
    
    def __getitem__(self, idx):
        hand = self.hands[idx]
        obj_path = self.object_paths[idx]
        E = self.Es[idx]
        nu = self.nus[idx]
        point_cloud = self.point_clouds[idx]
        scale = self.scales[idx]
        point_normal = self.point_normals[idx]
        point_to_region_id = self.point_to_region_ids[idx]
        hand_point_cloud = self.hand_point_clouds[idx]
        contact_point_id  = self.contact_point_ids[idx]
        gaussian_heatmap = self.gaussian_heatmaps[idx]
        force = self.forces[idx]
        motion = self.motions[idx]

        # motion_mask. 
        # for rigid body object, all 1
        # for deformable objects, 1 for keypoints, and 0 for non-keypoints. And we also need to multiply mask on motion to set those motion to zero for non-keypoints
        # TODO: add keypoint-wise mask for deformable objects
        motion_mask = torch.ones(motion.shape[0], 1)
        motion = motion * motion_mask

        # heatmap mask. TODO: see when to use heatmap in input. Currently we do not use it and set heatmap mask to zero
        heatmap_mask = torch.zeros(gaussian_heatmap.shape[0], 1)
        if random.random() < self.use_gaussian_map_ratio:
            heatmap_mask = torch.ones(gaussian_heatmap.shape[0], 1)
        
        num_points = point_cloud.shape[0]
        part_seg_indicator = torch.ones_like(gaussian_heatmap) # here part seg indicator is heatmap

        if random.random() < self.augment_part_seg_indicator_ratio:
            part_seg_indicator = torch.ones(num_points, 1)            
        else:
            part_seg_indicator = part_seg_indicator.max(dim=1, keepdim=True)[0]

        if not self.use_region:
            part_seg_indicator = torch.ones_like(gaussian_heatmap)
        
        physical_properties = torch.zeros(num_points, self.physcial_property_num)    # 6 physical properties
        physical_property_mask = torch.zeros(num_points, self.physcial_property_num)    # 6 physical properties
        if self.use_physics:
            physical_properties[:, 1] = 1/E
            physical_property_mask[:, 1] = 1
            physical_properties[:, 2] = nu
            physical_property_mask[:, 2] = 1

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
        ) # num_points * 12

        if self.use_scale:
            input_object_point_cloud = torch.cat([input_object_point_cloud, torch.tensor(float(scale)).repeat(num_points, 1)], dim=1)  # 12-13

        input_object_point_cloud = torch.concat(
            [
                input_object_point_cloud,
                physical_properties,
                physical_property_mask,
            ], dim=1)
        
        input_hand_point_cloud = hand_point_cloud

        force_map = torch.zeros(gaussian_heatmap.shape[0], 3)
        for j in range(self.max_num_contact_points):
            if contact_point_id[j] < CPPAD:
                force_map[gaussian_heatmap[:, j]>0, :] = force[j, :] / 10
                force_map[gaussian_heatmap[:, j]>0, :] *= (gaussian_heatmap[gaussian_heatmap[:, j]>0, j] / torch.max(gaussian_heatmap[:, j])).unsqueeze(-1).repeat(1,3) # normalize the force
        force_map /= torch.linalg.norm(force_map, dim=1).max()

        # return {
        #     "input_object_point_cloud": input_object_point_cloud,
        #     "input_hand_point_cloud": input_hand_point_cloud,
        #     "hand": hand,
        #     "point_cloud": point_cloud,
        #     "point_normal": point_normal,
        #     "point_to_region_id": point_to_region_id,
        #     "contact_point_id": contact_point_id,            
        #     "part_seg_indicator": part_seg_indicator,
        #     "heatmaps": gaussian_heatmap,
        #     "mix_heatmap": gaussian_heatmap.max(dim=1)[0],    # here we use mean rather than sum to avoid situations that more than one contact points in a region
        #     "force": force,
        #     "target_wrench": torch.zeros(6),    # dummy
        #     "forcemap": force_map,

        #     "scale": scale, #
        #     "f_coef": torch.tensor(1), # no f_coef for mpm
        #     "palm_pose": torch.zeros(6), # no gt palm pose for mpm
        #     "joint_pos": torch.zeros(16), # no gt joint pos for mpm
        #     "obj_path": obj_path, #
        # }

        output_dict = {
            "input_object_point_cloud": input_object_point_cloud,
            "input_hand_point_cloud": input_hand_point_cloud,
            "hand": hand,
            "point_cloud": point_cloud,
            "point_normal": point_normal,
            "point_to_region_id": point_to_region_id,
            "contact_point_id": contact_point_id,            
            "part_seg_indicator": part_seg_indicator,
            "heatmaps": gaussian_heatmap,
            "mix_heatmap": gaussian_heatmap.max(dim=1)[0],    # here we use mean rather than sum to avoid situations that more than one contact points in a region
            "force": force,
            "target_wrench": torch.zeros(6),    # dummy
            "forcemap": force_map,

            "scale": scale, #
            "f_coef": torch.tensor(1), # no f_coef for mpm
            "palm_pose": torch.zeros(6), # no gt palm pose for mpm
            "joint_pos": torch.zeros(16), # no gt joint pos for mpm
            "obj_path": obj_path, #
        } 

        for k,v in output_dict.items():
            if isinstance(v, np.ndarray):
                print(k)
                assert False
    
        return {
            "input_object_point_cloud": input_object_point_cloud,
            "input_hand_point_cloud": input_hand_point_cloud,
            "hand": hand,
            "point_cloud": point_cloud,
            "point_normal": point_normal,
            "point_to_region_id": point_to_region_id,
            "contact_point_id": contact_point_id,            
            "part_seg_indicator": part_seg_indicator,
            "heatmaps": gaussian_heatmap,
            "mix_heatmap": gaussian_heatmap.max(dim=1)[0],    # here we use mean rather than sum to avoid situations that more than one contact points in a region
            "force": force,
            "target_wrench": torch.zeros(6),    # dummy
            "forcemap": force_map,

            # "scale": scale, #
            # "f_coef": torch.tensor(1), # no f_coef for mpm
            # "palm_pose": torch.zeros(6), # no gt palm pose for mpm
            # "joint_pos": torch.zeros(16), # no gt joint pos for mpm
            "obj_path": obj_path, #
        }