'''
Author: Zhixuan Xu 
Last Modified: 2023.09.13
'''

import os 
import pybullet as p
import numpy as np 
import trimesh
import torch
import pytorch_kinematics as pk
from utils.opt_utils import (
    extract_colors_from_urdf,
    as_mesh,
    sample_init_palm_poses,
    generate_joint_combinations,
    pose_vector_to_transformation_matrix,
    transform_vectors
)
from scipy.spatial.transform import Rotation as R

class RobotHand:
    def __init__(self,
                 hand_urdf_path,
                 hand_mesh_path,
                 physics_client_id):
        
        # Initialize the path to the URDF file and the physics client ID
        self.physics_client_id = physics_client_id
        self.hand_urdf_path = hand_urdf_path
        self.hand_mesh_path = hand_mesh_path

        # Load the hand model and set basic information
        self.pb_load_hand()
        self.set_basic_hand_info()

        # Set the visualizer 
        self.set_hand_visualizer()

    def pb_load_hand(self):
        # Load the hand URDF into the physics client
        self.pb_hand = p.loadURDF(self.hand_urdf_path, 
                                  useFixedBase=1,
                                  flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS, # Important: Don't use p.URDF_MERGE_FIXED_LINKS, as it can cause issues with the tip link
                                  physicsClientId=self.physics_client_id)
                                  
        # Get the total number of joints, including fixed ones
        self.num_joints = p.getNumJoints(self.pb_hand)
        
        # Initialize variables for degree of freedom and joint indices
        self.dof = 0
        self.pb_joint_idx = []
        lower_joint_limits = []
        upper_joint_limits = []
        self.lower_joint_limits_w_fixed = []
        self.upper_joint_limits_w_fixed = []
        self.pb_joint_max_forces = []

        # print(f"All together {self.num_joints} joints (containing the fixed joints)!")

        # Mapping of joint type IDs to their namesor clarity
        joint_typeid_to_name = ["JOINT_REVOLUTE", "JOINT_PRISMATIC", "JOINT_SPHERICAL", "JOINT_PLANAR", "JOINT_FIXED"]
        
        # Iterate over all joints
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.pb_hand, i)
            # print(f"Joint {i}: {joint_info[1].decode('utf-8'), joint_typeid_to_name[joint_info[2]]}")
            self.lower_joint_limits_w_fixed.append(joint_info[8])
            self.upper_joint_limits_w_fixed.append(joint_info[9])
        
            # Filter out the fixed joints
            if joint_info[2] != p.JOINT_FIXED:
                self.pb_joint_idx.append(i)
                self.dof += 1
                lower_joint_limits.append(joint_info[8])
                upper_joint_limits.append(joint_info[9])
                self.pb_joint_max_forces.append(joint_info[10])
            

        # Convert joint limits to numpy arrays for easier calculations  
        self.pb_upper_joint_limits = np.array(upper_joint_limits)
        self.pb_lower_joint_limits = np.array(lower_joint_limits)

        # Calculate joint ranges and rest positions
        self.pb_joint_ranges = (self.pb_upper_joint_limits - self.pb_lower_joint_limits).tolist()
        self.pb_rest_joint_pos = ((self.pb_upper_joint_limits + self.pb_lower_joint_limits) / 2).tolist()

        # Convert back to lists for storage
        self.pb_upper_joint_limits = self.pb_upper_joint_limits.tolist()
        self.pb_lower_joint_limits = self.pb_lower_joint_limits.tolist()

        # print(f"All together {self.dof} joints are controllable: {self.pb_joint_idx}")
        # print(f"Lower joint limits are: {self.pb_lower_joint_limits}")
        # print(f"Upper joint limits are: {self.pb_upper_joint_limits}")

        self.pb_object = None 


    def set_basic_hand_info(self):
        '''
        This method is meant to set basic hand information.
        As it's a base class, this method is not implemented here.
        It should be implemented in the derived classes.
        '''
        raise NotImplementedError   
    

    def set_hand_visualizer(self, device="cpu"):
        '''
        Construct a trimesh visualizer for the robotic hand based on its URDF file.
        This method processes the URDF file to extract geometries of the links and 
        store them in a dictionary for later use in visualization.

        Parameters:
        - device (str): The device on which to perform tensor computations, typically "cpu" or "cuda". Default is "cpu".
        '''
        self.chain = pk.build_chain_from_urdf(open(self.hand_urdf_path).read())
        self.hand_mesh = {}
        link_colors_from_urdf = extract_colors_from_urdf(self.hand_urdf_path)

        def build_mesh_recurse(body):
            if (len(body.link.visuals) > 0 and body.link.visuals[0].geom_type is not None):
                link_vertices = []
                link_faces = []
                colors = []
                n_link_vertices = 0
                for visual in body.link.visuals:
                    scale = torch.tensor(
                        [1, 1, 1], dtype=torch.float, device=device)
                    mesh_path = None
                    if visual.geom_type == "box":
                        link_mesh = trimesh.primitives.Box(
                            extents=np.array(visual.geom_param))
                    elif visual.geom_type == "capsule":
                        link_mesh = trimesh.primitives.Capsule(
                            radius=visual.geom_param[0], height=visual.geom_param[1]*2).apply_translation((0, 0, -visual.geom_param[1]))
                    elif visual.geom_type == "sphere":
                        link_mesh = trimesh.primitives.Capsule(
                            radius=visual.geom_param, height=0).apply_translation((0, 0, 0))
                    elif visual.geom_type == "mesh":
                        print(visual.geom_param)
                        mesh_path = os.path.join(self.hand_mesh_path, visual.geom_param)
                        link_mesh = as_mesh(trimesh.load_mesh(mesh_path, process=False))
                        link_file_name = visual.geom_param
                        # if visual.geom_param[1] is not None:
                        #     print(visual.geom_param)
                        #     scale = (visual.geom_param[1]).to(dtype=torch.float, device=device)
                    link_hull = link_mesh.convex_hull
                    vertices = torch.tensor(
                        link_mesh.vertices, dtype=torch.float, device=device)
                    faces = torch.tensor(
                        link_mesh.faces, dtype=torch.float, device=device)
                    pos = visual.offset.to(dtype=torch.float, device=device)
                    vertices = vertices * scale
                    vertices = pos.transform_points(vertices)
                    link_vertices.append(vertices)
                    link_faces.append(faces + n_link_vertices)
                    n_link_vertices += len(vertices)
                link_vertices = torch.cat(link_vertices, dim=0)
                link_faces = torch.cat(link_faces, dim=0)
                self.hand_mesh[body.link.name] = {
                                                'mesh_path': mesh_path,
                                                'mesh': link_mesh,
                                                'hull': link_hull,
                                                'vertices': link_vertices,
                                                'faces': link_faces,
                                                'offset': pos.get_matrix()[:, :3, 3],
                                                'colors': colors 
                                                }
            for children in body.children:
                build_mesh_recurse(children)
        build_mesh_recurse(self.chain._root)

        # Save colors to self.hand_mesh
        for link_name in self.hand_mesh:
            self.hand_mesh[link_name]['colors'] = link_colors_from_urdf.get(link_name, [1.0, 0., 0., 1.0])  # default to silver color if not found

    def get_hand_part_trimesh(self, link_name):
        v = self.hand_mesh[link_name]['vertices']
        v = v.detach().cpu()
        f = self.hand_mesh[link_name]['faces'].detach().cpu()
        part_mesh = trimesh.Trimesh(vertices=v, faces=f)
        colors = np.array(self.hand_mesh[link_name]['colors']) * 255
        part_mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=colors)
        return part_mesh
        
    def get_hand_trimesh(self, joint_pos, X_w_b=torch.eye(4)):
        '''
        Get the trimesh representation of the robotic hand based on the provided joint positions and base transformation.

        Parameters:
        - joint_pos (list of float): Joint positions of the robot hand.
        - X_w_b (torch.tensor): A 4x4 transformation matrix representing the pose of the hand base in the world frame.

        Returns:
        - data (trimesh.Trimesh): A trimesh object representing the robotic hand in its current pose.
        '''
        self.current_status = self.chain.forward_kinematics(th=torch.tensor(joint_pos, dtype=torch.float).unsqueeze(0), world=pk.Transform3d(matrix=X_w_b))
        # Create an empty trimesh.Scene
        scene = trimesh.Scene()

        for link_name in self.hand_mesh:
            v = self.current_status[link_name].transform_points(
                self.hand_mesh[link_name]['vertices'])
            if len(v.shape) == 3:
                v = v[0]
            v = v.detach().cpu()
            f = self.hand_mesh[link_name]['faces'].detach().cpu()
            part_mesh = trimesh.Trimesh(vertices=v, faces=f)
            # Create a mesh with color for the link
            colors = np.array(self.hand_mesh[link_name]['colors']) * 255
            part_mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=colors)
            scene.add_geometry(part_mesh)
        return scene
    
    def get_hand_trimesh_hull(self, joint_pos, X_w_b=torch.eye(4)):
        '''
        Get the trimesh representation of the robotic hand based on the provided joint positions and base transformation.

        Parameters:
        - joint_pos (list of float): Joint positions of the robot hand.
        - X_w_b (torch.tensor): A 4x4 transformation matrix representing the pose of the hand base in the world frame.

        Returns:
        - data (trimesh.Trimesh): A trimesh object representing the robotic hand in its current pose.
        '''
        self.current_status = self.chain.forward_kinematics(th=torch.tensor(joint_pos, dtype=torch.float).unsqueeze(0), world=pk.Transform3d(matrix=X_w_b))
        # Create an empty trimesh.Scene
        scene = trimesh.Scene()
        scene_dict = {}
        for link_name in self.hand_mesh:
            part_mesh_hull = self.hand_mesh[link_name]['hull'].copy().apply_transform(self.current_status[link_name].get_matrix().squeeze(0).detach().cpu())
            colors = np.array(self.hand_mesh[link_name]['colors']) * 255
            part_mesh_hull.visual.vertex_colors = colors
            scene.add_geometry(part_mesh_hull)
            scene_dict[link_name] = part_mesh_hull
        return scene, scene_dict
    

    def pb_reset_joints(self, joint_pos):
        '''
        Reset the positions of all the joints in the robotic hand in the simulation.

        Parameters:
        - joint_pos (list of float): Desired positions for all joints of the robot hand.
        '''
        for idx, joint in zip(self.pb_joint_idx, joint_pos):
            p.resetJointState(self.pb_hand, idx, joint, physicsClientId=self.physics_client_id)

    def pb_reset_finger_joints(self, finger_name, finger_joint_pos):
        '''
        Reset the positions of the joints of one finger in the robotic hand in the simulation.

        Parameters:
        - finger_name (str): Name of the finger to reset.
        - finger_joint_pos (list of float): Desired positions for the finger joints of the robot hand.
        '''  
        finger_st_idx, finger_end_idx = self.pb_finger_name_to_joint_idx_range[finger_name]
        for idx, joint_pos in zip(range(finger_st_idx, finger_end_idx), finger_joint_pos):
            p.resetJointState(self.pb_hand, self.pb_joint_idx[idx], joint_pos, physicsClientId=self.physics_client_id)
        

    def pb_reset_base(self, pos, ori):
        '''
        Reset the position and orientation of the robot hand in the simulator.

        Parameters:
        - pos (list of float, len=3): The desired position [x, y, z] of the robot hand.
        - ori (list of float, len=4): The desired orientation (quaternion) [x, y, z, w] of the robot hand.
        '''
        p.resetBasePositionAndOrientation(self.pb_hand, pos, ori)
        
    def get_random_joint_poses(self, nof_joint_pos_samples):
        random_joint_poses = np.random.uniform(self.pb_lower_joint_limits, self.pb_upper_joint_limits, (nof_joint_pos_samples, self.dof))
        return random_joint_poses

    def pb_load_object(self, obj_urdf_path, pos=[0, 0, 0], ori=[0, 0, 0, 1]):
        '''
        Load an object into the simulator using its URDF path and set its initial position and orientation.

        Parameters:
        - obj_urdf_path (str): Path to the URDF file of the object to be loaded.
        - pos (list of float, len=3): The desired position [x, y, z] of the object.
        - ori (list of float, len=4): The desired orientation (quaternion) [x, y, z, w] of the object.
        '''
        self.pb_object = p.loadURDF(obj_urdf_path, 
                                    basePosition=pos,
                                    baseOrientation=ori,
                                    useFixedBase=1,
                                    physicsClientId=self.physics_client_id)


    def pb_get_joint_connections(self):
        """
        Get the joint connections for the given body.
        
        Returns:
        A list of tuples, where each tuple contains two joint indices indicating a connection.
        """
        joint_connections = []
        # Initialize parent joint indices for all links with -1 (indicating no parent)
        parent_joints = [-1] * (self.num_joints + 1)  # +1 because the base is considered as link -1
        # Populate the parent joint indices
        for joint_index in range(self.num_joints):
            joint_info = p.getJointInfo(self.pb_hand, joint_index)
            parent_link_index = joint_info[16]
            parent_joints[joint_index] = parent_link_index
        # Find the connections based on parent joint indices
        for joint_index, parent_link_index in enumerate(parent_joints):
            if parent_link_index != -1:  # If there is a parent
                # parent link index = parent joint index 
                joint_connections.append((parent_link_index, joint_index))
        return joint_connections
    

    def pb_set_and_get_joint_positions(self, base_pos=[0.0, 0.0, 0.0], base_ori=[0.0, 0.0, 0.0, 1.0], joint_pos=None):
        """
        Get the positions of all joint frames for the given body.
        
        Returns:
        A list of positions for each joint frame in the body. 
        """
        joint_positions = []
        self.pb_reset_base(base_pos, base_ori)
        if joint_pos is None:
            self.pb_reset_joints(self.pb_rest_joint_pos)
        else:
            self.pb_reset_joints(joint_pos)

        for joint_index in range(self.num_joints):
            # Get the joint info
            joint_info = p.getJointInfo(self.pb_hand, joint_index)
            # Parent link frame position and orientation
            if joint_info[16] == -1:
                parent_frame_pos = base_pos.copy()
                parent_frame_orn =base_ori.copy()
            else:
                parent_frame_pos, parent_frame_orn = p.getLinkState(self.pb_hand, joint_info[16])[0:2]
            # Joint frame position relative to the parent link frame
            joint_frame_pos = joint_info[14]
            # Convert the local joint position to world space
            world_joint_pos = p.multiplyTransforms(parent_frame_pos, parent_frame_orn, joint_frame_pos, [0, 0, 0, 1])[0]
            joint_positions.append(world_joint_pos)

        return joint_positions
    

    def sample_palm_poses(self,
                          obj_pc, 
                          obj_n,
                          sample_num=4,
                          jittering=False):
        '''
        Sample initial poses for the palm based on random points chosen from an object's point cloud.

        Parameters:
        - obj_pc (np.ndarray): The point cloud of the object.
        - obj_n (np.ndarray): Normal vectors corresponding to the points in the point cloud.
        - sample_num (int, optional): The number of points to randomly select from the point cloud for sampling palm poses. Default is 4.

        Returns:
        - palm_pose_list (list): A list of 4x4 transformation matrices representing the sampled palm poses.
        '''
        palm_pose_point_id = np.random.choice(obj_pc.shape[0], sample_num)
        palm_pose_list = sample_init_palm_poses(obj_pc[palm_pose_point_id],
                                                obj_n[palm_pose_point_id],
                                                self.min_palm_dis,
                                                self.max_palm_dis,
                                                ori_face_vector=self.palm_ori_face_direction,
                                                jittering=True)
        return palm_pose_list


    def finger_ee_pos_sampling(self, finger_name, grid_num=5):
        self.pb_reset_joints(self.pb_open_joint_pos)

        finger_joint_order_idx_st, finger_joint_order_idx_end = self.pb_finger_name_to_joint_idx_range[finger_name] 
        upper_finger_joint_limits = np.array(self.pb_upper_joint_limits)[finger_joint_order_idx_st: finger_joint_order_idx_end].reshape(-1)
        lower_finger_joint_limits = np.array(self.pb_lower_joint_limits)[finger_joint_order_idx_st: finger_joint_order_idx_end].reshape(-1)
        finger_joint_combinations = generate_joint_combinations(upper_finger_joint_limits, lower_finger_joint_limits, grid_num)

        pos_list = []
        for finger_joint in finger_joint_combinations:
            pos_list.append(self.fk(finger_joint, finger_name))
        
        return np.array(pos_list).reshape(-1, 3)
    


    def finger_joint_pos_sampling(self, finger_name, acceptable_surface_points, grid_num=100, threshold=0.003):

        finger_joint_order_idx_st, finger_joint_order_idx_end = self.pb_finger_name_to_joint_idx_range[finger_name]  
        upper_finger_joint_limits = np.array(self.pb_upper_joint_limits)[finger_joint_order_idx_st: finger_joint_order_idx_end].reshape(-1)
        lower_finger_joint_limits = np.array(self.pb_lower_joint_limits)[finger_joint_order_idx_st: finger_joint_order_idx_end].reshape(-1)
        
        finger_joint_combinations = generate_joint_combinations(upper_finger_joint_limits, lower_finger_joint_limits, grid_num)

        np.random.shuffle(finger_joint_combinations)

        for finger_joint in finger_joint_combinations:
            self.pb_reset_finger_joints(finger_name, finger_joint)
            p.performCollisionDetection()  

            self_collision_points = p.getContactPoints(self.pb_hand, self.pb_hand)
            if len(self_collision_points) > 0:
                return None  

            tip_pos =  self.fk(finger_joint, finger_name)
            if np.min(np.linalg.norm(acceptable_surface_points - np.array(tip_pos).reshape(-1, 3), axis=1)) > threshold:
                continue  
            object_collision_points = p.getContactPoints(self.pb_hand, self.pb_object)
            acceptable_contact_flag = True
            for object_collision_point in object_collision_points:
                detected_point_on_obj = np.array(object_collision_point[6]).reshape(-1, 3)
                if np.linalg.norm(detected_point_on_obj - tip_pos) > threshold:
                    acceptable_contact_flag = False
                    break
            if acceptable_contact_flag:
                p_idx = np.argmin(np.linalg.norm(acceptable_surface_points - np.array(tip_pos).reshape(-1, 3), axis=1))
                return finger_joint, p_idx
        return None 
    


    def close_hand_finger(self, finger_name, threshold=0.0025):
        '''
        Close the finger under the current palm pose to find valid contact point if exists. 
        
        First, set the joints to self.pb_open_joint_pos. 
        Then we set the target joint pos to self.pb_close_joint_pos in p.setJointMotorControlArray to try to close the finger.
        We continuously step the simulation and check:
            - any joint is out of its limit 
            - whether self collision occurs 
            - whether acceptable collision occurs 
        If any of the above 3 happens, we jump out of the loop. If it is because of acceptable collision and that
        the current joints are within limits and no self collision occurs, we return the finger_joint_pos.
        Otherwise, we return None.

        Parameters:
        - finger_name (str): Name of the finger for which to close.
        - threshold (float): The contact point can be acceptable if the distance is within threshold.
        
        Returns:
        - None: If no acceptable joint positions are found.
        - finger_joint_pos (np.array, shape=(finger_dof,)): The joint positions that allow the finger to reach the target point.
        '''
        
        self.pb_reset_joints(self.pb_open_joint_pos)
        finger_joint_order_idx_st, finger_joint_order_idx_end = self.pb_finger_name_to_joint_idx_range[finger_name]  # TODO: only change the closing one the revolute yaw one just done by sampling
        finger_joint_idx = self.pb_joint_idx[finger_joint_order_idx_st: finger_joint_order_idx_end]

        p.setJointMotorControlArray(bodyUniqueId=self.pb_hand,
                    jointIndices=finger_joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=self.pb_close_joint_pos[finger_joint_order_idx_st: finger_joint_order_idx_end],
                    forces=self.pb_joint_max_forces[finger_joint_order_idx_st: finger_joint_order_idx_end],
                    # may need to tune 
                    positionGains=[0.02]*(finger_joint_order_idx_end-finger_joint_order_idx_st), #[0.002+0.004*i for i in range(finger_joint_order_idx_end-finger_joint_order_idx_st)], 
                    velocityGains=[1.0]*(finger_joint_order_idx_end-finger_joint_order_idx_st)
                    )
        
        for i in range(960):

            p.stepSimulation()

            finger_joint_pos = []
            for joint_index in finger_joint_idx:
                joint_state = p.getJointState(self.pb_hand, joint_index)
                joint_position = joint_state[0]
                finger_joint_pos.append(joint_position)
                breakpoint()
                if joint_position < self.lower_joint_limits_w_fixed[joint_index] or joint_position > self.upper_joint_limits_w_fixed[joint_index]:
                    return None

            self_collision_points = p.getContactPoints(self.pb_hand, self.pb_hand)
                
            if len(self_collision_points) > 0:
                return None  
            
            # print(i, finger_joint_pos)
            object_collision_points = p.getContactPoints(self.pb_hand, self.pb_object)
            
            if len(object_collision_points) > 0:
                breakpoint()
                link_idx = self.pb_finger_name_to_tip_link_idx[finger_name]
                result = p.getLinkState(self.pb_hand, link_idx, computeLinkVelocity=0,
                                        computeForwardKinematics=1, physicsClientId=self.physics_client_id)
                tip_pos, quat = result[0], result[1]

                for object_collision_point in object_collision_points:
                    detected_point_on_hand = np.array(object_collision_point[5]).reshape(-1, 3)
                    if np.linalg.norm(detected_point_on_hand - np.array(tip_pos).reshape(-1, 3)) > self.tip_radius + threshold:
                        continue
                    if object_collision_point[8] <= 0:  # one more step to avoid inaccurate collision detection (no contact yet)
                        return finger_joint_pos
                    
        return None 


    def fk(self, joint_pos, finger_name, return_quat=False):
        '''
        Calculate the Forward Kinematics (FK) of the robot hand to find the position of a specific fingertip.
        
        Parameters:
        - joint_pos (list of float): Joint positions, with length equal to the number of joints.
        - finger_name (str): Name of the finger for which to compute the FK.

        Returns:
        - pos (list of float): The calculated position [x, y, z] of the specified fingertip.
        '''
        self.pb_reset_joints(joint_pos)
        link_idx = self.pb_finger_name_to_tip_link_idx[finger_name]
        result = p.getLinkState(self.pb_hand, link_idx, computeLinkVelocity=0,
                                computeForwardKinematics=1, physicsClientId=self.physics_client_id)
        pos, quat = result[0], result[1]
        if return_quat:
            return pos, quat
        else:
            return pos
    

    def ik(self, finger_name, pos, quat=None):
        '''
        Calculate the Inverse Kinematics (IK) of the robot hand to find the joint angles that place 
        a specific fingertip at a given position.

        Parameters:
        - finger_name (str): Name of the fingertip for which to compute the IK.
        - pos (list of float): Desired position [x, y, z] for the fingertip.

        Returns:
        - finger_joint_pos (list of float): The calculated joint positions that achieve the desired fingertip position.
        '''
        fingertip_idx = self.pb_finger_name_to_tip_link_idx[finger_name]
        if quat is None:
            joint_pb = p.calculateInverseKinematics(self.pb_hand, 
                                                    endEffectorLinkIndex=fingertip_idx, 
                                                    targetPosition=pos,
                                                    lowerLimits=self.pb_lower_joint_limits,
                                                    upperLimits=self.pb_upper_joint_limits,
                                                    jointRanges=self.pb_joint_ranges,
                                                    restPoses=self.pb_rest_joint_pos, 
                                                    maxNumIterations=1000,
                                                    residualThreshold=0.0001)
        else:
            joint_pb = p.calculateInverseKinematics(self.pb_hand, 
                                                    endEffectorLinkIndex=fingertip_idx, 
                                                    targetPosition=pos,
                                                    targetOrientation=quat,
                                                    lowerLimits=self.pb_lower_joint_limits,
                                                    upperLimits=self.pb_upper_joint_limits,
                                                    jointRanges=self.pb_joint_ranges,
                                                    restPoses=self.pb_rest_joint_pos, 
                                                    maxNumIterations=1000,
                                                    residualThreshold=0.0001)
        assert len(joint_pb) == self.dof, f'IK error, get {len(joint_pb)} joint values'
        idx_range = self.pb_finger_name_to_joint_idx_range[finger_name]
        finger_joint_pos = joint_pb[idx_range[0]: idx_range[1]]
        return finger_joint_pos


    def whether_within_joint_limits(self, finger_name, finger_joint_pos):
        '''
        Check if the joint positions of a specified finger are within their limits.
        
        Parameters:
        - finger_name (str): Name of the finger to be checked.
        - finger_joint_pos (list of float): Joint positions of the specified finger.

        Returns:
        - bool: True if the joint positions are within their limits, False otherwise.
        '''
        finger_begin_joint_idx, finger_end_joint_idx = self.pb_finger_name_to_joint_idx_range[finger_name]
        finger_lower_joint_limits = np.array(self.pb_lower_joint_limits[finger_begin_joint_idx: finger_end_joint_idx])
        finger_upper_joint_limits = np.array(self.pb_upper_joint_limits[finger_begin_joint_idx: finger_end_joint_idx])
        if (finger_lower_joint_limits < np.array(finger_joint_pos)).all() and (np.array(finger_joint_pos) < finger_upper_joint_limits).all():
            # print("whether_within_joint_limits: True")
            return True
        else:
            # print("whether_within_joint_limits: False")
            return False


    def whether_reach_target_pos(self, finger_name, finger_joint_pos, target_pos, threshold=0.005, target_quat=None):
        '''
        Check if the specified finger, given certain joint positions, can reach a target position.
        
        Parameters:
        - finger_name (str): Name of the finger to be checked.
        - finger_joint_pos (list of float): Joint positions of the specified finger.
        - target_pos (list of float): Desired position [x, y, z] for the fingertip.

        Returns:
        - bool: True if the fingertip reaches the target position, False otherwise.
        '''
        full_joints = np.array(self.pb_zero_joint_pos)
        finger_begin_joint_idx, finger_end_joint_idx = self.pb_finger_name_to_joint_idx_range[finger_name]
        full_joints[finger_begin_joint_idx: finger_end_joint_idx] = finger_joint_pos
        
        if target_quat is None:
            reach_pos = self.fk(full_joints.tolist(), finger_name=finger_name)
        else:
            reach_pos, reach_quat = self.fk(full_joints.tolist(), finger_name=finger_name, return_quat=True)

        if np.linalg.norm(target_pos - reach_pos) < threshold:
            # breakpoint()
            # print("whether_reach_target_pos: True")
            return True 
        else:
            # dis = np.linalg.norm(target_pos - reach_pos)
            # print(f"whether_reach_target_pos: False, dis={dis}")
            return False 


    def whether_self_collision(self, finger_joint_pos, finger_name=None):
        '''
        Check if there's a self-collision happening in the robot hand, given certain joint positions.
        
        Parameters:
        - finger_joint_pos (list of float): Joint positions to be checked. 
                                           Can either be for a single finger or for all joints.
        - finger_name (str, optional): If provided, checks for the specified finger. 
                                       If not provided, checks for all fingers.

        Returns:
        - bool: True if there's a self-collision, False otherwise.
        '''
        if not finger_name:  
            self.pb_reset_joints(finger_joint_pos)
        else:
            full_joints = np.array(self.pb_zero_joint_pos)
            finger_begin_joint_idx, finger_end_joint_idx = self.pb_finger_name_to_joint_idx_range[finger_name]
            full_joints[finger_begin_joint_idx: finger_end_joint_idx] = finger_joint_pos
            self.pb_reset_joints(full_joints)
        p.performCollisionDetection()  
        self_collision_points = p.getContactPoints(self.pb_hand, self.pb_hand)
        if len(self_collision_points) > 0:
            # sc = len(self_collision_points)
            # print(f"whether_self_collision: True, {sc}")
            return True 
        else:
            # print(f"whether_self_collision: False")
            return False


    def whether_acceptable_collision(self, accept_points=None, threshold=0.007):
        '''
        Check if the collision between the robot hand and the object is acceptable.
        The collision is acceptable if it occurs at specific points.
        
        Parameters:
        - accept_points (list or np.ndarray, optional): List of acceptable collision points.
          If not provided, no collision is acceptable.
        - threshold (float, optional): A threshold to determine the acceptability of the distance between actual 
          contact points and expected acceptable contact points. If the distance between any actual contact point 
          and its nearest acceptable contact point exceeds this threshold, the collision is considered unacceptable. 
          Default is 0.007.

        Returns:
        - bool: True if the collision is acceptable, False otherwise.
        '''
        p.performCollisionDetection()
        contact_points = p.getContactPoints(self.pb_hand, self.pb_object)
        # print(f"All together {len(contact_points)} contact points!")
        if len(contact_points) > 0:
            if accept_points is None:
                return False 
            else:
                if len(accept_points) < len(contact_points):
                    return False 
                else:
                    if isinstance(accept_points, list):
                        accept_points = np.array(accept_points)
                    accept_flag = np.zeros(len(accept_points))
                    for contact_point in contact_points:
                        detected_point_on_obj = np.array(contact_point[6]).reshape(-1, 3)
                        # print(f"Contact point distance to EXP: {np.linalg.norm(accept_points - detected_point_on_obj, axis=1)}")
                        within_region_flag = np.linalg.norm(accept_points - detected_point_on_obj, axis=1) < threshold
                        accept_flag = accept_flag + within_region_flag
                    if accept_flag.all():
                        return True
                    else:
                        return False
        else:
            if accept_points is None:
                return True
            else:
                return False


    def find_acceptable_contact_point_and_joint_pos(self, finger_name, p_to_reach, p_to_cont):
        '''
        Determine if a given contact point is acceptable for the current hand and palm pose.

        The method uses inverse kinematics (IK) to find the joint positions that allow a 
        finger to reach a specified point. Once the joint positions are determined, several 
        checks are performed. These checks determine whether the joint positions are within 
        acceptable joint limits, whether they indeed allow the finger to reach the target 
        position, and whether they lead to any self-collisions or other unacceptable 
        collisions with the object. If all conditions are satisfied, the joint positions are 
        returned as acceptable.

        Parameters:
        - finger_name (str): Name of the finger (e.g., "index", "middle").
        - p_to_reach (list, len=3): The 3D point that the tip of the finger should reach.
        - p_to_cont (list, len=3): The 3D point on the object's surface that the finger should contact.

        Returns:
        - None: If no acceptable joint positions are found.
        - finger_joint_pos (np.array, shape=(finger_dof,)): The joint positions that allow the finger to reach the target point.
        '''
        finger_joint_pos = self.ik(finger_name, p_to_reach)

        full_joint_pos = np.array(self.pb_zero_joint_pos)
        finger_begin_joint_idx, finger_end_joint_idx = self.pb_finger_name_to_joint_idx_range[finger_name]
        full_joint_pos[finger_begin_joint_idx: finger_end_joint_idx] = finger_joint_pos
        
        if self.whether_within_joint_limits(finger_name, finger_joint_pos) and \
           self.whether_reach_target_pos(finger_name, finger_joint_pos, p_to_reach) and \
           (not self.whether_self_collision(finger_joint_pos, finger_name)) and \
           self.whether_acceptable_collision([p_to_cont]):
            # print("~"*10 + "Find One Possible IK Solution" + "~"*10)
            return finger_joint_pos
        else:
            return None 
        

    def get_contact_point_Jacobian(self, joint_pos, finger_name, p_w_cont):
        '''
        Compute the Jacobian related to the contact point of a specified finger.
        
        Parameters:
        - joint_pos (list of float): Joint positions of the robot hand.
        - finger_name (str): Name of the finger to be checked.
        - p_w_cont (list of float): Contact point in the world frame.

        Returns:
        - lin_J_finger (np.array): Linear Jacobian related to the contact point. 
                                  The shape is (3, finger_dof).
        '''
        self.pb_reset_joints(joint_pos)

        link_idx = self.pb_finger_name_to_tip_link_idx[finger_name]
        result = p.getLinkState(self.pb_hand, link_idx, computeLinkVelocity=0,
                                computeForwardKinematics=1, physicsClientId=self.physics_client_id)
        pos, quat = result[0], result[1]
        X_w_ee = pose_vector_to_transformation_matrix(pos+quat)
        X_ee_w = np.linalg.inv(X_w_ee)
        p_w_cont_1 = np.ones((4, 1))
        p_w_cont_1[:3, 0] = p_w_cont
        p_ee_cont = X_ee_w @ p_w_cont_1
        lin_J, ang_J = p.calculateJacobian(self.pb_hand, link_idx, p_ee_cont[:3], joint_pos, [0]*self.dof, [0]*self.dof, self.physics_client_id)
        finger_begin_joint_idx, finger_end_joint_idx = self.pb_finger_name_to_joint_idx_range[finger_name]
        return np.array(lin_J)[:, finger_begin_joint_idx:finger_end_joint_idx]


    def get_random_ee_contact_force(self, joint_pos, palm_pose, finger_name, p_w_cont, sample_num=5, mimic=False):
        '''
        joint_pos: list, len=self.num_joints
        finger_name: str
        p_w_cont: list, len=3, contact point in the world frame 
        ---
        Return 
        contact_force: np.array, shape=(sample_num, 3), contact force in the world frame 
        '''
        finger_begin_joint_idx, finger_end_joint_idx = self.pb_finger_name_to_joint_idx_range[finger_name]
        finger_joint_num = finger_end_joint_idx - finger_begin_joint_idx

        finger_upper_torque_limit =  self.upper_joint_torque_limits[finger_begin_joint_idx: finger_end_joint_idx]
        finger_lower_torque_limit =  self.lower_joint_torque_limits[finger_begin_joint_idx: finger_end_joint_idx]
        
        J = self.get_contact_point_Jacobian(joint_pos, finger_name, p_w_cont)

        if mimic:  # for robotiq85
            J = self.mimic_jacobian_map(J)
            
        random_joint_torque = np.random.rand(finger_joint_num, sample_num) * (finger_upper_torque_limit - finger_lower_torque_limit).reshape(-1, 1) + finger_lower_torque_limit.reshape(-1, 1)
        contact_force = np.linalg.lstsq(J.T, random_joint_torque, rcond=None)[0]
        f_w = transform_vectors(palm_pose, contact_force.T)
        return f_w  # in the world frame 
    

    def sample_ee_contact_force_within_friction_cone(self, joint_pos, palm_pose, finger_name, p_w_cont, n_w_cont, friction_coefficient=0.5, first_sample_num=100, max_save_num=4):
        '''
        joint_pos: list, len=self.num_joints
        palm_pose: np.ndarray, shape=(4, 4)
        finger_name: str
        p_w_cont: list, len=3, contact point described in the world frame 
        n_w_cont: list, len=3, contact point face normal described in the world frame 
        friction_coefficient: float, default=0.5
        sample_num: int, default=4, for the contact point, sample number 
        ---
        Return 
        force_within_frication_cone / None: 
            None: no solution found with this first_sample_num 
            force_within_frication_cone: np.array, shape=(n, 3), n <= max_save_num

        '''
        f_w = self.get_random_ee_contact_force(joint_pos, palm_pose, finger_name, p_w_cont, sample_num=first_sample_num)  # shape=(first_sample_num, 3)
        f_n = (f_w @ n_w_cont).reshape(-1, 1) * n_w_cont
        f_t = f_w - f_n 
        within_fr_cone_id = (np.linalg.norm(f_t, axis=1) <= friction_coefficient * np.linalg.norm(f_n, axis=1)) & ((f_w @ n_w_cont) < 0)

        if f_w[within_fr_cone_id].shape[0] > 0: 
            return f_w[within_fr_cone_id][: min(max_save_num, f_w[within_fr_cone_id].shape[0])]
        else:
            return None 


    def find_acceptable_contact_point_and_wrench(self, finger_name, p_to_reach, p_to_cont, n_cont, obj_mesh, friction_coefficient=0.5, first_sample_num=100, max_save_num=4):
        '''
        The main function to judge whether the contact point is acceptable for the current hand and palm pose
        ---
        finger_name: str
        p_to_reach: list, len=3, the point for the tip center to reach
        p_to_cont: list, len=3, the point on the surface for the tip surface to contact 
        n_cont: list, len=3, the surface normal of the p_to_cont
        obj_mesh: trimesh
        ---
        Return
            - None: if didn't find any solution and force 
            - finger_joint_pos, executable_wrench: 
                - finger_joint_pos, np.array, shape=(finger_dof,)
                - executable_wrench, np.array, shape=(n, 6), n <= max_save_num
        '''
        finger_joint_pos = self.ik(finger_name, p_to_reach)

        full_joint_pos = np.array(self.zero_joint_pos)
        finger_begin_joint_idx, finger_end_joint_idx = self.finger_name_to_joint_idx_range[finger_name]
        full_joint_pos[finger_begin_joint_idx: finger_end_joint_idx] = finger_joint_pos
        
        if self.whether_within_joint_limits(finger_name, finger_joint_pos) and \
           self.whether_reach_target_pos(finger_name, finger_joint_pos, p_to_reach) and \
           (not self.whether_self_collision(finger_joint_pos, finger_name)) and \
           self.whether_acceptable_collision(obj_mesh, [p_to_cont]):
            # print("~"*10 + "Find One Possible IK Solution" + "~"*10)
            f_w = self.sample_ee_contact_force_within_friction_cone(full_joint_pos.tolist(), finger_name, p_to_cont, n_cont,  friction_coefficient=0.5, first_sample_num=100, max_save_num=4)
            if f_w is None:
                return None
            else:
                t_w = np.cross(np.reshape(p_to_cont, (1, -1)), f_w)
                executable_wrench = np.concatenate((f_w, t_w), axis=1)
                return finger_joint_pos, executable_wrench
        else:
            return None 
               
               

class PandaHand(RobotHand):
    def __init__(self,  #unicontact/UniContactNet/ContactSampling/assets/franka_hand/franka_hand.urdf
                 hand_urdf_path="assets/hands/franka/model.urdf", 
                 hand_mesh_path="assets/hands/franka/meshes/visual", physics_client_id=None):
        super().__init__(hand_urdf_path, hand_mesh_path, physics_client_id)


    def set_basic_hand_info(self):
        '''
        Set specific hand information for PandaHand.
        '''
        # Define the direction the palm is facing
        self.palm_ori_face_direction = np.array([0.0, 0.0, 1.0])
        self.min_palm_dis = 0.002
        self.max_palm_dis = 0.046
        self.max_operation_dis = 0.07

        # Define the mapping from finger names to the index of the tip link
        self.pb_finger_name_to_tip_link_idx = {"left": 2, "right": 4}
        self.tip_radius = 0.0

        # Define the mapping from finger names to the range of joint indices
        self.pb_finger_name_to_joint_idx_range = {"left": (0, 1), "right": (1, 2)}

        # Define the joint positions when they are at zero
        self.pb_zero_joint_pos = [0.04] * self.dof
        self.pb_open_joint_pos = [0.04] * self.dof
        self.pb_close_joint_pos = [0.0] * self.dof

        # Define the lower and upper torque limits for the joints
        self.lower_joint_torque_limits = np.array([-1.0] * self.dof)
        self.upper_joint_torque_limits = np.array([1.0] * self.dof)



class WSG50Hand(RobotHand):
    def __init__(self,  #unicontact/UniContactNet/ContactSampling/assets/wsg_50/model.urdf
                hand_urdf_path="assets/hands/wsg_50/model.urdf",
                hand_mesh_path="assets/hands/wsg_50/meshes",
                physics_client_id=None):
        super().__init__(hand_urdf_path, hand_mesh_path, physics_client_id)

    def set_basic_hand_info(self):
        '''
        Set specific hand information for PandaHand.
        '''
        # Define the direction the palm is facing
        self.palm_ori_face_direction = np.array([0.0, 0.0, 1.0])
        self.min_palm_dis = 0.000
        self.max_palm_dis = 0.072
        self.max_operation_dis = 0.11

        # Define the mapping from finger names to the index of the tip link
        self.pb_finger_name_to_tip_link_idx = {"left": 3, "right": 6}
        self.tip_radius = 0.0

        # Define the mapping from finger names to the range of joint indices
        self.pb_finger_name_to_joint_idx_range = {"left": (0, 1), "right": (1, 2)}

        # Define the joint positions when they are at zero
        self.pb_zero_joint_pos = [-0.055, 0.055] 
        self.pb_open_joint_pos = [-0.055, 0.055] 
        self.pb_close_joint_pos = [-0.0027, 0.0027]

        # Define the lower and upper torque limits for the joints
        self.lower_joint_torque_limits = np.array([-1.0] * self.dof)
        self.upper_joint_torque_limits = np.array([1.0] * self.dof)


class Robotiq2F85Hand(RobotHand):
    # assets/robotiq_arg85/urdf/robotiq_arg85_description.urdf
    def __init__(self, 
                 hand_urdf_path="assets/hands/robotiq_arg85/urdf/robotiq_arg85_description.urdf",
                 hand_mesh_path="assets/hands/robotiq_arg85/meshes",
                 physics_client_id=None):
        super().__init__(hand_urdf_path, hand_mesh_path, physics_client_id)

    def set_basic_hand_info(self):
        '''
        Set specific hand information for ShadowHand.
        '''
        # Cancel the collision detected when zero pose
        p.setCollisionFilterPair(self.pb_hand, self.pb_hand, 2, 4, 0)
        p.setCollisionFilterPair(self.pb_hand, self.pb_hand, 7, 10, 0)
        # p.setCollisionFilterPair(self.pb_hand, self.pb_hand, 18, 24, 0)

        # Define the direction the palm is facing
        self.palm_ori_face_direction = np.array([0.0, 0.0, 1.0])
        self.min_palm_dis = 0.005
        self.max_palm_dis = 0.065
        self.max_operation_dis = 0.08

        # Define the mapping from finger names to the index of the tip link
        self.pb_finger_name_to_tip_link_idx = {"left": 5, "right": 8}
        self.tip_radius = 0.0
        
        # Define the mapping from finger names to the range of joint indices
        self.pb_finger_name_to_joint_idx_range = {'left': (0, 6), 
                                                  'right': (0, 6)}

        # Define the joint positions when they are at zero   
        self.pb_lower_joint_limits = [-0.8]
        self.pb_upper_joint_limits = [0.0]
        self.pb_joint_ranges = [0.8]
        self.pb_rest_joint_pos = [-0.4]
        self.pb_zero_joint_pos = [0.0] * self.dof  # dof=6
        self.pb_open_joint_pos = [0.0] * self.dof
        self.pb_close_joint_pos = self.mimic_joint_map(-0.8)

        # Define the lower and upper torque limits for the joints
        self.lower_joint_torque_limits = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.upper_joint_torque_limits = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


    def mimic_joint_map(self, input_joint_pos):
        output_joint_pos = [input_joint_pos, input_joint_pos, -input_joint_pos, input_joint_pos, -input_joint_pos, input_joint_pos]
        return output_joint_pos
    
    def mimic_jacobian_map(self, input_jacobian):
        '''
        input: (3, 6)
        output: (3, 1)
        '''
        mapping_relations = np.array([1, 1, -1, 1, -1, 1])
        J_main = input_jacobian @ mapping_relations.reshape(-1, 1)
        J_full = np.zeros((3, 6))
        for i in range(6):
            J_full[:, i] = J_main.flatten() * mapping_relations[i]
        return J_full  
        
    def mimic_torque_map(self, input_joint_torques):
        '''
        (n, 6) -> (n, 6)
        '''
        source_joint_torque = input_joint_torques[:, 0]
        mimic_joint_torque = np.column_stack((source_joint_torque, source_joint_torque, -source_joint_torque, source_joint_torque, -source_joint_torque, source_joint_torque))
        return mimic_joint_torque

    
    def finger_ee_pos_sampling(self, finger_name, grid_num=5):
        finger_joint_order_idx_st, finger_joint_order_idx_end = self.pb_finger_name_to_joint_idx_range[finger_name]  
        finger_joint_order_idx_end = 1
        upper_finger_joint_limits = np.array(self.pb_upper_joint_limits)[finger_joint_order_idx_st: finger_joint_order_idx_end].reshape(-1)
        lower_finger_joint_limits = np.array(self.pb_lower_joint_limits)[finger_joint_order_idx_st: finger_joint_order_idx_end].reshape(-1)
        
        finger_joint_combinations = generate_joint_combinations(upper_finger_joint_limits, lower_finger_joint_limits, grid_num)

        pos_list = []
        for finger_joint in finger_joint_combinations:
            pos_list.append(self.fk(self.mimic_joint_map(finger_joint[0]), "left"))
            pos_list.append(self.fk(self.mimic_joint_map(finger_joint[0]), "right"))
        
        return np.array(pos_list).reshape(-1, 3)
    

    def finger_joint_pos_sampling(self, finger_name, acceptable_surface_points, grid_num=100, threshold=0.005):

        finger_joint_order_idx_st, finger_joint_order_idx_end = self.pb_finger_name_to_joint_idx_range[finger_name]  
        finger_joint_order_idx_end = 1
        upper_finger_joint_limits = np.array(self.pb_upper_joint_limits)[finger_joint_order_idx_st: finger_joint_order_idx_end].reshape(-1)
        lower_finger_joint_limits = np.array(self.pb_lower_joint_limits)[finger_joint_order_idx_st: finger_joint_order_idx_end].reshape(-1)
        
        finger_joint_combinations = generate_joint_combinations(upper_finger_joint_limits, lower_finger_joint_limits, grid_num)

        np.random.shuffle(finger_joint_combinations)

        for finger_joint in finger_joint_combinations:
            self.pb_reset_joints(self.mimic_joint_map(finger_joint[0]))
            p.performCollisionDetection()  

            self_collision_points = p.getContactPoints(self.pb_hand, self.pb_hand)
            if len(self_collision_points) > 0:
                return None  

            tip_pos_1 =  self.fk(self.mimic_joint_map(finger_joint[0]), "left")
            tip_pos_2 =  self.fk(self.mimic_joint_map(finger_joint[0]), "right")
            # breakpoint()
            if np.min(np.linalg.norm(acceptable_surface_points - np.array(tip_pos_1).reshape(-1, 3), axis=1)) > threshold and np.min(np.linalg.norm(acceptable_surface_points - np.array(tip_pos_2).reshape(-1, 3), axis=1)) > threshold:
                continue  
            object_collision_points = p.getContactPoints(self.pb_hand, self.pb_object)
            acceptable_contact_flag = True


            for object_collision_point in object_collision_points:
                detected_point_on_obj = np.array(object_collision_point[6]).reshape(-1, 3)
                # print(np.linalg.norm(detected_point_on_hand - tip_pos))
                if np.linalg.norm(detected_point_on_obj - tip_pos_1) > threshold and np.linalg.norm(detected_point_on_obj - tip_pos_2) > threshold:
                    acceptable_contact_flag = False
                    break
                    
            if acceptable_contact_flag and len(object_collision_points) > 0:
                p_idx_list = [-1, -1]

                if np.min(np.linalg.norm(acceptable_surface_points - np.array(tip_pos_1).reshape(-1, 3), axis=1)) < threshold:
                    p_idx_list[0] = np.argmin(np.linalg.norm(acceptable_surface_points - np.array(tip_pos_1).reshape(-1, 3), axis=1))
                if np.min(np.linalg.norm(acceptable_surface_points - np.array(tip_pos_2).reshape(-1, 3), axis=1)) < threshold:
                    p_idx_list[1] = np.argmin(np.linalg.norm(acceptable_surface_points - np.array(tip_pos_2).reshape(-1, 3), axis=1))    
                if p_idx_list == [-1, -1]:
                    return None 
                else:
                    return self.mimic_joint_map(finger_joint[0]), p_idx_list
        return None 
    



class Kinova3FHand(RobotHand):
    def __init__(self, 
                 hand_urdf_path="assets/hands/kinova_3f/model.urdf",
                 hand_mesh_path="assets/hands/kinova_3f/meshes",
                 physics_client_id=None):
        super().__init__(hand_urdf_path, hand_mesh_path, physics_client_id)

    def set_basic_hand_info(self):
        '''
        Set specific hand information for ShadowHand.
        '''
        # Define the direction the palm is facing
        self.palm_ori_face_direction = np.array([0.0, 0.0, 1.0])
        self.min_palm_dis = 0.002
        self.max_palm_dis = 0.1
        self.max_operation_dis = 0.13

        # Define the mapping from finger names to the index of the tip link
        self.pb_finger_name_to_tip_link_idx = {"a": 3, "b": 6,  "c": 9}
        # self.tip_radius = 0.0193
        # self.tip_radius = 0.015
        self.tip_radius = 0.0
        
        # Define the mapping from finger names to the range of joint indices
        self.pb_finger_name_to_joint_idx_range = {'a': (0, 2), 
                                                  'b': (2, 4), 
                                                  'c': (4, 6)}

        # Define the joint positions when they are at zero
        self.pb_zero_joint_pos = [0.0] * self.dof
        self.pb_open_joint_pos = self.pb_lower_joint_limits.copy()
        self.pb_close_joint_pos = self.pb_upper_joint_limits.copy()

        # Define the lower and upper torque limits for the joints
        self.lower_joint_torque_limits = np.array([-1.0] * self.dof)
        self.upper_joint_torque_limits = np.array([1.0] * self.dof)



class Robotiq3FHand(RobotHand):
    # assets/robotiq_3finger/urdf/robotiq_3finger_description.urdf
    def __init__(self, 
                 hand_urdf_path="assets/hands/robotiq_3finger/urdf/robotiq_3finger_description.urdf",
                 hand_mesh_path="assets/hands/robotiq_3finger/meshes/robotiq_3finger",
                 physics_client_id=None):
        super().__init__(hand_urdf_path, hand_mesh_path, physics_client_id)

    def set_basic_hand_info(self):
        '''
        Set specific hand information for ShadowHand.
        '''
        # Define the direction the palm is facing
        self.palm_ori_face_direction = np.array([0.0, 0.0, 1.0])
        self.min_palm_dis = 0.004
        self.max_palm_dis = 0.12
        self.max_operation_dis = 0.18

        # Define the mapping from finger names to the index of the tip link
        self.pb_finger_name_to_tip_link_idx = {"a": 7, "b": 12,  "c": 17}
        # self.tip_radius = 0.0193
        # self.tip_radius = 0.015
        self.tip_radius = 0.0
        
        # Define the mapping from finger names to the range of joint indices
        self.pb_finger_name_to_joint_idx_range = {'a': (0, 3), 
                                                  'b': (3, 7), 
                                                  'c': (7, 11)}

        # Define the joint positions when they are at zero
        self.pb_zero_joint_pos = [0.0] * self.dof
        self.pb_open_joint_pos = self.pb_upper_joint_limits.copy()
        self.pb_close_joint_pos = self.pb_lower_joint_limits.copy()

        # Define the lower and upper torque limits for the joints
        self.lower_joint_torque_limits = np.array([-1.0] * self.dof)
        self.upper_joint_torque_limits = np.array([1.0] * self.dof)
    


class AllegroHand(RobotHand):
    def __init__(self, 
                 hand_urdf_path="assets/hands/allegro_hand_right/allegro_hand_right_pb.urdf", 
                 hand_mesh_path="assets/hands/allegro_hand_right/meshes",
                 physics_client_id=None):
        super().__init__(hand_urdf_path,    hand_mesh_path, physics_client_id)


    def set_basic_hand_info(self):
        '''
        Set specific hand information for AllegroHand.
        '''
        # Cancel the collision between the thumb and the palm
        p.setCollisionFilterPair(self.pb_hand, self.pb_hand, 0, 17, 0)

        # Define the direction the palm is facing
        self.palm_ori_face_direction = np.array([0.0, 0.0, 1.0])
        self.min_palm_dis = 0.002
        self.max_palm_dis = 0.09
        self.max_operation_dis = 0.16

        # Define the mapping from finger names to the index of the tip link
        self.pb_finger_name_to_tip_link_idx = {"index": 5, "middle":10, "ring": 15, "thumb": 20}
        self.tip_radius = 0.014

        # Define the mapping from finger names to the range of joint indices
        self.pb_finger_name_to_joint_idx_range = {'index': (0, 4), 'middle': (4, 8), 
                                               'ring': (8, 12), 'thumb': (12, 16)}

        # Define the joint positions when they are at zero
        self.pb_zero_joint_pos = [0.0] * 12 + [0.3] + [0.0] * 3
        self.pb_open_joint_pos = self.pb_zero_joint_pos.copy()
        self.pb_close_joint_pos = self.pb_upper_joint_limits.copy()

        # Define the lower and upper torque limits for the joints
        self.lower_joint_torque_limits = np.array([-1.0] * self.dof)
        self.upper_joint_torque_limits = np.array([1.0] * self.dof)

        hand_wrist_p = [0.0, -0.075, -0.025]
        hand_wrist_q = R.from_euler('XYZ', [-90, 0.0, -90.0], degrees=True).as_quat().tolist()
        self.X_PalmWrsit = pose_vector_to_transformation_matrix(hand_wrist_p+hand_wrist_q)
    

class LeapHand(RobotHand):
    def __init__(self, 
                 hand_urdf_path="assets/hands/leap_hand/robot.urdf", 
                 hand_mesh_path="assets/hands/leap_hand",
                 physics_client_id=None):
        super().__init__(hand_urdf_path, hand_mesh_path, physics_client_id)


    def set_basic_hand_info(self):
        '''
        Set specific hand information for AllegroHand.
        '''
        # Cancel the collision between the thumb and the palm
        p.setCollisionFilterPair(self.pb_hand, self.pb_hand, 0, 17, 0)

        # Define the direction the palm is facing
        self.palm_ori_face_direction = np.array([0.0, 0.0, 1.0])
        self.min_palm_dis = 0.002
        self.max_palm_dis = 0.09
        self.max_operation_dis = 0.16

        # Define the mapping from finger names to the index of the tip link
        self.pb_finger_name_to_tip_link_idx = {"index": 5, "middle":10, "ring": 15, "thumb": 20}
        self.pb_finger_name_to_finger_idx = {"index": 0, "middle":1, "ring": 2, "thumb": 3}
        self.tip_radius = 0.0134

        # Define the mapping from finger names to the range of joint indices
        self.pb_finger_name_to_joint_idx_range = {'index': (0, 4), 'middle': (4, 8), 
                                               'ring': (8, 12), 'thumb': (12, 16)}

        # Define the joint positions when they are at zero
        self.pb_zero_joint_pos = [0.0] * 16
        self.pb_open_joint_pos = self.pb_zero_joint_pos.copy()
        self.pb_close_joint_pos = self.pb_upper_joint_limits.copy()

        # Define the lower and upper torque limits for the joints
        self.lower_joint_torque_limits = np.array([-1.0] * self.dof)
        self.upper_joint_torque_limits = np.array([1.0] * self.dof)

        hand_wrist_p = [0.0, -0.053, -0.025]
        hand_wrist_q = R.from_euler('XYZ', [-90, 0.0, -90.0], degrees=True).as_quat().tolist()
        self.X_PalmWrsit = pose_vector_to_transformation_matrix(hand_wrist_p+hand_wrist_q)


class ShadowHand(RobotHand):
    def __init__(self,
                 hand_urdf_path="assets/hands/shadow_hand/shadow_hand.urdf", 
                 hand_mesh_path="assets/hands/shadow_hand/meshes/collision", physics_client_id=None):
        super().__init__(hand_urdf_path, hand_mesh_path, physics_client_id)
    
    def set_basic_hand_info(self):
        '''
        Set specific hand information for ShadowHand.
        '''
        # Cancel the collision detected when zero pose
        p.setCollisionFilterPair(self.pb_hand, self.pb_hand, 8, 13, 0)
        p.setCollisionFilterPair(self.pb_hand, self.pb_hand, 13, 18, 0)
        p.setCollisionFilterPair(self.pb_hand, self.pb_hand, 18, 24, 0)

        # Define the direction the palm is facing
        self.palm_ori_face_direction = np.array([0.0, 0.0, 1.0])
        self.min_palm_dis = 0.002
        self.max_palm_dis = 0.07
        self.max_operation_dis = 0.12

        # Define the mapping from finger names to the index of the tip link
        self.pb_finger_name_to_tip_link_idx = {"thumb": 6, "index": 11,  "middle": 16, "ring": 21, "little": 27}
        self.tip_radius = 0.0076

        # Define the mapping from finger names to the range of joint indices
        self.pb_finger_name_to_joint_idx_range = {'thumb': (0, 5), 'index': (5, 9), 
                                               'middle': (9, 13), 'ring': (13, 17), 'little': (17, 22)}

        # Define the joint positions when they are at zero
        self.pb_zero_joint_pos = [0.0] * self.dof
        self.pb_open_joint_pos = [0.0] * self.dof
        self.pb_close_joint_pos = self.pb_upper_joint_limits.copy()

        # Define the lower and upper torque limits for the joints
        self.lower_joint_torque_limits = np.array([-1.0] * self.dof)
        self.upper_joint_torque_limits = np.array([1.0] * self.dof)
    


