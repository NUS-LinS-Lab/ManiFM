from optimization.RobotHand_nb_sdf import RobotHandNb
import numpy as np


class BarrettHandNb(RobotHandNb):
    def __init__(self, 
                hand_urdf_path,
                hand_mesh_path,
                hand_tm_urdf_path=None):
        super().__init__(hand_urdf_path, hand_mesh_path, hand_tm_urdf_path)

    def set_basic_hand_info(self):
        self.nb_finger_radius_low = 0.01
        self.nb_finger_radius_high = 0.014
        self.nb_finger_name_to_tip_link_name = {"index": "finger_1_tip", "middle":"finger_2_tip", "thumb": "finger_3_tip"}
        self.nb_finger_name_to_joint_idx_range = {'index': (6, 9), 'middle': (9, 12), 'thumb': (12, 14)}
        self.nb_rest_pose = np.array((0, 0, 0, 0, 0, -0.3,
                                -0.5, -0.2, 0, 0.5, -0.2, 0, -0.2, 0))
        self.collision_group_to_linknames_to_collision_radius = {
            "palm": {
                "base_link": 0.025,
            },
            "index": {
                "finger_1_tip": 0.0134-0.05, 
                "finger_1_dist_link": 0.0134, 
                "finger_1_med_link": 0.0184, 
                "finger_1_prox_link": 0.0184, 
            },
            "middle": {
                "finger_2_tip": 0.0134-0.05, 
                "finger_2_dist_link": 0.0134, 
                "finger_2_med_link": 0.0184,
                "finger_2_prox_link": 0.0184,  
            },
            "thumb": {
                "finger_3_tip": 0.0134-0.05, 
                "finger_3_dist_link": 0.0134, 
                "finger_3_med_link": 0.0184, 
            },
        }
        self.considered_collision_group_pairs = [
            ("index", "object"), ("index", "middle"), ("index", "thumb"),
            ("middle", "object"), ("middle", "thumb"), 
            ("thumb", "object"),
            ("palm", "object")
        ]