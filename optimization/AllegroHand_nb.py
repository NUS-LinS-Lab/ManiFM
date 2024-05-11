from optimization.RobotHand_nb_sdf import RobotHandNb
import numpy as np


class AllegroHandNb(RobotHandNb):
    def __init__(self, 
                hand_urdf_path,
                hand_mesh_path,
                hand_tm_urdf_path=None):
        super().__init__(hand_urdf_path, hand_mesh_path, hand_tm_urdf_path)

    def set_basic_hand_info(self):
        self.nb_finger_radius_low = 0.01
        self.nb_finger_radius_high = 0.0142
        self.nb_finger_name_to_tip_link_name = {"index": "link_3.0_tip", "thumb": "link_15.0_tip", "middle": "link_7.0_tip", "ring": "link_11.0_tip"}
        self.pb_finger_name_to_joint_idx_range = {'index': (0, 4), 'middle': (4, 8), 
                                               'ring': (8, 12), 'thumb': (12, 16)}
        self.nb_finger_name_to_joint_idx_range = {'index': (6, 10), 'thumb': (10, 14), 'middle': (14, 18), 'ring': (18, 22)}
        self.nb_rest_pose = np.array((0, 0, 0, 0, 0, -0.3,
                                0, 0.2, 0, 0, 0.5, 0.5, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0))
        
        self.collision_group_to_linknames_to_collision_radius = {
            "palm": {
                "palm_link": 0.025,
            },
            "index": {
                "link_3.0_tip": 0.0134-0.05, 
                "link_3.0": 0.0134, 
                "link_2.0": 0.0184, 
                "link_1.0": 0.0184,  
                "link_0.0": 0.0184,
            },
            "thumb": {
                "link_15.0_tip": 0.0134-0.05, 
                "link_15.0": 0.0134, 
                "link_14.0": 0.0184, 
                "link_13.0": 0.0184,  
                "link_12.0": 0.0184,
            },
            "middle": {
                "link_7.0_tip": 0.0134-0.05, 
                "link_7.0": 0.0134, 
                "link_6.0": 0.0184, 
                "link_5.0": 0.0184,  
                "link_4.0": 0.0184,
            },
            "ring": {
                "link_11.0_tip": 0.0134-0.05, 
                "link_11.0": 0.0134, 
                "link_10.0": 0.0184, 
                "link_9.0": 0.0184,  
                "link_8.0": 0.0184,
            },
        }
        self.considered_collision_group_pairs = [
            ("index", "object"), ("index", "middle"), ("index", "ring"), ("index", "thumb"),
            ("middle", "object"), ("middle", "ring"), ("middle", "thumb"), 
            ("ring", "object"), ("ring", "thumb"), 
            ("thumb", "object"),
            ("palm", "object")
        ]