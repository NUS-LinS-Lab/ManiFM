from optimization.RobotHand_nb_sdf import RobotHandNb
import numpy as np


class ShadowHandNb(RobotHandNb):
    def __init__(self, 
                hand_urdf_path,
                hand_mesh_path,
                hand_tm_urdf_path=None):
        super().__init__(hand_urdf_path, hand_mesh_path, hand_tm_urdf_path)

    def set_basic_hand_info(self):
        self.nb_finger_radius_low = 0.01
        self.nb_finger_radius_high = 0.014
        self.nb_finger_name_to_tip_link_name = {"index": "index_tip", "little": "little_tip", "middle":"middle_tip", "ring": "ring_tip", "thumb": "thumb_tip"}
        self.pb_finger_name_to_joint_idx_range = {'thumb': (0, 5), 'index': (5, 9), 
                                               'middle': (9, 13), 'ring': (13, 17), 'little': (17, 22)}
        self.nb_finger_name_to_joint_idx_range = {'index': (6, 10), 'little': (10, 15),'middle': (15, 19), 
                                               'ring': (19, 23), 'thumb': (23, 28)}
        self.nb_rest_pose = np.array((0, 0, 0, 0, 0.3, -0,
                                0, 0.2, 0, 0, 0.2, 0, 0, 0, 0,
                                0, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0.2, 0, 0, 0))
        self.collision_group_to_linknames_to_collision_radius = {
            "palm": {
                "palm": 0.025,
            },
            "index": {
                "index_tip": 0.0134-0.05, 
                "index_finger_distal": 0.0134, 
                "index_finger_middle": 0.0184, 
                "index_finger_proximal": 0.0184,  
                "index_finger_knuckle": 0.0184,  
            },
            "little": {
                "little_tip": 0.0134-0.05, 
                "little_finger_distal": 0.0134, 
                "little_finger_middle": 0.0134, 
                "little_finger_proximal": 0.0134,  
                "little_finger_knuckle": 0.0134,  
                "little_finger_metacarpal": 0.0134, 
            },
            "middle": {
                "middle_tip": 0.0134-0.05, 
                "middle_finger_distal": 0.0134, 
                "middle_finger_middle": 0.0184, 
                "middle_finger_proximal": 0.0184,  
                "middle_finger_knuckle": 0.0184,  
            },
            "ring": {
                "ring_tip": 0.0134-0.05, 
                "ring_finger_distal": 0.0134, 
                "ring_finger_middle": 0.0184, 
                "ring_finger_proximal": 0.0184,  
                "ring_finger_knuckle": 0.0184,  
            },
            "thumb": {
                "thumb_tip": 0.0134-0.05, 
                "thumb_distal": 0.0134, 
                "thumb_middle": 0.0134, 
                "thumb_hub": 0.0134, 
                "thumb_proximal": 0.0134, 
                "thumb_base": 0.0134, 
            },
        }
        self.considered_collision_group_pairs = [
            ("index", "object"), ("index", "little"), ("index", "middle"), ("index", "ring"), ("index", "thumb"),
            ("little", "object"), ("little", "middle"), ("little", "ring"), ("little", "thumb"),
            ("middle", "object"), ("middle", "ring"), ("middle", "thumb"), 
            ("ring", "object"), ("ring", "thumb"), 
            ("thumb", "object"),
            ("palm", "object")
        ]