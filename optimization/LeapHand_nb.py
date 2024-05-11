from optimization.RobotHand_nb_sdf import RobotHandNb
import numpy as np


class LeapHandNb(RobotHandNb):
    def __init__(self, 
                hand_urdf_path,
                hand_mesh_path,
                hand_tm_urdf_path=None):
        super().__init__(hand_urdf_path, hand_mesh_path, hand_tm_urdf_path)

    def set_basic_hand_info(self):
        self.nb_finger_radius_low = 0.01
        self.nb_finger_radius_high = 0.02
        self.nb_finger_name_to_tip_link_name = {"index": "index_ball0", "middle":"middle_ball0", "ring": "ring_ball0", "thumb": "thumb_ball0"}
        # self.nb_finger_name_to_tip_link_name = {"index": "index_tip", "middle":"middle_tip", "ring": "ring_tip", "thumb": "thumb_tip"}
        self.pb_finger_name_to_joint_idx_range = {'index': (0, 4), 'middle': (4, 8), 
                                        'ring': (8, 12), 'thumb': (12, 16)}
        self.nb_finger_name_to_joint_idx_range = {'index': (6, 10), 'middle': (14, 18), 
                                               'ring': (18, 22), 'thumb': (10, 14)}
        self.nb_rest_pose = np.array((0, 0, 0, 0, 0, -0.3,
                                # 0.2, 0, 0, 0, 0.2, 0.5, 0, 0,
                                # 0.2, 0, 0, 0, 0.2, 0, 0, 0))
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0))
        tol_radius = 0.05
        self.collision_group_to_linknames_to_collision_radius = {
            "palm": {
                "palm_ball1": 0.025,
                "palm_ball2": 0.025,
                "palm_ball3": 0.025,
                "palm_ball4": 0.025,
                "palm_ball5": 0.025,
            },
            "index": {
                "index_ball0": 0.0134-tol_radius, 
                "index_ball1": 0.0134, 
                "index_ball2": 0.0184, 
                "index_ball3": 0.0184,  
                "index_ball4": 0.0184,  
                "index_ball5": 0.0184, 
                "index_ball6": 0.0184,
                "index_ball7": 0.0184,
            },
            "middle": {
                "middle_ball0": 0.0134-tol_radius, 
                "middle_ball1": 0.0134, 
                "middle_ball2": 0.0184, 
                "middle_ball3": 0.0184,  
                "middle_ball4": 0.0184,  
                "middle_ball5": 0.0184, 
                "middle_ball6": 0.0184,
                "middle_ball7": 0.0184,
            },
            "ring": {
                "ring_ball0": 0.0134-tol_radius, 
                "ring_ball1": 0.0134, 
                "ring_ball2": 0.0184, 
                "ring_ball3": 0.0184,  
                "ring_ball4": 0.0184,  
                "ring_ball5": 0.0184, 
                "ring_ball6": 0.0184,
                "ring_ball7": 0.0184,
            },
            "thumb": {
                "thumb_ball0": 0.0134-tol_radius, 
                "thumb_ball1": 0.0134, 
                "thumb_ball2": 0.0134, 
                "thumb_ball3": 0.0134, 
                "thumb_ball4": 0.0134, 
                "thumb_ball5": 0.0134, 
                "thumb_ball6": 0.0134, 
                "thumb_ball7": 0.0134, 
            },
        }
        self.considered_collision_group_pairs = [
            ("index", "object"), ("index", "middle"), ("index", "ring"), ("index", "thumb"),
            ("middle", "object"), ("middle", "ring"), ("middle", "thumb"), 
            ("ring", "object"), ("ring", "thumb"), 
            ("thumb", "object"),
            ("palm", "object")
        ]