# robot hand class for nimble 
# we use nimble for accurate free joint Jacobian calculation

import nimblephysics as nimble
import torch
import numpy as np
from sklearn.svm import SVC
import cvxpy as cp
from time import time
import trimesh
import pytorch_kinematics as pk
import os
from optimization.ObjectModel import Contact, PointCloud
from scipy.spatial.transform import Rotation as R


class Link:
    def __init__(self, node):
        self.node = node
        self.name = node.getName()
        total_dof = node.getSkeleton().getNumDofs()
        self.jacobian = np.zeros((3, total_dof))
        self.update()
    
    def update(self):
        self.position = self.node.getTransform().translation()
        dep_jacobian = self.node.getWorldJacobian([0.0]*3)
        dep_dof = self.node.getNumDependentGenCoords()
        for i in range(dep_dof):
            idx = self.node.getDependentGenCoordIndex(i)
            self.jacobian[:, idx] = dep_jacobian[3:, i]


class RobotHandNb:
    def __init__(self,
                hand_urdf_path,
                hand_mesh_path,
                hand_tm_urdf_path=None):
        
        self.hand_urdf_path = hand_urdf_path
        self.hand_mesh_path = hand_mesh_path

        self.nb_load_hand()
        self.set_basic_hand_info()
        if hand_tm_urdf_path is not None:
            self.chain = pk.build_chain_from_urdf(open(hand_tm_urdf_path).read())
            self.init_hand_mesh()

    def init_hand_mesh(self):
        device='cpu'
        self.hand_mesh = {}

        def build_mesh_recurse(body):
            if(len(body.link.visuals) > 0 and body.link.visuals[0].geom_type == "mesh"):
                link_name = body.link.name
                link_vertices = []
                link_faces = []
                n_link_vertices = 0
                for visual in body.link.visuals:
                    scale = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
                    if visual.geom_type == "mesh":
                        link_mesh = trimesh.load_mesh(os.path.join(self.hand_mesh_path, visual.geom_param.split("/")[-1]), process=False)
                    else:
                        continue
                    vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
                    faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
                    pos = visual.offset.to(device)
                    vertices = vertices * scale
                    vertices = pos.transform_points(vertices)
                    link_vertices.append(vertices)
                    link_faces.append(faces + n_link_vertices)
                    n_link_vertices += len(vertices)
                link_vertices = torch.cat(link_vertices, dim=0)
                link_faces = torch.cat(link_faces, dim=0)
                self.hand_mesh[link_name] = {
                    'vertices': link_vertices,
                    'faces': link_faces,
                }
            for children in body.children:
                build_mesh_recurse(children)
        build_mesh_recurse(self.chain._root)

    def get_hand_mesh(self):
        joints = self.getJoints()[6:]
        pose = torch.eye(4)
        pose[:3, :3] = torch.tensor(R.from_rotvec(self.getJoints()[:3]).as_matrix())
        pose[:3, 3] = torch.tensor(self.getJoints()[3:6])
        self.current_status = self.chain.forward_kinematics(th=torch.tensor(joints, dtype=torch.float).unsqueeze(0), world=pk.Transform3d(matrix=pose))
        data = trimesh.Trimesh()
        for link_name in self.hand_mesh:
            v = self.current_status[link_name].transform_points(
                self.hand_mesh[link_name]['vertices'])
            if len(v.shape) == 3:
                v = v[0]
            v = v.detach().cpu()
            f = self.hand_mesh[link_name]['faces'].detach().cpu()
            data += trimesh.Trimesh(vertices=v, faces=f)
        return data

    def nb_load_hand(self):
        self.world = nimble.simulation.World()
        self.world.setTimeStep(1.)
        # self.gui = nimble.NimbleGUI(self.world)
        # self.gui.serve(8080)
        # print(self.hand_urdf_path)
        self.hand = self.world.loadSkeleton(self.hand_urdf_path)
        # print(self.hand)
        # print("1"*10)
        self.links = {}
        # print("LOADING HAND URDF")
        # print(f"All together {self.hand.getNumBodyNodes()} links, names are:")
        for i in range(self.hand.getNumBodyNodes()):
            link = self.hand.getBodyNode(i)
            # print(link.getName())
            self.links[link.getName()] = Link(link)
        # print(f"All together {len(self.world.getPositions())} dofs")
        self.ndof = len(self.world.getPositions())
        self.nb_lower_joint_limits = self.hand.getPositionLowerLimits()[6:]
        self.nb_upper_joint_limits = self.hand.getPositionUpperLimits()[6:]


    def pb_jointpos_basepose_to_nb_jointpos(self, pb_jointpos, basepose_vec):
        '''
        pb_jointpos: shape=(self.ndof-6, )
        basepose_vec: (xyz, xyzw) 

        return: nb_jointpos: shape=(self.ndof)
        '''

        basepose_pos = basepose_vec[:3]
        basepose_quat = basepose_vec[3:]
        basepose_rotvec = R.from_quat(basepose_quat).as_rotvec()
        
        nb_jointpos = np.zeros(self.ndof)
        nb_jointpos[:3] = basepose_rotvec
        nb_jointpos[3:6] = basepose_pos
        
        for finger in self.nb_finger_name_to_joint_idx_range.keys():
            nb_jointpos[self.nb_finger_name_to_joint_idx_range[finger][0]: self.nb_finger_name_to_joint_idx_range[finger][1]] = pb_jointpos[self.pb_finger_name_to_joint_idx_range[finger][0]: self.pb_finger_name_to_joint_idx_range[finger][1]]
        return torch.Tensor(nb_jointpos)


    def nb_jointpos_to_pb_jointpos_basepose(self, nb_jointpos, return_type="mat"):
        '''
        nb_jointpos: shape=(self.ndof, )
        return_type: "mat" for 4x4 matrix, "vec" for xyz+xyzw 

        '''
        if isinstance(nb_jointpos, torch.Tensor):
            nb_jointpos = nb_jointpos.numpy()

        basepose_rotvec = nb_jointpos[:3]
        basepose_pos = nb_jointpos[3:6]

        return_pose = None 
        if return_type == "mat":
            return_pose = np.eye(4)
            basepose_rotmat = R.from_rotvec(basepose_rotvec).as_matrix()
            return_pose[:3, :3] = basepose_rotmat
            return_pose[:3, 3] = basepose_pos
        elif return_type == "vec":
            basepose_quat = R.from_rotvec(basepose_rotvec).as_quat()
            return_pose = np.hstack((basepose_pos, basepose_quat))
        else:
            raise NotImplementedError("not supported return pose type")

        pb_jointpos = np.zeros(self.ndof-6)

        for finger in self.nb_finger_name_to_joint_idx_range.keys():
            pb_jointpos[self.pb_finger_name_to_joint_idx_range[finger][0]: self.pb_finger_name_to_joint_idx_range[finger][1]] = nb_jointpos[self.nb_finger_name_to_joint_idx_range[finger][0]: self.nb_finger_name_to_joint_idx_range[finger][1]]
        return pb_jointpos, return_pose



    def set_basic_hand_info(self):
        raise NotImplementedError   
        

    def updateLinks(self):
        for _, link in self.links.items():
            link.update()


    def setJoints(self, q):
        self.world.setPositions(q)
        self.updateLinks()

    def getJoints(self):
        return self.hand.getPositions()
    
    
    

    def getTips(self, finger_names):
        '''
        finger_names: a list of str
        return containing the tip position of the fingers 
        '''
        return [self.links[self.nb_finger_name_to_tip_link_name[finger_name]].position for finger_name in finger_names ]


    def getJacobianTips(self, finger_names):
        return [self.links[self.nb_finger_name_to_tip_link_name[finger_name]].jacobian for finger_name in finger_names]


    def set_pc(self, pc):
        self.pc = pc
        self.use_mask = False


    def set_mask(self, mask):
        assert self.pc.number == mask.shape[0], 'Number of points not match'

        svc = SVC(kernel='linear')
        svc.fit(self.pc.points, mask)

        self.use_mask = True
        self.mask_dist = 0.01
        self.mask_dir = svc.coef_[0] / np.linalg.norm(svc.coef_[0])
        self.mask_offset = svc.intercept_[0] / np.linalg.norm(svc.coef_[0])


    def moveJoints(self, dq):
        '''
        dq: shape=(self.ndof, )
        '''
        action = torch.zeros((self.ndof,))
        self.world.setVelocities(dq)
        state = torch.Tensor(self.world.getState())
        state = nimble.timestep(self.world, state, action)
        self.updateLinks()
        return state
    

    def opt_init(self, finger_names, wrench=np.zeros((6,)), mu=1, step=0.05, min_f=0.05, max_f=1.0, linear_f=8, tol_f=0.1, low_dis=0.03, high_dis=0.08, detailed_collision=False):
        self.num_contacts = len(finger_names)
        self.opt_finger_names = finger_names
        self.opt_tip_names = [self.nb_finger_name_to_tip_link_name[finger_name] for finger_name in finger_names]
        self.wrench = wrench
        self.mu = mu 
        self.step = step
        self.min_f = min_f
        self.max_f = max_f
        self.tol_f = tol_f
        self.linear_f = linear_f
        self.construct_friction_cone()
        self.low_distance = low_dis
        self.high_distance = high_dis
        self.detailed_collision = detailed_collision
        self.G = np.zeros((6, self.num_contacts*3))
        self.G_dp = np.zeros((6, self.num_contacts*3, self.num_contacts*3))
        self.contacts = {}
        self.out_of_col_flag = False
        tip_poses = self.getTips(finger_names)
        for finger_name, tip_pos in zip(finger_names, tip_poses):
            contact = Contact(tip_pos)
            self.contacts[finger_name] = contact
        


    def construct_friction_cone(self):
        self.friction_cone = np.empty((self.linear_f + 2, 4))
        for i in range(self.linear_f):
            nx = - self.mu * np.cos(i * 2 * np.pi / self.linear_f)
            ny = - self.mu * np.sin(i * 2 * np.pi / self.linear_f)
            self.friction_cone[i] = np.array((nx, ny, 1, -self.tol_f))
        self.friction_cone[-2] = np.array((0, 0, 1, - self.min_f - self.tol_f))
        self.friction_cone[-1] = np.array((0, 0, -1, self.max_f - self.tol_f))
        
    def pc_sdf(self, pc: PointCloud, pos):
        '''
        return the sdf of a pos, if too close, return the outward normal
        '''
        indice, knn = pc.kNN(pos)
        p0 = knn[0]
        n0 = pc.normals[indice[0]]
        dp = p0 - pos
        # shawn
        # sign = np.sign(np.dot(dp, n0))
        # sdf = np.linalg.norm(dp) * sign
        sdf = np.dot(dp, n0)
        return sdf, n0


        

    def grasp_map(self):
        # Compute the adjoint matrix, grasp map and its derivative
        self.dist = 0
        self.signed_dist = 0
        self.dist_dict = {}
        self.sign_dist_dict = {}

        for i, finger_name in enumerate(self.opt_finger_names):
            self.contacts[finger_name].grasp_map(self.pc)
            self.dist += self.contacts[finger_name].h
            self.dist_dict[finger_name] = self.contacts[finger_name].h
            self.signed_dist += self.contacts[finger_name].signed_h
            self.sign_dist_dict[finger_name] = self.contacts[finger_name].signed_h

            self.G[:, i*3:i*3+3] = self.contacts[finger_name].grasp
            self.G_dp[:, i*3:i*3+3, i*3:i*3+3] = self.contacts[finger_name].grasp_dp


    def force_opt(self):
        f = cp.Variable(3*self.num_contacts)
        constraints = []
        for i in range(self.num_contacts):
            constraints += [
                cp.NonPos(- self.friction_cone[:,:3] @ f[3*i: 3*i+3]
                 - self.friction_cone[:,3])
                ]
        res = self.G @ f - self.wrench
        objective = cp.Minimize(cp.norm(res))
        prob = cp.Problem(objective, constraints)
        # shawn
        try:
            try:
                try:
                    try:
                        prob.solve(solver=cp.ECOS)
                    except cp.SolverError:
                        prob.solve(solver=cp.SCS)
                except cp.SolverError:
                    prob.solve(solver=cp.OSQP)
            except cp.SolverError:
                prob.solve(solver=cp.CVXOPT)
        except cp.SolverError:
            return np.inf
        self.f = f.value
        return prob.value
    
    def point_opt(self):
        df = cp.Variable(self.num_contacts*3)
        dq = cp.Variable(self.ndof)  

        local_dp = []
        now_dist_dict = {}
        tip_jacobians = self.getJacobianTips(self.opt_finger_names)
        for i, finger_name in enumerate(self.opt_finger_names):
            dp = tip_jacobians[i] @ dq  
            local_dp.append(self.contacts[finger_name].R.transpose() @ dp)  
            now_dist_dict[finger_name] = self.sign_dist_dict[finger_name] - self.contacts[finger_name].n @ dp  

        
        local_dp = cp.hstack(local_dp)

        slack_vars = []
        constraints = []

        for i in range(self.num_contacts): 
            constraints += [cp.NonPos(- self.friction_cone[:,:3] @ (self.f[3*i: 3*i+3] + df[3*i: 3*i+3])
                - self.friction_cone[:,3])]
            

        if self.use_mask:
            tip_jacobians = self.getJacobianTips(self.opt_finger_names)
            tip_positions = self.getTips(self.opt_finger_names)
            for i, finger_name in enumerate(self.opt_finger_names):
                p = tip_positions[i]
                dp = tip_jacobians[i] @ dq  
                separate_fild = np.dot(p, self.mask_dir) + self.mask_offset

                if separate_fild > self.mask_dist:
                    constraints += [
                        cp.NonPos(self.mask_dist - separate_fild - self.mask_dir @ dp)  
                    ]
                else: 
                    slack_var = cp.Variable()
                    slack_vars.append(slack_var)
                    constraints += [self.mask_dist - separate_fild - self.mask_dir @ dp <= slack_var]


        for collision_pair in self.considered_collision_group_pairs: 
            collision_group_a, collision_group_b = collision_pair
            
            if collision_group_b == "object":
                for link_name, link_radius in self.collision_group_to_linknames_to_collision_radius[collision_group_a].items():
                    link = self.links[link_name]
                    J_link = link.jacobian
                    pos_link = link.position
                    sdf, delta_link = self.pc_sdf(self.pc, pos_link)
                    # constraints += [
                    #     cp.NonPos(link_radius - sdf - delta_link @ J_link @ dq)  # link_radius - sdf - delta_link @ J_link @ dq <= 0 --> sdf + J_link @ dq
                    # ] # sdf - delta_link @ J_link @ dq = link radius 
                    if sdf > link_radius or self.out_of_col_flag:
                        constraints += [
                            cp.NonPos(link_radius - sdf + delta_link @ J_link @ dq)  # link_radius - sdf - delta_link @ J_link @ dq <= 0 --> sdf + J_link @ dq
                        ] # sdf - delta_link @ J_link @ dq = link radius 
                    else: 
                        slack_var = cp.Variable()
                        slack_vars.append(slack_var)
                        constraints += [link_radius - (sdf - delta_link @ J_link @ dq) <= slack_var]
                    

            else: # self collision avoidance 
                for link_name_a, link_radius_a in self.collision_group_to_linknames_to_collision_radius[collision_group_a].items():
                    link_a = self.links[link_name_a]
                    J_link_a = link_a.jacobian
                    pos_link_a = link_a.position
                    for link_name_b, link_radius_b in self.collision_group_to_linknames_to_collision_radius[collision_group_b].items():
                        # shawn
                        # if (extract_end_number_from_str(link_name_a) != 0 and extract_end_number_from_str(link_name_b) != 0 and abs(extract_end_number_from_str(link_name_a) - extract_end_number_from_str(link_name_b)) > 1) or \
                        # ((extract_end_number_from_str(link_name_a) == 0 or extract_end_number_from_str(link_name_b) == 0) and abs(extract_end_number_from_str(link_name_a) - extract_end_number_from_str(link_name_b)) > 2):
                        #     continue
                        link_b = self.links[link_name_b]
                        J_link_b = link_b.jacobian
                        pos_link_b = link_b.position

                        dis = np.linalg.norm(pos_link_a - pos_link_b)
                        if dis > link_radius_a + link_radius_b + 0.02:
                            continue
                        delta_a = (pos_link_a - pos_link_b) / dis  
                        delta_b = (pos_link_b - pos_link_a) / dis  
                        if dis > link_radius_a + link_radius_b + 0.003 or self.out_of_col_flag: 
                            constraints += [
                                cp.NonPos(link_radius_a + link_radius_b + 0.003 - dis - delta_a @ J_link_a @ dq - delta_b @ J_link_b @ dq)  
                            ]
                        else:
                            slack_var = cp.Variable()
                            slack_vars.append(slack_var)
                            constraints += [link_radius_a + link_radius_b + 0.003 - dis - delta_a @ J_link_a @ dq - delta_b @ J_link_b @ dq <= slack_var]

        for slack_var in slack_vars:
            constraints += [slack_var >= 0]

        constraints += [
            cp.NonPos(cp.abs(dq) - self.step * np.ones(self.ndof)),   
            cp.NonPos(self.getJoints()[6:] + dq[6:] - self.nb_upper_joint_limits), 
            cp.NonPos(self.nb_lower_joint_limits - self.getJoints()[6:] - dq[6:])
        ]  

        residual =  self.G @ (self.f + df) + np.tensordot(self.G_dp, self.f, 1) @ local_dp - self.wrench 
        obj_func = cp.norm(residual) 
        for finger_name, finger_dist in now_dist_dict.items():
            obj_func += cp.norm(finger_dist - self.nb_finger_radius_low)
        for slack_var in slack_vars:
            obj_func += self.lambda_penalty * cp.pos(slack_var)

        # shawn
        if len(slack_vars) == 0: 
            self.out_of_col_flag = True
        objective = cp.Minimize(obj_func)

        prob = cp.Problem(objective, constraints)
        try:
            try:
                try:
                    try:
                        prob.solve(solver=cp.ECOS)
                    except cp.SolverError:
                        prob.solve(solver=cp.SCS)
                except cp.SolverError:
                    prob.solve(solver=cp.OSQP)
            except cp.SolverError:
                prob.solve(solver=cp.CVXOPT)
        except cp.SolverError:
            return dq.value, False
        
        if prob.status == 'infeasible':
            return dq.value, False
        else:
            return dq.value, True
        
    def point_opt_wo_force(self):
        dq = cp.Variable(self.ndof)  

        local_dp = []
        now_dist_dict = {}
        tip_jacobians = self.getJacobianTips(self.opt_finger_names)
        for i, finger_name in enumerate(self.opt_finger_names):
            dp = tip_jacobians[i] @ dq  
            local_dp.append(self.contacts[finger_name].R.transpose() @ dp)  
            now_dist_dict[finger_name] = self.sign_dist_dict[finger_name] - self.contacts[finger_name].n @ dp  

        
        local_dp = cp.hstack(local_dp)

        slack_vars = []
        constraints = []

        for collision_pair in self.considered_collision_group_pairs: 
            collision_group_a, collision_group_b = collision_pair
            
            if collision_group_b == "object":
                for link_name, link_radius in self.collision_group_to_linknames_to_collision_radius[collision_group_a].items():
                    link = self.links[link_name]
                    J_link = link.jacobian
                    pos_link = link.position
                    sdf, delta_link = self.pc_sdf(self.pc, pos_link)
                    # constraints += [
                    #     cp.NonPos(link_radius - sdf - delta_link @ J_link @ dq)  # link_radius - sdf - delta_link @ J_link @ dq <= 0 --> sdf + J_link @ dq
                    # ] # sdf - delta_link @ J_link @ dq = link radius 
                    if sdf > link_radius:
                        constraints += [
                            cp.NonPos(link_radius - sdf + delta_link @ J_link @ dq)  # link_radius - sdf - delta_link @ J_link @ dq <= 0 --> sdf + J_link @ dq
                        ] # sdf - delta_link @ J_link @ dq = link radius 
                    else: 
                        slack_var = cp.Variable()
                        slack_vars.append(slack_var)
                        constraints += [link_radius - (sdf - delta_link @ J_link @ dq) <= slack_var]

            else: # self collision avoidance 
                for link_name_a, link_radius_a in self.collision_group_to_linknames_to_collision_radius[collision_group_a].items():
                    link_a = self.links[link_name_a]
                    J_link_a = link_a.jacobian
                    pos_link_a = link_a.position
                    for link_name_b, link_radius_b in self.collision_group_to_linknames_to_collision_radius[collision_group_b].items():
                        # if (extract_end_number_from_str(link_name_a) != 0 and extract_end_number_from_str(link_name_b) != 0 and abs(extract_end_number_from_str(link_name_a) - extract_end_number_from_str(link_name_b)) > 1) or \
                        # ((extract_end_number_from_str(link_name_a) == 0 or extract_end_number_from_str(link_name_b) == 0) and abs(extract_end_number_from_str(link_name_a) - extract_end_number_from_str(link_name_b)) > 2):
                        #     continue
                        link_b = self.links[link_name_b]
                        J_link_b = link_b.jacobian
                        pos_link_b = link_b.position

                        dis = np.linalg.norm(pos_link_a - pos_link_b)
                        delta_a = (pos_link_a - pos_link_b) / dis  
                        delta_b = (pos_link_b - pos_link_a) / dis  
                        if dis > link_radius_a + link_radius_b + 0.01:
                            constraints += [
                                cp.NonPos(link_radius_a + link_radius_b + 0.01 - dis - delta_a @ J_link_a @ dq - delta_b @ J_link_b @ dq)  
                            ]
                        else:
                            slack_var = cp.Variable()
                            slack_vars.append(slack_var)
                            constraints += [link_radius_a + link_radius_b + 0.01 - dis - delta_a @ J_link_a @ dq - delta_b @ J_link_b @ dq <= slack_var]

        
        constraints += [
            cp.NonPos(cp.abs(dq) - self.step * np.ones(self.ndof)),   
            cp.NonPos(self.getJoints()[6:] + dq[6:] - self.nb_upper_joint_limits), 
            cp.NonPos(self.nb_lower_joint_limits - self.getJoints()[6:] - dq[6:])
        ]  

        obj_func = 0
        for finger_name, finger_dist in now_dist_dict.items():
            obj_func += cp.norm(finger_dist - self.nb_finger_radius_low)
        objective = cp.Minimize(obj_func)

        prob = cp.Problem(objective, constraints)
        try:
            try:
                try:
                    try:
                        prob.solve(solver=cp.ECOS)
                    except cp.SolverError:
                        prob.solve(solver=cp.SCS)
                except cp.SolverError:
                    prob.solve(solver=cp.OSQP)
            except cp.SolverError:
                prob.solve(solver=cp.CVXOPT)
        except cp.SolverError:
            return dq.value, False

        if prob.status == 'infeasible':
            return dq.value, False
        else:
            return dq.value, True
           
    def update_wo_force(self):
        self.grasp_map()
        
        if (np.array(list(self.sign_dist_dict.values())) <= self.nb_finger_radius_high).all() and (np.array(list(self.sign_dist_dict.values())) >= self.nb_finger_radius_low).all():
            return 'success'

        dq, flag_joint = self.point_opt_wo_force()
        if not flag_joint:
            return 'failure'

        self.moveJoints(dq)
        tip_poses = self.getTips(self.opt_finger_names)

        for i, finger_name in enumerate(self.opt_finger_names):
            self.contacts[finger_name].p = tip_poses[i]
        return 'continue'
    
    def update_ik(self):
        self.dist = 0
        self.dist_dict = {}
        self.dist_dir_dict = {}

        tip_poses = self.getTips(self.opt_finger_names)
        for i, finger_name in enumerate(self.opt_finger_names):
            p = tip_poses[i]
            target_p = self.finger_names2pos[finger_name]
            p_dis = np.linalg.norm(p - target_p)
            delta_tip = (target_p - p) / p_dis  

            self.dist += p_dis
            self.dist_dict[finger_name] = p_dis
            self.dist_dir_dict[finger_name] = delta_tip


        if (np.array(list(self.dist_dict.values())) <= self.nb_finger_radius_high).all() and (np.array(list(self.dist_dict.values())) >= self.nb_finger_radius_low).all():
            return 'success'

        dq = cp.Variable(self.ndof)  

        constraints = []

        if not self.detailed_collision:
            for name, link in self.links.items():
                J_link = link.jacobian
                pos_link = link.position
                sdf, delta_link = self.pc_sdf(self.pc, pos_link)
                if name in self.opt_tip_names:
                    constraints += [
                        cp.NonPos(- sdf - delta_link @ J_link @ dq + self.nb_finger_radius_low)  # -(distance_now+delta_dis)<= 0 -> distance_now+delta_dis >= self.nb_finger_radius_low # acceptable contact collision distance 
                    ]
                    for name2, tip2 in self.links.items():
                        if name == name2 or name2 not in self.opt_tip_names:
                            continue
                        pos_tip2 = tip2.position
                        sdf = np.linalg.norm(pos_link - pos_tip2)
                        delta_tip = (pos_link - pos_tip2) / sdf  
                        constraints += [
                            cp.NonPos(2 * self.nb_finger_radius_high - sdf - delta_tip @ J_link @ dq)  
                        ]
                else:
                    constraints += [
                        cp.NonPos(self.nb_finger_radius_high - sdf - delta_link @ J_link @ dq)
                    ]
        else: 
            for collision_pair in self.considered_collision_group_pairs: 
                collision_group_a, collision_group_b = collision_pair
                
                if collision_group_b == "object":
                    for link_name, link_radius in self.collision_group_to_linknames_to_collision_radius[collision_group_a].items():
                        link = self.links[link_name]
                        J_link = link.jacobian
                        pos_link = link.position
                        sdf, delta_link = self.pc_sdf(self.pc, pos_link)
                        constraints += [
                            cp.NonPos(link_radius - sdf - delta_link @ J_link @ dq)
                        ]
                else: # self collision avoidance 
                    for link_name_a, link_radius_a in self.collision_group_to_linknames_to_collision_radius[collision_group_a].items():
                        link_a = self.links[link_name_a]
                        J_link_a = link_a.jacobian
                        pos_link_a = link_a.position
                        for link_name_b, link_radius_b in self.collision_group_to_linknames_to_collision_radius[collision_group_b].items():
                            # if (extract_end_number_from_str(link_name_a) != 0 and extract_end_number_from_str(link_name_b) != 0 and abs(extract_end_number_from_str(link_name_a) - extract_end_number_from_str(link_name_b)) > 1) or \
                            # ((extract_end_number_from_str(link_name_a) == 0 or extract_end_number_from_str(link_name_b) == 0) and abs(extract_end_number_from_str(link_name_a) - extract_end_number_from_str(link_name_b)) > 2):
                            #     continue
                            link_b = self.links[link_name_b]
                            J_link_b = link_b.jacobian
                            pos_link_b = link_b.position

                            dis = np.linalg.norm(pos_link_a - pos_link_b)
                            delta_a = (pos_link_a - pos_link_b) / dis  
                            delta_b = (pos_link_b - pos_link_a) / dis  
                            constraints += [
                                cp.NonPos(link_radius_a + link_radius_b + 0.003 - dis - delta_a @ J_link_a @ dq - delta_b @ J_link_b @ dq)  
                            ]

        
        constraints += [
            cp.NonPos(cp.abs(dq) - self.step * np.ones(self.ndof)),   
            cp.NonPos(self.getJoints()[6:] + dq[6:] - self.nb_upper_joint_limits), 
            cp.NonPos(self.nb_lower_joint_limits - self.getJoints()[6:] - dq[6:])
        ]  

        obj_func = 0

        tip_jacobians = self.getJacobianTips(self.opt_finger_names)
        for i, finger_name in enumerate(self.opt_finger_names):
            dp = tip_jacobians[i] @ dq  
            obj_func += cp.norm(self.dist_dict[finger_name] - self.dist_dir_dict[finger_name] @ dp - self.nb_finger_radius_low)

        objective = cp.Minimize(obj_func)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)

        if prob.status == 'infeasible':
            flag_joint = False
        else:
            flag_joint = True

        if not flag_joint:
            return 'failure'
        
        self.moveJoints(dq.value)

        return 'continue'

    def update(self, wrench_threshold=1e-6): #shawn
        self.grasp_map()
        self.value = self.force_opt() 
        
        if self.value < wrench_threshold and (np.array(list(self.sign_dist_dict.values())) <= self.nb_finger_radius_high).all() and (np.array(list(self.sign_dist_dict.values())) >= self.nb_finger_radius_low).all():
            return 'success'

        dq, flag_joint = self.point_opt()
        if not flag_joint:
            return 'failure'

        self.moveJoints(dq)
        tip_poses = self.getTips(self.opt_finger_names)

        for i, finger_name in enumerate(self.opt_finger_names):
            self.contacts[finger_name].p = tip_poses[i]

        return 'continue'
    
    def wrench_opt_evaluate(self):
        dist = 0
        tip_poses = self.getTips(self.opt_finger_names)
        for i, finger_name in enumerate(self.opt_finger_names):
            dist += np.linalg.norm(self.contacts[finger_name].p - tip_poses[i])
        dist /= self.num_contacts

        collisions = []
        for _, link in self.links.items():
            pos_link = link.position
            sdf, _ = self.pc_sdf(self.pc, pos_link)
            collisions.append(sdf)
        collision = min(collisions)
        return self.value, dist, collision 
    
    def contact_wrench_jointpos_opt(self, finger_names, wrench=np.zeros((6,)), step_num=100 , mu=1, step=0.05, min_f=0.05, max_f=1.0, low_dis=0.03, high_dis=0.08, evaluate=True, detailed_collision=False, lambda_penalty=10):
        '''
        finger_names: a list of str
        
        '''
        # init contact 
        self.lambda_penalty = lambda_penalty
        self.opt_init(finger_names, wrench=wrench, mu=mu, step=step, min_f=min_f, max_f=max_f, low_dis=low_dis, high_dis=high_dis, detailed_collision=detailed_collision)

        start = time()
        # t = trange(step_num, desc='Point Opt', leave=True)
        t = range(step_num)
        pose_list = []
        dist_list = []
        dist_dict_list = []
        iters = 0
        pose_list.append(self.getJoints())
        for i in t:
            flag = self.update()
            # t.set_description(f"Value: {self.value}, Dist: {self.dist}", refresh=True)
            pose_list.append(self.getJoints())
            dist_list.append(self.signed_dist)
            dist_dict_list.append(self.sign_dist_dict)
            iters = i
            if flag != 'continue':
                break
        end = time()

        self.grasp_map() # shawn
        value = self.force_opt()

        finger_force = {}

        local_f = self.f
        grasp_map = self.G
        opt_wrench = self.G @ self.f
        tip_positions = self.getTips(self.opt_finger_names)
        
        for i, finger_name in enumerate(self.opt_finger_names):
            contact_force = grasp_map[:3, 3*i:3*i+3] @ local_f[3*i:3*i+3]
            finger_force.update({finger_name: (tip_positions[i], contact_force)})

        info = {'flag': flag, 'value': value, 'iters': iters,'time': end-start, 'joints': self.getJoints(), 'wrench': wrench, 'pose_list': pose_list, 'finger_force': finger_force, "opt_wrench": opt_wrench}
        return info

    def contact_jointpos_opt(self, finger_names, wrench=np.zeros((6,)), step_num=100 , mu=1, step=0.05, min_f=0.05, max_f=1.0, low_dis=0.03, high_dis=0.08, evaluate=False, detailed_collision=False):
        '''
        finger_names: a list of str
        
        '''
        # init contact 
        self.opt_init(finger_names, wrench=wrench, mu=mu, step=step, min_f=min_f, max_f=max_f, low_dis=low_dis, high_dis=high_dis, detailed_collision=detailed_collision)

        start = time()
        t = range(step_num)
        iters = 0
        pose_list = []
        pose_list.append(self.getJoints())
        for i in t:
            flag = self.update_wo_force()
            pose_list.append(self.getJoints())
            iters = i
            if flag != 'continue':
                break
        end = time()
    
        self.grasp_map()
        value = self.force_opt()

        finger_force = {}

        local_f = self.f
        grasp_map = self.G
        opt_wrench = self.G @ self.f
        tip_positions = self.getTips(self.opt_finger_names)
        
        for i, finger_name in enumerate(self.opt_finger_names):
            contact_force = grasp_map[:3, 3*i:3*i+3] @ local_f[3*i:3*i+3]
            finger_force.update({finger_name: (tip_positions[i], contact_force)})

        info = {'flag': flag, 'value': value, 'iters': iters,'time': end-start, 'joints': self.getJoints(), 'wrench': wrench, 'pose_list': pose_list, 'finger_force': finger_force, "opt_wrench": opt_wrench}
        return info
    
    def ik_jointpos_opt(self, finger_names2pos, step_num=100, step=0.08, detailed_collision=True):
        '''
        finger_names2pos: a dict, e.g., {
            "index": [x, y, z],
            "middle": ...,
            ...
        }
        '''
        finger_names = list(finger_names2pos.keys())
        self.opt_init(finger_names, step=step)
        self.finger_names2pos = finger_names2pos
        self.num_contacts = len(finger_names)
        self.opt_finger_names = finger_names
        self.opt_tip_names = [self.nb_finger_name_to_tip_link_name[finger_name] for finger_name in finger_names]
        self.detailed_collision = detailed_collision
        
        start = time()
        # t = trange(step_num, desc='IK Opt', leave=True)
        for i in range(step_num):
            flag = self.update_ik()
            # t.set_description(f"Dist: {self.dist}", refresh=True)
            if flag != 'continue':
                break
        end = time()

        # print(f'Time: {end-start}')
        return flag
    

# class AllegroHandNb(RobotHandNb):
#     def __init__(self, 
#                 hand_urdf_path="./assets/hands/allegro_hand_right/allegro_hand_right_nb.urdf", 
#                 hand_mesh_path="./assets/hands/allegro_hand_right/meshes",
#                 ):
#         super().__init__(hand_urdf_path,  hand_mesh_path)

#     def set_basic_hand_info(self):
#         self.nb_finger_radius_low = 0.01
#         self.nb_finger_radius_high = 0.0142
#         # self.nb_finger_name_to_tip_link_name = {"index": "link_3.0_tip", "middle":"link_7.0_tip", "ring": "link_11.0_tip", "thumb": "link_15.0_tip"}
#         self.nb_finger_name_to_tip_link_name = {"index": "link_3.0_tip", "thumb": "link_15.0_tip"}

#         # self.nb_finger_name_to_joint_idx_range = {'index': (6, 10), 'middle': (14, 18), 
#         #                                        'ring': (18, 22), 'thumb': (10, 14)}
#         # self.pb_finger_name_to_joint_idx_range = {'index': (0, 4), 'middle': (4, 8), 
#         #                                        'ring': (8, 12), 'thumb': (12, 16)}
#         self.nb_finger_name_to_joint_idx_range = {'index': (6, 10), 'thumb': (10, 14)}
#         self.pb_finger_name_to_joint_idx_range = {'index': (0, 4), 'thumb': (12, 16)}
#         self.nb_rest_pose = np.array((0, -1.5, 0, 0, 0, -0.5,
#                                 0, 0, 0, 0, 0, 0, 0, 0,
#                                 0, 0, 0, 0, 0, 0, 0, 0))
        
#         self.collision_group_to_linknames_to_collision_radius = {
#             "palm": {
#                 "palm_ball1": 0.025,
#                 "palm_ball2": 0.025,
#                 "palm_ball3": 0.025,
#                 "palm_ball4": 0.025,
#                 "palm_ball5": 0.025,
#             },
#             "index": {
#                 "index_ball0": 0.0134-0.05, 
#                 "index_ball1": 0.0134, 
#                 "index_ball2": 0.0184, 
#                 "index_ball3": 0.0184,  
#                 "index_ball4": 0.0184,  
#                 "index_ball5": 0.0184, 
#                 "index_ball6": 0.0184,
#                 "index_ball7": 0.0184,
#             },
#             "middle": {
#                 "middle_ball0": 0.0134-0.05, 
#                 "middle_ball1": 0.0134, 
#                 "middle_ball2": 0.0184, 
#                 "middle_ball3": 0.0184,  
#                 "middle_ball4": 0.0184,  
#                 "middle_ball5": 0.0184, 
#                 "middle_ball6": 0.0184,
#                 "middle_ball7": 0.0184,
#             },
#             "ring": {
#                 "ring_ball0": 0.0134-0.05, 
#                 "ring_ball1": 0.0134, 
#                 "ring_ball2": 0.0184, 
#                 "ring_ball3": 0.0184,  
#                 "ring_ball4": 0.0184,  
#                 "ring_ball5": 0.0184, 
#                 "ring_ball6": 0.0184,
#                 "ring_ball7": 0.0184,
#             },
#             "thumb": {
#                 "thumb_ball0": 0.0134-0.05, 
#                 "thumb_ball1": 0.0134, 
#                 "thumb_ball2": 0.0134, 
#                 "thumb_ball3": 0.0134, 
#                 "thumb_ball4": 0.0134, 
#                 "thumb_ball5": 0.0134, 
#                 "thumb_ball6": 0.0134, 
#                 "thumb_ball7": 0.0134, 
#             },
#         }
#         self.considered_collision_group_pairs = [
#             ("index", "object"), ("index", "middle"), ("index", "ring"), ("index", "thumb"),
#             ("middle", "object"), ("middle", "ring"), ("middle", "thumb"), 
#             ("ring", "object"), ("ring", "thumb"), 
#             ("thumb", "object"),
#             ("palm", "object")
#         ]
        
# class BarrettHandNb(RobotHandNb):
#     def __init__(self, 
#                 hand_urdf_path="./assets/hands/bhand_model/bhand_model.urdf", 
#                 hand_mesh_path="./assets/hands/bhand_model/meshes",
#                 ):
#         super().__init__(hand_urdf_path,  hand_mesh_path)

#     def set_basic_hand_info(self):
#         self.nb_finger_radius_low = 0.01
#         self.nb_finger_radius_high = 0.0142
#         self.nb_finger_name_to_tip_link_name = {"index": "finger_1_dist_link", "middle":"finger_1_dist_link", "thumb": "finger_1_dist_link"}
#         self.nb_finger_name_to_joint_idx_range = {'index': (6, 9), 'middle': (9, 12), 'thumb': (12, 15)}
#         self.nb_rest_pose = np.array((0, -1.5, 0, 0, 0, -0.5,
#                                 0, 0, 0, 0, 0, 0, 0, 0,
#                                 0, 0, 0, 0, 0, 0, 0, 0))
        
#         self.collision_group_to_linknames_to_collision_radius = {
#             "palm": {
#                 "palm_ball1": 0.025,
#                 "palm_ball2": 0.025,
#                 "palm_ball3": 0.025,
#                 "palm_ball4": 0.025,
#                 "palm_ball5": 0.025,
#             },
#             "index": {
#                 "index_ball0": 0.0134-0.05, 
#                 "index_ball1": 0.0134, 
#                 "index_ball2": 0.0184, 
#                 "index_ball3": 0.0184,  
#                 "index_ball4": 0.0184,  
#                 "index_ball5": 0.0184, 
#                 "index_ball6": 0.0184,
#                 "index_ball7": 0.0184,
#             },
#             "middle": {
#                 "middle_ball0": 0.0134-0.05, 
#                 "middle_ball1": 0.0134, 
#                 "middle_ball2": 0.0184, 
#                 "middle_ball3": 0.0184,  
#                 "middle_ball4": 0.0184,  
#                 "middle_ball5": 0.0184, 
#                 "middle_ball6": 0.0184,
#                 "middle_ball7": 0.0184,
#             },
#             "ring": {
#                 "ring_ball0": 0.0134-0.05, 
#                 "ring_ball1": 0.0134, 
#                 "ring_ball2": 0.0184, 
#                 "ring_ball3": 0.0184,  
#                 "ring_ball4": 0.0184,  
#                 "ring_ball5": 0.0184, 
#                 "ring_ball6": 0.0184,
#                 "ring_ball7": 0.0184,
#             },
#             "thumb": {
#                 "thumb_ball0": 0.0134-0.05, 
#                 "thumb_ball1": 0.0134, 
#                 "thumb_ball2": 0.0134, 
#                 "thumb_ball3": 0.0134, 
#                 "thumb_ball4": 0.0134, 
#                 "thumb_ball5": 0.0134, 
#                 "thumb_ball6": 0.0134, 
#                 "thumb_ball7": 0.0134, 
#             },
#         }
#         self.considered_collision_group_pairs = [
#             ("index", "object"), ("index", "middle"), ("index", "ring"), ("index", "thumb"),
#             ("middle", "object"), ("middle", "ring"), ("middle", "thumb"), 
#             ("ring", "object"), ("ring", "thumb"), 
#             ("thumb", "object"),
#             ("palm", "object")
#         ]

# class LeapHandNb(RobotHandNb):
#     def __init__(self, 
#                  hand_urdf_path="./assets/hands/leap_hand/robot_nb.urdf", 
#                  hand_mesh_path="./assets/hands/leap_hand",
#                 ):
#         super().__init__(hand_urdf_path, hand_mesh_path)

#     def set_basic_hand_info(self):
#         self.nb_finger_radius_low = 0.01
#         self.nb_finger_radius_high = 0.014
#         self.nb_finger_name_to_tip_link_name = {"index": "index_ball0", "middle":"middle_ball0", "ring": "ring_ball0", "thumb": "thumb_ball0"}
#         self.nb_finger_name_to_joint_idx_range = {'index': (6, 10), 'middle': (14, 18), 
#                                                'ring': (18, 22), 'thumb': (10, 14)}
#         self.pb_finger_name_to_joint_idx_range = {'index': (0, 4), 'middle': (4, 8), 
#                                                'ring': (8, 12), 'thumb': (12, 16)}
#         self.nb_rest_pose = np.array((0, -1.5, 0, 0, 0, -0.5,
#                                 0, 0, 0, 0, 0.3, 0, 0, 0,
#                                 0, 0, 0, 0, 0, 0, 0, 0))
#         self.collision_group_to_linknames_to_collision_radius = {
#             "palm": {
#                 "palm_ball1": 0.025,
#                 "palm_ball2": 0.025,
#                 "palm_ball3": 0.025,
#                 "palm_ball4": 0.025,
#                 "palm_ball5": 0.025,
#             },
#             "index": {
#                 "index_ball0": 0.0134-0.05, 
#                 "index_ball1": 0.0134, 
#                 "index_ball2": 0.0184, 
#                 "index_ball3": 0.0184,  
#                 "index_ball4": 0.0184,  
#                 "index_ball5": 0.0184, 
#                 "index_ball6": 0.0184,
#                 "index_ball7": 0.0184,
#             },
#             "middle": {
#                 "middle_ball0": 0.0134-0.05, 
#                 "middle_ball1": 0.0134, 
#                 "middle_ball2": 0.0184, 
#                 "middle_ball3": 0.0184,  
#                 "middle_ball4": 0.0184,  
#                 "middle_ball5": 0.0184, 
#                 "middle_ball6": 0.0184,
#                 "middle_ball7": 0.0184,
#             },
#             "ring": {
#                 "ring_ball0": 0.0134-0.05, 
#                 "ring_ball1": 0.0134, 
#                 "ring_ball2": 0.0184, 
#                 "ring_ball3": 0.0184,  
#                 "ring_ball4": 0.0184,  
#                 "ring_ball5": 0.0184, 
#                 "ring_ball6": 0.0184,
#                 "ring_ball7": 0.0184,
#             },
#             "thumb": {
#                 "thumb_ball0": 0.0134-0.05, 
#                 "thumb_ball1": 0.0134, 
#                 "thumb_ball2": 0.0134, 
#                 "thumb_ball3": 0.0134, 
#                 "thumb_ball4": 0.0134, 
#                 "thumb_ball5": 0.0134, 
#                 "thumb_ball6": 0.0134, 
#                 "thumb_ball7": 0.0134, 
#             },
#         }
#         self.considered_collision_group_pairs = [
#             ("index", "object"), ("index", "middle"), ("index", "ring"), ("index", "thumb"),
#             ("middle", "object"), ("middle", "ring"), ("middle", "thumb"), 
#             ("ring", "object"), ("ring", "thumb"), 
#             ("thumb", "object"),
#             ("palm", "object")
#         ]

# class ShadowHandNb(RobotHandNb):
#     def __init__(self, 
#                  hand_urdf_path="./assets/hands/leap_hand/robot_nb.urdf", 
#                  hand_mesh_path="./assets/hands/leap_hand",
#                 ):
#         super().__init__(hand_urdf_path, hand_mesh_path)

#     def set_basic_hand_info(self):
#         self.nb_finger_radius_low = 0.01
#         self.nb_finger_radius_high = 0.014
#         self.nb_finger_name_to_tip_link_name = {"index": "index_tip", "little": "little_tip", "middle":"middle_tip", "ring": "ring_tip", "thumb": "thumb_tip"}
#         self.nb_finger_name_to_joint_idx_range = {'index': (6, 10), 'little': (10, 15),'middle': (15, 19), 
#                                                'ring': (19, 23), 'thumb': (23, 28)}
#         self.nb_rest_pose = np.array((0, 0, 0, 0, 0, 0,
#                                 0, 0, 0, 0, 0, 0, 0, 0,
#                                 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0))
#         self.collision_group_to_linknames_to_collision_radius = {
#             "palm": {
#                 "palm": 0.025,
#             },
#             "index": {
#                 "index_tip": 0.0134-0.05, 
#                 "index_finger_distal": 0.0134, 
#                 "index_finger_middle": 0.0184, 
#                 "index_finger_proximal": 0.0184,  
#                 "index_finger_knuckle": 0.0184,  
#             },
#             "little": {
#                 "little_tip": 0.0134-0.05, 
#                 "little_finger_distal": 0.0134, 
#                 "little_finger_middle": 0.0134, 
#                 "little_finger_proximal": 0.0134,  
#                 "little_finger_knuckle": 0.0134,  
#                 "little_finger_metacarpal": 0.0134, 
#             },
#             "middle": {
#                 "middle_tip": 0.0134-0.05, 
#                 "middle_finger_distal": 0.0134, 
#                 "middle_finger_middle": 0.0184, 
#                 "middle_finger_proximal": 0.0184,  
#                 "middle_finger_knuckle": 0.0184,  
#             },
#             "ring": {
#                 "ring_tip": 0.0134-0.05, 
#                 "ring_finger_distal": 0.0134, 
#                 "ring_finger_middle": 0.0184, 
#                 "ring_finger_proximal": 0.0184,  
#                 "ring_finger_knuckle": 0.0184,  
#             },
#             "thumb": {
#                 "thumb_tip": 0.0134-0.05, 
#                 "thumb_distal": 0.0134, 
#                 "thumb_middle": 0.0134, 
#                 "thumb_hub": 0.0134, 
#                 "thumb_proximal": 0.0134, 
#                 "thumb_base": 0.0134, 
#             },
#         }
#         self.considered_collision_group_pairs = [
#             ("index", "object"), ("index", "little"), ("index", "middle"), ("index", "ring"), ("index", "thumb"),
#             ("little", "object"), ("little", "middle"), ("little", "ring"), ("little", "thumb"),
#             ("middle", "object"), ("middle", "ring"), ("middle", "thumb"), 
#             ("ring", "object"), ("ring", "thumb"), 
#             ("thumb", "object"),
#             ("palm", "object")
#         ]
        
