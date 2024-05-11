import numpy as np 
import trimesh 
import pytorch3d.ops
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import curve_fit
from scipy.linalg import eigh
import seaborn as sns
from utils.opt_utils import (
    normalize,
    hat,
    local_frame,
    quadric,
    as_mesh,
    get_transform_matrix,
    sample_cone,
    create_urdf_from_obj
)

class Contact: 
    def __init__(self, p=None, n=None):
        self.p = p  
        self._p = None  
        self.h = None  
        self.n = n  
        self.t1 = None
        self.t2 = None
        self.k1 = None  
        self.k2 = None
        self.R = np.zeros((3, 3))
        self.grasp = np.zeros((6, 3))
        self.grasp_dp = np.zeros((6, 3, 3))

    def normalize(self):
        self.n = normalize(self.n)
        self.t2 = normalize(np.cross(self.n, self.t1))
        self.t1 = normalize(np.cross(self.t2, self.n))

    def project(self, pc):
        self._p, self.n, k1, k2, self.t1 = pc.project(self.p)
        self.h = np.linalg.norm(self.p - self._p)
        sign = np.sign(np.dot(self._p - self.p, self.n))
        self.signed_h = sign * self.h

        self.k1 = 1 / (self.h + 1 / k1)
        self.k2 = 1 / (self.h + 1 / k2)
        self.normalize()

    def grasp_map_only(self):
        self.n = normalize(self.n)
        r = np.random.rand(3)
        self.t1 = normalize(np.cross(r, self.n))
        self.t2 = normalize(np.cross(self.n, self.t1))
        self.R[:, 0] = self.t1
        self.R[:, 1] = self.t2
        self.R[:, 2] = self.n

        self.grasp[:3, :] = self.R
        self.grasp[3:, :] = hat(self.p) @ self.R
        return self.grasp

    def grasp_map(self, pc):   # approximation, and with gradient
        self.project(pc)

        self.R[:, 0] = self.t1
        self.R[:, 1] = self.t2
        self.R[:, 2] = self.n

        self.grasp[:3, :] = self.R
        self.grasp[3:, :] = hat(self.p) @ self.R

        R_dt1 = np.zeros((3, 3))
        R_dt1[:, 0] = self.k1 * self.n
        R_dt1[:, 2] = - self.k1 * self.t1

        R_dt2 = np.zeros((3, 3))
        R_dt2[:, 1] = self.k2 * self.n
        R_dt2[:, 2] = - self.k2 * self.t2

        grasp_dt1 = np.zeros((6, 3))
        grasp_dt1[:3, :] = R_dt1
        grasp_dt1[3:, :] = hat(self.p) @ R_dt1 + hat(self.t1) @ self.R
        
        grasp_dt2 = np.zeros((6, 3))
        grasp_dt2[:3, :] = R_dt2
        grasp_dt2[3:, :] = hat(self.p) @ R_dt2 + hat(self.t2) @ self.R

        grasp_dn = np.zeros((6, 3))
        grasp_dn[3:, :] = hat(self.n) @ self.R

        self.grasp_dp[:, 0, :] = grasp_dt1  
        self.grasp_dp[:, 1, :] = grasp_dt2
        self.grasp_dp[:, 2, :] = grasp_dn


class PointCloud:
    def __init__(self, points, normals, k_nn=20):
        self.points = points
        self.normals = normals
        self.number = points.shape[0]
        self.k_nn = k_nn
        self.knn_model = NearestNeighbors(n_neighbors=self.k_nn, algorithm='auto', metric='euclidean')
        self.knn_model.fit(self.points)

    def kNN(self, p):
        _, indices = self.knn_model.kneighbors([p])
        knn_points = [self.points[indices[0][i]]
                       for i in range(len(indices[0]))]
        knn_points = np.array(knn_points)
        return indices[0], knn_points  

    def project(self, p):
        indice, knn = self.kNN(p)

        p0 = knn[0]
        n0 = self.normals[indice[0]]
        R = local_frame(n0)

        local_knn = []
        # for _p in knn:
        #     dp = _p - p0
        #     local_dp = R.transpose() @ dp
        #     local_knn.append(local_dp)
        for _p_i in range(len(knn)):
            _p = knn[_p_i]
            _p_idx = indice[_p_i]
            _n = self.normals[_p_idx]
            if n0 @ _n >= 0:
                dp = _p - p0
                local_dp = R.transpose() @ dp
                local_knn.append(local_dp)

        local_knn = np.array(local_knn) 

        (a, b, c), _ = curve_fit(quadric, (local_knn[:, 0], local_knn[:, 1]), local_knn[:, 2])

        A = np.array([[a * 2, b], 
                    [b, c * 2]])
        curv, ev = eigh(A)

        k1 = curv[0]
        k2 = curv[1]
        t1 = ev[0][0] * R[:, 0] + ev[0][1] * R[:, 1]

        return p0, n0, k1, k2, t1  


class ObjectModel:
    def __init__(self,
                 decomposed_mesh_path,
                 scale=[1.0, 1.0, 1.0],
                 sample_point_num=2048,
                 tip_retreat_dis=0.0):
        '''
        Initialize the ObjectModel with a given mesh path and parameters.

        Parameters:
        - decomposed_mesh_path (str): Path to the mesh file for the object.
        - scale (list, optional): Scale factors in x, y, z directions. Default is [1.0, 1.0, 1.0].
        - sample_point_num (int, optional): Number of points to sample from the mesh surface. Default is 2048.
        - tip_retreat_dis (float, optional): Distance to retreat the sampled points from the mesh surface. Default is 0.0.
        '''

        self.mesh_path = decomposed_mesh_path
        self.scale = scale
        self.sample_point_num = sample_point_num
        self.tip_retreat_dis = tip_retreat_dis
        self.mesh = as_mesh(trimesh.load(self.mesh_path, force="mesh", process=False)).apply_transform(get_transform_matrix(scale=self.scale))
        self.sample_camera_pc()
        self.cluster()
        

    def sample_pc(self):
        '''
        Sample a point cloud from the object's mesh surface.

        The method samples points from the object's mesh surface, computes the normals at these points, 
        and then retreats the points away from the surface by a specified distance.
        '''

        sphere_points = trimesh.sample.sample_surface_sphere(10 * self.sample_point_num)
        ray_directions = - sphere_points

        rmi = trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh, scale_to_box=False)
        locations, index_ray, index_tri =  rmi.intersects_location(sphere_points, ray_directions, multiple_hits=False)

        # ABANDONED: random sampling
        # intersect_num = locations.shape[0]
        # intersect_choose_id = np.random.choice(intersect_num, self.sample_point_num, replace=False)
        
        # Farthest points sampling
        points_torch = torch.tensor(locations).unsqueeze(0)
        locations_choose_torch, intersect_choose_id_torch = pytorch3d.ops.sample_farthest_points(points_torch, K=self.sample_point_num)
        intersect_choose_id = intersect_choose_id_torch.squeeze(0).detach().cpu().numpy()
        
        locations = locations[intersect_choose_id]
        index_tri = index_tri[intersect_choose_id]

        self.point_idx_to_surface_points = locations
        self.point_idx_to_face_idx = index_tri
        self.point_idx_to_face_normals = self.mesh.face_normals[self.point_idx_to_face_idx]
        self.point_idx_to_tip_center_reach_points = self.point_idx_to_surface_points + self.point_idx_to_face_normals * self.tip_retreat_dis


    def sample_camera_pc(self, camera_ray_num=16384, min_sphere_radius=0.75, max_sphere_radius=1.25):
        '''
        Sample a point cloud from the object's mesh surface using multiple camera positions.

        The method samples points from the object's mesh surface using rays emitted from 6 predefined 
        camera positions on a sphere. The sphere undergoes a random rotation, and each camera position 
        emits a specified number of rays within a 60-degree cone facing the origin. The method then 
        computes the normals at these points, and then retreats the points away from the surface 
        by a specified distance.
        '''
        
        # Step 1: Define fixed camera positions
        fixed_camera_positions = np.array([
            [0, 0, 1],  # North pole
            [0, 0, -1],  # South pole
            [1, 0, 0],  # Right
            [-1, 0, 0],  # Left
            [0, 1, 0],  # Front
            [0, -1, 0]  # Back
        ])
        
        # Step 2: Randomly generate a rotation for the entire sphere
        rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
        # rotation = np.eye(3)

        # Step 3: Apply the rotation to the fixed camera positions
        rotated_camera_positions = fixed_camera_positions @ rotation.T
        sphere_radius = np.random.uniform(min_sphere_radius, max_sphere_radius, size=(fixed_camera_positions.shape[0], 1))
        rotated_camera_positions = sphere_radius * rotated_camera_positions

        # Initialize lists to store results from all cameras
        all_locations = []
        all_index_tri = []
        
        rmi = trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh, scale_to_box=False)
        
        for camera_position in rotated_camera_positions[0:6]:
            # Step 4: Generate rays for each camera position
            cone_directions = sample_cone(-camera_position, angle=np.radians(60), count=camera_ray_num)
            
            # Step 5: Compute intersections, face index, face normals for each camera
            locations, index_ray, index_tri = rmi.intersects_location(np.tile(camera_position, (camera_ray_num, 1)), cone_directions, multiple_hits=False)
            
            all_locations.append(locations)
            all_index_tri.append(index_tri)

        self.camera_positions = rotated_camera_positions.copy()
        self.camera_all_surface_points = all_locations.copy()
        self.mesh_face_normals = self.mesh.face_normals.copy()
        self.camera_all_face_normals_id = all_index_tri.copy()

        # Step 6: Merge intersection points from all cameras
        all_locations = np.concatenate(all_locations, axis=0)
        all_index_tri = np.concatenate(all_index_tri, axis=0)

        # insure we have enough points 
        if all_locations.shape[0] < self.sample_point_num:
            sphere_points = trimesh.sample.sample_surface_sphere(10 * self.sample_point_num)
            ray_directions = - sphere_points

            rmi = trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh, scale_to_box=False)
            locations, index_ray, index_tri =  rmi.intersects_location(sphere_points, ray_directions, multiple_hits=False)

            all_locations = np.concatenate([all_locations, locations], axis=0)
            all_index_tri = np.concatenate([all_index_tri, index_tri], axis=0)

        # Step 7: Farthest points sampling
        points_torch = torch.tensor(all_locations).unsqueeze(0)

        locations_choose_torch, intersect_choose_id_torch = pytorch3d.ops.sample_farthest_points(points_torch, K=self.sample_point_num)
        
        intersect_choose_id = intersect_choose_id_torch.squeeze(0).detach().cpu().numpy()
        
        locations = all_locations[intersect_choose_id]
        index_tri = all_index_tri[intersect_choose_id]
        
        # Step 8: Store results
        self.point_idx_to_surface_points = locations
        self.point_idx_to_face_idx = index_tri
        self.point_idx_to_face_normals = self.mesh.face_normals[self.point_idx_to_face_idx]
        self.point_idx_to_tip_center_reach_points = self.point_idx_to_surface_points + self.point_idx_to_face_normals * self.tip_retreat_dis
        self.pc = PointCloud(self.point_idx_to_surface_points, -self.point_idx_to_face_normals)


    def cluster(self, local_weight=1, n_clusters=32):
        '''
        Cluster the sampled point cloud based on the points' positions and normals.

        The method clusters the sampled point cloud using KMeans clustering. The clustering is based on 
        the points' positions, their normals, and a cross product of the position and normal.

        Parameters:
        - local_weight (float, optional): Weight factor for the positions in the clustering. Default is 1.
        - n_clusters (int, optional): Number of clusters to form using KMeans clustering. Default is 16.
        '''

        t = np.empty_like(self.point_idx_to_surface_points) 
        for i in range(self.point_idx_to_surface_points.shape[0]):
            t[i] = normalize(np.cross(self.point_idx_to_surface_points[i], self.point_idx_to_face_normals[i]))
        score = np.hstack([50 * self.point_idx_to_surface_points, self.point_idx_to_face_normals, 10 * t]) 
        # score = np.hstack([self.point_idx_to_face_normals, t]) 
        kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=0)
        self.num_regions = n_clusters
        self.point_idx_to_region_idx = kmeans.fit_predict(score)


    def visualize_segmented_pointcloud(self):
        '''
        Visualize a segmented point cloud with each segment having a distinct color.

        This function creates a visualization for the point cloud where each region 
        or segment is colored differently. It generates random colors for each segment 
        and maps them to the points belonging to that segment. The visualization is 
        created using trimesh.

        Returns:
        - trimesh.points.PointCloud: A colored trimesh point cloud object.
        '''

        # Generate random colors for regions
        # region_to_color = np.random.randint(256, size=(self.num_regions, 3))
        region_to_color = np.array(np.array(sns.color_palette("husl", self.num_regions)) * 255, dtype=np.uint8)

        # Map every point index to its corresponding region color
        colors_for_points = region_to_color[self.point_idx_to_region_idx]

        # Add alpha channel (255) to the colors
        vertex_color = np.hstack((colors_for_points, 255 * np.ones((self.sample_point_num, 1), dtype=np.uint8)))
        
        # Create point cloud using trimesh
        pc_trimesh = trimesh.points.PointCloud(self.point_idx_to_surface_points)
        pc_trimesh.colors = vertex_color

        return pc_trimesh


    def create_urdf(self):
        '''
        Generate a URDF file based on the mesh associated with this object.

        This method creates a URDF representation for the robot model using the mesh 
        specified in the class. The generated URDF file will be saved in the same directory 
        as the mesh with the same name but with a ".urdf" extension.

        Returns:
        - str: The path to the generated URDF file.
        '''
        #! make the urdf_path different every time, otherwise bullet may load the same cache object
        # self.urdf_path = self.mesh_path[:-4] + str(self.scale[0]).split(".")[-1] + ".urdf"
        self.urdf_path = "./mesh.obj"
        create_urdf_from_obj(self.mesh_path, self.urdf_path, self.scale)
        return self.urdf_path
    

    def filter_points_by_dis(self, sphere_center, sphere_radius, sample_point_num):
        '''
        Filter points from the point cloud that are within a specified radius 
        of a given sphere's center.

        This method retains points from the point cloud if they are within a 
        defined radius from the sphere's center. If the number of retained points 
        exceeds the `sample_point_num`, a random subset of those points is returned.

        Parameters:
        - sphere_center (np.ndarray, shape=(3,)): Center of the sphere used as a reference.
        - sphere_radius (float): Radius of the sphere. Points within this distance from 
          the sphere's center will be retained.
        - sample_point_num (int): Maximum number of points to retain after filtering. 
          If more points are found, a random subset of this size is returned.

        Returns:
        - np.ndarray: Indices of the filtered points from the point cloud.
        '''

        filterd_idx_full = np.where(np.linalg.norm(self.point_idx_to_tip_center_reach_points - sphere_center, axis=1) < sphere_radius)[0]
        save_idx_num = min(filterd_idx_full.shape[0], sample_point_num)
        filterd_idx = np.random.choice(filterd_idx_full, save_idx_num)

        return filterd_idx