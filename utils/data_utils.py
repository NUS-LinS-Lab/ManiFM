import numpy as np
import torch
import scipy.sparse as sp
import cvxpy as cp
import trimesh
from utils.opt_utils import normalize, hat, compute_grasp_maps


def normal_add_noise_torch(normals, sigma=torch.pi/6, return_noise=False):
    n = normals.size(0)
    random_directions = torch.randn_like(normals)
    rotation_axes = torch.cross(normals, random_directions)
    rotation_axes = torch.nn.functional.normalize(rotation_axes, p=2, dim=1)

    angles = torch.randn(n) * sigma 

    cos_angles = torch.cos(angles).unsqueeze(-1)
    sin_angles = torch.sin(angles).unsqueeze(-1)
    one_minus_cos_angles = 1 - cos_angles
    
    zeros = torch.zeros(n, 1)
    rx = rotation_axes[:, 0].unsqueeze(1)
    ry = rotation_axes[:, 1].unsqueeze(1)
    rz = rotation_axes[:, 2].unsqueeze(1)

    cross_product_matrices = torch.cat([
        torch.cat([zeros, -rz, ry], dim=1).unsqueeze(1),
        torch.cat([rz, zeros, -rx], dim=1).unsqueeze(1),
        torch.cat([-ry, rx, zeros], dim=1).unsqueeze(1)
    ], dim=1).view(n, 3, 3)
    
    rotated_normals = cos_angles * normals + sin_angles * torch.bmm(cross_product_matrices, normals.unsqueeze(-1)).squeeze(-1) + one_minus_cos_angles * torch.bmm(rotation_axes.unsqueeze(-1), torch.bmm(rotation_axes.unsqueeze(1), normals.unsqueeze(-1))).squeeze(-1)
    return torch.nn.functional.normalize(rotated_normals, p=2, dim=1)

sphere_points = torch.from_numpy(trimesh.sample.sample_surface_sphere(360) * 5).to(dtype=torch.float)

def replace_noisy_points(noisy_points_torch, contact_points, remove_points_num=200):
    distances = torch.cdist(sphere_points, contact_points).mean(dim=1)
    furthest_point_idx = distances.argmax()
    furthest_point = sphere_points[furthest_point_idx]
    
    distances_to_extended = torch.norm(noisy_points_torch - furthest_point, dim=1)

    _, nearest_idxs = torch.topk(distances_to_extended, remove_points_num, largest=False)

    all_idxs = torch.arange(noisy_points_torch.size(0))
    mask = torch.ones_like(all_idxs, dtype=torch.bool)
    mask[nearest_idxs] = False
    remaining_idxs = all_idxs[mask]
    random_idxs_to_pad = remaining_idxs[torch.randperm(remaining_idxs.size(0))[:remove_points_num]]

    noisy_points_torch[nearest_idxs] = noisy_points_torch[random_idxs_to_pad]

    return noisy_points_torch, nearest_idxs, random_idxs_to_pad


def pc_add_noise_torch(points, sigma=0.01, return_noise=False):
    '''
    Args:
        points: shape=(n, 3)
        std: float 
    '''
    noise = torch.normal(torch.zeros(points.shape), sigma)
    points += noise
    if return_noise:
        return points, noise
    else: 
        return points 


def normal_add_noise_torch(normals, sigma=torch.pi/6, return_noise=False):
    n = normals.size(0)
    random_directions = torch.randn_like(normals)
    rotation_axes = torch.cross(normals, random_directions)
    rotation_axes = torch.nn.functional.normalize(rotation_axes, p=2, dim=1)

    angles = torch.randn(n) * sigma 

    cos_angles = torch.cos(angles).unsqueeze(-1)
    sin_angles = torch.sin(angles).unsqueeze(-1)
    one_minus_cos_angles = 1 - cos_angles
    
    zeros = torch.zeros(n, 1)
    rx = rotation_axes[:, 0].unsqueeze(1)
    ry = rotation_axes[:, 1].unsqueeze(1)
    rz = rotation_axes[:, 2].unsqueeze(1)

    cross_product_matrices = torch.cat([
        torch.cat([zeros, -rz, ry], dim=1).unsqueeze(1),
        torch.cat([rz, zeros, -rx], dim=1).unsqueeze(1),
        torch.cat([-ry, rx, zeros], dim=1).unsqueeze(1)
    ], dim=1).view(n, 3, 3)
    
    rotated_normals = cos_angles * normals + sin_angles * torch.bmm(cross_product_matrices, normals.unsqueeze(-1)).squeeze(-1) + one_minus_cos_angles * torch.bmm(rotation_axes.unsqueeze(-1), torch.bmm(rotation_axes.unsqueeze(1), normals.unsqueeze(-1))).squeeze(-1)
    return torch.nn.functional.normalize(rotated_normals, p=2, dim=1)


def generate_contact_frame(pos, normal): 
    """Generate contact frame, whose z-axis aligns with the normal direction (inward to the object)
    """
    up = normalize(np.random.rand(3))
    z = normalize(normal)
    x = normalize(np.cross(up, z))
    y = normalize(np.cross(z, x))

    result = np.eye(4)
    result[0:3, 0] = x
    result[0:3, 1] = y
    result[0:3, 2] = z
    result[0:3, 3] = pos
    return result


def adj_T(frame):
    """Compute the adjoint matrix for the contact frame
    """
    assert frame.shape[0] == frame.shape[1] == 4, 'Frame needs to be 4x4'

    R = frame[0:3, 0:3]
    p = frame[0:3, 3]
    result = np.zeros((6, 6))
    result[0:3, 0:3] = R
    result[3:6, 0:3] = hat(p) @ R
    result[3:6, 3:6] = R
    return result


def compute_grasp_map(contact_pos, contact_normal, soft_contact=False):
    """ Computes the grasp map for all contact points.
    Check chapter 5 of http://www.cse.lehigh.edu/~trink/Courses/RoboticsII/reading/murray-li-sastry-94-complete.pdf for details.
    Args:
        contact_pos: location of contact in the object frame
        contact_normal: surface normals at the contact location, point inward !!!, N x 3, in the object frame
        soft_contact: whether use soft contact model. Defaults to False.
    Returns:
        G: grasp map for the contacts
    """
    n_point = contact_pos.shape[0]

    # Compute the contact basis B
    if soft_contact:
        B = np.zeros((6, 4))
        B[0:3, 0:3] = np.eye(3)
        B[5, 3] = 1
    else: # use point contact w/ friction
        B = np.zeros((6, 3))
        B[0:3, 0:3] = np.eye(3)

    # Compute the contact frames, adjoint matrix, and grasp map
    contact_frames = []
    grasp_maps = []
    for pos, normal in zip(contact_pos, contact_normal):
        contact_frame = generate_contact_frame(pos, normal)
        contact_frames.append(contact_frame)

        adj_matrix = adj_T(contact_frame)
        grasp_map = adj_matrix @ B
        grasp_maps.append(grasp_map)

    G = np.hstack(grasp_maps)
    assert G.shape == (6, n_point * B.shape[1]), 'Grasp map shape does not match'
    
    return G


def from_wrench_to_contact_force(point_cloud, point_normal, origin_contact_point_id, target_wrench, mu=0.5):
    contact_pos = []
    contact_normal = []
    for contact_point in origin_contact_point_id:
        if contact_point == -1:
            continue
        contact_pos.append(point_cloud[contact_point][:3])
        contact_normal.append(point_normal[contact_point][:3])
    
    if len(contact_pos)==0:
        info = "No contact points."
    else:
        contact_pos = np.stack(contact_pos)
        contact_normal = np.stack(contact_normal)
        w_ext = target_wrench
        num_contact = len(contact_pos)

        f = cp.Variable(3 * num_contact)
        G = compute_grasp_map(contact_pos, -contact_normal)

        linear_f = 8
        min_f = 0.01
        max_f = 1.0
        tol_f = 0.0
        friction_cone = np.empty((linear_f + 2, 4))
        for i in range(linear_f):
            nx = - mu * np.cos(i * 2 * np.pi / linear_f)
            ny = - mu * np.sin(i * 2 * np.pi / linear_f)
            friction_cone[i] = np.array((nx, ny, 1, -tol_f))
        friction_cone[-2] = np.array((0, 0, 1, - min_f - tol_f))
        friction_cone[-1] = np.array((0, 0, -1, max_f - tol_f))
        
        constraints = []
        for i in range(num_contact):
            constraints += [
                    cp.NonPos(- friction_cone[:,:3] @ f[3*i: 3*i+3]
                    - friction_cone[:,3])
                ]
        res = G @ f - w_ext
        objective = cp.Minimize(cp.norm(res))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        f_array = f.value.reshape(-1, 3)
        f_global_array = np.zeros((num_contact, 3))
        G_array = G.reshape(6, len(contact_pos), 3).transpose(1, 0, 2)
        for i in range(num_contact):
            f_global_array[i] = (G_array[i] @ f_array[i])[:3]
        return prob.value, f_global_array


def generate_random_wrenches(contact_points, contact_normals, friction_coefficient=1, min_force=0.1, max_force=1, sample_num=10, return_force=False):
    n_points = contact_points.shape[0]

    # Precompute grasp maps for all contact points
    grasp_maps = compute_grasp_maps(contact_points, contact_normals)  # (n_points, 6, 3)

    # Sample normal forces within the specified range
    normal_forces = np.random.uniform(min_force, max_force, (sample_num, n_points))

    # Sample tangential force components within the friction cone
    angles = np.random.uniform(0, 2 * np.pi, (sample_num, n_points))
    magnitudes = normal_forces * friction_coefficient * np.sqrt(np.random.uniform(0, 1, (sample_num, n_points)))
    tangential_forces = np.stack((magnitudes * np.cos(angles), magnitudes * np.sin(angles)), axis=-1)

    # Combine forces to form wrenches in the local frame
    forces = np.concatenate((tangential_forces, normal_forces[..., np.newaxis]), axis=-1)  # (sample_num, n_points, 3)

    # Map local frame forces to object frame wrenches
    object_frame_wrenches = np.einsum('nab,knb->kna', grasp_maps, forces)  # shape: (sample_num, n_points, 6)

    # Sum the wrenches for each sample
    sample_wrenches = np.sum(object_frame_wrenches, axis=1)

    if return_force:
        return sample_wrenches, object_frame_wrenches[:, :, :3]
    else:
        return sample_wrenches


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': total_params, 'trainable_params': trainable_params}


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def edge_to_adj(edge, node_num):
    adj = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 0], edge[:, 1])),
                        shape=(node_num, node_num),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = row_normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj