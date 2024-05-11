
import os 
import numpy as np 
import coacd
import trimesh
import xml.etree.ElementTree as ET
import torch 
import scipy
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from pytorch3d.ops import knn_points
import re
import potpourri3d as pp3d
from multiprocessing import Manager
# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing.dummy import Pool


######################################################################################
###################################### MESH ##########################################
######################################################################################



def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def as_mesh_with_colors(scene_or_mesh):
    """
    Convert a possible scene to a mesh while preserving color information.

    If conversion occurs, the returned mesh has vertex, face, and color data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            all_vertices = []
            all_faces = []
            all_vertex_colors = []
            face_offset = 0

            for g in scene_or_mesh.geometry.values():

                all_vertices.append(g.vertices)
                all_faces.append(g.faces + face_offset)  # Offset faces
                face_offset += len(g.vertices)

                # If the geometry has vertex colors, use them
                if hasattr(g.visual, 'vertex_attributes') and 'color' in g.visual.vertex_attributes:
                    all_vertex_colors.append(g.visual.vertex_attributes['color'])
                # If not, use the baseColorFactor from the material as the default color
                elif 'baseColorTexture' in g.visual.material._data:
                    mesh = scene_or_mesh.dump(concatenate=True)
                    return mesh
                    # uv = g.visual.vertex_attributes['uv']
                    # image = g.visual.material._data['baseColorTexture']
                    # vertex_colors = []
                    # for coord in uv:
                    #     x, y = coord
                    #     x = int(x * (image.width - 1))
                    #     y = int((1 - y) * (image.height - 1))  # Reverse Y because UV and image coordinates differ
                    #     color = image.getpixel((x, y))
                    #     vertex_colors.append(color)
                    # all_vertex_colors.append(vertex_colors)
                elif 'baseColorFactor' in g.visual.material._data:
                    color = g.visual.material._data['baseColorFactor']
                    all_vertex_colors.append(np.tile(color, (len(g.vertices), 1)))
                else:
                    # If no colors defined at all, use a default (e.g., white)
                    all_vertex_colors.append(np.full((len(g.vertices), 4), [255, 255, 255, 255], dtype=np.uint8))

            # Concatenate all the vertices, faces, and vertex colors
            all_vertices = np.vstack(all_vertices)
            all_faces = np.vstack(all_faces)
            all_vertex_colors = np.vstack(all_vertex_colors)

            mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
            mesh.visual.vertex_colors = all_vertex_colors

    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh

    return mesh


def get_transform_matrix(scale=[1.0, 1.0, 1.0], translation=[0, 0, 0], orientation=[0, 0, 0], seq="ZYX", degrees=True):
    # scale matrix
    scale_matrix = np.eye(4)
    scale_matrix[0, 0] = scale[0]
    scale_matrix[1, 1] = scale[1]
    scale_matrix[2, 2] = scale[2]

    # translation_matrix
    translation_matrix = np.eye(4)
    translation_matrix[0, 3] = translation[0]
    translation_matrix[1, 3] = translation[1]
    translation_matrix[2, 3] = translation[2]
    
    # rotation_matrix  ZYX Euler (intrisic)
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = R.from_euler(seq, orientation,degrees=degrees).as_matrix()

    # combine
    transform_matrix = translation_matrix @ rotation_matrix 
    transform_matrix_with_scale = transform_matrix @ scale_matrix
    return transform_matrix_with_scale


def create_urdf_from_obj(obj_filename, urdf_filename, scale=[1, 1, 1]):
    '''
    Generate a URDF file from a given OBJ file.

    This function creates a simple URDF representation with a single link.
    The visual and collision geometries of the link are based on the provided OBJ file. 
    The link has a default inertial property, which can be modified later if needed.

    Parameters:
    - obj_filename (str): The path to the input OBJ file which defines the shape of the robot link.
    - urdf_filename (str): The path where the generated URDF file will be saved.
    - scale (list of floats, optional): Scaling factors in the x, y, z directions for the OBJ mesh. Default is [1, 1, 1].

    Returns:
    None. The URDF file is saved to the specified path.
    '''
    with open(urdf_filename, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<robot name="my_robot">\n')
        f.write('  <link name="base_link">\n')
        
        # Inertial part
        f.write('    <inertial>\n')
        f.write('      <mass value="1.0" />\n')
        f.write('      <origin xyz="0 0 0" rpy="0 0 0" />\n')
        f.write('      <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0" />\n')
        f.write('    </inertial>\n')
        
        # Visual part with scale
        f.write('    <visual>\n')
        f.write('      <geometry>\n')
        f.write(f'        <mesh filename="{obj_filename}" scale="{scale[0]} {scale[1]} {scale[2]}"/>\n')
        f.write('      </geometry>\n')
        f.write('    </visual>\n')
        
        # Collision part with scale
        f.write('    <collision>\n')
        f.write('      <geometry>\n')
        f.write(f'        <mesh filename="{obj_filename}" scale="{scale[0]} {scale[1]} {scale[2]}"/>\n')
        f.write('      </geometry>\n')
        f.write('    </collision>\n')
        
        f.write('  </link>\n')
        f.write('</robot>\n')



def urdf_add_scale_to_mesh(input_urdf_path, output_urdf_path, scale=[1.0, 1.0, 1.0]):
    """
    Modify the scale of all meshes in the visual and collision tags of a URDF file.

    Parameters:
    - input_urdf_path (str): Path to the input URDF file.
    - output_urdf_path (str): Path where the modified URDF file will be saved.
    - scale (list of float): A list of three float values representing the scale factors for the x, y, and z axes.

    Returns:
    None

    Example:
    urdf_add_scale_to_mesh("input.urdf", "output.urdf", [1.5, 1.5, 1.5])
    """

    # Parse the URDF file
    tree = ET.parse(input_urdf_path)
    root = tree.getroot()

    # Iterate through all visual and collision tags
    for elem in root.findall(".//visual/geometry/mesh") + root.findall(".//collision/geometry/mesh"):
        # Set the scale attribute
        elem.set("scale", " ".join(map(str, scale)))

    # Save the modified URDF to the output path
    tree.write(output_urdf_path)

def ConvexDecompositionIntoOBJ(input_obj_path, output_dir_path):
    '''
    # Convex Decomposition with CoACD
    ---
    Args:
        - input_obj_path: str 
        - output_dir_path: str
    '''

    obj_base_name = os.path.basename(input_obj_path)

    # use trimesh to load the mesh
    mesh = trimesh.load(input_obj_path, force="mesh", process=False)
    imesh = coacd.Mesh()
    imesh.vertices = mesh.vertices
    imesh.indices = mesh.faces

    # run coacd 
    parts = coacd.run_coacd(imesh)
    decomposed_all_parts_trimesh = trimesh.Scene()
    for p in parts:
        part_trimesh = trimesh.Trimesh(np.array(p[0]), np.array(p[1]).reshape((-1, 3)))
        part_trimesh.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        decomposed_all_parts_trimesh.add_geometry(part_trimesh)
    
    output_object_obj_path = os.path.join(output_dir_path, obj_base_name[:-4]+".obj")
    decomposed_all_parts_trimesh.export(output_object_obj_path)
    

def ConvexDecompositionIntoURDF(input_obj_path, output_dir_path):
    '''
    # Convex Decomposition with CoACD
    ---
    Args:
        - input_obj_path: str 
        - output_dir_path: str
    ---
    Returns:
        - output_object_obj_path: str 
        - output_object_urdf_path: str 
    ---
    Directory tree:
        - output_dir_path
            - original .obj name without extension as a directory 
                - meshes 
                    - decomposed_0.obj
                    - decomposed_1.obj
                    - decomposed_2.obj
                - object.urdf 
                - decomposed_all.obj
    ---
    Example:
    input_obj_path = "/home/xzhixuan/Projects/unicontact/UniContact_04/assets/geometry_files/cup.obj"
    output_dir_path = "/home/xzhixuan/Projects/unicontact/UniContact_04/assets/geometry_files"
    output_object_obj_path, output_object_urdf_path = ConvexDecompositionIntoURDF(input_obj_path, output_dir_path)
    '''

    # use trimesh to load the mesh
    mesh = trimesh.load(input_obj_path)
    imesh = coacd.Mesh()
    imesh.vertices = mesh.vertices
    imesh.indices = mesh.faces

    # run coacd 
    parts = coacd.run_coacd(imesh)
    mesh_parts = []
    decomposed_all_parts_trimesh = trimesh.Scene()
    for p in parts:
        part_trimesh = trimesh.Trimesh(np.array(p[0]), np.array(p[1]).reshape((-1, 3)))
        part_trimesh.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        mesh_parts.append(part_trimesh)
        decomposed_all_parts_trimesh.add_geometry(part_trimesh)
    
    # the object name without extension 
    obj_base_name = os.path.basename(input_obj_path)
    if '.' in obj_base_name and not obj_base_name.startswith('.'):
        obj_base_name = obj_base_name.rpartition('.')[0]

    # create the output dir 
    os.makedirs(os.path.join(output_dir_path, obj_base_name, "meshes"), exist_ok=True)

    # write the .obj
    output_object_obj_path = os.path.join(output_dir_path, obj_base_name, f"decomposed_all.obj")
    decomposed_all_parts_trimesh.export(output_object_obj_path)
    for obj_idx, m in enumerate(mesh_parts):
        export_path = os.path.join(output_dir_path, obj_base_name, "meshes", f"decomposed_{obj_idx}.obj")
        m.export(export_path)
    
    # write the urdf file 
    urdf_s = f"""<?xml version="1.0"?>
<robot name="{ obj_base_name }_skeleton">
    <link name="world" />
    <link name="{ obj_base_name }_link0">
        <inertial>
            <mass value="1.0" />
            <origin xyz="0 0 0" rpy="0 0 0" />
            <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0" />
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0" />
            <geometry>
                <mesh filename="meshes/decomposed_0.obj"/>
            </geometry>
        </visual>
        <collision group="default">
            <origin xyz="0 0 0" rpy="0 0 0" />
            <geometry>
                <mesh filename="meshes/decomposed_0.obj"/>
            </geometry>
        </collision>
    </link>
    <joint name="{ obj_base_name }_joint0" type="fixed">
        <origin xyz="0 0 0" rpy="0 0 0" />
        <parent link="world" />
        <child link="{ obj_base_name }_link0" />
    </joint>
"""
    for i in range(1, len(mesh_parts)):
        urdf_s += f"""    <link name="{ obj_base_name }_link{ i }">
        <inertial>
            <mass value="100.0" />
            <origin xyz="0 0 0" rpy="0 0 0" />
            <inertia ixx="10.0" ixy="0" ixz="0" iyy="10.0" iyz="0" izz="10.0" />
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0" />
            <geometry>
                <mesh filename="meshes/decomposed_{ i }.obj"/>
            </geometry>
        </visual>
        <collision group="default">
            <origin xyz="0 0 0" rpy="0 0 0" />
            <geometry>
                <mesh filename="meshes/decomposed_{ i }.obj"/>
            </geometry>
        </collision>
    </link>
    <joint name="{ obj_base_name }_joint{ i }" type="fixed">
        <origin xyz="0 0 0" rpy="0 0 0" />
        <parent link="{ obj_base_name }_link{ i - 1}" />
        <child link="{ obj_base_name }_link{ i }" />
    </joint>
"""
    urdf_s = urdf_s + "</robot>"
    output_object_urdf_path = os.path.join(output_dir_path, obj_base_name, "object.urdf")
    with open(output_object_urdf_path, "w") as _file:
        _file.write(urdf_s)

    return output_object_obj_path, output_object_urdf_path


def ConvexDecompositionIntoURDFwithScale(input_obj_path, output_dir_path, scale=[1.0, 1.0, 1.0]):
    '''
    # Convex Decomposition with CoACD
    ---
    Args:
        - input_obj_path: str 
        - output_dir_path: str
        - scale: list of float, default [1.0, 1.0, 1.0]
    ---
    Returns:
        - output_object_obj_path: str 
        - output_object_urdf_path: str 
    ---
    Directory tree:
        - output_dir_path
            - original .obj name without extension as a directory 
                - meshes 
                    - decomposed_0.obj
                    - decomposed_1.obj
                    - decomposed_2.obj
                - object.urdf 
                - decomposed_all.obj
    ---
    Example:
    input_obj_path = "/home/xzhixuan/Projects/unicontact/UniContact_04/assets/geometry_files/cup.obj"
    output_dir_path = "/home/xzhixuan/Projects/unicontact/UniContact_04/assets/geometry_files"
    output_object_obj_path, output_object_urdf_path = ConvexDecompositionIntoURDF(input_obj_path, output_dir_path)
    '''

    # use trimesh to load the mesh
    mesh = trimesh.load(input_obj_path)
    imesh = coacd.Mesh()
    imesh.vertices = mesh.vertices
    imesh.indices = mesh.faces

    # run coacd 
    parts = coacd.run_coacd(imesh)
    mesh_parts = []
    decomposed_all_parts_trimesh = trimesh.Scene()
    for p in parts:
        part_trimesh = trimesh.Trimesh(np.array(p[0]), np.array(p[1]).reshape((-1, 3)))
        part_trimesh.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        mesh_parts.append(part_trimesh)
        decomposed_all_parts_trimesh.add_geometry(part_trimesh)
    
    # the object name without extension 
    obj_base_name = os.path.basename(input_obj_path)
    if '.' in obj_base_name and not obj_base_name.startswith('.'):
        obj_base_name = obj_base_name.rpartition('.')[0]

    # create the output dir 
    os.makedirs(os.path.join(output_dir_path, obj_base_name, "meshes"), exist_ok=True)

    # write the .obj
    output_object_obj_path = os.path.join(output_dir_path, obj_base_name, f"decomposed_all.obj")
    decomposed_all_parts_trimesh.export(output_object_obj_path)
    for obj_idx, m in enumerate(mesh_parts):
        export_path = os.path.join(output_dir_path, obj_base_name, "meshes", f"decomposed_{obj_idx}.obj")
        m.export(export_path)
    
    scale_str = f"{scale[0]} {scale[1]} {scale[2]}"
    # write the urdf file 
    urdf_s = f"""<?xml version="1.0"?>
<robot name="{ obj_base_name }_skeleton">
    <link name="world" />
    <link name="{ obj_base_name }_link0">
        <inertial>
            <mass value="1.0" />
            <origin xyz="0 0 0" rpy="0 0 0" />
            <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0" />
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0" />
            <geometry>
                <mesh filename="meshes/decomposed_0.obj" scale="{scale_str}"/>
            </geometry>
        </visual>
        <collision group="default">
            <origin xyz="0 0 0" rpy="0 0 0" />
            <geometry>
                <mesh filename="meshes/decomposed_0.obj" scale="{scale_str}"/>
            </geometry>
        </collision>
    </link>
    <joint name="{ obj_base_name }_joint0" type="fixed">
        <origin xyz="0 0 0" rpy="0 0 0" />
        <parent link="world" />
        <child link="{ obj_base_name }_link0" />
    </joint>
"""
    for i in range(1, len(mesh_parts)):
        urdf_s += f"""    <link name="{ obj_base_name }_link{ i }">
        <inertial>
            <mass value="100.0" />
            <origin xyz="0 0 0" rpy="0 0 0" />
            <inertia ixx="10.0" ixy="0" ixz="0" iyy="10.0" iyz="0" izz="10.0" />
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0" />
            <geometry>
                <mesh filename="meshes/decomposed_{ i }.obj" scale="{scale_str}"/>
            </geometry>
        </visual>
        <collision group="default">
            <origin xyz="0 0 0" rpy="0 0 0" />
            <geometry>
                <mesh filename="meshes/decomposed_{ i }.obj" scale="{scale_str}"/>
            </geometry>
        </collision>
    </link>
    <joint name="{ obj_base_name }_joint{ i }" type="fixed">
        <origin xyz="0 0 0" rpy="0 0 0" />
        <parent link="{ obj_base_name }_link{ i - 1}" />
        <child link="{ obj_base_name }_link{ i }" />
    </joint>
"""
    urdf_s = urdf_s + "</robot>"
    output_object_urdf_path = os.path.join(output_dir_path, obj_base_name, "object.urdf")
    with open(output_object_urdf_path, "w") as _file:
        _file.write(urdf_s)

    return output_object_obj_path, output_object_urdf_path

######################################################################################
#################################### VISUALIZE #######################################
######################################################################################


def vis_world_axis():
    '''
    Generate a visual representation of the world coordinate axis using trimesh.

    Returns:
    - world_axis_mesh (trimesh.Trimesh): A trimesh object representing the world coordinate axis.
    '''
    return trimesh.creation.axis(origin_size=0.005, axis_radius=0.002, axis_length=1.0)


def vis_point(p, radius=0.005, color=[0, 0, 255, 255]):
    '''
    Generate a visual representation of a point in the form of a trimesh object.

    Parameters:
    - p (np.ndarray): The 3D coordinates of the point.
    - radius (float, optional): The radius of the sphere that represents the point in the visualization. Default is 0.005.

    Returns:
    - p_mesh (trimesh.Trimesh): A trimesh object representing the point.
    '''
    contact_sphere = trimesh.creation.capsule(height=0.0, radius=radius, transform=get_transform_matrix(translation=p))
    contact_sphere.visual.face_colors = color
    return contact_sphere


def vis_vector(start_point, vector, length=0.4, cyliner_r=0.0015, color=[255, 255, 100, 245]):
    '''
    start_points: np.ndarray, shape=(3,)
    vectors: np.ndarray, shape=(3,)
    length: cylinder length 
    '''
    normalized_vector = normalize(vector)
    end_point = start_point + length * normalized_vector

    # create a mesh for the force
    force_cylinder = trimesh.creation.cylinder(radius=cyliner_r, 
                                               segment=np.array([start_point, end_point]))
    
    # create a mesh for the arrowhead
    cone_transform = sample_transform_w_normals(end_point, normalized_vector, 0, ori_face_vector=np.array([0.0, 0.0, 1.0]))
    arrowhead_cone = trimesh.creation.cone(radius=2*cyliner_r, 
                                           height=4*cyliner_r, 
                                           transform=cone_transform)
    # combine the two meshes into one
    force_mesh = force_cylinder + arrowhead_cone 
    force_mesh.visual.face_colors = color

    return force_mesh

def vis_cone(cone_tip_point, normalized_vector, cone_height=0.3, cone_radius=0.15):
    cone_bottom_point = cone_tip_point - cone_height * normalized_vector
    cone_transform = sample_transform_w_normals(cone_bottom_point, normalized_vector, 0, ori_face_vector=np.array([0.0, 0.0, 1.0]))
    cone = trimesh.creation.cone(radius=cone_radius, 
                                           height=cone_height, 
                                           transform=cone_transform)
    return cone 

def vis_in_jupyter(scene, light=2, height=600):
    from IPython import display
    from trimesh import viewer
    as_html = viewer.notebook.scene_to_html(scene).replace('scene.background=new THREE.Color(0xffffff);', f'scene.background=new THREE.Color(0xffffff);addlight=new THREE.DirectionalLight(0x404040,{light}); addlight.position.set(0, 1, 0); scene.add(addlight);addlight2=new THREE.DirectionalLight(0x404040,{light}); addlight2.position.set(0, -1, 0); scene.add(addlight2); addlight3=new THREE.DirectionalLight(0x404040,{light}); addlight3.position.set(1, 0, 0);scene.add(addlight3);addlight4=new THREE.DirectionalLight(0x404040,{light}); addlight4.position.set(-1, 0, 0);scene.add(addlight4);addlight5=new THREE.DirectionalLight(0x404040,{light}); addlight5.position.set(0, 0, 1);scene.add(addlight5);addlight6=new THREE.DirectionalLight(0x404040,{light}); addlight6.position.set(0, 0, -1);scene.add(addlight6);')
    srcdoc = as_html.replace('"', "&quot;")
    embedded = display.HTML(
        " ".join(
            [
                '<div><iframe srcdoc="{srcdoc}"',
                'width="100%" height="{height}px"',
                'style="border:none;"></iframe></div>',
            ]
        ).format(srcdoc=srcdoc, height=height)
    )
    return embedded
  
def extract_colors_from_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    global_materials = {}
    for material in root.findall('material'):
        name = material.attrib['name']
        color_elem = material.find('color')
        if color_elem is not None and 'rgba' in color_elem.attrib:
            rgba = [float(c) for c in color_elem.attrib['rgba'].split()]
            global_materials[name] = rgba

    link_colors = {}

    for link in root.iter('link'):
        link_name = link.attrib['name']
        visual = link.find('./visual')
        if visual is not None:
            material = visual.find('./material')
            if material is not None:
                # 如果材料直接包含颜色定义
                color = material.find('color')
                if color is not None and 'rgba' in color.attrib:
                    rgba = [float(c) for c in color.attrib['rgba'].split()]
                    link_colors[link_name] = rgba
                # 否则，查找全局材料定义
                elif 'name' in material.attrib:
                    material_name = material.attrib['name']
                    if material_name in global_materials:
                        link_colors[link_name] = global_materials[material_name]

    return link_colors


def extrude_mesh(surface_mesh, extrude_distance=0.0001):
    """
    Extrude a surface mesh to create a volumetric mesh.

    Args:
    surface_mesh (trimesh.Trimesh): The original surface mesh.
    extrude_distance (float): The distance to extrude the mesh, creating thickness.

    Returns:
    trimesh.Trimesh: A volumetric mesh representing the extruded surface.
    """

    # Duplicate and offset the original mesh to create the extruded layer
    extruded_mesh = surface_mesh.copy()
    extruded_mesh.vertices[:, 2] += extrude_distance  # Assuming Z is the extrusion direction
    extruded_mesh.faces = np.fliplr(extruded_mesh.faces)  # Reverse vertex order for correct normals

    # Combine the original and extruded meshes
    combined_vertices = np.vstack([surface_mesh.vertices, extruded_mesh.vertices])
    combined_faces = np.vstack([surface_mesh.faces, extruded_mesh.faces + len(surface_mesh.vertices)])

    # Indices of the original and extruded vertices
    orig_v = surface_mesh.faces
    extr_v = orig_v + len(surface_mesh.vertices)

    # Create side faces using array operations
    # Each face is represented by [v1, v2, v2_ext], [v2_ext, v1_ext, v1] etc.
    side_faces = np.hstack([
        orig_v[:, np.newaxis, :], 
        np.roll(orig_v, -1, axis=1)[:, np.newaxis, :], 
        np.roll(extr_v, -1, axis=1)[:, np.newaxis, :]
    ]).reshape(-1, 3)

    side_faces = np.vstack([side_faces, np.roll(side_faces, 1, axis=1)])

    # Combine the top, bottom, and side faces
    combined_faces = np.vstack([combined_faces, side_faces])

    # Create the final volumetric mesh
    volumetric_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)

    return volumetric_mesh

def sample_parabolic_surface(a, b, c, x_limits, y_limits, grid_num):
    """
    Sample a parabolic surface defined by z = a * x^2 + b * x * y + c * y^2.

    Args:
    a, b, c (float): Coefficients of the surface equation.
    x_limits (tuple): A tuple of (min, max) for x.
    y_limits (tuple): A tuple of (min, max) for y.
    grid_num (int): Number of points to sample along each axis.

    Returns:
    np.ndarray: An array of shape (n, 3) containing sampled points.
    """

    # Create a grid of x and y values
    x = np.linspace(x_limits[0], x_limits[1], grid_num)
    y = np.linspace(y_limits[0], y_limits[1], grid_num)
    x_grid, y_grid = np.meshgrid(x, y)

    # Calculate z values based on the surface equation
    z_grid = a * x_grid**2 + b * x_grid * y_grid + c * y_grid**2

    # Flatten the grids and stack them into a (n, 3) array
    points = np.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=-1)

    return points

def create_trimesh_from_grid(points, grid_num):
    """
    Create a triangular mesh from a regular grid.

    Args:
    points np.ndarray
    grid_num (int): The number of points sampled along each axis of the grid.

    Returns:
    trimesh.Trimesh: A trimesh object representing the surface.
    """

    # Generate vertex indices for a grid
    vertex_indices = np.arange(grid_num * grid_num).reshape(grid_num, grid_num)

    # Generate two sets of triangles for each cell in the grid
    # Triangle 1: (top left, top right, bottom left)
    # Triangle 2: (top right, bottom right, bottom left)
    faces1 = np.stack([
        vertex_indices[:-1, :-1].ravel(), 
        vertex_indices[:-1, 1:].ravel(), 
        vertex_indices[1:, :-1].ravel()
    ], axis=-1)
    
    faces2 = np.stack([
        vertex_indices[:-1, 1:].ravel(), 
        vertex_indices[1:, 1:].ravel(), 
        vertex_indices[1:, :-1].ravel()
    ], axis=-1)

    # Concatenate the two sets of faces to form the complete set of faces
    faces = np.concatenate([faces1, faces2])
    point_mesh = trimesh.Trimesh(vertices=points, faces=faces)
    point_mesh = extrude_mesh(point_mesh)

    return point_mesh

######################################################################################
#################################### SAMPLING ########################################
######################################################################################

def generate_joint_combinations(upper_finger_joint_limits, lower_finger_joint_limits, grid_num):
    '''
    Generate all possible joint angle combinations for a finger based on specified grid and joint limits.
    
    Parameters:
    - upper_finger_joint_limits (np.ndarray): An array of shape (finger_joint_num,) specifying the upper limits for each joint angle.
    - lower_finger_joint_limits (np.ndarray): An array of shape (finger_joint_num,) specifying the lower limits for each joint angle.
    - grid_num (int): Number of divisions for each joint range. Determines how many values between the joint limits to consider.
    
    Returns:
    - joint_combinations (np.ndarray): An array of shape (grid_num^finger_joint_num, finger_joint_num) containing all possible combinations of joint angles.
    '''
    
    finger_joint_num = upper_finger_joint_limits.shape[0]
    
    # Generate a list of arrays, each containing the grid points for one joint
    joint_grids = [np.linspace(lower_finger_joint_limits[i], upper_finger_joint_limits[i], grid_num) for i in range(finger_joint_num)]
    
    # Generate a meshgrid for all joint combinations
    mesh = np.meshgrid(*joint_grids)
    
    # Reshape and stack to get the desired output format
    joint_combinations = np.stack(mesh, axis=-1).reshape(-1, finger_joint_num)
    
    return joint_combinations


def find_closest_k_points(source_points, target_points, save_point_num=10):
    '''
    Find save_point_num points in target_points, which has the closest distance to one of the source_points.
  
    Parameters:
    - source_points (np.ndarray): An array of shape (m, 3) representing m source points.
    - target_points (np.ndarray): An array of shape (n ,3) representing n target points.
    - save_point_num (int): The number of closest points to retrieve from target_points.
    
    Returns:
    - closest_k_points (np.ndarray): An array of shape (min(save_point_num, n), 3) representing 
      the closest points from target_points to any of the source_points.
    '''

    # Compute pairwise distances using broadcasting
    distances = np.linalg.norm(source_points[:, np.newaxis, :] - target_points[np.newaxis, :, :], axis=-1)

    # Find the minimum distance for each target_point
    min_distances = np.min(distances, axis=0)

    # Sort target_points by their distance to source_points
    sorted_indices = np.argsort(min_distances)
    
    # Select the closest save_point_num points
    closest_k_points = target_points[sorted_indices[:save_point_num]]
    indices = sorted_indices[:save_point_num]
    
    return closest_k_points, indices


def vector_mag_normalize(a, norm_min=0.2):
    '''
    Normalize the last dim (considering the zero magnitude situation) and multiply the last dim with 
    a random factor shape=(..., 1) in (0, 1).

    Parameters:
    - a (numpy.ndarray): Input array with shape (..., 3).
    - norm_min (float): The minimal sample magnitude to avoid too small vector/force 

    Returns:
    - numpy.ndarray: Normalized and scaled array with shape (..., 3).
    '''
    
    # Compute the magnitudes of vectors along the last dimension
    magnitudes = np.linalg.norm(a, axis=-1, keepdims=True)
    
    # Normalize vectors
    normalized = a / (magnitudes + 1e-15)
    
    # Multiply by random factor
    random_factor = norm_min + (1 - norm_min) * np.random.rand(*normalized.shape[:-1], 1)
    scaled = normalized * random_factor
    
    return scaled



def find_valid_forces(arm_forces, tip_forces, contact_normals, f_coef=0.5):
    '''
    Finds valid force combinations that satisfy the friction cone conditions.
    
    Parameters:
    - arm_forces (numpy.ndarray): Sampled forces with shape (force_sample_num, 3).
    - tip_forces (numpy.ndarray): Sampled forces for n tips with shape (n, force_sample_num, 3).
    - contact_normals (numpy.ndarray): Contact normals for n tips with shape (n, 3).
    - f_coef (float): Friction coefficient. Default is 0.5.
    - k (int): Number of valid force combinations to return. Default is 10.
    
    Returns:
    - arm_forces_list, list of (3,), 
    - tip_forces_list,  list of (n, 3), 
    - added_tip_forces_list list of (n, 3), 
    '''

    arm_forces = vector_mag_normalize(arm_forces)
    tip_forces = vector_mag_normalize(tip_forces)

    # Expand arm_forces for broadcasting
    arm_forces_expanded = arm_forces[:, np.newaxis, np.newaxis, :]  # shape=()
    
    # Expand tip_forces for broadcasting
    tip_forces_expanded = tip_forces[np.newaxis, ...]

    # Compute the combined forces
    combined_forces = arm_forces_expanded + tip_forces_expanded  # shape: (100, n, 100, 3)

    # Compute the normal force component
    force_normals = np.sum(combined_forces * contact_normals[np.newaxis, :, np.newaxis, :], axis=-1)

    # Check if the force direction is opposite to the normal
    is_opposite_to_normal = force_normals < 0

    # Compute the tangential force component
    force_normals_expanded = force_normals[..., np.newaxis]
    force_tangential_vectors = combined_forces - force_normals_expanded * contact_normals[np.newaxis, :, np.newaxis, :]
    force_tangential_magnitudes = np.linalg.norm(force_tangential_vectors, axis=-1)
    
    # Check the friction cone condition
    valid_friction = force_tangential_magnitudes <= f_coef * np.abs(force_normals)

    # All conditions must be satisfied
    all_valid_mask = np.logical_and(is_opposite_to_normal, valid_friction) 
    valid_arm_finger_mask = np.any(all_valid_mask, axis=-1) 
    valid_arm_mask = np.all(valid_arm_finger_mask, axis=-1) 
    valid_arm_indices = np.where(valid_arm_mask)[0]

    arm_forces_list = []
    tip_forces_list = []
    added_tip_forces_list = []

    for valid_arm_idx in valid_arm_indices:

        arm_forces_list.append(arm_forces[valid_arm_idx])
        finger_force_list = []

        for finger_id in range(tip_forces.shape[0]):
            valid_force_id = np.random.choice(np.where(all_valid_mask[valid_arm_idx, finger_id])[0], 1) # shape=
            finger_force_list.append(tip_forces[finger_id, valid_force_id])
        finger_force_array = np.array(finger_force_list).reshape(-1, 3)   # (n_contact_finger, 3)
        added_finge_force_array = finger_force_array + arm_forces[valid_arm_idx]  # (n_contact_finger, 3)

        tip_forces_list.append(finger_force_array)
        added_tip_forces_list.append(added_finge_force_array)

    return arm_forces_list, tip_forces_list, added_tip_forces_list


def sample_cone(direction, angle, count):
    '''
    Sample directions within a cone.
    
    Args:
    - direction: The main direction of the cone.
    - angle: The angle of the cone in radians.
    - count: Number of samples to generate.
    
    Returns:
    - samples: An array of shape (count, 3) containing the sampled directions.
    '''
    
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)
    
    # Generate random axes perpendicular to the direction
    random_vectors = np.random.randn(count, 3)
    axes = np.cross(direction, random_vectors)
    axes = axes / np.linalg.norm(axes, axis=1)[:, np.newaxis]
    
    # Generate random angles within [0, angle/2]
    angles = np.random.uniform(0, angle / 2, count)
    
    # Create a rotation matrix for each angle around the respective axis
    rot_mats = np.array([trimesh.transformations.rotation_matrix(a, ax)[:3, :3] for a, ax in zip(angles, axes)])
    
    # Rotate the direction by each rotation matrix
    samples = np.dot(rot_mats, direction)
    
    return samples




######################################################################################
#################################### SPATIAL #########################################
######################################################################################
def normalize(x):
    '''
    Normalize the input vector. If the magnitude of the vector is zero, a small value is added to prevent division by zero.

    Parameters:
    - x (np.ndarray): Input vector to be normalized.

    Returns:
    - np.ndarray: Normalized vector.
    '''
    if len(x.shape) == 1:
        mag = np.linalg.norm(x)
        if mag == 0:
            mag = mag + 1e-10
        return x / mag
    else: 
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        return x / norms


def hat(vectors):
    if vectors.ndim == 1 or (vectors.ndim == 2 and vectors.shape[0] == 3):
        # Handle a single vector
        v = vectors.flatten()  # Ensuring the vector is one-dimensional
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    elif vectors.ndim == 2 and vectors.shape[1] == 3:
        # Handle a batch of vectors
        result = np.zeros((vectors.shape[0], 3, 3))
        result[:, 0, 1] = -vectors[:, 2]
        result[:, 0, 2] = vectors[:, 1]
        result[:, 1, 0] = vectors[:, 2]
        result[:, 1, 2] = -vectors[:, 0]
        result[:, 2, 0] = -vectors[:, 1]
        result[:, 2, 1] = vectors[:, 0]
        return result
    else:
        raise ValueError("Input should be a 3-element vector or an array of 3-element vectors.")

def compute_grasp_maps(contact_points, contact_normals):  # inward normals!
    n = contact_normals.shape[0]
    n_normalized = normalize(contact_normals)
    r = np.random.rand(n, 3)
    t1 = normalize(np.cross(r, n_normalized))
    t2 = normalize(np.cross(n_normalized, t1))
    
    R = np.empty((n, 3, 3))
    R[:, :, 0] = t1
    R[:, :, 1] = t2
    R[:, :, 2] = n_normalized  # z is the local frame inward 
    
    grasp_maps = np.empty((n, 6, 3))
    grasp_maps[:, :3, :] = R
    p_hat = hat(contact_points)
    grasp_maps[:, 3:, :] = np.matmul(p_hat, R)

    return grasp_maps

def local_frame(n):
    n = normalize(n)
    r = np.random.rand(3)
    t1 = normalize(np.cross(r, n))
    t2 = normalize(np.cross(n, t1))

    R = np.zeros((3, 3))
    R[:, 0] = t1
    R[:, 1] = t2
    R[:, 2] = n

    return R

def quadric(data, a, b, c):
    x, y = data
    return a * (x ** 2) + b * x * y + c * (y ** 2)

def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.arccos(cosine_angle)
    return angle

def normalize_torch(x):
    '''
    Normalize considering zero magnitude situation
    ---
    x: torch.tensor
    '''
    mag = torch.linalg.norm(x)
    if mag == 0:
        mag = mag + 1e-10
    return x / mag

def transform_matrix_to_6D(transform_matrix, seq="ZYX"):
    '''
    transform_matrix: torch.tensor, shape=(4, 4)
    seq: str, len=3, 3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or {'x', 'y', 'z'} for extrinsic rotations 
    '''
     
    rot = torch.tensor(R.from_matrix(transform_matrix.detach().cpu()[:3, :3]).as_euler(seq), dtype=torch.float)
    trans_vec = torch.cat((transform_matrix[:3, 3], rot))
    return trans_vec

def are_quaternions_close(q1, q2, threshold):
    """
    Check if two quaternions are close to each other based on a threshold.
    
    Parameters:
    - q1, q2: The two quaternions to be compared. Each should be an array of shape (4,).
    - threshold: The cosine of half of the maximum allowed angle difference between the two quaternions.
    
    Returns:
    - True if the quaternions are close to each other, False otherwise.
    """
    # Compute the dot product of the two quaternions
    dot_product = np.dot(q1, q2)
    
    # Ensure the dot product is in the range [-1, 1] due to potential numerical errors
    dot_product = np.clip(dot_product, -1, 1)
    
    # Compare the absolute value of the dot product with the threshold
    return abs(dot_product) >= threshold


def sample_transform_w_normals(new_palm_center, new_face_vector, sample_roll, ori_face_vector=np.array([1.0, 0.0, 0.0])):
    '''
    Compute the transformation matrix from the original palm pose to a new palm pose.
    
    Parameters:
    - new_palm_center (np.ndarray): The point of the palm center [x, y, z].
    - new_face_vector (np.ndarray): The direction vector representing the new palm facing direction.
    - sample_roll (float): The roll angle in range [0, 2*pi).
    - ori_face_vector (np.ndarray): The original direction vector representing the palm facing direction. Default is [1.0, 0.0, 0.0].

    Returns:
    - rst_transform (np.ndarray): A 4x4 transformation matrix.
    '''

    rot_axis = np.cross(ori_face_vector, normalize(new_face_vector))
    rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-16)
    rot_ang = np.arccos(np.clip(np.dot(ori_face_vector, new_face_vector), -1.0, 1.0))

    if rot_ang > 3.1415 or rot_ang < -3.1415: 
        rot_axis = np.array([1.0, 0.0, 0.0]) if not np.isclose(ori_face_vector, np.array([1.0, 0.0, 0.0])).all() else np.array([0.0, 1.0, 0.0])
    
    rot = R.from_rotvec(rot_ang * rot_axis).as_matrix()
    roll_rot = R.from_rotvec(sample_roll * new_face_vector).as_matrix()

    final_rot = roll_rot @ rot
    rst_transform = np.eye(4)
    rst_transform[:3, :3] = final_rot
    rst_transform[:3, 3] = new_palm_center
    return rst_transform

def vector_jittering(vec, max_angle=np.pi/6):
    '''
    Jittering a vector in a cone. Keep the magnitude and change the direction a little bit.

    Parameters:
    - vec (np.ndarray): Input vector with shape (3,).
    - max_angle (float): The maximum deviation angle in radians. Default is pi/6.

    Returns:
    - vec_jit (np.ndarray): Jittered vector with shape (3,).
    '''
    
    # Generate a random axis for rotation
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    
    # Generate a random angle within [0, max_angle]
    angle = np.random.uniform(0, max_angle)
    
    # Compute the rotation
    rotation = R.from_rotvec(angle * axis)
    
    # Apply the rotation
    vec_jit = rotation.apply(vec)
    
    return vec_jit


def sample_init_palm_poses(sampled_points, 
                           face_normals, 
                           min_palm_dis = 0.04,
                           max_palm_dis = 0.12,
                           ori_face_vector = np.array([1.0, 0.0, 0.0]),
                           jittering=False):
    '''
    Sample initial poses for the palm based on points sampled from a mesh surface.
    
    Parameters:
    - sampled_points (np.ndarray): Points sampled from the mesh surface.
    - face_normals (np.ndarray): Normal vectors corresponding to the sampled points.
    - min_palm_dis (float): The minimum distance from the point to the palm center.
    - max_palm_dis (float): The maximum distance from the point to the palm center.
    - ori_face_vector (np.ndarray): The original direction vector representing the palm facing direction. Default is [1.0, 0.0, 0.0].

    Returns:
    - t_list (list): A list of 4x4 transformation matrices representing the sampled palm poses.
    '''
    sample_point_num = sampled_points.shape[0]
    retreat_dis = np.random.rand(sample_point_num) * (max_palm_dis - min_palm_dis) + min_palm_dis
    base_points = sampled_points + retreat_dis.reshape(-1, 1) * face_normals

    t_list = []
    for i in range(sample_point_num):
        sample_point = base_points[i]
        sample_normal = - face_normals[i]
        if jittering:
            sample_normal = vector_jittering(sample_normal)
        sample_roll = np.random.rand() * 2 * np.pi   # rad 
        t = sample_transform_w_normals(sample_point, sample_normal, sample_roll, ori_face_vector)
        t_list.append(t)
        
    return t_list

def iterate_init_palm_poses(sampled_points, 
                           face_normals, 
                           min_palm_dis = 0.04,
                           max_palm_dis = 0.12,
                           sample_roll_num = 5,
                           ori_face_vector = np.array([1.0, 0.0, 0.0]),
                           jittering=False):
    '''
    Sample initial poses for the palm based on points sampled from a mesh surface.
    
    Parameters:
    - sampled_points (np.ndarray): Points sampled from the mesh surface.
    - face_normals (np.ndarray): Normal vectors corresponding to the sampled points.
    - min_palm_dis (float): The minimum distance from the point to the palm center.
    - max_palm_dis (float): The maximum distance from the point to the palm center.
    - ori_face_vector (np.ndarray): The original direction vector representing the palm facing direction. Default is [1.0, 0.0, 0.0].

    Returns:
    - t_list (list): A list of 4x4 transformation matrices representing the sampled palm poses.
    '''
    sample_point_num = sampled_points.shape[0]
    retreat_dis = np.random.rand(sample_point_num) * (max_palm_dis - min_palm_dis) + min_palm_dis
    base_points = sampled_points + retreat_dis.reshape(-1, 1) * face_normals

    t_list = []
    for i in range(sample_point_num):
        for j in range(sample_roll_num):
            sample_point = base_points[i]
            sample_normal = - face_normals[i]
            if jittering:
                sample_normal = vector_jittering(sample_normal)
            sample_roll = np.random.rand() * 2 * np.pi   # rad 
            t = sample_transform_w_normals(sample_point, sample_normal, sample_roll, ori_face_vector)
            t_list.append(t)
        
    return t_list

def sample_arm_base_pose_vector(sphere_center, sphere_radius_min=0.2, sphere_radius_max=0.8, sample_num=100, up_vector=[0.0, 1.0, 0.0]):
    """
    Sample base positions and orientations inside a partially hollow sphere using vectorized operations.
    
    Parameters:
    - sphere_center (list or numpy array of shape (3,)): Center of the sphere.
    - sphere_radius_min (float): Minimum radius of the hollow sphere.
    - sphere_radius_max (float): Maximum radius of the hollow sphere.
    - sample_num (int): Number of samples to generate.
    - up_vector (list or numpy array of shape (3,)): Upward direction vector. Orientations will be generated to keep this direction upward.
    
    Returns:
    - numpy array of shape (sample_num, 7): Each row contains position (first 3 values) and orientation quaternion (next 4 values).
    """
    
    # Normalize the up vector
    up_vector = np.array(up_vector) / np.linalg.norm(up_vector)
    
    # 1. Randomly generate all distances, angles, and rotation angles
    r = np.random.uniform(sphere_radius_min, sphere_radius_max, sample_num)
    theta = np.random.uniform(0, 2 * np.pi, sample_num)  # azimuthal angle
    phi = np.arccos(2 * np.random.uniform(size=sample_num) - 1)  # polar angle
    rotation_angles = np.random.uniform(0, 2 * np.pi, sample_num)
    
    # 2. Convert spherical coordinates to Cartesian coordinates using vectorized operations
    x = r * np.sin(phi) * np.cos(theta) + sphere_center[0]
    y = r * np.sin(phi) * np.sin(theta) + sphere_center[1]
    z = r * np.cos(phi) + sphere_center[2]
    
    # 3. Generate quaternions for all samples
    rotations = R.from_rotvec(np.outer(rotation_angles, up_vector))
    quaternions = rotations.as_quat()
    
    # Concatenate positions and quaternions
    result = np.column_stack((x, y, z, quaternions))
    
    return result

def pose_vector_to_transformation_matrix(pose_vector):
    '''
    Convert a pose vector containing position and quaternion into a 4x4 transformation matrix.
    
    Parameters:
    - pose_vector (list of float): A list of 7 elements where the first 3 elements are the position [x, y, z]
      and the next 4 elements are the quaternion [qx, qy, qz, qw].

    Returns:
    - transformation_matrix (np.array): A 4x4 matrix representing the pose.
    '''
    # Extract position and quaternion from pose vector
    position = pose_vector[:3]
    quaternion = pose_vector[3:]
    
    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    
    # Construct transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position
    
    return transformation_matrix


def transformation_matrix_to_pose_vector(transformation_matrix):
    '''
    Convert a 4x4 transformation matrix into a pose vector containing position and quaternion.
    
    Parameters:
    - transformation_matrix (np.array): A 4x4 matrix representing the pose.

    Returns:
    - pose_vector (list of float): A list of 7 elements where the first 3 elements are the position [x, y, z]
      and the next 4 elements are the quaternion [qx, qy, qz, qw].
    '''
    # Extract rotation matrix and position from transformation matrix
    rotation_matrix = transformation_matrix[:3, :3]
    position = transformation_matrix[:3, 3]
    
    # Convert rotation matrix to quaternion
    quaternion = R.from_matrix(rotation_matrix).as_quat()
    
    # Construct pose vector
    pose_vector = np.concatenate((position, quaternion))
    
    return pose_vector.tolist()


def tranform_points(transform_matrix, points_to_transform):
    '''
    Given the points described in frame A and the transform matrix from frame A to frame B.
    Calculate the transformed points described in frame B.

    Args:
    - transform_matrix: np.ndarray, shape=(4, 4). Transformation matrix from frame A to frame B.
    - points_to_transform: np.ndarray, shape=(n, 3). Points described in frame A.

    Returns:
    - np.ndarray, shape=(n, 3). Transformed points described in frame B.
    '''
    # Convert points to homogeneous coordinates
    homogeneous_points = np.hstack((points_to_transform, np.ones((points_to_transform.shape[0], 1))))
    
    # Transform points using the transformation matrix
    transformed_homogeneous_points = homogeneous_points.dot(transform_matrix.T)
    
    # Convert back to 3D coordinates by removing the last column
    transformed_points = transformed_homogeneous_points[:, :3]
    
    return transformed_points


def transform_vectors(transform_matrix, vectors_to_transform):
    '''
    Given the vectors described in frame A and the transform matrix from frame A to frame B.
    Calculate the transformed vectors described in frame B.

    Args:
    - transform_matrix: np.ndarray, shape=(4, 4). Transformation matrix from frame A to frame B.
    - vectors_to_transform: np.ndarray, shape=(n, 3). Vectors described in frame A.

    Returns:
    - np.ndarray, shape=(n, 3). Transformed vectors described in frame B.
    '''
    # Convert vectors to "pseudo-homogeneous" coordinates (appending a 0)
    pseudo_homogeneous_vectors = np.hstack((vectors_to_transform, np.zeros((vectors_to_transform.shape[0], 1))))
    
    # Transform vectors using the transformation matrix
    transformed_pseudo_homogeneous_vectors = pseudo_homogeneous_vectors.dot(transform_matrix.T)
    
    # Convert back to 3D coordinates by removing the last column
    transformed_vectors = transformed_pseudo_homogeneous_vectors[:, :3]
    
    return transformed_vectors

def wrench_to_pointcloud_motion_vectorized(object_pointcloud, wrenches, mass=1, rot_inertia=np.eye(3), start_v=np.array([0.0, 0.0, 0.0]), start_w=np.array([0.0, 0.0, 0.0]), timestep=1):
    wrench_dim = wrenches.ndim
    if wrench_dim == 1:
        wrenches = wrenches[np.newaxis, :]

    forces = wrenches[:, :3]
    torques = wrenches[:, 3:]

    # Compute linear and angular accelerations
    linear_accs = forces / mass
    angular_accs = np.linalg.inv(rot_inertia) @ torques.T

    # Calculate displacements
    translations = start_v * timestep + 0.5 * linear_accs * timestep**2
    rotations = R.from_rotvec((start_w * timestep + 0.5 * angular_accs.T * timestep**2).reshape(-1, 3))
    rotation_matrices = rotations.as_matrix()

    # Apply transformations to the point cloud
    moved_pointclouds = np.einsum('ijk,kl->ijl', rotation_matrices, object_pointcloud.T).transpose(0, 2, 1) + translations[:, np.newaxis, :]
    
    pointcloud_motions = moved_pointclouds - object_pointcloud[np.newaxis, :, :]
    if wrench_dim == 1:
        return moved_pointclouds[0], pointcloud_motions[0]
    else:
        return moved_pointclouds, pointcloud_motions


def pc_keypoint_jacobian(object_pc, object_n, keypoints):
    '''
    object_pc: (n, 3)
    object_n: (n, 3)
    keypoints: (k, 3)
    ---
    return: J: (n, k * 3, 3)
    '''
    sample_point_num_k = keypoints.shape[0]
    object_point_num_n = len(object_pc)
    grasp_maps = compute_grasp_maps(object_pc, object_n) # (n, 6, 3)
    grasp_maps_col = grasp_maps.transpose(0, 2, 1).reshape(-1, 6) # 3 * (6,)
    _, J = wrench_to_pointcloud_motion_vectorized(keypoints, grasp_maps_col)
    J = J.reshape(object_point_num_n, 3, sample_point_num_k, 3)  
    J = J.transpose(0, 2, 3, 1)  #  (n, k, 3, 3)
    J = J.reshape(object_point_num_n, -1, 3)  #  (n, k * 3, 3)
    return J


# https://github.com/ClayFlannigan/icp/blob/master/icp.py
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, corresponded_points=True, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        if not corresponded_points:
            # find the nearest neighbors between the current source and destination points
            distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
            # compute the transformation between the current source and nearest destination points
            T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)
        else: 
            distances = np.sum(np.linalg.norm(src[:m,:].T - dst[:m,:].T, axis=-1))
            T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,:].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i


def best_fit_transform_torch(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Batch of Nxm torch tensor of corresponding points
      B: Batch of Nxm torch tensor of corresponding points
    Returns:
      T: (batch, m+1, m+1) homogeneous transformation matrix that maps A on to B
      R: (batch, m, m) rotation matrix
      t: (batch, m, 1) translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[2]

    # translate points to their centroids
    centroid_A = torch.mean(A, dim=1, keepdim=True)
    centroid_B = torch.mean(B, dim=1, keepdim=True)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = torch.matmul(AA.transpose(-2, -1), BB)
    U, S, Vt = torch.linalg.svd(H)
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))

    # special reflection case
    det_R = torch.det(R)
    reflections = det_R < 0
    Vt[reflections, m-1, :] *= -1
    R[reflections] = torch.matmul(Vt[reflections].transpose(-2, -1), U[reflections].transpose(-2, -1))

    # translation
    t = centroid_B.transpose(-2, -1) - torch.matmul(R, centroid_A.transpose(-2, -1))

    # homogeneous transformation
    T = torch.eye(m+1).repeat(A.shape[0], 1, 1).to(A.device)
    T[:, :m, :m] = R
    T[:, :m, m] = t.squeeze()
    
    return T, R, t


def nearest_neighbor_torch(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src using PyTorch3D's knn
    Input:
        src: Batch of Nxm torch tensor of points
        dst: Batch of Nxm torch tensor of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    # Using PyTorch3D's knn. knn_points returns squared distances, so we take the sqrt.
    knn_result = knn_points(src, dst, K=1)
    distances = torch.sqrt(knn_result.dists[..., 0])  # Squared distances, take sqrt for Euclidean
    indices = knn_result.idx[..., 0]

    return distances, indices


def icp_torch(A, B, corresponded_points=True, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Batch of Nxm torch tensor of source mD points
        B: Batch of Nxm torch tensor of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape
    device = A.device 
    # get number of dimensions
    m = A.shape[2]

    # make points homogeneous, copy them to maintain the originals
    src = torch.ones((A.shape[0], m+1, A.shape[1]), device=device)
    dst = torch.ones((B.shape[0], m+1, B.shape[1]), device=device)
    src[:, :m, :] = A.transpose(-2, -1)
    dst[:, :m, :] = B.transpose(-2, -1)

    # apply the initial pose estimation
    if init_pose is not None:
        src = torch.matmul(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        if not corresponded_points:
            # find the nearest neighbors between the current source and destination points
            distances, indices = nearest_neighbor_torch(src[:, :m, :].transpose(-2, -1), dst[:, :m, :].transpose(-2, -1))
            # compute the transformation between the current source and nearest destination points
            T,_,_ = best_fit_transform_torch(src[:, :m, :].transpose(-2, -1), torch.gather(dst[:, :m, :], 2, indices.unsqueeze(-1).expand(-1, -1, m)))
        else: 
            distances = torch.norm(src[:, :m, :].transpose(-2, -1) - dst[:, :m, :].transpose(-2, -1), dim=-1)
            T,_,_ = best_fit_transform_torch(src[:, :m, :].transpose(-2, -1), dst[:, :m, :].transpose(-2, -1))

        # update the current source
        src = torch.matmul(T, src)

        # check error
        mean_error = torch.mean(distances)
        if torch.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform_torch(A, src[:, :m, :].transpose(-2, -1))

    return T, distances, i


######################################################################################
############################### FILE MANAGEMENT ######################################
######################################################################################
def find_path_by_uid(file_path, uid):
    with open(file_path, 'r') as file:
        for line in file:
            if uid in line:
                return line.strip()  
    return None  

def extract_end_number_from_str(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    else:
        return None
######################################################################################
################################ WRENCH SPACE ########################################
######################################################################################
def generate_3d_friction_cone(normal, friction_coefficient, num_samples=1024):
    angle = np.arctan(friction_coefficient)
    base_radius = np.linalg.norm(normal) * np.tan(angle)
    t = np.linspace(0, 2*np.pi, num_samples)
    circle_points = np.array([base_radius * np.cos(t), base_radius * np.sin(t), np.zeros(num_samples)]).T

    normal_unit = normal / np.linalg.norm(normal)
    up_vector = np.array([0, 0, 1])

    if np.allclose(normal_unit, up_vector) or np.allclose(normal_unit, -up_vector):
        rotated_points = circle_points
    else:
        rotation_axis = np.cross(up_vector, normal_unit)
        if np.linalg.norm(rotation_axis) < 1e-6:
            # Choose a different up_vector that is not parallel to normal_unit
            for axis in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]:
                if not np.allclose(normal_unit, axis) and not np.allclose(normal_unit, -axis):
                    up_vector = axis
                    break
            rotation_axis = np.cross(up_vector, normal_unit)
        
        rotation_angle = np.arccos(np.clip(np.dot(up_vector, normal_unit), -1.0, 1.0))
        rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis, point=np.zeros(3))
        rotated_points = trimesh.transform_points(circle_points, rotation_matrix)

    cone_vectors = rotated_points + normal
    return cone_vectors


def generate_6d_wrench(contact_point, normal, friction_coefficient, num_samples=8):  # wrench basis 
    forces = generate_3d_friction_cone(normal, friction_coefficient, num_samples)
    moments = np.cross(contact_point, forces)
    wrenches = np.hstack((forces, moments))
    return wrenches

def combine_wrenches_6d(contact_points, normals, friction_coefficient):
    wrenches = [generate_6d_wrench(cp, n, friction_coefficient) for cp, n in zip(contact_points, normals)]
    combined_wrenches = np.vstack(wrenches)
    return combined_wrenches

def find_subspace_convex_hull(points):
    """
    Finds the convex hull of points possibly lying in a subspace of the higher-dimensional space.
    """
    try:
        hull = ConvexHull(points, qhull_options='QJ')
        return hull
    except scipy.spatial.qhull.QhullError:
        # Fallback to Delaunay triangulation in case of degeneracies
        del_tri = Delaunay(points, qhull_options='QJ')
        return del_tri.convex_hull
    
def generate_random_barycentric_coordinates(dim, n_samples):
    """
    Generate random barycentric coordinates for multiple points in a simplex of dimension 'dim'.
    """
    random_values = np.random.rand(n_samples, dim - 1)
    random_values.sort(axis=1)
    barycentric_coords = np.diff(np.concatenate([np.zeros((n_samples, 1)), random_values, np.ones((n_samples, 1))], axis=1), axis=1)
    return barycentric_coords

def sample_points_in_hull(hull, n_samples=1000):
    """
    Sample points inside a convex hull in 6D using uniform sampling on the surface and interpolation.
    """
    points = hull.points
    centroid = np.mean(points, axis=0)
    simplex_indices = np.random.choice(len(hull.simplices), size=n_samples)
    simplexes = hull.points[hull.simplices[simplex_indices]]

    barycentric_coords = generate_random_barycentric_coordinates(simplexes.shape[1], n_samples)
    surface_points = np.einsum('ijk,ij->ik', simplexes, barycentric_coords)

    fractions = np.random.rand(n_samples, 1)
    inside_points = surface_points + fractions * (centroid - surface_points)

    return inside_points


def wrench_space_sampling(contact_points, normals, friction_coefficient, num_sample=1024):
    '''
    E.g.
    contact_points = [[0.2, 0.2, 0], [-0.2, 0.2, 0]]  # Example contact points in 3D
    normals = [[-1, 0, 0], [1, 0, 0]]  # Normals at contact points
    friction_coefficient = 0.5  # Example friction coefficient

    '''
    combined_wrenches_6d = combine_wrenches_6d(contact_points, normals, friction_coefficient)
    combined_wrenches_6d = combined_wrenches_6d / (np.linalg.norm(combined_wrenches_6d, axis=-1, keepdims=True)+1e-10)
    combined_wrenches_6d = np.vstack((combined_wrenches_6d, np.zeros(6,)))

    hull_6d = find_subspace_convex_hull(combined_wrenches_6d)
    sampled_points_6d = sample_points_in_hull(hull_6d, num_sample)
    sampled_points_6d = sampled_points_6d / (np.linalg.norm(sampled_points_6d, axis=-1, keepdims=True)+1e-10)
    return sampled_points_6d

def visualize_6d_points_trimesh(points, f_color=[255, 0, 0], t_color=[0, 0, 255]):
    """
    Visualize a n*6 numpy array as two sets of 3D points using trimesh.
    The first three columns are visualized as red points, and the last three as blue points.
    """
    # Create two point clouds
    cloud1 = trimesh.points.PointCloud(points[:, :3], colors=f_color)
    cloud2 = trimesh.points.PointCloud(points[:, 3:], colors=t_color)

    return cloud1, cloud2


def generate_random_wrenches(contact_points, contact_normals, friction_coefficient=1, min_force=0.1, max_force=1, sample_num=10):
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

    return sample_wrenches


def define_finger_orders(idx):
    x1, x2, x3, x4 = "index",  "middle", "ring", "thumb"
    if len(idx) == 1:
        contact_finger_names_list = [[x1],[x2],[x3],[x4]]
    elif len(idx) == 2:
        contact_finger_names_list = [[x1,x2],[x1,x3],[x1,x4],[x2,x1],[x2,x3],[x2,x4]
                        ,[x3,x1],[x3,x2],[x3,x4],[x4,x1],[x4,x2],[x4,x3]]
    elif len(idx) == 3:
        contact_finger_names_list = [[x1,x2,x3],[x1,x2,x4],[x1,x3,x2],[x1,x3,x4],[x1,x4,x2],[x1,x4,x3]
                        ,[x2,x1,x3],[x2,x1,x4],[x2,x3,x1],[x2,x3,x4],[x2,x4,x1],[x2,x4,x3]
                        ,[x3,x1,x2],[x3,x1,x4],[x3,x2,x1],[x3,x2,x4],[x3,x4,x1],[x3,x4,x2]
                        ,[x4,x1,x2],[x4,x1,x3],[x4,x2,x1],[x4,x2,x3],[x4,x3,x1],[x4,x3,x2]]
    elif len(idx) == 4:
        
        contact_finger_names_list = [[x1,x2,x3,x4],[x1,x2,x4,x3],[x1,x3,x2,x4],[x1,x3,x4,x2],[x1,x4,x2,x3],[x1,x4,x3,x2]
                        ,[x2,x1,x3,x4],[x2,x1,x4,x3],[x2,x3,x1,x4],[x2,x3,x4,x1],[x2,x4,x1,x3],[x2,x4,x3,x1]
                        ,[x3,x1,x2,x4],[x3,x1,x4,x2],[x3,x2,x1,x4],[x3,x2,x4,x1],[x3,x4,x1,x2],[x3,x4,x2,x1]
                        ,[x4,x1,x2,x3],[x4,x1,x3,x2],[x4,x2,x1,x3],[x4,x2,x3,x1],[x4,x3,x1,x2],[x4,x3,x2,x1]]
    elif len(idx) == 0:
        raise ValueError("Length of idx is zero.")
        
    return contact_finger_names_list


def select_top_contacts(origin_idx, pred_hmap):
    contacts_num = origin_idx.shape[0]
    # print(contacts_num)
    if contacts_num > 4:
        points_values = pred_hmap[origin_idx].reshape(-1)
        order = np.argsort(points_values)
        chosen_4 = order[-4:]
        chosen_idx = origin_idx[chosen_4]
        # import pdb;pdb.set_trace()
    else:
        chosen_idx = origin_idx
    return chosen_idx


def geodesic_distance_from_pcd(point_cloud, keypoint_index):
    solver = pp3d.PointCloudHeatSolver(point_cloud)

    # Compute the geodesic distance to point 4
    dists = solver.compute_distance(keypoint_index)

    return torch.from_numpy(dists).float()


def get_heatmap(point_cloud, keypoint_index, distance="geodesic", max_value = 10.0):
    # distance: "l2" or "geodesic"
    # Set the standard deviation to be one-thirtieth of the longest edge
    ranges = torch.max(point_cloud, dim=0)[0] - torch.min(point_cloud, dim=0)[0]
    longest_edge = torch.max(ranges)
    std_dev = longest_edge / 25

    # Extract keypoint coordinates
    keypoint = point_cloud[keypoint_index]
    if distance == "l2":
        # Compute the L2 distance from the keypoint to all other points
        distances = torch.norm(point_cloud - keypoint, dim=1)
    elif distance == "geodesic":
        # Compute the geodesic distance from the keypoint to all other points
        distances = geodesic_distance_from_pcd(point_cloud, keypoint_index)
    heatmap_values = torch.exp(-0.5 * (distances / std_dev) ** 2)
    heatmap_values /= torch.max(heatmap_values)

    # rgb = torch.zeros(point_cloud.shape[0], 3)
    # rgb[:, 0] = heatmap_values.clone()
    # save_pcd_as_pcd(point_cloud, rgb, "test/pcd.pcd")

    heatmap_values *= max_value

    return heatmap_values


def calculate_iou(heatmap1, heatmap2, threshold1, threshold2):
    # Apply threshold to create binary masks
    mask1 = heatmap1 > threshold1
    mask2 = heatmap2 > threshold2

    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Calculate IoU
    iou = intersection / union if union != 0 else 0
    return iou


def generate_heatmap_from_tip_pos(pc, tip_pos):
    """
    pc: np.array (2048,3),
    tip_pos: list of np.array [(3,), ...]
    """
    heatmaps = []
    for pos in tip_pos:
        dist = np.linalg.norm(pc - pos, axis=-1)
        point_idx = np.argsort(dist)[0]
        pc_th = torch.from_numpy(pc).float()
        heatmap = get_heatmap(pc_th, point_idx, distance="geodesic", max_value=10)
        heatmaps.append(heatmap.numpy())

    mixed_heatmap = np.max(heatmaps, axis=0)
    return mixed_heatmap


def normalize_heatmap(hmap):
    """
    hmap: np.array of (2048,)
    """
    return (hmap - hmap.min()) / (hmap.max() - hmap.min())


def get_hand_icp_pose_from_points_and_finger_names(args):
    hand, hand_nb, pc, chosen_contact_ids, contact_finger_names, pred_heatmap, candidates, lock = args
    contact_point_set = pc.points[chosen_contact_ids]
    hand.pb_reset_base([0, 0, 0], [0, 0, 0, 1])
    hand.pb_reset_joints(hand.pb_rest_joint_pos)
    rest_fingre_tip_points = np.array([hand.fk(hand.pb_rest_joint_pos, finger_name=f) for f in contact_finger_names])
    init_T, init_dis, init_icp_iter = icp(rest_fingre_tip_points, contact_point_set, corresponded_points=True)
    init_T[:3, 3] = init_T[:3, 3] # * 2

    init_hand_nb_joint_pos = hand_nb.nb_rest_pose.copy()
    init_palm_rotvec = R.from_matrix(init_T[:3, :3]).as_rotvec() 
    init_palm_pos = init_T[:3, 3]
    init_hand_nb_joint_pos[:3] = init_palm_rotvec
    init_hand_nb_joint_pos[3:6] = init_palm_pos

    with lock:
        # print(init_hand_nb_joint_pos)
        hand_nb.setJoints(init_hand_nb_joint_pos)
        tip_position = hand_nb.getTips(contact_finger_names)
        # print(tip_position)
        # print("==============================")

    icp_heatmap = generate_heatmap_from_tip_pos(pc.points, tip_position)
    norm_icp_heatmap = normalize_heatmap(icp_heatmap)
    thres_icp = np.percentile(norm_icp_heatmap, 70)

    norm_pred_heatmap = normalize_heatmap(pred_heatmap)
    thres_pred = np.percentile(norm_pred_heatmap, 70)

    IoU_pred_icp = calculate_iou(norm_pred_heatmap, norm_icp_heatmap, thres_pred, thres_icp)

    candidates.append((init_hand_nb_joint_pos, contact_finger_names, IoU_pred_icp, norm_pred_heatmap, norm_icp_heatmap, thres_pred, thres_icp))


def generate_icp_poses(hand, hand_nb, pc, chosen_idx, contact_finger_names_list, pred_hmap):
    manager = Manager()
    lock = manager.Lock()
    candidates = manager.list()
    pool = Pool()
    args = [(hand, hand_nb, pc, chosen_idx, contact_finger_names, pred_hmap, candidates, lock) for contact_finger_names in contact_finger_names_list]
    pool.map(get_hand_icp_pose_from_points_and_finger_names, args)
    candidates = list(candidates)
    return candidates


def select_best_candidate(candidates):

    sorted_cand = sorted(candidates, key=lambda x:x[2])
    init_pose = sorted_cand[-1][0]
    contact_finger_names = sorted_cand[-1][1]
    return init_pose, contact_finger_names

