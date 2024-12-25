import open3d as o3d
import numpy as np
import trimesh
import matplotlib.pyplot as plt

def preprocess_mesh(file_path):
    # 读取网格
    mesh = o3d.io.read_triangle_mesh(file_path)
    print(f"原始网格顶点数量: {len(mesh.vertices)}")
    # 去除未引用的顶点
    mesh = remove_unreferenced_vertices(mesh)
    print(f"去除未引用顶点后的网格顶点数量: {len(mesh.vertices)}")
    return mesh

def remove_unreferenced_vertices(mesh):
    # 删除未引用的顶点
    mesh.remove_unreferenced_vertices()
    return mesh

def compute_hip_height(mesh):
    # 获取网格的所有顶点
    vertices = np.asarray(mesh.vertices)
    z_min = vertices[:, 2].min()
    z_max = vertices[:, 2].max()
    hip_height = z_max - z_min
    # 保持单位为米
    return hip_height, z_min, z_max

def compute_hip_width(mesh, z_max, threshold=0.1):
    vertices = np.asarray(mesh.vertices)
    top_region_threshold = z_max - threshold
    top_region_vertices = vertices[vertices[:, 2] > top_region_threshold]
    if len(top_region_vertices) > 0:
        hip_width = top_region_vertices[:, 1].max() - top_region_vertices[:, 1].min()
        # 保持单位为米
        return hip_width
    else:
        return None

def compute_body_length(mesh, z_max, threshold=0.1):
    vertices = np.asarray(mesh.vertices)
    top_region_threshold = z_max - threshold
    top_region_vertices = vertices[vertices[:, 2] > top_region_threshold]
    if len(top_region_vertices) > 0:
        body_length = top_region_vertices[:, 0].max() - top_region_vertices[:, 0].min()
        # 保持单位为米
        return body_length
    else:
        return None

def convert_open3d_to_trimesh(o3d_mesh):
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)

    if len(vertices) == 0 or len(faces) == 0:
        print("转换后的 Trimesh 对象为空或无效。")
        return None

    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return trimesh_mesh

def compute_volume_trimesh(trimesh_mesh):
    if trimesh_mesh is None:
        print("无法计算体积，因为 Trimesh 网格无效。")
        return None

    if trimesh_mesh.is_watertight:
        volume = trimesh_mesh.volume  # 体积以立方米为单位
        return volume
    else:
        print("Trimesh 网格不是水密的，尝试使用凸包计算体积。")
        hull = trimesh_mesh.convex_hull
        volume = hull.volume
        print("使用凸包计算体积。")
        return volume

def visualize_density(densities):
    plt.hist(densities, bins=50)
    plt.xlabel("Density")
    plt.ylabel("Frequency")
    plt.title("Point Density Distribution")
    plt.show()

def visualize_mesh_trimesh(trimesh_mesh, title="Trimesh Mesh"):
    if trimesh_mesh is not None:
        trimesh_scene = trimesh.Scene(trimesh_mesh)
        trimesh_scene.show()
    else:
        print("无法可视化，因为 Trimesh 网格为空。")

def compute_volume_voxel(mesh, voxel_size=0.01):
    """
    使用体素化方法计算体积
    Args:
        mesh: Open3D网格对象
        voxel_size: 体素大小（米）
    Returns:
        volume_m3: 体积（立方米）
    """
    # 创建体素网格
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)

    # 获取体素数量
    voxels = voxel_grid.get_voxels()
    voxel_count = len(voxels)

    # 计算体积（体素数量 * 每个体素的体积）
    voxel_volume = voxel_size ** 3  # 每个体素的体积（立方米）
    total_volume = voxel_count * voxel_volume

    return total_volume, voxel_grid

def compute_volume_bbox(mesh):
    """
    使用边界框计算体积
    Args:
        mesh: Open3D网格对象
    Returns:
        volume_m3: 体积（立方米）
    """
    # 获取网格的边界框
    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound = np.array(bbox.min_bound)
    max_bound = np.array(bbox.max_bound)

    # 计算边界框体积
    length = max_bound[0] - min_bound[0]
    width = max_bound[1] - min_bound[1]
    height = max_bound[2] - min_bound[2]

    volume = length * width * height
    return volume

def compute_surface_area_open3d(mesh):
    """
    使用 Open3D 计算表面积
    Args:
        mesh: Open3D网格对象
    Returns:
        surface_area_m2: 表面积（平方米）
    """
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    # 计算每个三角形的面积
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross_prod = np.cross(v1 - v0, v2 - v0)
    triangle_areas = np.linalg.norm(cross_prod, axis=1) / 2

    total_area = triangle_areas.sum()  # 表面积以平方米为单位
    return total_area

def compute_surface_area_trimesh(trimesh_mesh):
    """
    使用 Trimesh 计算表面积
    Args:
        trimesh_mesh: Trimesh网格对象
    Returns:
        surface_area_m2: 表面积（平方米）
    """
    if trimesh_mesh is None:
        print("无法计算表面积，因为 Trimesh 网格无效。")
        return None

    surface_area = trimesh_mesh.area  # 表面积以平方米为单位
    return surface_area

def main():
    # 使用牛数据库中已合成好的牛点云，过滤噪点、重建并进行孔洞修复的最终效果
    # file_path = "rigidObj/originalCow_normal_rigidObj_fixed.ply"

    # 使用宁老师的方法进行改进，重新合成点云与重建刚体，并进行孔洞修复的最终效果
    file_path = "../img2pcd/model_fixed.ply"
    mesh = preprocess_mesh(file_path)

    # 可选：进一步处理原始网格，例如修复
    mesh = remove_unreferenced_vertices(mesh)

    # 计算臀高
    hip_height, z_min, z_max = compute_hip_height(mesh)
    print(f"臀高/十字部高(hip height): {hip_height:.2f} m")

    # 计算髋宽
    hip_width = compute_hip_width(mesh, z_max)
    if hip_width:
        print(f"髋宽(hip joint width): {hip_width:.2f} m")
    else:
        print("髋宽(hip joint width): N/A")

    # 计算体长
    body_length = compute_body_length(mesh, z_max)
    if body_length:
        print(f"体长(body length): {body_length:.2f} m")
    else:
        print("体长(body length): N/A")

    # 使用体素化方法计算体积
    volume_voxel, voxel_grid = compute_volume_voxel(mesh, voxel_size=0.01)
    print(f"体积 (体素化方法): {volume_voxel:.2f} m³")
    o3d.visualization.draw_geometries([voxel_grid], window_name="体素化结果",
                                      width=2000, height=2000,
                                      mesh_show_back_face=True)
    # visualize_mesh_trimesh(voxel_grid)

    # 使用边界框方法计算体积
    volume_bbox = compute_volume_bbox(mesh)
    print(f"体积 (边界框方法): {volume_bbox:.2f} m³")

    # 使用 Trimesh 计算体积（更为准确）
    trimesh_mesh = convert_open3d_to_trimesh(mesh)
    volume_trimesh = compute_volume_trimesh(trimesh_mesh)
    if volume_trimesh:
        print(f"体积 (Trimesh 方法): {volume_trimesh:.2f} m³")

    # 计算表面积使用 Open3D
    surface_area_o3d = compute_surface_area_open3d(mesh)
    print(f"表面积 (Open3D 方法): {surface_area_o3d:.2f} m²")

    # 计算表面积使用 Trimesh
    surface_area_trimesh = compute_surface_area_trimesh(trimesh_mesh)
    if surface_area_trimesh:
        print(f"表面积 (Trimesh 方法): {surface_area_trimesh:.2f} m²")

    # 可视化（可选）
    # 方法1：使用 Trimesh 可视化（需要 pyglet）
    visualize_mesh_trimesh(trimesh_mesh, "Trimesh结果")

    # 方法2：使用 Open3D 可视化
    # o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    main()