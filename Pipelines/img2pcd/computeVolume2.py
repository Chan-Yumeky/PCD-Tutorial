import open3d as o3d
import numpy as np


def compute_volume_with_voxelization(mesh, voxel_size=0.01):
    """
    使用体素化方法计算网格的体积。

    Args:
        mesh (open3d.geometry.TriangleMesh): 输入的三角网格。
        voxel_size (float): 体素的大小（单位：米）。

    Returns:
        float: 近似体积（单位：立方米）。
    """
    # 检查输入网格是否有效
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError("输入网格为空或无效！")

    # 计算网格的包围盒
    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound

    # 创建体素网格
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
    o3d.visualization.draw_geometries([voxel_grid],
                                      width=2000, height=2000,
                                      mesh_show_back_face=True)
    # 计算体素的总数（每个体素的体积为 voxel_size^3）
    filled_voxel_count = len(voxel_grid.get_voxels())
    volume = filled_voxel_count * (voxel_size ** 3)

    return volume


if __name__ == "__main__":
    # 加载网格
    mesh = o3d.io.read_triangle_mesh("model_fixed.ply")
    # mesh = o3d.io.read_triangle_mesh("../surface reconstruction/rigidObj/originalCow_normal_rigidObj_fixed.ply")
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print("网格为空或无效，请检查输入文件！")
        exit()

    # 确保网格是闭合的（体素化方法要求闭合网格）
    if not mesh.is_watertight():
        print("网格不是闭合的，请修复网格！")
        exit()

    # 设置体素大小并计算体积
    voxel_size = 0.01  # 体素大小为 1cm
    volume = compute_volume_with_voxelization(mesh, voxel_size=voxel_size)
    print(f"使用体素化方法计算的体积: {volume:.4f} 立方米")
