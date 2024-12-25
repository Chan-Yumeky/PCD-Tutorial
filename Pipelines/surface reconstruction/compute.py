import trimesh
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pyvista as pv


def load_mesh(file_path):
    # 读取网格文件
    mesh = trimesh.load_mesh(file_path)
    if mesh.is_empty:
        print(f"网格加载失败: {file_path}")
    return mesh


def compute_volume_trimesh(mesh):
    # 计算网格体积
    if mesh.is_watertight:
        volume = mesh.volume  # 体积以立方米为单位
        volume_cm3 = volume * 1e6  # 转换为立方厘米
        print(f"体积: {volume_cm3:.2f} 立方厘米")
        return volume_cm3
    else:
        print("网格不是水密的，无法准确计算体积")
        return None


def compute_hip_height(vertices):
    # 计算臀高（Z轴最大最小值之差）
    z_min = vertices[:, 2].min()
    z_max = vertices[:, 2].max()
    hip_height = (z_max - z_min) * 100  # 转换为厘米
    return hip_height, z_min, z_max


def compute_shoulder_width(vertices, z_max, threshold=0.1):
    # 计算肩宽（顶部区域的宽度）
    top_region_points = vertices[vertices[:, 2] > (z_max - threshold)]
    if len(top_region_points) > 0:
        shoulder_width = top_region_points[:, 0].max() - top_region_points[:, 0].min()
        shoulder_width_cm = shoulder_width * 100  # 转换为厘米
        return shoulder_width_cm
    else:
        return None


def repair_mesh(mesh):
    # 修复网格（填补孔洞、去除重复顶点、处理退化面）
    mesh = mesh.fill_holes()  # 填补孔洞
    mesh = mesh.merge_vertices()  # 合并重复顶点
    mesh = mesh.remove_degenerate_faces()  # 移除退化三角形
    mesh.apply_translation(-mesh.centroid)  # 将网格移到坐标原点
    return mesh


def visualize_mesh(mesh):
    # 可视化网格（使用 Trimesh 或 Open3D）
    try:
        mesh.show()
    except:
        # 如果 Trimesh 可视化失败，尝试 Open3D
        print("Trimesh 可视化失败，使用 Open3D 尝试")
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d.visualization.draw_geometries([o3d_mesh])


def main():
    file_path = "rigidObj/originalCow_fixed(12-0).ply"  # 替换为你的网格文件路径
    mesh = load_mesh(file_path)

    if mesh is None or mesh.is_empty:
        print("网格加载失败，退出程序。")
        return

    # 获取网格的顶点
    vertices = mesh.vertices

    # 计算体积
    compute_volume_trimesh(mesh)

    # 计算臀高
    hip_height, z_min, z_max = compute_hip_height(vertices)
    print(f"臀高: {hip_height:.2f} cm")

    # 计算肩宽
    shoulder_width = compute_shoulder_width(vertices, z_max)
    if shoulder_width:
        print(f"肩宽: {shoulder_width:.2f} cm")
    else:
        print("肩宽计算失败，未找到顶部区域的点")

    # 修复网格
    # mesh = repair_mesh(mesh)

    # 可视化网格
    # 可视化原始网格模型和孔洞
    p = pv.Plotter()
    p.add_mesh(mesh, color=True)
    p.enable_eye_dome_lighting()
    p.show()


if __name__ == "__main__":
    main()
