import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull


def compute_sliced_volume_with_hull(mesh, axis="x", slice_thickness=0.1, save_hulls=False, output_prefix="slice"):
    """
    计算网格通过切片累积体积的方法，并保存每个切片的凸包化网格。

    Args:
        mesh: 输入的 Open3D 网格对象。
        axis: 切片方向，可选值 "x", "y", "z"。
        slice_thickness: 切片厚度，控制切片精度。
        save_hulls: 是否保存每个切片的凸包化网格。
        output_prefix: 保存凸包化网格文件的前缀。

    Returns:
        total_volume: 总体积。
        all_hulls: 所有切片的凸包化网格的合并结果。
    """
    # 获取点云数据
    points = np.asarray(mesh.vertices)

    # 确定切片的范围
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    min_coord = points[:, axis_idx].min()
    max_coord = points[:, axis_idx].max()

    total_volume = 0.0  # 初始化体积
    all_hulls = []  # 保存每个切片的凸包网格

    current_min = min_coord
    slice_index = 0  # 切片编号
    while current_min < max_coord:
        current_max = current_min + slice_thickness
        # 提取当前切片的点
        slice_points = points[
            (points[:, axis_idx] >= current_min) & (points[:, axis_idx] < current_max)
            ]
        if len(slice_points) > 3:  # 至少需要4个点才能计算凸包
            # 计算当前切片的凸包
            hull = ConvexHull(slice_points)
            slice_volume = hull.volume
            total_volume += slice_volume

            # 将凸包点和面转为 Open3D 格式
            hull_mesh = o3d.geometry.TriangleMesh()
            hull_mesh.vertices = o3d.utility.Vector3dVector(slice_points[hull.vertices])
            hull_mesh.triangles = o3d.utility.Vector3iVector(hull.simplices)

            # 添加到所有凸包列表
            all_hulls.append(hull_mesh)

            # 保存当前凸包网格（可选）
            if save_hulls:
                o3d.io.write_triangle_mesh(f"./hulls/{output_prefix}_hull_{slice_index}.ply", hull_mesh)
                print(f"Saved: hulls/{output_prefix}_hull_{slice_index}.ply")

        current_min += slice_thickness
        slice_index += 1

    # 合并所有凸包网格为一个整体
    if len(all_hulls) > 0:
        merged_hulls = all_hulls[0]
        for hull in all_hulls[1:]:
            merged_hulls += hull
        return total_volume, merged_hulls

    return total_volume, None


if __name__ == "__main__":
    # 加载网格
    mesh = o3d.io.read_triangle_mesh("model.ply")

    # 确保网格为三角形网格
    mesh.compute_vertex_normals()

    # 调用切片体积计算函数，并保存每个凸包化网格
    volume, merged_hull = compute_sliced_volume_with_hull(
        mesh, axis="x", slice_thickness=0.004, save_hulls=False, output_prefix="slice"
    )
    print(f"切片累积体积: {volume:.2f} 立方米")

    # 保存合并后的凸包网格
    if merged_hull is not None:
        o3d.io.write_triangle_mesh("merged_hull.ply", merged_hull)
        print("合并后的凸包网格已保存为 merged_hull.ply")

    o3d.visualization.draw_geometries([merged_hull],
                                      width=2000, height=2000,
                                      mesh_show_back_face=True)
