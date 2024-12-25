# 点云预处理，其实就是对点云无关点进行降噪（距离滤波 -> 统计滤波 -> 半径滤波）

import numpy as np
import open3d as o3d
from datetime import datetime

# 读取点云
pcd = o3d.io.read_point_cloud("usablePCD/originalCow.ply")
if pcd.is_empty():
    print("Failed to load point cloud.")
    exit()
print("原始点云:", len(pcd.points))
# 可视化结果
o3d.visualization.draw_geometries([pcd])

# 设置距离阈值
distance_threshold = 1.4  # 根据离主体较远的点云团的距离设定，比如0.8米
# 计算主体的质心
centroid = np.mean(np.asarray(pcd.points), axis=0)
# 如果原始点云有颜色信息，我们需要同时保留颜色
original_points = np.asarray(pcd.points)
if pcd.has_colors():
    original_colors = np.asarray(pcd.colors)  # 提取颜色信息
else:
    original_colors = None  # 没有颜色信息
# 过滤距离超过阈值的点和对应颜色
filtered_points = []
filtered_colors = []
print("->正在距离滤波...")
# 遍历点和颜色（如果有颜色信息）
for i, point in enumerate(original_points):
    if np.linalg.norm(point - centroid) < distance_threshold:
        filtered_points.append(point)
        if original_colors is not None:
            filtered_colors.append(original_colors[i])
# 创建新点云并赋予颜色
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
if original_colors is not None:
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
print("距离滤波后点云:", len(filtered_pcd.points))
# 可视化结果
o3d.visualization.draw_geometries([filtered_pcd])

# 统计滤波
# 设置邻域点数和标准差比率
nb_neighbors = 20  # 邻域点数
std_ratio = 2.0    # 标准差比率
cl, ind = filtered_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
print("->正在统计滤波...")
# 使用索引保留有效点
filtered_pcd_statistical = filtered_pcd.select_by_index(ind)
print("统计滤波后点云:", len(filtered_pcd_statistical.points))
# 可视化结果
o3d.visualization.draw_geometries([filtered_pcd_statistical])

# 半径滤波
# 设置邻域点数和半径
nb_points = 16      # 邻域点数
radius = 0.045       # 半径
cl, ind = filtered_pcd_statistical.remove_radius_outlier(nb_points=nb_points, radius=radius)
print("->正在半径滤波...")
# 使用索引保留有效点
filtered_pcd_radius = filtered_pcd_statistical.select_by_index(ind)
print("半径滤波后点云:", len(filtered_pcd_radius.points))
# 可视化结果
o3d.visualization.draw_geometries([filtered_pcd_radius])

# 获取当前时间戳并格式化为字符串
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# 保存最终点云
output_filename = f"./usablePCD/{timestamp}.ply"
o3d.io.write_point_cloud(output_filename, filtered_pcd_radius)
print(f"点云已保存为: {output_filename}")

# # 获取当前时间戳并格式化为字符串
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # 保存最终点云
# output_filename = f"./usablePCD/filtered_downsampled_{timestamp}.pcd"
# o3d.io.write_point_cloud(output_filename, down_pcd)
# print(f"点云已保存为: {output_filename}")
