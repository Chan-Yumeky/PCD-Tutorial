# 单独做法线估计处理
import open3d as o3d
from datetime import datetime

# 读取点云
pcd = o3d.io.read_point_cloud("./usablePCD/filtered_originalCow.ply")
if pcd.is_empty():
    print("Failed to load point cloud.")
    exit()

print("原始点云点数:", len(pcd.points))

# 可视化原始点云
o3d.visualization.draw_geometries([pcd], window_name="原始点云")

# 法线估计
print("-> 正在进行法线估计...")
search_radius = 0.02  # 搜索半径
max_nn = 30           # 邻居点数量

# 进行法线估计
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=max_nn))

# 调整法线方向一致性（可选）
pcd.orient_normals_consistent_tangent_plane(k=30)

print("法线估计后点云点数:", len(pcd.points))
# 可视化带法线的点云
print("-> 可视化法线估计结果...")
o3d.visualization.draw_geometries([pcd], window_name="法线估计后的点云", point_show_normal=False)

# 获取当前时间戳并格式化为字符串
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# 保存带法线的点云
output_filename = f"../surface reconstruction/cow_normal{timestamp}.ply"
o3d.io.write_point_cloud(output_filename, pcd)
print(f"带法线的点云已保存至: {output_filename}")
