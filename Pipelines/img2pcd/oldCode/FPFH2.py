import open3d as o3d
import numpy as np
import copy

from numpy import array

# 加载点云
source = o3d.io.read_point_cloud("../raw_PCD/filtered/left.pcd")
target = o3d.io.read_point_cloud("../raw_PCD/filtered/right.pcd")

# 创建全局坐标轴
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

# 可视化应用初始变换后的点云
o3d.visualization.draw_geometries([source, target, axis], window_name="初始点云")

# 下采样
print("-> 正在进行体素下采样...")
voxel_size = 0.01  # 设置下采样体素大小
source_down = source.voxel_down_sample(voxel_size)
target_down = target.voxel_down_sample(voxel_size)

# 进行法线估计
print("-> 正在进行法线估计...")
search_radius = voxel_size * 2  # 搜索半径
max_nn = 30  # 邻居点数量
source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=max_nn))
source_down.orient_normals_towards_camera_location(camera_location=array([0.0, 0.0, 0.0]))
target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=max_nn))
target_down.orient_normals_towards_camera_location(camera_location=array([0.0, 0.0, 0.0]))
# # 调整法线方向一致性（可选）
source_down.orient_normals_consistent_tangent_plane(k=20)
# target_down.orient_normals_consistent_tangent_plane(k=20)
# 设置FPFH特征的搜索半径
radius_feature = voxel_size * 5  # FPFH特征的搜索半径

# 计算FPFH特征
print("-> 计算源点云的FPFH特征...")
source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
print("-> 计算目标点云的FPFH特征...")
target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

# # 使用新的转换矩阵作为初始变换
# trans_init = np.array([
#     [0.988, 0.092, 0.122, 0.320],
#     [0.146, -0.343, -0.928, -1.546],
#     [-0.044, 0.935, -0.352, -1.637],
#     [0.0, 0.0, 0.0, 1.0]
# ])
#
# # 应用初始变换
# source_down.transform(trans_init)
#
# # 可视化应用初始变换后的点云
# o3d.visualization.draw_geometries([source_down, target_down, axis], window_name="应用初始变换后的点云")

# 设置RANSAC的距离阈值
distance_threshold = voxel_size * 1.5

# 使用RANSAC进行粗配准
print("-> 正在进行RANSAC粗配准...")
result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source_down, target_down, source_fpfh, target_fpfh, mutual_filter=True,
    max_correspondence_distance=distance_threshold,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=4,
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 500)
)

# 应用RANSAC获得的粗配准变换
source_down.transform(result_ransac.transformation)
print("RANSAC粗配准完成。")

# 可视化RANSAC粗配准结果，包括全局坐标轴
o3d.visualization.draw_geometries([source_down, target_down, axis], window_name="RANSAC粗配准结果")

# 设置 ICP 的精细配准距离阈值（略小于 RANSAC 配准阈值）
icp_distance_threshold = voxel_size * 0.4

# 执行 ICP 精细配准，使用 Point-to-Plane（点对面）方法
print("-> 正在进行ICP精细配准...")
result_icp = o3d.pipelines.registration.registration_icp(
    source_down, target_down, icp_distance_threshold, result_ransac.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

print("ICP 精细配准完成。")
print("ICP 配准结果变换矩阵：")
print(result_icp.transformation)

# 应用 ICP 获得的精细配准变换
source_down.transform(result_icp.transformation)

# 可视化 ICP 配准结果，包括全局坐标轴
o3d.visualization.draw_geometries([source_down, target_down, axis], window_name="ICP 精细配准结果")
