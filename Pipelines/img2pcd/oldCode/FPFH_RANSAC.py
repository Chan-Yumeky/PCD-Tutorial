import open3d as o3d
import numpy as np
import copy

# 加载点云
source = o3d.io.read_point_cloud("../raw_PCD/test_raw/cloud_bin_1.pcd")
target = o3d.io.read_point_cloud("../raw_PCD/test_raw/cloud_bin_0.pcd")

source_temp = copy.deepcopy(source)
target_temp = copy.deepcopy(target)

o3d.visualization.draw_geometries([source_temp, target_temp], window_name="初始点云")

# -------------------------- 点云空间变换 ------------------------
print("\n->采用欧拉角进行点云旋转")
R1 = source_temp.get_rotation_matrix_from_xyz((np.pi/18, np.pi + np.pi/15, 0))
print("旋转矩阵：\n", R1)
source_temp.rotate(R1)    # 不指定旋转中心
print("\n质心：", source_temp.get_center())
source_temp.translate((-0.25, 0, -0.45))
print(source_temp)
print(f'pcd_tx质心：{source_temp.get_center()}')

# 可视化平移后的点云
o3d.visualization.draw_geometries([source_temp, target_temp], window_name="空间变换后的点云")

# # 使用新的转换矩阵作为初始变换
# trans_init = np.array([[1, 0, 0, -0.25],
#                        [0, 1, 0, 0.0],
#                        [0, 0, 1, -0.45],
#                        [0, 0, 0, 1.0]
#                        ])
#
# # 应用初始变换
# source_temp.transform(trans_init)
#
# # 可视化应用初始变换后的点云
# o3d.visualization.draw_geometries([source_temp, target_temp], window_name="应用初始变换后的点云")

# 下采样
print("->正在体素下采样...")
voxel_size = 0.02
source_down = source_temp.voxel_down_sample(voxel_size)
target_down = target_temp.voxel_down_sample(voxel_size)
print("->正在可视化下采样点云")

# 进行法线估计
search_radius = voxel_size * 2  # 搜索半径
max_nn = 30  # 邻居点数量
source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=max_nn))
target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=max_nn))

# 可视化带法线的点云
print("-> 可视化法线估计结果...")
o3d.visualization.draw_geometries([source_down, target_down], window_name="下采样、法线估计后的点云",
                                  point_show_normal=False)

# 设置FPFH特征的搜索半径
radius_feature = voxel_size * 5  # FPFH特征的搜索半径
# 计算FPFH特征
print("-> 计算源点云的FPFH特征...")
source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
print("-> 计算目标点云的FPFH特征...")
target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
# 设置RANSAC的距离阈值
distance_threshold = voxel_size * 1.5
# 使用RANSAC进行粗配准
print("-> 正在进行RANSAC粗配准...")
result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source_down, target_down, source_fpfh, target_fpfh, mutual_filter=True,
    max_correspondence_distance=distance_threshold,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=3,
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
)
# 应用RANSAC获得的粗配准变换
source_down.transform(result_ransac.transformation)
print("RANSAC粗配准完成。")
# 可视化粗配准结果
o3d.visualization.draw_geometries([source_down, target_down], window_name="RANSAC粗配准结果")


# 设置 ICP 的精细配准距离阈值（略小于 RANSAC 配准阈值）
icp_distance_threshold = 0.02
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
# 可视化 ICP 配准结果
o3d.visualization.draw_geometries([source_down, target_down], window_name="ICP 精细配准结果")
