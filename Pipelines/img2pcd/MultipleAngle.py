# 用姿态图估计进行配准的源码，改进版本在Pose.py中

import open3d as o3d
import numpy as np
import copy


# 多视角点云配准
# 多视角配准是在全局空间中对齐多个几何形状的过程。比较有代表性的是，输入是一组几何形状 { P i }
# （可以是点云或者RGBD图像）。输出是一组刚性变换{ T i }
# 变换后的点云 { T i P i }可以在全局空间中对齐。

# 输入
# 第一部分是从三个文件中读取三个点云数据，这三个点云将降采样和可视化，可以看出他们三个是不对齐的。
def load_point_clouds(voxel_size=0.0):
    pcds = []
    for i in range(3):
        pcd = o3d.io.read_point_cloud(f'raw_PCD/test_raw/remain_ground/cloud_bin_{i}.pcd')
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        # 计算法向量
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        if i == 1:
            # -------------------------- 点云旋转 ------------------------
            R2 = pcd_down.get_rotation_matrix_from_xyz((np.pi / 18, np.pi + np.pi / 15, 0))
            print("旋转矩阵：\n", R2)
            pcd_down.rotate(R2)  # 不指定旋转中心
            # -------------------------- 点云平移 ------------------------
            pcd_down.translate((-0.25, 0, -0.3))
        if i == 2:
            # -------------------------- 点云旋转 ------------------------
            R3 = pcd_down.get_rotation_matrix_from_xyz((-np.pi / 2 + np.pi / 36, 0, 0))
            print("旋转矩阵：\n", R3)
            pcd_down.rotate(R3)  # 不指定旋转中心
            # -------------------------- 点云平移 ------------------------
            pcd_down.translate((-0.35, -0.35, 0.68))
        pcds.append(pcd_down)
    return pcds


voxel_size = 0.02
pcds_down = load_point_clouds(voxel_size)
o3d.visualization.draw_geometries(pcds_down)


# 姿态图
# 姿态图有两个关键的基础：节点和边。节点是与姿态矩阵Ti关联的一组几何体Pi,
# 通过该矩阵能够将Pi转换到全局空间。集合{ T i }是一组待优化的未知的变量
# PoseGraph.nodes是PoseGraphNode的列表。我们设P0的空间是全局空间
# 因此T0是单位矩阵。其他的姿态矩阵通过累加相邻节点之间的变换来初始化。相邻节点通常都有着大规模的重叠并且能够通过Point-to-plane ICP来配准。

# 下面的脚本创造了具有三个节点和三个边的姿态图。
# 这些边里，两个是odometry edges（uncertain = False），一个是loop closure edge（uncertain = True）。
def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph


print("Full registration ...")
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(pcds_down,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)
# Open3d使用函数global_optimization进行姿态图估计，可以选择两种类型的优化算法，分别是GlobalOptimizationGaussNewton和GlobalOptimizationLevenbergMarquardt。
# 比较推荐后一种的原因是因为它具有比较好的收敛性。GlobalOptimizationConvergenceCriteria类可以用来设置最大迭代次数和别的优化参数。
# GlobalOptimizationOption定于了两个参数。max_correspondence_distance定义了对应阈值。edge_prune_threshold是修剪异常边缘的阈值。reference_node是被视为全局空间的节点ID。
print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)
# 全局优化在姿态图上执行两次。
# 第一遍将考虑所有边缘的情况优化原始姿态图的姿态，并尽量区分不确定边缘之间的错误对齐。这些错误对齐将会产生小的 line process weights，他们将会在第一遍被剔除。
# 第二遍将会在没有这些边的情况下运行，产生更紧密地全局对齐效果。在这个例子中，所有的边都将被考虑为真实的匹配，所以第二遍将会立即终止。

# 可视化操作
# 使用```draw_geometries``函数可视化变换点云。
print("Transform points and display")
for point_id in range(len(pcds_down)):
    print(pose_graph.nodes[point_id].pose)
    pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
o3d.visualization.draw_geometries(pcds_down)

# 得到合并的点云
# PointCloud是可以很方便的使用+来合并两组点云成为一个整体。
# 合并之后，将会使用voxel_down_sample进行重新采样。建议在合并之后对点云进行后处理，因为这样可以减少重复的点后者较为密集的点。
pcds = load_point_clouds(voxel_size)
pcd_combined = o3d.geometry.PointCloud()
for point_id in range(len(pcds)):
    pcds[point_id].transform(pose_graph.nodes[point_id].pose)
    pcd_combined += pcds[point_id]
pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
# o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
o3d.visualization.draw_geometries([pcd_combined_down])
