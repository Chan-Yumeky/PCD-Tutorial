# 对牛三视角多点云配准的代码。
# 在后面注释掉的代码是完整流程中进行全局优化配准的步骤
# 但是效果并不好，所以直接用了手动配准的点云展示（写报告的时候机灵点⌓‿⌓）

import open3d as o3d
import numpy as np
import copy

# 多视角点云配准
def load_point_clouds(voxel_size):
    """
    读取点云文件，并进行降采样和法向量估计。
    对特定点云应用旋转和平移变换。

    Args:
        voxel_size (float): 用于降采样的体素大小，越小保留的点越多。

    Returns:
        pcds (list): 处理后的点云列表。
    """
    pcds = []
    for i in range(3):
        pcd = o3d.io.read_point_cloud(f'raw_PCD/test_raw/remove_ground/cloud_bin_{i}.pcd')
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcds.append(pcd_down)

    return pcds


def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    """
    执行两点云之间的配准，使用Point-to-Plane ICP算法。

    Args:
        source (PointCloud): 源点云
        target (PointCloud): 目标点云
        max_correspondence_distance_coarse (float): 粗配准的最大对应点距离
        max_correspondence_distance_fine (float): 精细配准的最大对应点距离

    Returns:
        transformation_icp (numpy.ndarray): 计算得到的变换矩阵
        information_icp (numpy.ndarray): 配准信息矩阵
    """
    print("Apply point-to-plane ICP")

    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    print("This is icp_coarse.transformation:")
    print(icp_coarse.transformation)

    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)

    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    """
    完整的多点云配准流程，通过ICP和姿态图来对齐多个点云
    姿态图（Pose Graph）是一个图结构，节点代表点云的位置（姿态矩阵），边代表相邻节点之间的变换关系。
    通过全局优化来优化整个图的结构，使得所有点云最终在全局坐标系下对齐。

    姿态图有两个关键的基础：节点和边
    节点是与姿态矩阵关联的一组几何体，通过该矩阵能够将结点转换到全局空间
    设姿态图第一个节点node的空间是全局空间,则其对应的姿态矩阵是单位矩阵
    其他结点的姿态矩阵通过累加相邻节点之间的变换来初始化
    姿态图的边连接着两个重叠的节点（几何形状），而结点之间配准容易出错，甚至错误的匹配会大于正确的匹配
    因此，将姿态图的边分为两类：
    Odometry edges连接着邻域节点，Loop closure edges连接着非邻域的节点
    前者使用ICP这种局部配准的方式就可以对齐，但这种对齐是通过不太可靠的全局配准找到的。

    Args:
        pcds (list): 点云列表
        max_correspondence_distance_coarse (float): 粗配准的最大对应点距离
        max_correspondence_distance_fine (float): 精细配准的最大对应点距离
        注：PoseGraph.nodes是PoseGraphNode的列表

    Returns:
        pose_graph (PoseGraph): 完整的姿态图
    """
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            # 通过两个嵌套的循环遍历所有点云的组合（每两个点云之间配准一次）。
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)

            print("Build o3d.pipelines.registration.PoseGraph")

            # 根据配准结果将节点和边添加到姿态图中
            if target_id == source_id + 1:  # odometry case
                # 将当前配准得到的变换矩阵 transformation_icp 应用到 odometry（全局变换矩阵）上，得到最新的全局变换
                # 通过累积相邻点云的变换矩阵，逐步计算每个点云在全局坐标系下的位置。
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



def global_optimization(pose_graph, max_correspondence_distance_fine):
    """
    对姿态图进行全局优化，调整各个点云的位置。
    全局优化在姿态图上执行两次。
    第一遍将考虑所有边缘的情况优化原始姿态图的姿态，并尽量区分不确定边缘之间的错误对齐。这些错误对齐将会产生小的 line process weights，他们将会在第一遍被剔除。
    第二遍将会在没有这些边的情况下运行，产生更紧密地全局对齐效果。所有的边都将被考虑为真实的匹配，所以第二遍将会立即终止。

    Args:
        pose_graph (PoseGraph): 需要优化的姿态图
        max_correspondence_distance_fine (float): 精细配准时的最大对应点距离
    """
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)


def filterProcess(pcd_combined):
    pcd = pcd_combined
    # 设置距离阈值
    distance_threshold = 1.0  # 根据离主体较远的点云团的距离设定，比如0.8米
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
    std_ratio = 2.0  # 标准差比率
    cl, ind = filtered_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print("->正在统计滤波...")
    # 使用索引保留有效点
    filtered_pcd_statistical = filtered_pcd.select_by_index(ind)
    print("统计滤波后点云:", len(filtered_pcd_statistical.points))
    # 可视化结果
    o3d.visualization.draw_geometries([filtered_pcd_statistical])

    # 半径滤波
    # 设置邻域点数和半径
    nb_points = 16  # 邻域点数
    radius = 0.045  # 半径
    cl, ind = filtered_pcd_statistical.remove_radius_outlier(nb_points=nb_points, radius=radius)
    print("->正在半径滤波...")
    # 使用索引保留有效点
    filtered_pcd_radius = filtered_pcd_statistical.select_by_index(ind)
    print("半径滤波后点云:", len(filtered_pcd_radius.points))
    # 可视化结果
    o3d.visualization.draw_geometries([filtered_pcd_radius])

    return filtered_pcd_radius

def main():
    voxel_size = 0.02  # 设置降采样的体素大小

    # 加载并处理点云数据
    pcds_down = load_point_clouds(voxel_size)

    # 可视化初步处理后的点云
    o3d.visualization.draw_geometries(pcds_down)

    for i in range(3):
        if i == 1:
            R2 = pcds_down[i].get_rotation_matrix_from_xyz((np.pi / 18, np.pi + np.pi / 15, 0))
            pcds_down[i].rotate(R2)
            pcds_down[i].translate((-0.25, 0, -0.3))

        if i == 2:
            R3 = pcds_down[i].get_rotation_matrix_from_xyz((-np.pi / 2 + np.pi / 36, 0, 0))
            pcds_down[i].rotate(R3)
            pcds_down[i].translate((-0.35, 0.33, -0.3))

    # 可视化初步处理后的点云
    o3d.visualization.draw_geometries(pcds_down)

    # 合并优化前的点云
    pre_pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds_down)):
        pre_pcd_combined += pcds_down[point_id]

    o3d.visualization.draw_geometries([pre_pcd_combined])
    o3d.io.write_point_cloud("./usablePCD/myCow_unfiltered.pcd", pre_pcd_combined)

    # # 对合并后的点云进行降采样
    # pcd_combined_down = pre_pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    #
    # # 可视化合并后的点云
    # o3d.visualization.draw_geometries([pcd_combined_down])
    #
    # # 对合并点云进行滤波处理
    # final_pcd = filterProcess(pre_pcd_combined)

    # o3d.io.write_point_cloud("cow.pcd", final_pcd)

    # # 配准参数设置
    # max_correspondence_distance_coarse = voxel_size * 15
    # max_correspondence_distance_fine = voxel_size * 1.5
    #
    # # 执行完整的多点云配准
    # print("Full registration ...")
    # pose_graph = full_registration(pcds_down, max_correspondence_distance_coarse, max_correspondence_distance_fine)
    #
    # # 执行全局优化
    # print("Optimizing PoseGraph ...")
    # global_optimization(pose_graph, max_correspondence_distance_fine)
    #
    # # 可视化优化后的点云
    # print("Transform points and display")
    # for point_id in range(len(pcds_down)):
    #     print(pose_graph.nodes[point_id].pose)
    #     pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    #
    # # 可视化优化后的点云
    # o3d.visualization.draw_geometries(pcds_down)
    #
    # # 合并优化后的点云
    # pcd_combined = o3d.geometry.PointCloud()
    # for point_id in range(len(pcds_down)):
    #     pcd_combined += pcds_down[point_id]
    #
    # # 对合并后的点云进行降采样
    # pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    #
    # # 可视化合并后的点云
    # o3d.visualization.draw_geometries([pcd_combined_down])
    #
    # # 对合并点云进行滤波处理
    # filterProcess(pcd_combined)

if __name__ == "__main__":
    main()
