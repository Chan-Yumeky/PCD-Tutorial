#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import copy
import time



def draw_registration_result(source: o3d.geometry.PointCloud, 
                             target: o3d.geometry.PointCloud, 
                             transformation: np.ndarray):
    """
    Visualize the alignment of two point clouds (source and target) after applying 
    a transformation to the source point cloud. This function provides a simple way 
    to assess how well the two point clouds overlap after registration.

    Args:
        source (o3d.geometry.PointCloud): The source point cloud that will be transformed 
                                          and aligned to the target.
        target (o3d.geometry.PointCloud): The target point cloud that remains fixed and is 
                                          used as the reference for alignment.
        transformation (np.ndarray): A 4x4 transformation matrix that defines the rigid 
                                     transformation (rotation + translation) to be applied 
                                     to the source point cloud. This matrix should be of 
                                     shape (4, 4) and in homogeneous coordinates.

    Behavior:
        1. The source point cloud is transformed using the provided transformation matrix.
        2. Both point clouds are visualized in a 3D viewer.
           - The `source` point cloud is colored **yellow**.
           - The `target` point cloud is colored **cyan**.
        3. The function creates temporary copies of the original point clouds to avoid 
           modifying the originals.
        4. The degree of overlap between the source and target point clouds indicates 
           the quality of the registration.

    Example:
        # >>> import open3d as o3d
        # >>> import numpy as np
        # >>> # Load example point clouds
        # >>> source = o3d.io.read_point_cloud("source.pcd")
        # >>> target = o3d.io.read_point_cloud("target.pcd")
        # >>> # Define a transformation matrix
        # >>> transformation = np.array([[0.862, 0.011, -0.507, 0.5],
        #                                [-0.139, 0.967, -0.215, 0.7],
        #                                [0.487, 0.255, 0.835, -1.4],
        #                                [0.0, 0.0, 0.0, 1.0]])
        # >>> # Visualize the registration result
        # >>> draw_registration_result(source, target, transformation)

    Note:
        The transformation matrix should be in homogeneous coordinates, where the
        last row is `[0, 0, 0, 1]`. The first three rows define rotation and translation.
    """
    # Make deep copies of the source and target to avoid modifying the originals
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    # Set colors for the point clouds: yellow for the source and cyan for the target
    # source_temp.paint_uniform_color([1, 0.706, 0])  # Yellow for source
    # target_temp.paint_uniform_color([0, 0.651, 0.929])  # Cyan for target
    
    # Apply the transformation matrix to the source point cloud
    source_temp.transform(transformation)
    
    # Visualize the aligned point clouds
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])


def load_point_clouds() -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Load the source and target point clouds from specified file paths.

    Returns:
        tuple: A tuple containing the source and target point clouds.
    """
    # Load the source and target point clouds from .pcd files
    # source = o3d.io.read_point_cloud("../data/cloud_bin_0.pcd")
    # target = o3d.io.read_point_cloud("../data/cloud_bin_1.pcd")
    source = o3d.io.read_point_cloud("../usablePCD/filtered/chair_front.pcd")
    target = o3d.io.read_point_cloud("../usablePCD/filtered/chair_top.pcd")
    return source, target


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float) -> tuple[o3d.geometry.PointCloud, 
                                                                                     o3d.pipelines.registration.Feature]:
    """
    Downsample the point cloud and compute FPFH features for registration.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        voxel_size (float): The voxel size to use for downsampling.

    Returns:
        tuple: A tuple containing the downsampled point cloud and the computed FPFH features.
    """
    # Downsample the point cloud using the specified voxel size（下采样）
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Estimate normals for the downsampled point cloud（法线估计）
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Compute the Fast Point Feature Histogram (FPFH) for the downsampled point cloud（FPFH提取特征获得初始变换矩阵）
    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,
                                                               o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size: float) -> tuple[o3d.geometry.PointCloud, 
                                                o3d.geometry.PointCloud, 
                                                o3d.geometry.PointCloud, 
                                                o3d.geometry.PointCloud, 
                                                o3d.pipelines.registration.Feature, 
                                                o3d.pipelines.registration.Feature]:
    """
    Load and preprocess the source and target point clouds.

    Args:
        voxel_size (float): The voxel size to use for downsampling.

    Returns:
        tuple: A tuple containing the original source and target point clouds, 
               the downsampled source and target point clouds, and their corresponding FPFH features.
    """
    print(":: Load two point clouds and disturb initial pose.")
    # Load the source and target point clouds
    source, target = load_point_clouds()

    # Disturb the initial pose of the source point cloud by applying a transformation
    trans_init = np.asarray([[0.988, 0.092, 0.122, 0.320],
                             [0.146, -0.343, -0.928, -1.546],
                             [-0.044, 0.935, -0.352, 1.637],
                             [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    
    # Visualize the disturbed source and the target point clouds
    draw_registration_result(source, target, np.identity(4))

    # Preprocess both source and target point clouds (downsampling and FPFH feature extraction)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down: o3d.geometry.PointCloud, 
                                target_down: o3d.geometry.PointCloud, 
                                source_fpfh: o3d.pipelines.registration.Feature, 
                                target_fpfh: o3d.pipelines.registration.Feature, 
                                voxel_size: float) -> o3d.pipelines.registration.RegistrationResult:
    """
    Perform RANSAC-based global registration between two downsampled point clouds.

    Args:
        source_down (o3d.geometry.PointCloud): Downsampled source point cloud.
        target_down (o3d.geometry.PointCloud): Downsampled target point cloud.
        source_fpfh (o3d.pipelines.registration.Feature): FPFH feature of the source point cloud.
        target_fpfh (o3d.pipelines.registration.Feature): FPFH feature of the target point cloud.
        voxel_size (float): Voxel size used for downsampling.

    Returns:
        o3d.pipelines.registration.RegistrationResult: The registration result containing transformation matrix.
    """
    # Define the distance threshold for RANSAC based on voxel size
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    
    # Perform RANSAC-based feature matching for global registration
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    return result


def refine_registration(source: o3d.geometry.PointCloud, 
                        target: o3d.geometry.PointCloud, 
                        voxel_size: float, 
                        result_ransac: o3d.pipelines.registration.RegistrationResult) -> o3d.pipelines.registration.RegistrationResult:
    """
    Refine the registration result using point-to-plane ICP (Iterative Closest Point).

    Args:
        source (o3d.geometry.PointCloud): Original source point cloud.
        target (o3d.geometry.PointCloud): Original target point cloud.
        voxel_size (float): Voxel size used for downsampling.
        result_ransac (o3d.pipelines.registration.RegistrationResult): Initial transformation obtained from RANSAC.

    Returns:
        o3d.pipelines.registration.RegistrationResult: The refined registration result.
    """
    # Define a smaller distance threshold for ICP refinement
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point clouds")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    
    # Perform point-to-plane ICP to refine the alignment
    result = o3d.pipelines.registration.registration_icp(source, 
                                                         target, 
                                                         distance_threshold, 
                                                         result_ransac.transformation, 
                                                         o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    return result


def execute_fast_global_registration(source_down: o3d.geometry.PointCloud, 
                                     target_down: o3d.geometry.PointCloud, 
                                     source_fpfh: o3d.pipelines.registration.Feature, 
                                     target_fpfh: o3d.pipelines.registration.Feature, 
                                     voxel_size: float) -> o3d.pipelines.registration.RegistrationResult:
    """
    Perform Fast Global Registration (FGR) on two downsampled point clouds.

    Args:
        source_down (o3d.geometry.PointCloud): Downsampled source point cloud.
        target_down (o3d.geometry.PointCloud): Downsampled target point cloud.
        source_fpfh (o3d.pipelines.registration.Feature): FPFH feature of the source point cloud.
        target_fpfh (o3d.pipelines.registration.Feature): FPFH feature of the target point cloud.
        voxel_size (float): Voxel size used for downsampling.

    Returns:
        o3d.pipelines.registration.RegistrationResult: The registration result containing transformation matrix.
    """
    # Define the distance threshold for Fast Global Registration (FGR)
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
    
    # Perform Fast Global Registration (FGR) based on feature matching
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold))
    
    return result


def main():
    """
    Main function to demonstrate point cloud registration. It includes:
    1. Global registration using RANSAC.
    2. Local refinement using ICP.
    3. Fast global registration (FGR).
    """
    # Set the voxel size for downsampling
    voxel_size = 0.01  # 1cm for this dataset

    # Load and preprocess the source and target point clouds
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)

    # ------------------------------------
    # RANSAC-based Global Registration
    # ------------------------------------
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)

    # ------------------------------------
    # Local refinement using ICP
    # ------------------------------------
    result_icp = refine_registration(source, target, voxel_size, result_ransac)
    print(result_icp)
    draw_registration_result(source, target, result_icp.transformation)

    # ------------------------------------
    # Fast global registration (FGR)
    # ------------------------------------
    start = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print(result_fast)
    draw_registration_result(source_down, target_down, result_fast.transformation)


if __name__ == "__main__":
    main()
