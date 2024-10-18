#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import copy
import time


def draw_registration_result_with_bounding_box(source: o3d.geometry.PointCloud,
                                               target: o3d.geometry.PointCloud,
                                               transformation: np.ndarray):
    """
    Visualize the alignment of two point clouds (source and target) after applying
    a transformation to the source point cloud. The function also visualizes the
    oriented bounding box (OBB) for each point cloud.
    """
    # Make deep copies of the source and target to avoid modifying the originals
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    # Set colors for the point clouds: yellow for the source and cyan for the target
    source_temp.paint_uniform_color([1, 0.706, 0])  # Yellow for source
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Cyan for target

    # Apply the transformation matrix to the source point cloud
    source_temp.transform(transformation)

    # Compute Oriented Bounding Box (OBB) for both point clouds
    obb_source = source_temp.get_oriented_bounding_box()
    obb_source.color = (1, 0, 0)  # Red for source bounding box
    obb_target = target_temp.get_oriented_bounding_box()
    obb_target.color = (0, 1, 0)  # Green for target bounding box

    # Visualize the aligned point clouds with their bounding boxes
    o3d.visualization.draw_geometries([source_temp, target_temp, obb_source, obb_target],
                                      zoom=0.4459,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])


def load_point_clouds() -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Load the source and target point clouds from specified file paths.
    """
    # Load the source and target point clouds from .pcd files
    source = o3d.io.read_point_cloud("../data/cloud_bin_0.pcd")
    target = o3d.io.read_point_cloud("../data/cloud_bin_1.pcd")

    return source, target


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float) -> tuple[o3d.geometry.PointCloud,
o3d.pipelines.registration.Feature]:
    """
    Downsample the point cloud and compute FPFH features for registration.
    """
    # Downsample the point cloud using the specified voxel size
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Estimate normals for the downsampled point cloud
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Compute the Fast Point Feature Histogram (FPFH) for the downsampled point cloud
    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,
                                                               o3d.geometry.KDTreeSearchParamHybrid(
                                                                   radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size: float) -> tuple[o3d.geometry.PointCloud,
o3d.geometry.PointCloud,
o3d.geometry.PointCloud,
o3d.geometry.PointCloud,
o3d.pipelines.registration.Feature,
o3d.pipelines.registration.Feature]:
    """
    Load and preprocess the source and target point clouds.
    """
    print(":: Load two point clouds and disturb initial pose.")
    # Load the source and target point clouds
    source, target = load_point_clouds()

    # Disturb the initial pose of the source point cloud by applying a transformation
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)

    # Visualize the disturbed source and the target point clouds
    draw_registration_result_with_bounding_box(source, target, np.identity(4))

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
    Main function to demonstrate point cloud registration with bounding box display.
    """
    # Set the voxel size for downsampling
    voxel_size = 0.05  # 5cm for this dataset

    # Load and preprocess the source and target point clouds
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)

    # ------------------------------------
    # RANSAC-based Global Registration
    # ------------------------------------
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print(result_ransac)
    draw_registration_result_with_bounding_box(source, target, result_ransac.transformation)

    # ------------------------------------
    # Refine the registration result using ICP
    # ------------------------------------
    result_icp = refine_registration(source, target, voxel_size, result_ransac)
    print(result_icp)
    draw_registration_result_with_bounding_box(source, target, result_icp.transformation)


if __name__ == "__main__":
    main()
