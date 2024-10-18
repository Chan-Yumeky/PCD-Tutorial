#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import copy

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
    source_temp = copy.deepcopy(source)  # Deep copy to avoid modifying the original
    target_temp = copy.deepcopy(target)
    
    # Set colors for visualization
    source_temp.paint_uniform_color([1, 0.706, 0])  # Yellow for source
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Cyan for target
    
    # Apply the transformation to the source point cloud
    source_temp.transform(transformation)
    
    # Render the point clouds in the Open3D viewer
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def load_point_clouds() -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Load the source and target point clouds from specified file paths.

    Returns:
        tuple: A tuple containing the source and target point clouds.
    """
    source = o3d.io.read_point_cloud("../data/cloud_bin_0.pcd")
    target = o3d.io.read_point_cloud("../data/cloud_bin_1.pcd")
    return source, target

def main():
    # Load point clouds
    source, target = load_point_clouds()

    # Define the registration threshold and initial transformation
    threshold = 0.02
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])

    # Visualize the initial alignment
    draw_registration_result(source, target, trans_init)

    # Perform ICP registration
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation)

    # Point-to-point ICP
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(source, target, reg_p2p.transformation)

    # Increase iterations for better registration
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(source, target, reg_p2p.transformation)

    # Point-to-plane ICP
    print("Apply point-to-plane ICP")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    draw_registration_result(source, target, reg_p2l.transformation)

if __name__ == "__main__":
    main()
