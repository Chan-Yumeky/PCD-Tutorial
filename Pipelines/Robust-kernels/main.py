#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import copy


# ---------------------------------------------
#       Function to Visualize Point Clouds 
# ---------------------------------------------
def draw_registration_result(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, transformation: np.ndarray):
    """
    Visualizes the alignment of two point clouds (source and target).
    
    Args:
        source (open3d.geometry.PointCloud): The source point cloud.
        target (open3d.geometry.PointCloud): The target point cloud.
        transformation (numpy.ndarray): A 4x4 transformation matrix applied to the source point cloud.
        
    This function paints the source point cloud in orange and the target in blue, 
    and applies the given transformation to the source for visualization.
    """
    source_temp = copy.deepcopy(source)  # Create a copy of the source point cloud
    target_temp = copy.deepcopy(target)  # Create a copy of the target point cloud
    source_temp.paint_uniform_color([1, 0.706, 0])  # Color source point cloud orange
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Color target point cloud blue
    source_temp.transform(transformation)  # Apply transformation to source

    # Visualize the transformed source and target point clouds
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


# ---------------------------------------------
#       Function to Add Noise to Point Cloud 
# ---------------------------------------------
def apply_noise(pcd: o3d.geometry.PointCloud, mu: float, sigma: float) -> o3d.geometry.PointCloud:
    """
    Adds Gaussian noise to the points of a point cloud.
    
    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud to which noise will be added.
        mu (float): The mean of the Gaussian noise.
        sigma (float): The standard deviation of the Gaussian noise.
    
    Returns:
        open3d.geometry.PointCloud: The noisy point cloud.
    """
    noisy_pcd = copy.deepcopy(pcd)  # Create a copy of the input point cloud
    points = np.asarray(noisy_pcd.points)  # Extract point cloud data as a numpy array

    # Add Gaussian noise to the points
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)  # Set the noisy points back to the point cloud

    return noisy_pcd


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
    # ---------------------------------------------
    #       Load Example Point Clouds 
    # ---------------------------------------------
    # Load point clouds
    source, target = load_point_clouds()

    # Initial transformation matrix for alignment
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                            [-0.139, 0.967, -0.215, 0.7],
                            [0.487, 0.255, 0.835, -1.4], 
                            [0.0, 0.0, 0.0, 1.0]])

    # Visualize the initial alignment using the initial transformation matrix
    draw_registration_result(source, target, trans_init)

    # ---------------------------------------------
    #       Add Noise to Source Point Cloud 
    # ---------------------------------------------
    # Add Gaussian noise to the source point cloud
    mu, sigma = 0, 0.1  # Mean (mu) and standard deviation (sigma) of the noise
    source_noisy = apply_noise(source, mu, sigma)  # Apply noise to the source

    # Visualize the noisy source point cloud
    print("Source PointCloud + noise:")
    o3d.visualization.draw_geometries([source_noisy],
                                    zoom=0.4459,
                                    front=[0.353, -0.469, -0.809],
                                    lookat=[2.343, 2.217, 1.809],
                                    up=[-0.097, -0.879, 0.467])

    # ---------------------------------------------
    #       Vanilla ICP: Small Threshold 
    # ---------------------------------------------
    # Vanilla ICP: Point-to-plane ICP with a small threshold
    threshold = 0.02  # Maximum allowable distance between correspondences for ICP
    print("Vanilla point-to-plane ICP, threshold={}:".format(threshold))

    # Create a point-to-plane ICP object (basic version)
    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    # Perform ICP registration with the noisy source point cloud
    reg_p2l = o3d.pipelines.registration.registration_icp(source_noisy, target,
                                                        threshold, trans_init,
                                                        p2l)

    # Output ICP results
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)

    # Visualize the result after applying the transformation from ICP
    draw_registration_result(source, target, reg_p2l.transformation)

    # ---------------------------------------------
    #       Tuning ICP: Larger Threshold 
    # ---------------------------------------------
    # Tuning ICP: Using a larger threshold for a looser ICP
    threshold = 1.0  # Increase threshold for ICP
    print("Vanilla point-to-plane ICP, threshold={}:".format(threshold))

    # Perform ICP registration again with a higher threshold
    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    reg_p2l = o3d.pipelines.registration.registration_icp(source_noisy, target,
                                                        threshold, trans_init,
                                                        p2l)

    # Output ICP results with the higher threshold
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)

    # Visualize the result with the looser threshold
    draw_registration_result(source, target, reg_p2l.transformation)

    # ---------------------------------------------
    #       Robust ICP: Using Tukey Loss 
    # ---------------------------------------------
    # Robust ICP: Using a robust loss function to handle outliers
    print("Robust point-to-plane ICP, threshold={}:".format(threshold))

    # Use a Tukey loss function for robust registration to better handle noise/outliers
    loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
    print("Using robust loss:", loss)

    # Apply robust loss in point-to-plane ICP
    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

    # Perform robust ICP registration
    reg_p2l = o3d.pipelines.registration.registration_icp(source_noisy, target,
                                                        threshold, trans_init,
                                                        p2l)

    # Output results for the robust ICP
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)

    # Visualize the result after applying robust ICP transformation
    draw_registration_result(source, target, reg_p2l.transformation)


if __name__ == "__main__":
    main()
