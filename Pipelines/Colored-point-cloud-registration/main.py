#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import copy


def draw_registration_result_original_color(source: o3d.geometry.PointCloud, 
                                            target: o3d.geometry.PointCloud, 
                                            transformation: np.ndarray):
    """
    Visualizes the source and target point clouds after applying the given transformation to the source.

    Args:
        source (o3d.geometry.PointCloud): The source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.
        transformation (np.ndarray): A 4x4 transformation matrix to align the source to the target.
    """
    # Deep copy of the source to avoid modifying the original point cloud
    source_temp = copy.deepcopy(source)
    
    # Apply the transformation matrix to the source
    source_temp.transform(transformation)
    
    # Visualize the transformed source and target point clouds in 3D
    o3d.visualization.draw_geometries([source_temp, target],
                                      zoom=0.5,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7745, 2.2305, 0.9787],
                                      up=[0.3109, -0.5878, -0.7468])


def load_point_clouds() -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Load the source and target point clouds from specified file paths.

    Returns:
        tuple: A tuple containing the source and target point clouds.
    """
    # Load the source point cloud from a .ply file
    source = o3d.io.read_point_cloud("../data/frag_115.ply")
    target = o3d.io.read_point_cloud("../data/frag_116.ply")

    return source, target


def main():
    # -------------------
    # Load point clouds
    # -------------------
    print("1. Load two point clouds and show initial pose")
    
    # Load the source and target point clouds
    source, target = load_point_clouds()
    
    # Identity transformation matrix (no initial alignment)
    current_transformation = np.identity(4)
    
    # Draw the initial alignment of the two point clouds
    draw_registration_result_original_color(source, target, current_transformation)


    # -------------------
    # Point-to-plane ICP
    # -------------------
    print("2. Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. Distance threshold 0.02.")
    
    # Apply Point-to-Plane ICP for registration with a distance threshold of 0.02
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, 0.02, current_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    # Output the result of ICP registration
    print(result_icp)
    
    # Draw the point clouds after Point-to-Plane ICP registration
    draw_registration_result_original_color(source, target, result_icp.transformation)
    

    # -------------------
    # Colored point cloud registration
    # -------------------
    # Implementation based on the paper:
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # "Colored Point Cloud Registration Revisited", ICCV 2017
    voxel_radius = [0.04, 0.02, 0.01]  # Different scales for multi-resolution registration
    max_iter = [50, 30, 14]            # Max iterations for each scale
    current_transformation = np.identity(4)  # Reset transformation to identity
    
    print("3. Colored point cloud registration")
    
    # Iterate over multiple scales for coarse-to-fine registration
    for scale in range(3):
        iter = max_iter[scale]         # Max iterations for the current scale
        radius = voxel_radius[scale]   # Voxel size for downsampling
        
        print([iter, radius, scale])

        # 3-1. Downsample the source and target point clouds using the voxel size
        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        # 3-2. Estimate normals for the downsampled point clouds
        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        # 3-3. Apply colored ICP for fine registration using colors and geometry
        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
        
        # Update the transformation with the result from the current scale
        current_transformation = result_icp.transformation
        print(result_icp)
    
    # Draw the final alignment after multi-scale colored ICP
    draw_registration_result_original_color(source, target, result_icp.transformation)


if __name__ == "__main__":
    main()
