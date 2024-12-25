import open3d as o3d
import open3d.cpu.pybind.core as o3c
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import math
import copy


def makePCD(png, intrinsics, rotate, move):
    # 读取深度图
    # tum_data = o3d.data.SampleTUMRGBDImage()
    depth = o3d.t.io.read_image(png)

    # OPENGL = o3c.Tensor([[1, 0, 0, 0],
    #                    [0, -1, 0, 0],
    #                    [0, 0, -1, 0],
    #                    [0, 0, 0, 1]])
    # left [6.2,0,1]

    extrinsics = o3c.Tensor([[1, 0, 0, 0],
                             [0, 0, -1, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]])

    # 生成点云
    pcd = o3d.t.geometry.PointCloud.create_from_depth_image(
        depth=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        depth_scale=1000.0,
        depth_max=3.5
    )
    cl, ind = pcd.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_mask(ind)

    pcd.estimate_normals(max_nn=20)
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))

    r = Rotation.from_euler("xyz", rotate, degrees=True)
    trs = np.eye(4)
    trs[:3, :3] = r.as_matrix()
    trs[:3, 3] = move
    pcd.transform(trs)

    return pcd
    # o3d.io.write_point_cloud('left.ply',pcd.to_legacy())


if __name__ == "__main__":

    # sample_ply_data = o3d.data.DemoCropPointCloud()

    # 设置相机内参
    fx = 364.032
    fy = 364.032
    cx = 258.891
    cy = 209.32

    intrinsic = o3c.Tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]])

    # left = makePCD('./raw_images/left_depth.png',intrinsic,[-6.2,0,-1])
    # right = makePCD('./raw_images/right_depth.png',intrinsic,[-2.5,0,1.2])
    # top = makePCD('./raw_images/top_depth.png',intrinsic,[0,0,0])
    # [-4,0,-1],[0.331151,-2.45074,1.00474]
    left = makePCD('./raw_images/left_depth.png', intrinsic, [-3.9, 0.53, 0], [0.438113, -2.42936, 0.979995])
    right = makePCD('./raw_images/right_depth.png', intrinsic, [-2.63, -1.23, -168.44], [-0.327399, 2.17402, 0.884261])
    top = makePCD('./raw_images/top_depth.png', intrinsic, [-86.72, 1.42, 2.55], [-0.001563, -0.130949, 2.973])

    # top
    points = top.point.positions.numpy()
    indices = np.where((points[:, 2] >= 0.5))[0]
    top = top.select_by_index(indices)

    maxZ = np.max(points[:, 2])

    print('maxZ', maxZ)

    # 左
    points = left.point.positions.numpy()
    indices = np.where((points[:, 2] >= 0.00) & (points[:, 2] < maxZ))[0]
    left = left.select_by_index(indices)

    points = right.point.positions.numpy()
    indices = np.where((points[:, 2] >= 0.00) & (points[:, 2] < maxZ))[0]
    right = right.select_by_index(indices)

    pcd = left.append(right)

    normals = pcd.point.normals.numpy()
    indices = np.where((normals[:, 2] < 0.6))[0]

    # print("indices",indices)
    # down1 = down.select_by_index(indices)

    pcd = pcd.select_by_index(indices)
    pcd = pcd.append(top)

    cl, ind = pcd.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)

    o3d.io.write_point_cloud('pcd.ply', pcd.to_legacy())
    # o3d.io.write_point_cloud('left.ply',left.to_legacy())
    # o3d.io.write_point_cloud('right.ply',right.to_legacy())
    # o3d.io.write_point_cloud('top.ply',top.to_legacy())

    maxX = np.max(points[:, 0])
    minX = np.min(points[:, 0])

    print('minX', minX, maxX)

    i = 0

    pcd1 = o3d.t.geometry.PointCloud()
    while (minX < maxX):
        # box = o3d.geometry.OrientedBoundingBox.create_from_points()
        points = pcd.point.positions.numpy()
        indices = np.where((points[:, 0] >= minX) & (points[:, 0] < minX + 0.1) & (points[:, 1] < 1))[0]
        # print('len',minX,indices,len(points))
        if (len(indices) > 0):
            m = pcd.select_by_index(indices)
            pcd = pcd.select_by_index(indices, invert=True)

            p = m.point.positions.numpy()
            avgZ = np.mean(p[:, 2])

            print('avg', i, avgZ)

            if (avgZ > 0.1):
                if (pcd1.is_empty()):
                    pcd1 = m
                else:
                    pcd1 = pcd1.append(m)
            # o3d.io.write_point_cloud(f'm_{i}.ply',m.to_legacy())
        minX += 0.1
        i += 1
    o3d.io.write_point_cloud('pcd1.ply', pcd1.to_legacy())

    # 泊松
    poisson_mesh = \
    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd1.to_legacy(), depth=8, width=0, scale=1.1,
                                                              linear_fit=False)[0]

    # volume = poisson_mesh.get_volume()
    triangle_clusters, cluster_n_triangles, cluster_area = (poisson_mesh.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    print('cluster_n_triangles', cluster_n_triangles)
    mesh = copy.deepcopy(poisson_mesh)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < 30000
    mesh.remove_triangles_by_mask(triangles_to_remove)

    mesh.remove_vertices_by_index(mesh.get_non_manifold_vertices())
    mesh = mesh.remove_non_manifold_edges()

    hull, _ = mesh.compute_convex_hull()
    volume = hull.get_volume()

    o3d.io.write_triangle_mesh('model.ply', mesh, write_vertex_colors=False)
    o3d.io.write_triangle_mesh('hull.ply', hull, write_vertex_colors=False)

    print('体积', volume)

# 可视化点云
# o3d.visualization.draw([pcd])
