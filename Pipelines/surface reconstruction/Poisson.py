import open3d as o3d
import numpy as np

# --------------------------- 加载点云 ---------------------------
print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud(f"rigidObj/originalCow_normal.ply")
# pcd = o3d.io.read_point_cloud(f"rigidObj/cow_normal.ply")
print("原始点云：", pcd)
o3d.visualization.draw_geometries([pcd])
# ==============================================================

# ------------------------- Ball pivoting --------------------------
print('run Poisson surface reconstruction')
radius = 0.02   # 搜索半径
max_nn = 50         # 邻域内用于估算法线的最大点数
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))     # 执行法线估计

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
print(mesh)
o3d.visualization.draw_geometries([mesh])
# ==============================================================
# output_filename = "./rigidObj/originalCow_normal_rigidObj.ply"
# o3d.io.write_triangle_mesh(output_filename, mesh)