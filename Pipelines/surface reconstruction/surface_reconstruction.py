import open3d as o3d
import numpy as np

# pcd = o3d.io.read_point_cloud("cow.pcd")
pcd = o3d.io.read_point_cloud("myCow_unfiltered.pcd")
o3d.visualization.draw_geometries([pcd], width=2000, height=2000,
                                  window_name="原始点云",
                                  mesh_show_back_face=False)  # 可视化点云
alpha = 0.04
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)  # 执行单一alpah阈值
mesh.compute_vertex_normals()  # 计算mesh的法线
o3d.visualization.draw_geometries([mesh], window_name="单一alpah阈值的结果",
                                  width=2000, height=2000,
                                  mesh_show_back_face=True)

output_filename = "./rigidObj/myCow_unfiltered(Alpha).ply"
o3d.io.write_triangle_mesh(output_filename, mesh)
