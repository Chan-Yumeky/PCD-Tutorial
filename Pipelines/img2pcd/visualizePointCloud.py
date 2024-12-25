import open3d as o3d

# pcd = o3d.io.read_point_cloud("usablePCD/filtered_originalCow.ply")
# pcd = o3d.io.read_point_cloud("../surface reconstruction/cow_normal.ply")
# print(pcd)
# o3d.visualization.draw_geometries([pcd])

# mesh = o3d.io.read_triangle_mesh('../surface reconstruction/originalCow(8-0).ply')
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
#
# mesh = o3d.io.read_triangle_mesh('../surface reconstruction/originalCow(12-0).ply')
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
#
# mesh = o3d.io.read_triangle_mesh('../surface reconstruction/myCow(12-0).ply')
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
#
# mesh = o3d.io.read_triangle_mesh('../surface reconstruction/myCow(12-1).ply')
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

mesh = o3d.io.read_triangle_mesh('../surface reconstruction/rigidObj/testMyCow.ply')
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)