# 尝试多点云配准的代码，可以不管

import copy
import numpy as np
import open3d as o3d

print("->正在加载点云1... ")
pcd1 = o3d.io.read_point_cloud("raw_PCD/test_raw/cloud_bin_0.pcd")
print(pcd1)
print("->正在加载点云2...")
pcd2 = o3d.io.read_point_cloud("raw_PCD/test_raw/cloud_bin_1.pcd")
print("->正在加载点云2...")
print(pcd2)
pcd3 = o3d.io.read_point_cloud("raw_PCD/test_raw/cloud_bin_2_new.pcd")
print(pcd3)
print("->正在同时可视化两个点云...")
o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])

pcd1_temp = copy.deepcopy(pcd1)
pcd2_temp = copy.deepcopy(pcd2)
pcd3_temp = copy.deepcopy(pcd3)
# -------------------------- 点云旋转 ------------------------
print("\n->采用欧拉角进行点云旋转")
R2 = pcd2_temp.get_rotation_matrix_from_xyz((np.pi/18, np.pi + np.pi/15, 0))
print("旋转矩阵：\n", R2)
pcd2_temp.rotate(R2)    # 不指定旋转中心
print("\n质心：", pcd2_temp.get_center())
# o3d.visualization.draw_geometries([pcd1_temp, pcd2_temp, pcd3_temp])

# -------------------------- 点云平移 ------------------------
print("\n->沿Z轴平移1m")
pcd2_temp.translate((-0.25, 0, -0.45))
print(pcd2_temp)
print(f'pcd_tx质心：{pcd2_temp.get_center()}')
# o3d.visualization.draw_geometries([pcd1_temp, pcd2_temp])

# -------------------------- 点云旋转 ------------------------
print("\n->采用欧拉角进行点云旋转")
R3 = pcd3_temp.get_rotation_matrix_from_xyz((-np.pi/2 + np.pi/36, 0, 0))
print("旋转矩阵：\n", R3)
pcd3_temp.rotate(R3)    # 不指定旋转中心
print("\n->pcd3_temp质心：", pcd3_temp.get_center())
# o3d.visualization.draw_geometries([pcd1_temp,pcd3_temp])

# -------------------------- 点云平移 ------------------------
# pcd3_temp.translate((-0.35, -0.3, 0.6))
pcd3_temp.translate((-0.35, 0.4, -0.3))
print(pcd3_temp)
print(f'pcd3_temp质心：{pcd3_temp.get_center()}')
o3d.visualization.draw_geometries([pcd1_temp, pcd3_temp])

# o3d.visualization.draw_geometries([pcd2_temp, pcd3_temp])
o3d.visualization.draw_geometries([pcd1_temp, pcd2_temp, pcd3_temp])