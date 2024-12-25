# 这个版本的图转点云是自己拍摄RGB图和深度图情况下使用的（获取的时候已经裁剪过并进行等比缩放适配）

import open3d as o3d
import matplotlib.pyplot as plt
import os

# 初始化计数器
pcd_counter = 0

# 检查并创建保存目录
pcd_output_dir = "./raw_PCD/"
if not os.path.exists(pcd_output_dir):
    os.makedirs(pcd_output_dir)

# 读取 RGB 和深度图像
color_raw = o3d.io.read_image("./raw_images/left_rgb.png")
depth_raw = o3d.io.read_image("./raw_images/left_depth.png")

# 创建一个 RGBD 图像
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
print(rgbd_image)

# 使用 matplotlib 显示图像
plt.subplot(1, 2, 1)
plt.title('RGB Image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Depth Image')
plt.imshow(rgbd_image.depth)
plt.show()

# RGBD 转 PCD，并显示
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# 进行点云数据的翻转操作（避免图像倒置）
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# 使用计数器生成唯一的文件名
pcd_filename = os.path.join(pcd_output_dir, f"pointcloud_{pcd_counter:04d}.pcd")

# 保存 PCD 点云文件
# o3d.io.write_point_cloud(pcd_filename, pcd)
# print(f"PCD 点云文件已保存: {pcd_filename}")

# 显示点云数据
o3d.visualization.draw_geometries([pcd])

# 递增计数器
# pcd_counter += 1
