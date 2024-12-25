# 这个版本的图转点云是老师给牛的图片时用的（还没做图片尺寸适配的情况）

import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import os

# 裁剪并缩放RGB图像，使其保持比例为目标尺寸 512x424
def crop_and_resize_rgb(rgb_image, target_width=512, target_height=424):
    h, w, _ = rgb_image.shape
    aspect_ratio_rgb = w / h
    aspect_ratio_target = target_width / target_height

    if aspect_ratio_rgb > aspect_ratio_target:
        new_width = int(h * aspect_ratio_target)
        start_x = (w - new_width) // 2
        rgb_cropped = rgb_image[:, start_x:start_x + new_width]
    else:
        new_height = int(w / aspect_ratio_target)
        start_y = (h - new_height) // 2
        rgb_cropped = rgb_image[start_y:start_y + new_height, :]

    rgb_resized = cv2.resize(rgb_cropped, (target_width, target_height))
    return rgb_resized

# 初始化计数器
pcd_counter = 0

# 检查并创建保存目录
pcd_output_dir = "./raw_PCD/"
if not os.path.exists(pcd_output_dir):
    os.makedirs(pcd_output_dir)

# 读取 RGB 和深度图像
color_raw = cv2.imread("./raw_images/top_rgb.png")  # 使用 OpenCV 读取 RGB 图像
depth_raw = o3d.io.read_image("./raw_images/top_depth.png")  # Open3D 读取深度图像

# 调整 RGB 图像的尺寸以匹配深度图像的分辨率
color_resized = crop_and_resize_rgb(color_raw, target_width=512, target_height=424)

# 将调整后的 RGB 图像转换为 Open3D 图像格式
color_o3d = o3d.geometry.Image(color_resized)

# 创建 RGBD 图像
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_raw)
print(rgbd_image)

# 使用 matplotlib 显示图像
plt.subplot(1, 2, 1)
plt.title('RGB Image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Depth Image')
plt.imshow(rgbd_image.depth)
plt.show()

# RGBD 转换为 PCD，并显示
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# 翻转点云（避免图像倒置）
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# 显示点云数据
o3d.visualization.draw_geometries([pcd])

# 使用计数器生成唯一的文件名
pcd_filename = os.path.join(pcd_output_dir, f"pointcloud_{pcd_counter:04d}.pcd")

# 保存 PCD 点云文件
o3d.io.write_point_cloud(pcd_filename, pcd)
print(f"PCD 点云文件已保存: {pcd_filename}")

# 递增计数器
pcd_counter += 1
