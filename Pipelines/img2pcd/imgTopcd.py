import open3d as o3d
import matplotlib.pyplot as plt

color_raw = o3d.io.read_image("./img/cat.jpg")
depth_raw = o3d.io.read_image("./img/cat.jpg")
# 创建一个rgbd图像
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
print(rgbd_image)

# 使用matplotlib显示图像
plt.subplot(1, 2, 1)
plt.title('Redwood grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Redwood depth image')
plt.imshow(rgbd_image.depth)
plt.show()

# rgbd 转 pcd，并显示
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])


# # image的读取与写入
# img = o3d.io.read_image("../test_data/lena_color.jpg")
# print(img)
# '''
# 输出：
# Image of size 512x512, with 3 channels.
# Use numpy.asarray to access buffer data.
# '''
# # 写入(这里是拷贝)一份新的image数据
# o3d.io.write_image("copy_of_lena_color.jpg", img)
