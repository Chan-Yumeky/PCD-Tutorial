# 算法会基于 法线方向 或 顶点密度 来判断是否填补某个区域。
# 如果地面上的点密度较低或法线方向与周围区域有差异，算法可能认为地面是一个无效区域或不需要修复的部分。

import pymeshfix as mf
import pyvista as pv
# 读取并展示原始网格模型
# cow = pv.read("rigidObj/myCow(12-3).ply")
# cow = pv.read("rigidObj/originalCow(12-0).ply")
# cow = pv.read("rigidObj/myCow(Alpha).ply")
# cow = pv.read("rigidObj/myCow_unfiltered(Alpha).ply")
cow = pv.read("rigidObj/originalCow_normal_rigidObj.ply")
# cow = pv.read("../img2pcd/model.ply")
# cow = pv.read("rigidObj/originalCow(8-0).ply")
cow.plot()
# 使用 pymeshfix 进行孔洞提取和修复
meshfix = mf.MeshFix(cow)
holes = meshfix.extract_holes()
# 可视化原始网格模型和孔洞
p = pv.Plotter()
p.add_mesh(cow, color=True)
p.add_mesh(holes, color="r", line_width=8)
p.enable_eye_dome_lighting()
p.show()
# 使用 pymeshfix 修复网格
meshfix.repair(verbose=True)
# 提取修复后的网格模型
meshfix.mesh.plot()

# 保存修复完成后的三维点云模型
output_filename = "./rigidObj/originalCow_normal_rigidObj_fixed.ply"
# output_filename = "../img2pcd/model_fixed.ply"
meshfix.mesh.save(output_filename)
# # 加载修复完成后的三维点云模型并可视化
# repaired_cow = pv.read(output_filename)
# repaired_cow.plot()


# holes1 = meshfix.show_feature_edges()
# p1 = pv.Plotter()
# p1.add_mesh(cow, color=True)
# p1.add_mesh(holes, color="g", line_width=8)
# p1.enable_eye_dome_lighting()
# p1.show()