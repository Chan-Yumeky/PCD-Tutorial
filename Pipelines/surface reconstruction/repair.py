import pymeshfix as mf
import pyvista as pv


def load_mesh(filepath):
    """
    读取并返回网格模型。
    Args:
        filepath (str): 网格文件路径。

    Returns:
        pv.PolyData: 加载的网格模型。
    """
    try:
        mesh = pv.read(filepath)
        mesh.plot()
        print(f"加载网格: {mesh.n_faces_strict} faces")
        return mesh
    except Exception as e:
        print(f"加载网格失败: {e}")
        return None


def visualize_mesh_with_holes(mesh):
    """
    可视化网格及其孔洞。
    Args:
        mesh (pv.PolyData): 原始网格。
    """
    try:
        meshfix = mf.MeshFix(mesh)
        holes = meshfix.extract_holes()

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color=True)
        plotter.add_mesh(holes, color="r", line_width=8)
        plotter.enable_eye_dome_lighting()
        plotter.show()
    except Exception as e:
        print(f"Error visualizing mesh with holes: {e}")


def repair_mesh(mesh, output_filename):
    """
    修复网格孔洞并保存。
    Args:
        mesh (pv.PolyData): 原始网格。
        output_filename (str): 修复后的网格保存路径。

    Returns:
        pv.PolyData: 修复后的网格。
    """
    try:
        meshfix = mf.MeshFix(mesh)
        meshfix.repair(verbose=True)
        repaired_mesh = meshfix.mesh

        repaired_mesh.save(output_filename)
        print(f"修复的网格已保存到{output_filename}")
        return repaired_mesh
    except Exception as e:
        print(f"修复网格失败: {e}")
        return None


def main():
    # 输入文件路径
    input_filepath = "rigidObj/originalCow_normal_rigidObj.ply"
    output_filepath = "rigidObj/originalCow_normal_rigidObj_fixed.ply"

    # 加载网格
    mesh = load_mesh(input_filepath)
    if mesh is None:
        return

    # 可视化网格及其孔洞
    visualize_mesh_with_holes(mesh)

    # 修复网格并保存
    repaired_mesh = repair_mesh(mesh, output_filepath)
    if repaired_mesh is not None:
        repaired_mesh.plot()


if __name__ == "__main__":
    main()
