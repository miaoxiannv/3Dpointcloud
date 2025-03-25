import open3d as o3d
import numpy as np
import json

def load_json_to_pointcloud(json_path):
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 获取position和intensity数组
    points_array = data["position"]
    intensity_array = data["intensity"]

    # 将一维数组重塑为Nx3的数组，每行代表一个点的(x,y,z)坐标
    points = np.array(points_array).reshape(-1, 3)

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 根据intensity设置颜色
    # 将intensity值归一化到0-1之间
    intensity_normalized = np.array(intensity_array)
    intensity_normalized = (intensity_normalized - np.min(intensity_normalized)) / (np.max(intensity_normalized) - np.min(intensity_normalized))

    # 使用jet颜色映射：低强度为蓝色，高强度为红色
    colors = np.zeros((len(points), 3))
    colors[:, 0] = intensity_normalized  # R通道
    colors[:, 2] = 1 - intensity_normalized  # B通道

    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def main():
    # 加载JSON数据并转换为点云
    pcd = load_json_to_pointcloud('data.json')

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加点云到可视化器
    vis.add_geometry(pcd)

    # 设置渲染选项
    opt = vis.get_render_option()
    opt.point_size = 2.0  # 设置点的大小
    opt.background_color = np.asarray([0, 0, 0])  # 设置背景为黑色

    # 运行可视化
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main() 