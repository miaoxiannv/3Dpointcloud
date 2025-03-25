import open3d as o3d
import numpy as np
import json
import matplotlib.pyplot as plt

def load_and_process_pointcloud(json_path):
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    points_array = data["position"]
    intensity_array = data["intensity"]
    points = np.array(points_array).reshape(-1, 3)
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 地面分割
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.3,
                                           ransac_n=3,
                                           num_iterations=1000)
    
    # 分离地面点云和非地面点云
    ground_cloud = pcd.select_by_index(inliers)
    non_ground_cloud = pcd.select_by_index(inliers, invert=True)
    
    # 对非地面点云进行聚类分割
    labels = np.array(non_ground_cloud.cluster_dbscan(eps=0.5, min_points=10))
    max_label = labels.max()
    
    # 为不同类别设置不同颜色
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    non_ground_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    # 设置地面点云为灰色
    ground_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    
    return ground_cloud, non_ground_cloud, labels

def visualize_segmented_pointcloud(ground_cloud, non_ground_cloud):
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 添加点云
    vis.add_geometry(ground_cloud)
    vis.add_geometry(non_ground_cloud)
    
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0, 0, 0])
    
    # 设置默认视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

def main():
    ground_cloud, non_ground_cloud, labels = load_and_process_pointcloud('data.json')
    
    # 打印分割结果统计
    print(f"地面点云数量: {len(ground_cloud.points)}")
    print(f"检测到的物体数量: {len(np.unique(labels))}")
    
    # 可视化
    visualize_segmented_pointcloud(ground_cloud, non_ground_cloud)

if __name__ == "__main__":
    main()