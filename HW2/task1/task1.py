import os
import numpy as np
import open3d as o3d
import datetime


def icp_core(points_ref, points_newscan):
    """
    Solve transformation from points_newscan to points_ref, T1_2
    :param points_ref: numpy array, size = n x 3, n is num of point
    :param points_newscan: numpy array, size = n x 3, n is num of point
    :return: transformation matrix T, size = 4x4
    
    Note: point cloud should be in same size. Point with same index are corresponding points.
          For example, points_ref[i] and points_newscan[i] are a pair of cooresponding points.
    
    """
    assert points_ref.shape == points_newscan.shape, 'point cloud size not match'
    
    T1_2 = np.eye(4)

    # Compute centroids of points_ref and points_newscan
    centroid_ref = np.average(points_ref, axis=0)
    centroid_newscan = np.average(points_newscan, axis=0)

    # Zero-centering points_ref and points_newscan
    zero_ctr_pts_ref = points_ref - centroid_ref
    zero_ctr_pts_newscan = points_newscan - centroid_newscan

    # Define matrix H for which the rotation matrix (R) to be obtained by maximising Trace(RH)
    H = zero_ctr_pts_newscan.T @ zero_ctr_pts_ref

    # Get the rotation matrix (R) from the resulting components of SVD on matrix H
    U, S, Vh = np.linalg.svd(H)
    R = Vh.T @ U.T

    # Get the translation vector (t)
    t = centroid_ref - R @ centroid_newscan

    T1_2[:3, :3] = R
    T1_2[:3, 3] = t
    
    return T1_2


def solve_icp_with_known_correspondence(points_ref, points_newscan):
    # Solve for transformation matrix
    T1_2 = icp_core(points_ref, points_newscan)
    print('------------ transformation matrix T1_2 ------------')
    print(T1_2)

    # Make the points_newscan into homogeneous coordinates and apply the transformation T1_2
    n_row = points_newscan.shape[0]
    pts_newscan_h = np.hstack((points_newscan, np.ones((n_row, 1)))).T
    points_newscan_transformed = T1_2 @ pts_newscan_h
    points_newscan_transformed = points_newscan_transformed.T[:, :3]

    # Visualization
    mean_distance = mean_dist(points_newscan_transformed, points_ref)
    print('mean_error= ' + str(mean_distance))

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points_ref)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points_newscan)
    pcd2_transformed = o3d.geometry.PointCloud()
    pcd2_transformed.points = o3d.utility.Vector3dVector(points_newscan_transformed)
    
    pcd1.paint_uniform_color([1, 0, 0])  # Red for reference cloud
    pcd2.paint_uniform_color([0, 1, 0])  # Green for original cloud
    pcd2_transformed.paint_uniform_color([0, 0, 1])  # Blue for transformed cloud
    
    o3d.visualization.draw_geometries([pcd1, pcd2, pcd2_transformed, axis_pcd])


def solve_icp_without_known_correspondence(points_ref, points_newscan, n_iter, threshold):
    points_newscan_temp = points_newscan.copy()
    T_1_2accumulated = np.eye(4)

    # viz
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points_ref)
    pcd1.paint_uniform_color([0, 0, 1])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(axis_pcd)
    vis.add_geometry(pcd1)
    
    total_time_cost = 0

    for i in range(n_iter):
        start_time = datetime.datetime.now()

        # Use KD-tree for more efficient searching of nearest neighbour
        points_ref_kdtree = o3d.geometry.KDTreeFlann(pcd1)
        points_ref_corresponded = np.zeros_like(points_ref)

        # Transform the new scan point cloud such that its centroid aligns with that of reference point cloud
        # Reference of this approach:
        #   - YouTube video: ICP & Point Cloud Registration - Part 2: Unknown Data Association (Cyrill Stachniss, 2021)
        #   - URL: https://youtu.be/ktRqKxddjJk?t=1645
        centroid_ref = np.average(points_ref, axis=0)
        centroid_newscan = np.average(points_newscan_temp, axis=0)
        points_newscan_centroid_ref_aligned = points_newscan_temp - centroid_newscan + centroid_ref

        # Iterate over all newscan points to find the corresponding closest points in reference point cloud
        for j, pt in enumerate(points_newscan_centroid_ref_aligned):
            _, idx, _ = points_ref_kdtree.search_knn_vector_3d(pt, 1)
            points_ref_corresponded[j] = points_ref[idx[0]]

        # Solve ICP for current iteration
        T1_2_cur = icp_core(points_ref_corresponded, points_newscan_temp)
        
        end_time = datetime.datetime.now()
        time_difference = (end_time - start_time).total_seconds()
        total_time_cost += time_difference
        
        # Update the the homeogeneous transformation matrix that describes the
        # transformation from points_newscan to points_ref
        T_1_2accumulated = T1_2_cur @ T_1_2accumulated
        
        print('-----------------------------------------')
        print('iteration = ' + str(i+1))
        print('time cost = ' + str(time_difference) + 's')
        print('total time cost = ' + str(total_time_cost) + 's')        
        print('T1_2_cur = ')
        print(T1_2_cur)
        print('accumulated T = ')
        print(T_1_2accumulated)
        
        # Make the points_newscan_temp into homogeneous coordinates and apply the transformation T1_2_cur
        n_row = points_newscan_temp.shape[0]
        pts_newscan_h = np.hstack((points_newscan_temp, np.ones((n_row, 1)))).T
        points_newscan_transformed = T1_2_cur @ pts_newscan_h
        points_newscan_temp = points_newscan_transformed.T[:, :3]
        
        mean_distance = mean_dist(points_ref, points_newscan_temp)
        print('mean_error= ' + str(mean_distance))

        # Update visualization
        pcd2_transed = o3d.geometry.PointCloud()
        pcd2_transed.points = o3d.utility.Vector3dVector(points_newscan_temp)
        pcd2_transed.paint_uniform_color([1, 0, 0])
        vis.add_geometry(pcd2_transed)
        vis.poll_events()
        vis.update_renderer()
        vis.remove_geometry(pcd2_transed)

        if mean_distance < 0.00001 or mean_distance < threshold:
            print('------- fully converged! -------')
            break
        
        if i == n_iter - 1:
            print('------- reach iteration limit -------')

    print('time cost: ' + str(total_time_cost) + ' s')
    
    vis.destroy_window()
    
    # Final visualization
    pcd2_final = o3d.geometry.PointCloud()
    pcd2_final.points = o3d.utility.Vector3dVector(points_newscan_temp)
    pcd2_final.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([axis_pcd, pcd1, pcd2_final])


def mean_dist(point_cloud1, point_cloud2):
    dis_array = []
    for i in range(point_cloud1.shape[0]):
        dif = point_cloud1[i] - point_cloud2[i]
        dis = np.linalg.norm(dif)
        dis_array.append(dis)
        
    return np.mean(np.array(dis_array))


def main(task, data_dir):
    print('start hw program')
    reference_pcd_path = os.path.join(data_dir, 'bunny1.ply')
    new_scan_pcd_path = os.path.join(data_dir, 'bunny2.ply')

    pcd_ref = o3d.io.read_point_cloud(reference_pcd_path)
    pcd_newscan = o3d.io.read_point_cloud(new_scan_pcd_path)

    points_ref = np.array(pcd_ref.points)
    points_newscan = np.array(pcd_newscan.points)

    if task == 1:
        solve_icp_with_known_correspondence(points_ref, points_newscan)
    elif task == 2:
        solve_icp_without_known_correspondence(points_ref, points_newscan, n_iter=60, threshold=0.1)


if __name__ == '__main__':
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('-t', '--task', type=int, required=True)
    parser.add_argument('-d', '--data-dir', required=True)
    args = parser.parse_args()

    main(args.task, args.data_dir)
