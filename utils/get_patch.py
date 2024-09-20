import open3d as o3d
import numpy as np
import os
import argparse


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, patch_size, color):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    npoint = 6
    if N < npoint:
        idxes = np.hstack((np.tile(np.arange(N), npoint//N), np.random.randint(N, size=npoint%N)))
        return point[idxes, :]

    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point, color


def knn_patch(pcd_name, patch_size=2048):
    pcd = o3d.io.read_point_cloud(pcd_name)

    # nomalize pc and set up kdtree
    points = pc_normalize(np.array(pcd.points))
    color = np.array(pcd.colors)
    pcd.points = o3d.utility.Vector3dVector(points)

    # Build KDTrees using FLANN
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    fps_point, color = farthest_point_sample(points, patch_size, color)
    point_size = fps_point.shape[0]

    patch_list = []
    for i in range(point_size):
        # the k nearest neighbors of the anchor
        [_, idx, dis] = kdtree.search_knn_vector_3d(fps_point[i], patch_size)
        patch_list.append(np.asarray(points)[idx[:], :])
    return np.array(patch_list)


def main(config):
    objs = os.walk(config.path)
    for path, dir_list, file_list in objs:
        for obj in file_list:
            pcd_name = os.path.join(path, obj)
            npy_name = os.path.join(config.out_path, obj.split('.ply')[0] + '.npy')
            patch = knn_patch(pcd_name)
            np.save(npy_name, patch)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='input')
    parser.add_argument('--out_path', type=str, default='output')
    config = parser.parse_args()

    main(config)
