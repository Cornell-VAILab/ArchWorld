# Reference: https://github.com/CUT3R/CUT3R/blob/main/eval/mv_recon/utils.py

import numpy as np
from scipy.spatial import cKDTree as KDTree


def umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].

    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t


def umeyama_ransac(X, Y, n_iterations=1000, inlier_threshold=0.1, min_samples=100, seed=None):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets using RANSAC.

    Estimates c, R and t such as c * R @ X + t ~ Y with outlier rejection.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].
    n_iterations : int
        Number of RANSAC iterations.
    inlier_threshold : float
        Distance threshold to consider a point as an inlier.
    min_samples : int
        Minimum number of samples to use for each RANSAC iteration.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    # Set random seed for reproducibility
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    n_points = X.shape[1]
    best_inliers = 0
    best_c, best_R, best_t = None, None, None
    
    # Ensure min_samples doesn't exceed available points
    min_samples = min(min_samples, n_points)
    
    for _ in range(n_iterations):
        # Randomly sample points
        sample_indices = rng.choice(n_points, min_samples, replace=False)
        X_sample = X[:, sample_indices]
        Y_sample = Y[:, sample_indices]
        
        # Estimate transformation on sample
        try:
            c, R, t = umeyama(X_sample, Y_sample)
        except:
            continue
        
        # Transform all points and compute errors
        X_transformed = c * R @ X + t
        errors = np.linalg.norm(Y - X_transformed, axis=0)
        
        # Count inliers
        inliers = np.sum(errors < inlier_threshold)
        
        # Update best model if this is better
        if inliers > best_inliers:
            best_inliers = inliers
            best_c, best_R, best_t = c, R, t
    
    # Refine with all inliers
    if best_c is not None:
        X_transformed = best_c * best_R @ X + best_t
        errors = np.linalg.norm(Y - X_transformed, axis=0)
        inlier_mask = errors < inlier_threshold
        
        if np.sum(inlier_mask) > 0:
            best_c, best_R, best_t = umeyama(X[:, inlier_mask], Y[:, inlier_mask])
    else:
        # Fallback to standard umeyama if RANSAC failed
        best_c, best_R, best_t = umeyama(X, Y)
    
    return best_c, best_R, best_t


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
    acc = np.mean(distances)

    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)

        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=-1)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)

        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median


def compute_iou(pred_vox, target_vox):
    # Get voxel indices
    v_pred_indices = [voxel.grid_index for voxel in pred_vox.get_voxels()]
    v_target_indices = [voxel.grid_index for voxel in target_vox.get_voxels()]

    # Convert to sets for set operations
    v_pred_filled = set(tuple(np.round(x, 4)) for x in v_pred_indices)
    v_target_filled = set(tuple(np.round(x, 4)) for x in v_target_indices)

    # Compute intersection and union
    intersection = v_pred_filled & v_target_filled
    union = v_pred_filled | v_target_filled

    # Compute IoU
    iou = len(intersection) / len(union)
    return iou
