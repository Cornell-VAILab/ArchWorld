import os
import os.path as osp
import csv
import json
import argparse
import glob
import shutil
from collections import defaultdict
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile
import cv2
import torch.nn.functional as F

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Project imports (must follow rootutils.setup_root which sets pythonpath)
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from depth_anything_3.api import DepthAnything3  # noqa: E402
from pi3.models.pi3 import Pi3  # noqa: E402
from visual_util import predictions_to_glb, integrate_camera_into_scene, run_skyseg, download_file_from_url  # noqa: E402
from read_write_model import read_model_cameras_images, qvec2rotmat  # noqa: E402
from eval_utils import umeyama_ransac, accuracy, completion  # noqa: E402

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
np.random.seed(42)

# ---------------------------------------------------------------------------- #
#                                 Helpers                                      #
# ---------------------------------------------------------------------------- #

def pointcloud_to_glb(points: np.ndarray, colors: np.ndarray = None, output_path: str = "", cameras: list = None):
    """Save point cloud as GLB file using trimesh."""
    if not TRIMESH_AVAILABLE:
        print("trimesh not available, skipping GLB save")
        return
    
    # Create trimesh scene
    scene = trimesh.Scene()

    # Create trimesh point cloud
    point_cloud = trimesh.PointCloud(vertices=points)
    
    # Add colors if provided
    if colors is not None:
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
        point_cloud.colors = colors
    
    scene.add_geometry(point_cloud)

    # Add cameras if provided
    if cameras:
        # Calculate scale for camera size
        if len(points) > 0:
            lower = np.percentile(points, 5, axis=0)
            upper = np.percentile(points, 95, axis=0)
            scene_scale = np.linalg.norm(upper - lower)
        else:
            scene_scale = 1.0

        for cam in cameras:
            # cam is dict with 'cam_to_world' (4x4)
            c2w = cam['cam_to_world']
            # Color: black for GT cameras
            color = (0, 0, 0) 
            integrate_camera_into_scene(scene, c2w, color, scene_scale)

    scene.export(output_path, file_type='glb')
    print(f"  Saved GT GLB to {output_path}")

def resolve_depth_path(base_depth_dir: str, image_path: str, had_wildcard: bool = False):
    """Resolve depth map path checking all wildcard expansions.
    Tries: path+.npy/.npz, path-with-extension-replaced-by-.npy/.npz, and flat basename/stem variants.
    """
    if '*' in base_depth_dir:
        candidate_dirs = glob.glob(base_depth_dir)
    else:
        candidate_dirs = [base_depth_dir]
    image_name = osp.basename(image_path)
    image_stem = osp.splitext(image_name)[0]

    # 1. Try preserving relative structure (e.g. subfolders)
    for d in candidate_dirs:
        if image_path and image_path != image_name:
            p_rel_npy = osp.join(d, image_path + ".npy")
            if osp.exists(p_rel_npy):
                return p_rel_npy
            p_rel_npz = osp.join(d, image_path + ".npz")
            if osp.exists(p_rel_npz):
                return p_rel_npz
            # Try stem-based path (replace extension with .npy/.npz)
            rel_no_ext = osp.join(osp.dirname(image_path), image_stem)
            p_stem_npy = osp.join(d, rel_no_ext + ".npy")
            if osp.exists(p_stem_npy):
                return p_stem_npy
            p_stem_npz = osp.join(d, rel_no_ext + ".npz")
            if osp.exists(p_stem_npz):
                return p_stem_npz

    # 2. Try flat structure: basename + .npy/.npz and stem + .npy/.npz
    for d in candidate_dirs:
        for name in (image_name, image_stem):
            p_npy = osp.join(d, name + ".npy")
            if osp.exists(p_npy):
                return p_npy
            p_npz = osp.join(d, name + ".npz")
            if osp.exists(p_npz):
                return p_npz
    return None

def load_depth_map(depth_path: str, expected_shape):
    """Load depth map from .npy or .npz file."""
    if depth_path.endswith('.npz'):
        arr = np.load(depth_path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            for key in ('depth', 'arr_0'):
                if key in arr:
                    depthmap = arr[key]
                    break
            else:
                first_key = list(arr.files)[0]
                depthmap = arr[first_key]
        else:
            depthmap = arr
    else:
        depthmap = np.load(depth_path)
    
    depthmap = depthmap.reshape(expected_shape)
    depthmap[~np.isfinite(depthmap)] = -1
    return depthmap

# ---------------------------------------------------------------------------- #
#                                 GT Generation                                #
# ---------------------------------------------------------------------------- #


def generate_gt_points(scene_name, scene_info, images_metadata, output_dir=None, verbose=False, pred_masks_dir=None):
    """
    Generate Ground Truth point cloud for a scene.
    Returns:
        fused_pointcloud: np.ndarray (N, 3)
        gt_samples: List[Dict] containing info for correspondence
    """
    recon_path = scene_info['recon_path']
    image_folder = scene_info['image_folder']
    depth_folder = scene_info['depth_folder']
    
    # Expand wildcards
    if '*' in image_folder:
        expanded = glob.glob(image_folder)
        if not expanded:
            if verbose:
                print(f"  ERROR: No paths found for wildcard: {image_folder}")
            return None, None
        image_folder = expanded[0]
    
    depth_folder_has_wildcard = '*' in depth_folder
    
    # Load COLMAP
    if not osp.exists(recon_path):
        if verbose:
            print(f"  ERROR: COLMAP sparse reconstruction not found at {recon_path}")
        return None, None

    try:
        cameras, images = read_model_cameras_images(recon_path)
        name_to_id = {img.name: img_id for img_id, img in images.items()}
    except Exception as e:
        if verbose:
            print(f"  ERROR: Failed to load COLMAP data: {e}")
        return None, None
        
    # Prepare reference frame (First image alphabetically)
    if images_metadata is None:
        # Scan directory for images if no metadata provided
        if verbose:
            print(f"  Scanning {image_folder} for images...")
        candidates_paths = []
        # Recursive search for common image extensions
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            candidates_paths.extend(glob.glob(osp.join(image_folder, "**", ext), recursive=True))
        
        # Create metadata format consistent with existing code
        images_metadata = [{'Image Path': p} for p in candidates_paths]
        if verbose:
            print(f"  Found {len(images_metadata)} images in directory.")

    images_metadata_sorted = sorted(images_metadata, key=lambda x: x['Image Path'])
    if not images_metadata_sorted:
        if verbose:
            print("  No images found.")
        return None, None
        
    ref_image_path = images_metadata_sorted[0]['Image Path']
    ref_colmap_id = None
    ref_basename = osp.basename(ref_image_path)
    
    # Try relative path first
    if image_folder in ref_image_path:
        ref_rel_path = osp.relpath(ref_image_path, image_folder)
        if ref_rel_path in name_to_id:
            ref_colmap_id = name_to_id[ref_rel_path]
            
    if ref_colmap_id is None:
        for colmap_img_name in name_to_id.keys():
            if osp.basename(colmap_img_name) == ref_basename:
                ref_colmap_id = name_to_id[colmap_img_name]
                break
                
    if ref_colmap_id is None:
        R_ref, t_ref = np.eye(3), np.zeros(3)
    else:
        qvec_ref = images[ref_colmap_id].qvec
        tvec_ref = images[ref_colmap_id].tvec
        R_ref = qvec2rotmat(qvec_ref)
        t_ref = tvec_ref

    # Sample images: select spatially closest to reference
    candidates = list(images_metadata)

    if ref_colmap_id is not None:
        if verbose:
            print(f"  Sorting {len(candidates)} images by spatial distance to reference...")
        
        # Calculate Ref Center
        qvec_ref = images[ref_colmap_id].qvec
        tvec_ref = images[ref_colmap_id].tvec
        R_ref_mat = qvec2rotmat(qvec_ref)
        center_ref = -R_ref_mat.T @ tvec_ref
        
        candidates_with_dist = []
        
        for img_meta in candidates:
            image_path = img_meta['Image Path']
            
            rel_path = None
            if image_folder in image_path:
                rel_path = osp.relpath(image_path, image_folder)
            
            # Try to find ID
            cid = None
            if rel_path and rel_path in name_to_id:
                cid = name_to_id[rel_path]
            else:
                # Fallback to basename match
                basename = osp.basename(image_path)
                for name, iid in name_to_id.items():
                    if osp.basename(name) == basename:
                        cid = iid
                        break
            
            if cid is not None:
                # Calculate Center
                img_colmap = images[cid]
                R = qvec2rotmat(img_colmap.qvec)
                t = img_colmap.tvec
                center = -R.T @ t
                dist = np.linalg.norm(center - center_ref)
                candidates_with_dist.append((dist, img_meta))
            else:
                # If not in COLMAP, push to end
                candidates_with_dist.append((float('inf'), img_meta))
                
        # Sort
        candidates_with_dist.sort(key=lambda x: x[0])
        candidates = [x[1] for x in candidates_with_dist]
    
    all_points_list = []
    all_colors_list = []
    gt_samples = []
    all_images = []
    valid_image_paths = []
    processed_count = 0
    
    for img_meta in candidates:
        if processed_count >= 50:
            break
            
        image_path = img_meta['Image Path']
        
        # Get relative path
        if image_folder in image_path:
            rel_path = osp.relpath(image_path, image_folder)
        else:
            basename = osp.basename(image_path)
            rel_path = None
            for colmap_img_name in name_to_id.keys():
                if osp.basename(colmap_img_name) == basename:
                    rel_path = colmap_img_name
                    break
            if rel_path is None:
                continue
                
        # Full path
        if osp.isabs(rel_path) and osp.exists(rel_path):
            full_image_path = rel_path
        else:
            full_image_path = osp.join(image_folder, rel_path)
            
        if not osp.exists(full_image_path):
            # Fallback: search recursively under image_folder for the basename
            basename = osp.basename(full_image_path)
            matches = glob.glob(osp.join(image_folder, '**', basename), recursive=True)
            if not matches:
                continue
            full_image_path = matches[0]
            
        # Load image
        try:
            img = Image.open(full_image_path)
            img_array = np.array(img)
            if img_array.ndim == 2:
                img_array = np.stack([img_array]*3, axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:,:,:3]
            width, height = img.size
        except Exception:
            continue

        # Load depth
        depth_path = resolve_depth_path(depth_folder, rel_path, depth_folder_has_wildcard)
        if depth_path is None:
            continue
            
        try:
            depthmap = load_depth_map(depth_path, (height, width))
        except Exception:
            continue

        if pred_masks_dir is not None:
            img_basename = osp.basename(full_image_path)
            mask_path = osp.join(pred_masks_dir, scene_name, img_basename + ".npy")
            if osp.exists(mask_path):
                try:
                    gt_mask = np.load(mask_path).astype(bool)
                    while gt_mask.ndim > 2:
                        gt_mask = gt_mask[0]
                    if gt_mask.shape != (height, width):
                        gt_mask = cv2.resize(gt_mask.astype(np.uint8), (width, height),
                                             interpolation=cv2.INTER_NEAREST).astype(bool)
                    depthmap = depthmap * gt_mask
                except Exception:
                    pass

        # Get COLMAP params
        if rel_path not in name_to_id:
            basename = osp.basename(rel_path)
            matches = [iid for iid, im in images.items() if osp.basename(im.name) == basename]
            if len(matches) != 1:
                continue
            colmap_image_id = matches[0]
        else:
            colmap_image_id = name_to_id[rel_path]
            
        qvec = images[colmap_image_id].qvec
        tvec = images[colmap_image_id].tvec
        R = qvec2rotmat(qvec)
        extr = np.concatenate([R, tvec[:, None]], axis=1)
        
        cam_id = images[colmap_image_id].camera_id
        cam = cameras[cam_id]
        if cam.model != "PINHOLE":
            continue
            
        fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
        K = np.eye(3, dtype=np.float32)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fy, cx, cy

        # Unproject
        try:
            depth_input = depthmap[np.newaxis, ..., np.newaxis]
            intrinsic_input = K[np.newaxis, :, :]
            extrinsic_input = extr[np.newaxis, :, :]
            
            pointmap = unproject_depth_map_to_point_map(
                depth_map=depth_input,
                intrinsics_cam=intrinsic_input,
                extrinsics_cam=extrinsic_input
            )[0]

            valid_mask = depthmap > 1e-4
            valid_pts = pointmap[valid_mask]
            valid_colors = img_array[valid_mask]
            
            if len(valid_pts) > 0:
                valid_mask_pts = np.isfinite(valid_pts).all(axis=1) & (np.linalg.norm(valid_pts, axis=1) > 1e-4)
                valid_pts = valid_pts[valid_mask_pts]
                valid_colors = valid_colors[valid_mask_pts]
                if len(valid_pts) > 0:
                    all_points_list.append(valid_pts)
                    all_colors_list.append(valid_colors)
                    
                    gt_samples.append({
                        'image_path': full_image_path,
                        'orig_stem': osp.splitext(osp.basename(full_image_path))[0],
                        'depth_path': depth_path,
                        'depth_map': depthmap,
                        'K': K,
                        'R_cam2world': extr[:3, :3].T,
                        'extrinsic': extr,
                        'R_ref': R_ref,
                        't_ref': t_ref,
                    })
                    
                    all_images.append(img)
                    valid_image_paths.append(rel_path)
                    processed_count += 1
        except Exception as e:
            if verbose:
                print(f"Unprojection failed: {e}")
            continue
            
    if not all_points_list:
        return None, None
        
    fused_pointcloud = np.concatenate(all_points_list, axis=0)
    fused_colors = np.concatenate(all_colors_list, axis=0) if all_colors_list else None
    
    # Transform to Reference Camera Frame (consistent with gt.py)
    # P_cam = P_world @ R_ref.T + t_ref
    fused_pointcloud = fused_pointcloud @ R_ref.T + t_ref
    
    # Rotate 180 deg around X
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ], dtype=np.float32)
    fused_pointcloud = fused_pointcloud @ rotation_matrix.T
    
    # Prepare Cameras for Visualization
    transformed_cameras = []
    
    # Construct T_world_to_out (4x4)
    # 1. World -> Ref: R_ref, t_ref
    T_w2ref = np.eye(4)
    T_w2ref[:3, :3] = R_ref
    T_w2ref[:3, 3] = t_ref
    
    # 2. Ref -> Flip: R_x
    T_ref2out = np.eye(4)
    T_ref2out[:3, :3] = rotation_matrix
    
    # Combined: World -> Out
    T_w2out = T_ref2out @ T_w2ref
    
    for s in gt_samples:
        ext = s['extrinsic']
        if ext.shape == (3, 4):
            ext = np.vstack([ext, [0,0,0,1]])
            
        c2w = np.linalg.inv(ext)
        c2w_out = T_w2out @ c2w
        
        transformed_cameras.append({
            'cam_to_world': c2w_out
        })

    # Save artifacts if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save GLB
        scene_name_safe = scene_name.replace('/', '_').replace('\\', '_')
        glb_path = osp.join(output_dir, f"{scene_name_safe}_gt.glb")
        pointcloud_to_glb(fused_pointcloud, fused_colors, glb_path, cameras=None)
        
        scene_images_dir = osp.join(output_dir, f"{scene_name_safe}_images")
        os.makedirs(scene_images_dir, exist_ok=True)
        for i, (img, rel_path) in enumerate(zip(all_images, valid_image_paths)):
            basename = osp.basename(rel_path)
            if not basename:
                basename = f"image_{i:03d}.jpg"
            if not any(basename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                basename = f"{basename}.jpg"
            basename = f"{i:02d}_{basename}"
            img.save(osp.join(scene_images_dir, basename))
            gt_samples[i]['image_path'] = osp.join(scene_images_dir, basename)
            
    return fused_pointcloud, gt_samples

# ---------------------------------------------------------------------------- #
#                               Prediction Logic                               #
# ---------------------------------------------------------------------------- #

def run_inference(model, image_paths, device, dtype, model_name):
    """Run model and return raw outputs."""
    model_name = model_name.replace('-ft', '')
    with torch.no_grad():
        with torch.amp.autocast(dtype=dtype, device_type="cuda"):
            images = load_and_preprocess_images(image_paths).to(device).unsqueeze(0)
            if model_name == "vggt":
                aggregated_tokens_list, ps_idx = model.aggregator(images)
                depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                point_maps = None
            elif model_name == "da3":
                image_list = torch.unbind(images.squeeze(0), dim=0)
                image_list = [(image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for image in image_list]
                out = model.inference(image_list, process_res=518)
                depth_map = torch.tensor(out.depth).unsqueeze(0).unsqueeze(-1)
                if out.conf is not None:
                    depth_conf = torch.tensor(out.conf).unsqueeze(0)
                else:
                    depth_conf = torch.ones_like(depth_map.squeeze(-1).unsqueeze(0))
                extrinsics = torch.tensor(out.extrinsics).unsqueeze(0)
                intrinsics = torch.tensor(out.intrinsics).unsqueeze(0)
                point_maps = None
            elif model_name == "pi3":
                pred = model(images)
                pred_pts = pred['points'].squeeze(0)  # (N, h, w, 3)
                pred_pts = F.interpolate(
                    pred_pts.permute(0, 3, 1, 2), (518, 518),
                    mode="bilinear", align_corners=False, antialias=True
                ).permute(0, 2, 3, 1)
                pred_conf = pred['conf'].squeeze(0)  # (N, H, W, 1)
                pred_conf = F.interpolate(
                    pred_conf.permute(0, 3, 1, 2), (518, 518),
                    mode="bilinear", align_corners=False, antialias=True
                ).squeeze(1)  # (N, 518, 518)
                extrinsics = pred['camera_poses'][0]
                depth_map, intrinsics = None, None
                point_maps = torch.tensor(pred_pts).unsqueeze(0)
                S, H, W, _ = pred_pts.shape
                depth_conf = pred_conf.unsqueeze(0)  # (1, N, 518, 518)

        return {
            'images': images,
            'depth_map': depth_map,
            'depth_conf': depth_conf,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'point_maps': point_maps
        }

def apply_sky_masks(pred_data, image_paths, sky_masks_dir, scene_name):
    """Zero out predicted depth/conf pixels where sky masks indicate sky (mask == 0)."""
    scene_sky_dir = osp.join(sky_masks_dir, scene_name)
    if not osp.isdir(scene_sky_dir):
        print(f"  [sky_mask] No sky masks found for {scene_name}, skipping.")
        return

    has_point_maps = pred_data.get('point_maps') is not None
    arr  = pred_data['point_maps'] if has_point_maps else pred_data['depth_map']
    conf = pred_data['depth_conf']   # (1, S, H, W)

    _, H, W = arr.shape[1], arr.shape[2], arr.shape[3]
    arr_np  = arr.cpu().float().numpy()
    conf_np = conf.cpu().float().numpy()

    applied = 0
    for i, img_path in enumerate(image_paths):
        img_name = osp.basename(img_path)
        if len(img_name) > 3 and img_name[:2].isdigit() and img_name[2] == '_':
            img_name = img_name[3:]
        mask_path = osp.join(scene_sky_dir, img_name)
        if not osp.exists(mask_path):
            continue
        sky_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 0=sky, 255=non-sky
        if sky_mask is None:
            continue
        if sky_mask.shape != (H, W):
            sky_mask = cv2.resize(sky_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        is_sky = sky_mask < 128
        arr_np[0, i][is_sky]  = 0.0
        conf_np[0, i][is_sky] = 0.0
        applied += 1

    if has_point_maps:
        pred_data['point_maps'] = torch.from_numpy(arr_np).to(arr.device)
    else:
        pred_data['depth_map'] = torch.from_numpy(arr_np).to(arr.device)
    pred_data['depth_conf'] = torch.from_numpy(conf_np).to(conf.device)
    print(f"  [sky_mask] Applied sky masks to {applied}/{len(image_paths)} images.")


def apply_filtration_masks(pred_data, image_paths, scene_name, masks_dir):
    """Zero out predicted depth/point-map pixels outside the SAM building mask."""
    scene_masks_dir = osp.join(masks_dir, scene_name)
    if not osp.isdir(scene_masks_dir):
        print(f"  [filter_pred] No masks found for {scene_name}, skipping.")
        return

    has_point_maps = pred_data.get('point_maps') is not None
    arr  = pred_data['point_maps'] if has_point_maps else pred_data['depth_map']
    conf = pred_data['depth_conf']   # (1, S, H, W)

    _, H, W = arr.shape[1], arr.shape[2], arr.shape[3]

    arr_np  = arr.cpu().float().numpy()
    conf_np = conf.cpu().float().numpy()

    applied = 0
    for i, img_path in enumerate(image_paths):
        img_stem  = osp.basename(img_path)
        # generate_gt_points prefixes saved images with "{i:02d}_"; strip it so the
        # lookup matches the original mask filename (e.g. "00_img.JPG" → "img.JPG").
        if len(img_stem) > 3 and img_stem[:2].isdigit() and img_stem[2] == '_':
            img_stem = img_stem[3:]
        mask_path = osp.join(scene_masks_dir, img_stem + ".npy")
        if not osp.exists(mask_path):
            continue
        mask = np.load(mask_path).astype(bool)   # (H_img, W_img)
        if mask.shape != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        arr_np[0, i][~mask]  = 0.0
        conf_np[0, i][~mask] = 0.0
        applied += 1

    if has_point_maps:
        pred_data['point_maps'] = torch.from_numpy(arr_np).to(arr.device)
    else:
        pred_data['depth_map'] = torch.from_numpy(arr_np).to(arr.device)
    pred_data['depth_conf'] = torch.from_numpy(conf_np).to(conf.device)
    print(f"  [filter_pred] Applied masks to {applied}/{len(image_paths)} images.")


def compute_element_chamfer_metrics(gt_samples, pred_points_aligned, scene_name,
                                    sam_features_dir, pred_masks_dir=None):
    """Compute per-architectural-element Chamfer distance (REW-24).

    For each feature tag found in SAM masks, reprojects masked depth pixels to 3D
    (same transform as generate_gt_points) and computes accuracy/completion against
    the full aligned pred point cloud.

    If pred_masks_dir is given, GT element points are additionally restricted to
    pixels inside the filtration (building) mask, matching what was done to the
    pred point cloud via apply_filtration_masks.

    Returns a flat dict keyed by elem_{feature}_{metric}.
    """
    mask_dir = osp.join(sam_features_dir, scene_name, 'mask')
    if not osp.exists(mask_dir):
        return {}

    if pred_masks_dir and not osp.isdir(osp.join(pred_masks_dir, scene_name)):
        print(f"  [elem_chamfer] No filtration masks for {scene_name}, skipping element metrics.")
        return {}

    rot_x_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    feature_points = defaultdict(list)

    for sample in gt_samples:
        stem = sample['orig_stem']
        depth_map = sample['depth_map']
        K = sample['K']
        extrinsic = sample['extrinsic']
        R_ref = sample['R_ref']
        t_ref = sample['t_ref']

        H, W = depth_map.shape
        y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        Z = depth_map
        P_cam = np.stack([
            (x_grid - K[0, 2]) * Z / K[0, 0],
            (y_grid - K[1, 2]) * Z / K[1, 1],
            Z,
        ], axis=-1)

        R_inv = extrinsic[:3, :3].T
        t_inv = -R_inv @ extrinsic[:3, 3]
        P_world = P_cam @ R_inv.T + t_inv
        P_final = (P_world @ R_ref.T + t_ref) @ rot_x_180.T

        valid = (Z > 1e-4) & np.isfinite(Z) & np.isfinite(P_final).all(axis=-1)

        # Restrict GT element points to the filtration (building) mask if provided,
        # matching the filtering applied to pred points via apply_filtration_masks.
        if pred_masks_dir:
            img_basename = osp.basename(sample['image_path'])
            if len(img_basename) > 3 and img_basename[:2].isdigit() and img_basename[2] == '_':
                img_basename = img_basename[3:]
            filt_path = osp.join(pred_masks_dir, scene_name, img_basename + ".npy")
            if osp.exists(filt_path):
                try:
                    filt_mask = np.load(filt_path).astype(bool)
                    while filt_mask.ndim > 2:
                        filt_mask = filt_mask[0]
                    if filt_mask.shape != (H, W):
                        filt_mask = cv2.resize(filt_mask.astype(np.uint8), (W, H),
                                               interpolation=cv2.INTER_NEAREST).astype(bool)
                    valid = valid & filt_mask
                except Exception:
                    pass

        for mask_file in glob.glob(osp.join(mask_dir, f"{glob.escape(stem)}_*.npy")):
            feature = osp.basename(mask_file)[len(stem) + 1:-4]
            try:
                raw_mask = np.load(mask_file).astype(bool)
                while raw_mask.ndim > 2:
                    raw_mask = raw_mask[0]
                if raw_mask.shape != (H, W):
                    raw_mask = cv2.resize(raw_mask.astype(np.uint8), (W, H),
                                          interpolation=cv2.INTER_NEAREST).astype(bool)
            except Exception:
                continue
            pts = P_final[valid & raw_mask]
            if len(pts) > 0:
                feature_points[feature].append(pts)

    if not feature_points:
        return {}

    out = {}
    for feature, pts_list in feature_points.items():
        elem_gt_pts = np.concatenate(pts_list, axis=0)
        if len(elem_gt_pts) < 10:
            continue
        try:
            acc, acc_med = accuracy(elem_gt_pts, pred_points_aligned)
            comp, comp_med = completion(elem_gt_pts, pred_points_aligned)
            out[f'elem_{feature}_acc_mean'] = acc
            out[f'elem_{feature}_acc_med']  = acc_med
            out[f'elem_{feature}_comp_mean'] = comp
            out[f'elem_{feature}_comp_med']  = comp_med
            out[f'elem_{feature}_score']     = (acc + comp) / 2.0
            out[f'elem_{feature}_n_pts']     = len(elem_gt_pts)
        except Exception as e:
            print(f"  [elem_chamfer] '{feature}': {e}")
    return out


def compute_alignment_with_correspondence(gt_samples, pred_data, scene_output_dir,
                                          max_total_points=None, n_iterations=1000, image_paths=None):
    pred_confs  = pred_data['depth_conf'].cpu().squeeze(0).numpy()   # (S, H, W)

    # --- Branch: point-map model (pi3) vs depth-map model (vggt/da3) ---
    has_point_maps = pred_data.get('point_maps') is not None
    if has_point_maps:
        # Pi3: world-space points already computed — shape (S, H, W, 3)
        pred_world_pts = pred_data['point_maps'].cpu().squeeze(0).numpy()
        S, H_pred, W_pred, _ = pred_world_pts.shape
    else:
        pred_depths     = pred_data['depth_map'].cpu().squeeze(0).squeeze(-1).numpy()  # (S, H, W)
        pred_extrinsics = pred_data['extrinsics'].cpu().squeeze(0).float().numpy()
        pred_intrinsics = pred_data['intrinsics'].cpu().squeeze(0).float().numpy()
        S, H_pred, W_pred = pred_depths.shape

    # Match gt_samples to prediction indices
    if image_paths is not None and len(image_paths) == S:
        gt_by_path = {osp.normpath(s["image_path"]): s for s in gt_samples}
        def get_sample(i):
            return gt_by_path.get(osp.normpath(image_paths[i]))
    else:
        def get_sample(i):
            return gt_samples[i] if i < len(gt_samples) else None

    correspondences_gt   = []
    correspondences_pred = []

    print(f"  Computing alignment from {S} images with per-pixel correspondence...")

    for i in range(S):
        sample = get_sample(i)
        if sample is None:
            continue

        # --- GT side (identical for all models) ---
        gt_depth    = sample['depth_map']
        gt_extrinsic = sample['extrinsic']
        R_ref       = sample['R_ref']
        t_ref       = sample['t_ref']
        K_gt        = sample['K']

        gt_depth_resized = cv2.resize(gt_depth, (W_pred, H_pred), interpolation=cv2.INTER_NEAREST)
        y, x = np.meshgrid(np.arange(H_pred), np.arange(W_pred), indexing='ij')

        K_gt_scaled = K_gt.copy()
        scale_x = W_pred / gt_depth.shape[1]
        scale_y = H_pred / gt_depth.shape[0]
        K_gt_scaled[0, 0] *= scale_x
        K_gt_scaled[0, 2] *= scale_x
        K_gt_scaled[1, 1] *= scale_y
        K_gt_scaled[1, 2] *= scale_y

        fx_gt, fy_gt = K_gt_scaled[0, 0], K_gt_scaled[1, 1]
        cx_gt, cy_gt = K_gt_scaled[0, 2], K_gt_scaled[1, 2]

        Z_gt = gt_depth_resized
        X_gt = (x - cx_gt) * Z_gt / fx_gt
        Y_gt = (y - cy_gt) * Z_gt / fy_gt
        P_cam_gt = np.stack([X_gt, Y_gt, Z_gt], axis=-1)

        R_gt_inv = gt_extrinsic[:3, :3].T
        t_gt_inv = -gt_extrinsic[:3, :3].T @ gt_extrinsic[:3, 3]
        P_world_gt = P_cam_gt @ R_gt_inv.T + t_gt_inv

        rot_x_180 = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float32)
        P_final_gt = (P_world_gt @ R_ref.T + t_ref) @ rot_x_180.T

        # --- Pred side ---
        if has_point_maps:
            # Pi3: directly use the world-space point map for this frame
            P_world_pred = pred_world_pts[i]           # (H, W, 3)
            Z_pred = np.linalg.norm(P_world_pred, axis=-1)   # used only for mask
        else:
            pred_depth    = pred_depths[i]
            pred_K        = pred_intrinsics[i]
            pred_extrinsic = pred_extrinsics[i]

            fx, fy = pred_K[0, 0], pred_K[1, 1]
            cx, cy = pred_K[0, 2], pred_K[1, 2]
            Z_pred = pred_depth
            X_pred = (x - cx) * Z_pred / fx
            Y_pred = (y - cy) * Z_pred / fy
            P_cam_pred = np.stack([X_pred, Y_pred, Z_pred], axis=-1)

            R_pred_inv = pred_extrinsic[:3, :3].T
            t_pred_inv = -pred_extrinsic[:3, :3].T @ pred_extrinsic[:3, 3]
            P_world_pred = P_cam_pred @ R_pred_inv.T + t_pred_inv

        # --- Shared masking + subsampling ---
        mask = (Z_gt > 1e-4) & np.isfinite(Z_gt) & (Z_pred > 1e-4) & np.isfinite(Z_pred)
        if pred_confs.ndim == 3:
            mask = mask & (pred_confs[i] > 1e-4)

        pts_pred_flat = P_world_pred[mask]
        pts_gt_flat   = P_final_gt[mask]

        if len(pts_pred_flat) > 5000:
            idx = np.random.choice(len(pts_pred_flat), 5000, replace=False)
            pts_pred_flat = pts_pred_flat[idx]
            pts_gt_flat   = pts_gt_flat[idx]

        if len(pts_pred_flat) > 0:
            correspondences_pred.append(pts_pred_flat)
            correspondences_gt.append(pts_gt_flat)

    if not correspondences_pred:
        return None, None

    pts_pred_all = np.concatenate(correspondences_pred, axis=0)
    pts_gt_all   = np.concatenate(correspondences_gt,   axis=0)

    if max_total_points is not None and len(pts_pred_all) > max_total_points:
        idx = np.random.choice(len(pts_pred_all), max_total_points, replace=False)
        pts_pred_all = pts_pred_all[idx]
        pts_gt_all   = pts_gt_all[idx]

    print(f"  Alignment based on {len(pts_pred_all)} corresponding points.")
    c, R, t = umeyama_ransac(pts_pred_all.T, pts_gt_all.T, n_iterations=n_iterations)
    return c, R, t

# ---------------------------------------------------------------------------- #
#                               Evaluation Main                                #
# ---------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="End-to-End Evaluation: GT Generation -> Prediction -> Metrics")
    
    script_dir = osp.dirname(osp.abspath(__file__))
    project_root = osp.dirname(script_dir) 
    
    parser.add_argument(
        "--geocoded_csv",
        type=str,
        default="/share/elor/ak2798/ArchitecturalFairness/gt_scenes_valid.csv",
    )
    parser.add_argument(
        "--images_csv",
        type=str,
        default=osp.join(project_root, "data", "image_scales_metadata.csv"),
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="/share/elor/ak2798/ArchitecturalFairness/points",
        help="Base output directory where GT and Pred GLBs are stored"
    )
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument(
        "--sam_features_dir",
        type=str,
        default="/share/elor/yz864/ECCV2026/SAM_features",
        help="Directory containing per-scene selected_images.json files"
    )
    parser.add_argument("--output_csv", type=str, default="evaluation_results")
    parser.add_argument("--no_ransac", action="store_true")
    parser.add_argument("--filter_pred", action="store_true",
                        help="Zero out predicted depths outside SAM building masks.")
    parser.add_argument("--pred_masks_dir", type=str,
                        default="/share/elor/ak2798/ArchitecturalFairness/filtration_masks",
                        help="Root dir of per-scene per-image SAM masks (used with --filter_pred).")
    parser.add_argument("--force", action="store_true", help="Recompute all scenes, ignoring existing CSV and metrics.json")
    parser.add_argument('--model', type=str, default='vggt', choices=['vggt', 'da3', 'pi3', 'vggt-ft', 'pi3-ft'])
    parser.add_argument('--sky_masks_dir', type=str,
                        default='/share/elor/ak2798/ArchitecturalFairness/sky_masks',
                        help='Shared directory for sky segmentation masks, reused across models.')
    args = parser.parse_args()
    args.output_base = f"{args.output_base}_{args.model}"
    args.output_csv = f"{args.output_csv}_{args.model}.csv"
    base_model = args.model.replace('-ft', '')
    # 1. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Loading {args.model.upper()} model on {device}...")
    if args.model == 'vggt':
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    elif args.model == 'vggt-ft':
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        ckpt_path = "/share/phoenix/nfs06/S9/yl4355/logs/finetune_vggt/finetune_full-full_data-refine_data-v2/ckpts/checkpoint_100.pt"
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt)))
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded fine-tuned VGGT weights from {ckpt_path}")
    elif args.model == 'da3':
        model = DepthAnything3.from_pretrained("depth-anything/da3-base").to(device)
    elif args.model == 'pi3':
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device)
    elif args.model == 'pi3-ft':
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device)
        ckpt_path = "/share/phoenix/nfs06/S9/yl4355/logs/finetune_pi3/finetune_release_mix/ckpts/checkpoint.pt"
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt)))
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded fine-tuned Pi3 weights from {ckpt_path}")
    model.eval()
    # 2. Load CSV Data
    print("Loading CSVs...")
    scenes_info = {}
    with open(args.geocoded_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_name = row['Category Name']
            if args.scene and scene_name != args.scene:
                continue
            
            scenes_info[scene_name] = {
                'image_folder': row['Image Folder'],
                'depth_folder': row['Depth Map Folder'],
                'recon_path': row['Recon Path'],
            }

    print(f"Found {len(scenes_info)} scenes to process.")

    results = []

    # Load existing results to skip (unless --force)
    processed_scenes_csv = set()
    if not args.force and osp.exists(args.output_csv):
        try:
            df_existing = pd.read_csv(args.output_csv)
            # Filter out AVERAGE row if present
            df_existing = df_existing[df_existing['scene'] != 'AVERAGE']
            processed_scenes_csv = set(df_existing['scene'].tolist())
            results.extend(df_existing.to_dict('records'))
            print(f"Loaded {len(results)} existing results from {args.output_csv}")
        except Exception as e:
            print(f"Error reading existing CSV: {e}")
            processed_scenes_csv = set()
    else:
        processed_scenes_csv = set()
    
    for scene_name, scene_info in tqdm(scenes_info.items()):
        scene_output_dir = osp.join(args.output_base, scene_name)
        metrics_json_path = osp.join(scene_output_dir, "metrics.json")
        
        if not args.force:
            if scene_name in processed_scenes_csv:
                print(f"[{scene_name}] Already in CSV. Skipping...")
                continue

            if osp.exists(metrics_json_path):
                print(f"[{scene_name}] Already processed (metrics.json exists). Skipping...")
                try:
                    with open(metrics_json_path, 'r') as f:
                        prev_res = json.load(f)
                    results.append(prev_res)
                except Exception:
                    pass
                continue

        print(f"\nProcessing {scene_name}...")
        
        images_metadata = None
        sam_json_path = osp.join(args.sam_features_dir, scene_name, "selected_images.json")
        if osp.exists(sam_json_path):
            with open(sam_json_path, 'r', encoding='utf-8') as f:
                sam_data = json.load(f)
            scene_info = dict(scene_info)
            scene_info['image_folder'] = sam_data['image_folder']
            scene_info['recon_path'] = sam_data['recon_path']
            images_metadata = [
                {'Image Path': osp.join(sam_data['image_folder'], b)}
                for b in sam_data['basenames']
            ]
            print(f"  Using {len(images_metadata)} selected images from {sam_json_path}")

        gt_points, gt_samples = generate_gt_points(
            scene_name,
            scene_info,
            images_metadata,
            output_dir=scene_output_dir,
            verbose=True,
            pred_masks_dir=args.pred_masks_dir if args.filter_pred else None,
        )
        if gt_points is None or len(gt_points) < 100:
            print(f"[{scene_name}] Failed to generate valid GT points")
            continue
            
        gt_samples.sort(key=lambda x: x['image_path'])
        image_paths = [s['image_path'] for s in gt_samples]
        pred_data = run_inference(model, image_paths, device, dtype, args.model)
        if args.filter_pred:
            apply_filtration_masks(pred_data, image_paths, scene_name, args.pred_masks_dir)
        if base_model != 'vggt':
            apply_sky_masks(pred_data, image_paths, args.sky_masks_dir, scene_name)
        c, R, t = compute_alignment_with_correspondence(gt_samples, pred_data, scene_output_dir)

        if c is None or not np.isfinite(c):
            print(f"[{scene_name}] Alignment failed (invalid scale: {c})")
            continue
        pred_extrinsics = pred_data['extrinsics'].cpu().float()
        if base_model == "pi3":
            pred_extrinsics = torch.tensor(np.linalg.inv(pred_extrinsics.numpy())[:, :3, :]).unsqueeze(0)
            point_maps_np = pred_data['point_maps'].cpu().squeeze(0).numpy()
            conf_np = pred_data['depth_conf'].cpu().squeeze(0).numpy()
            clean_mask = (conf_np > 1e-4) & np.isfinite(point_maps_np).all(axis=-1)
            pred_points_clean = point_maps_np[clean_mask].reshape(-1, 3)
            point_maps = torch.tensor(point_maps_np)

        else:
            pred_depths = pred_data['depth_map'].cpu()
            pred_intrinsics = pred_data['intrinsics'].cpu().float()
            point_maps = torch.tensor(unproject_depth_map_to_point_map(
                pred_depths.squeeze(0),
                pred_extrinsics.squeeze(0),
                pred_intrinsics.squeeze(0)
            ))
        
        raw_pred_path = osp.join(scene_output_dir, f"{scene_name}_pred.glb")
        depth_conf = pred_data['depth_conf'].cpu()
        if base_model == 'vggt':
            depth_conf_processed = torch.sigmoid(torch.log(depth_conf)) * 2 - 1 + 1e-8
        else:
            # DA3 conf is ≥1.0 for all detected surfaces including sky.
            # Sky pixels were zeroed (=0) by apply_sky_masks / out.sky.
            # Set those zeros to -1 so predictions_to_glb's (conf >= 0) filter
            # excludes them — matching how VGGT's transform naturally goes negative.
            depth_conf_processed = depth_conf.clone()
            depth_conf_processed[depth_conf_processed <= 0] = -1.0
        images_np = pred_data['images'].cpu().numpy().squeeze(0).transpose((0, 2, 3, 1))
        depth_conf_np = depth_conf_processed.numpy().squeeze(0)

        if base_model != 'vggt':
            # Run ONNX sky segmenter on images_np (518×518, same space as depth_conf_np).
            if not os.path.exists("skyseg.onnx"):
                print("Downloading skyseg.onnx...")
                download_file_from_url(
                    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx"
                )
            import onnxruntime
            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = 4
            skyseg_session = onnxruntime.InferenceSession("skyseg.onnx", sess_options=sess_options)
            S_seg, H_seg, W_seg = depth_conf_np.shape
            for si in range(S_seg):
                frame_rgb = (images_np[si] * 255).astype(np.uint8)   # (H, W, 3) RGB
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                result = run_skyseg(skyseg_session, [320, 320], frame_bgr)  # 320×320 uint8
                result_resized = cv2.resize(result, (W_seg, H_seg), interpolation=cv2.INTER_LINEAR)
                sky_mask = result_resized >= 20   # True = sky; lower than default 32 to catch boundary halos
                depth_conf_np[si][sky_mask] = -1.0

            # Filter bottom 15% of confident (non-sky) pixels by confidence.
            # Must compute percentile only on positive values — sky pixels are already -1
            # and would skew a global percentile, defeating the filter.
            pos_vals = depth_conf_np[depth_conf_np > 0]
            if len(pos_vals) > 0:
                conf_cutoff = np.percentile(pos_vals, 60)
                low_conf = (depth_conf_np > 0) & (depth_conf_np < conf_cutoff)
                depth_conf_np[low_conf] = -1.0

        preds_for_viz = {
            "world_points_from_depth": point_maps.numpy(),
            "depth_conf": depth_conf_np,
            "images": images_np,
            "extrinsic": pred_extrinsics.squeeze(0).numpy()
        }
        
        cache_dir = raw_pred_path + "_cache"
        if osp.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(osp.join(cache_dir, "images"), exist_ok=True)
        for img_path in image_paths:
            shutil.copy2(img_path, osp.join(cache_dir, "images", osp.basename(img_path)))

        # Pre-populate sky masks from shared cache so predictions_to_glb reuses them.
        # Shared masks are stored by original basename (no prefix); cache images use the
        # {i:02d}_ prefix, so copy with the prefixed name so predictions_to_glb finds them.
        shared_sky_dir = osp.join(args.sky_masks_dir, scene_name)
        cache_sky_dir = osp.join(cache_dir, "sky_masks")
        if osp.isdir(shared_sky_dir):
            shared_masks = {osp.basename(f): f for f in glob.glob(osp.join(shared_sky_dir, "*")) if osp.isfile(f)}
            os.makedirs(cache_sky_dir, exist_ok=True)
            populated = 0
            for img_path in image_paths:
                img_base = osp.basename(img_path)
                stripped = img_base[3:] if (len(img_base) > 3 and img_base[:2].isdigit() and img_base[2] == '_') else img_base
                src = shared_masks.get(stripped)
                if src:
                    shutil.copy2(src, osp.join(cache_sky_dir, img_base))
                    populated += 1
            print(f"  Pre-populated {populated} sky masks from shared cache.")

        try:
            if base_model == "pi3":
                print(TRIMESH_AVAILABLE)
                if TRIMESH_AVAILABLE:
                    max_pts = 500000
                    S, H_pm, W_pm, _ = point_maps_np.shape
                    images_resized = np.stack([
                        cv2.resize(images_np[i], (W_pm, H_pm), interpolation=cv2.INTER_LINEAR)
                        for i in range(S)
                    ], axis=0)

                    colors_flat = (images_resized[clean_mask] * 255).astype(np.uint8)

                    if len(pred_points_clean) > max_pts:
                        idx = np.random.choice(len(pred_points_clean), max_pts, replace=False)
                        verts = pred_points_clean[idx]
                        cols = colors_flat[idx]
                    else:
                        verts = pred_points_clean
                        cols = colors_flat

                    pc = trimesh.PointCloud(vertices=verts.astype(np.float32), colors=cols)
                    # with open(raw_pred_path, 'wb') as f:
                    #     f.write(trimesh.exchange.gltf.export_glb(trimesh.Scene([pc])))

                    glb_scene = trimesh.Scene([pc])
                    glb_scene.export(raw_pred_path)
            elif base_model == "vggt":
                pred_points_clean = None
                scene = predictions_to_glb(
                    preds_for_viz,
                    conf_thres=0,
                    mask_sky=True,
                    target_dir=cache_dir,
                    prediction_mode="Depth Map",
                    show_cam=False
                )
                scene.export(raw_pred_path)

                # Save newly generated sky masks to shared cache for reuse by other models.
                # Strip the run-specific {i:02d}_ prefix so masks are keyed by original basename,
                # making them stable across VGGT/DA3/Pi3 runs which assign different indices.
                os.makedirs(shared_sky_dir, exist_ok=True)
                saved = 0
                for f in glob.glob(osp.join(cache_sky_dir, "*")):
                    if osp.isfile(f):
                        dst_name = osp.basename(f)
                        if len(dst_name) > 3 and dst_name[:2].isdigit() and dst_name[2] == '_':
                            dst_name = dst_name[3:]
                        dst = osp.join(shared_sky_dir, dst_name)
                        if not osp.exists(dst):
                            shutil.copy2(f, dst)
                            saved += 1
                if saved:
                    print(f"  Saved {saved} new sky masks to {shared_sky_dir}")

                # Extract clean vertices
                vertices_list = []
                for geom in scene.geometry.values():
                    if isinstance(geom, trimesh.PointCloud):
                        vertices_list.append(geom.vertices)
                if vertices_list:
                    pred_points_clean = np.concatenate(vertices_list, axis=0)
            else:
                # DA3/other non-VGGT depth models: sky pixels have conf=-1 in depth_conf_processed
                # (set above), so predictions_to_glb's (conf >= 0) filter excludes them.
                # mask_sky=False: ONNX multiplication would reset -1→0, defeating the filter.
                pred_points_clean = None
                scene = predictions_to_glb(
                    preds_for_viz,
                    conf_thres=0,
                    mask_sky=False,
                    target_dir=cache_dir,
                    prediction_mode="Depth Map",
                    show_cam=False
                )
                scene.export(raw_pred_path)
                vertices_list = []
                for geom in scene.geometry.values():
                    if isinstance(geom, trimesh.PointCloud):
                        vertices_list.append(geom.vertices)
                if vertices_list:
                    pred_points_clean = np.concatenate(vertices_list, axis=0)
            print(f"  Saved Raw Pred GLB to {raw_pred_path}")

        finally:
            if osp.exists(cache_dir):
                shutil.rmtree(cache_dir)
            
        if pred_points_clean is None:
            print(f"[{scene_name}] Failed to extract clean pred points")
            continue
        MAX_PRED_POINTS = 500000
        if len(pred_points_clean) > MAX_PRED_POINTS:
            idx = np.random.choice(len(pred_points_clean), MAX_PRED_POINTS, replace=False)
            pred_points_clean = pred_points_clean[idx]
        pred_points_aligned = (c * R @ pred_points_clean.T + t).T
        combined_glb_path = osp.join(scene_output_dir, f"{scene_name}_aligned_combined.glb")
        if TRIMESH_AVAILABLE:
            save_N = 500000
            if len(gt_points) > save_N:
                idx = np.random.choice(len(gt_points), save_N, replace=False)
                p_gt = gt_points[idx]
            else:
                p_gt = gt_points
            pc_gt = trimesh.PointCloud(vertices=p_gt)
            pc_gt.colors = np.tile([0, 255, 0, 255], (len(p_gt), 1)).astype(np.uint8)
            if len(pred_points_aligned) > save_N:
                idx = np.random.choice(len(pred_points_aligned), save_N, replace=False)
                p_pred = pred_points_aligned[idx]
            else:
                p_pred = pred_points_aligned
            pc_pred = trimesh.PointCloud(vertices=p_pred)
            pc_pred.colors = np.tile([255, 0, 0, 255], (len(p_pred), 1)).astype(np.uint8)
            
            scene_comb = trimesh.Scene([pc_gt, pc_pred])
            scene_comb.export(combined_glb_path, file_type='glb')
            print(f"  Saved Combined Aligned GLB to {combined_glb_path}")

        acc, acc_med = accuracy(gt_points, pred_points_aligned)
        comp, comp_med = completion(gt_points, pred_points_aligned)
        
        print("  Computing per-element Chamfer metrics...")
        elem_stats = compute_element_chamfer_metrics(
            gt_samples, pred_points_aligned, scene_name, args.sam_features_dir,
            pred_masks_dir=args.pred_masks_dir if args.filter_pred else None)
        if elem_stats:
            features_found = [k[5:k.index('_', 5)] for k in elem_stats if k.startswith('elem_') and '_score' in k]
            print(f"  Element metrics computed for: {features_found}")
        else:
            print("  No element metrics (no SAM masks found for this scene)")

        res = {
            "scene": scene_name,
            "Acc-mean": acc,
            "Acc-med": acc_med,
            "Comp-mean": comp,
            "Comp-med": comp_med,
            "Score": (acc + comp) / 2.0,
            **elem_stats,
        }
        results.append(res)
        os.makedirs(scene_output_dir, exist_ok=True)
        metrics_json_path = osp.join(scene_output_dir, "metrics.json")
        with open(metrics_json_path, 'w') as f:
            json.dump(res, f, indent=4)
            
        print(f"[{scene_name}] Score: {res['Score']:.4f} (Acc: {acc:.4f}, Comp: {comp:.4f})")
        
    if results:
        df = pd.DataFrame(results)
        avgs = df.mean(numeric_only=True).to_dict()
        avgs['scene'] = 'AVERAGE'
        df_final = pd.concat([df, pd.DataFrame([avgs])], ignore_index=True)
        
        print("\nSummary:")
        print(df_final.tail(1))
        df_final.to_csv(args.output_csv, index=False)
        print(f"Saved to {args.output_csv}")

        pass
    else:
        print("No valid results computed.")


if __name__ == "__main__":
    main()
