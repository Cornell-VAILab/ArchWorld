# --- repo bootstrap: make the sibling `eval` package importable ---------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE)))
# eval/ modules import each other flatly, so that directory goes on the path too
for _p in (_REPO_ROOT, _os.path.join(_REPO_ROOT, "eval"), _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# -----------------------------------------------------------------------------
import os
import os.path as osp
import csv
import glob
import json
import gc
import time
import torch
import shutil
import numpy as np
import multiprocessing
from queue import Empty as QueueEmpty
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Setup root to allow imports from vggt
root = _REPO_ROOT

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from eval.visual_util import predictions_to_glb
from loss import VGGTAttentionIntervention
from eval.read_write_model import read_model_cameras_images, qvec2rotmat
from eval.evaluate import load_depth_map, pointcloud_to_glb, resolve_depth_path, compute_alignment_with_correspondence
from eval.eval_utils import accuracy, completion

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

PATCH_SIZE = 14
PATCH_START_IDX = 5
NUM_IMAGES = int(os.environ.get("NUM_IMAGES", "50"))
# Override image source: use this folder for listing/copying images (e.g. pts_500m scene). Still use CSV for recon/depth.
IMAGE_FOLDER_ENV = os.environ.get("IMAGE_FOLDER", "").strip()
# Base dir for pts_500m scene images: <PTS_500M_IMAGE_BASE>/<scene_name>/<scene_name>_images
PTS_500M_IMAGE_BASE = Path(os.environ.get("PTS_500M_IMAGE_BASE", ""))
SOURCE_IMAGE_DIR = os.environ.get("SOURCE_IMAGE_DIR", "")
PTS_500M_SCENE_DIR = Path(SOURCE_IMAGE_DIR).parent
# Possible COLMAP sparse recon locations for pts_500m (same image naming as SOURCE_IMAGE_DIR)
PTS_500M_RECON_CANDIDATES = [
    PTS_500M_SCENE_DIR / "dense" / "0" / "sparse",
    PTS_500M_SCENE_DIR / "sparse",
]
PTS_500M_DEPTH_DIR = PTS_500M_SCENE_DIR / "depth"
SAM_MASK_DIR = os.environ.get("SAM_MASK_DIR", "")
# Dir for "artificial architecture" SAM masks (used only when early-layer/combined code is re-enabled)
SAM_ARCHITECTURE_DIR = Path(os.environ.get("SAM_ARCHITECTURE_DIR", ""))
SCENE_NAME = os.environ.get("SCENE_NAME", "Al-Qibli_Chapel").strip() or "Al-Qibli_Chapel"
IMAGE_LIST_JSON = "image_list.json"
# Adaptive per NUM_IMAGES (e.g. images_50 when NUM_IMAGES=50)
IMAGES_DIR = f"images_{NUM_IMAGES}"
# Set to "1" to skip baseline pass: only run intervention pass -> less memory, faster (no pred_base.glb)
SKIP_BASELINE_PASS = os.environ.get("SKIP_BASELINE_PASS", "").strip() == "1"
# When set, skip base eval and copy base metrics from <path>/<scene_name>/metrics.json (e.g. metric_eval output)
METRIC_EVAL_BASE = os.environ.get("METRIC_EVAL_BASE", "").strip()
# (Commented out: early-layer / combined arch+early; we use SAM-only + per-scene tag from ATTENTION_STATS_DIR.)
# USE_EARLY_LAYER_MASK = os.environ.get("USE_EARLY_LAYER_MASK", "1").strip() == "1"
# EARLY_LAYER_INDEX = int(os.environ.get("EARLY_LAYER_INDEX", "3"))
# EARLY_LAYER_TOP_FRACTION = float(os.environ.get("EARLY_LAYER_TOP_FRACTION", "0.2"))
# EXCLUDE_PADDING_FROM_EARLY_MASK = os.environ.get("EXCLUDE_PADDING_FROM_EARLY_MASK", "1").strip() == "1"
# USE_COMBINED_ARCH_EARLY_MASK = os.environ.get("USE_COMBINED_ARCH_EARLY_MASK", "0").strip() == "1"
# Timeout (seconds) for alignment + evaluation per pass; 0 = no timeout (run in-process, same as evaluate.py).
# When >0 we use a subprocess and kill on timeout; subprocess adds pickle overhead.
EVAL_TIMEOUT_SEC = int(os.environ.get("EVAL_TIMEOUT_SEC", "0"))
# Set to "1" to skip alignment + accuracy/completion evaluation (faster; no eval_metrics_*.json).
SKIP_EVAL = os.environ.get("SKIP_EVAL", "").strip() == "1"
PROBE = os.environ.get("PROBE", "").strip() == "1"   # measure peak-mem + wall-clock of base vs intervention passes
TOP_K_ELEMENTS = int(os.environ.get("TOP_K_ELEMENTS", "1"))   # >1: steer the union of the top-k most-attended arch elements instead of a single anchor
# Alignment: match metric_eval/evaluate.py exactly (same procedure, same speed).
# evaluate.py uses 5000 per image (inside compute_alignment_with_correspondence), no global cap, 1000 RANSAC iters.
ALIGNMENT_MAX_POINTS = int(os.environ.get("ALIGNMENT_MAX_POINTS", "0"))  # 0 = no cap (match evaluate.py)
ALIGNMENT_N_ITERATIONS = int(os.environ.get("ALIGNMENT_N_ITERATIONS", "1000"))  # match evaluate.py default
# When set (e.g. to internal_analysis/attention/out), per-scene stats.json is read to pick the most-attended arch tag (max sum) for SAM mask suffix.
ATTENTION_STATS_DIR = os.environ.get("ATTENTION_STATS_DIR", "").strip()
# When set (e.g. 1), compute most-attended feature per layer from that layer's attention and intervene with per-layer features; save to intervention_feature.json
PER_LAYER_INTERVENTION = os.environ.get("PER_LAYER_INTERVENTION", "").strip() in ("1", "true", "yes")
# Per-image building validity masks (bool .npy, native image resolution). Keep only masked-in ("valid") points
# during GT fusion, alignment (RANSAC correspondences), and metric evaluation.
VALIDITY_MASKS_BASE = os.environ.get("VALIDITY_MASKS_BASE", "").strip()
USE_VALIDITY_MASKS = os.environ.get("USE_VALIDITY_MASKS", "1").strip().lower() in ("1", "true", "yes")


import re as _re


def _sanitize_scene_dir_name(name: str) -> str:
    """Safe subfolder name (same rule as run_pass_batch_large_error / analyze_best_worst_per_scene)."""
    s = _re.subn(r'[^\w\-., ()]', "_", name)[0]
    return s.strip("_") or "scene"


def load_validity_masks(scene_name, image_paths, masks_base=None):
    """Return {image_path: bool ndarray (H_img, W_img)} of per-image building validity masks.
    Images in images_<N>/ are named "NNN_<original>", so the lookup strips that prefix.
    Missing masks are simply absent from the dict — downstream code falls back to the unmasked path."""
    if masks_base is None:
        masks_base = VALIDITY_MASKS_BASE
    out = {}
    if not masks_base:
        return out
    base = Path(masks_base)
    cand_dirs = [base / scene_name]
    sanitized = _sanitize_scene_dir_name(scene_name)
    if sanitized != scene_name:
        cand_dirs.append(base / sanitized)
    scene_dir = next((d for d in cand_dirs if d.is_dir()), None)
    if scene_dir is None:
        return out
    for p in image_paths:
        bn = osp.basename(p)
        m = _re.match(r'^\d{3,}_(.+)$', bn)
        original = m.group(1) if m else bn
        candidates = [
            scene_dir / f"{original}.npy",
            scene_dir / f"{Path(original).stem}.npy",
        ]
        for c in candidates:
            if c.exists():
                try:
                    out[p] = np.load(str(c)).astype(bool)
                except Exception:
                    pass
                break
    return out


def _resize_mask_nearest(mask_bool, target_h, target_w):
    """Resize a (H,W) bool mask to (target_h, target_w) using nearest-neighbor."""
    import cv2 as _cv2
    if mask_bool.shape == (target_h, target_w):
        return mask_bool
    resized = _cv2.resize(mask_bool.astype(np.uint8), (target_w, target_h), interpolation=_cv2.INTER_NEAREST)
    return resized.astype(bool)


def compute_alignment_with_correspondence_masked(
    gt_samples, pred_data, scene_output_dir,
    max_total_points=None, n_iterations=1000, image_paths=None,
    validity_masks_by_path=None,
):
    """Same as metric_eval compute_alignment_with_correspondence but also ANDs in each image's validity mask
    (resized to (H_pred, W_pred)) before sampling correspondences. When a sample has no validity mask the behavior
    matches the upstream function for that image."""
    import cv2 as _cv2
    from eval.eval_utils import umeyama_ransac as _umeyama_ransac

    pred_depths = pred_data['depth_map'].cpu().squeeze(0).squeeze(-1).numpy()
    # Processed confidence (sigmoid(log(raw))*2-1+1e-8 > 1e-5 ≈ raw > 1). MVS GT is inherently sparse
    # (~13% of pixels have valid depth — that's the "confidence filter" baked into the GT). This is the
    # analog on VGGT's dense pred, giving comparable sparsity between the two evaluated sets.
    _conf_raw = pred_data['depth_conf'].cpu()
    pred_confs = (torch.sigmoid(torch.log(_conf_raw)) * 2 - 1 + 1e-8).squeeze(0).numpy()
    pred_extrinsics = pred_data['extrinsics'].cpu().squeeze(0).float().numpy()
    pred_intrinsics = pred_data['intrinsics'].cpu().squeeze(0).float().numpy()

    S, H_pred, W_pred = pred_depths.shape

    if image_paths is not None and len(image_paths) == S:
        gt_by_path = {osp.normpath(s["image_path"]): s for s in gt_samples}
        def _get_sample(i): return gt_by_path.get(osp.normpath(image_paths[i]))
        def _path_for(i): return osp.normpath(image_paths[i])
    else:
        def _get_sample(i): return gt_samples[i] if i < len(gt_samples) else None
        def _path_for(i): return osp.normpath(gt_samples[i]["image_path"]) if i < len(gt_samples) else None

    vmasks = validity_masks_by_path or {}
    correspondences_gt, correspondences_pred = [], []
    n_vm_applied = 0
    for i in range(S):
        sample = _get_sample(i)
        if sample is None:
            continue

        gt_depth = sample['depth_map']
        gt_extrinsic = sample['extrinsic']
        R_ref = sample['R_ref']; t_ref = sample['t_ref']
        pred_depth = pred_depths[i]
        pred_K = pred_intrinsics[i]; pred_extrinsic = pred_extrinsics[i]

        gt_depth_resized = _cv2.resize(gt_depth, (W_pred, H_pred), interpolation=_cv2.INTER_NEAREST)
        y, x = np.meshgrid(np.arange(H_pred), np.arange(W_pred), indexing='ij')
        fx, fy = pred_K[0, 0], pred_K[1, 1]
        cx, cy = pred_K[0, 2], pred_K[1, 2]
        Z_pred = pred_depth
        X_pred = (x - cx) * Z_pred / fx
        Y_pred = (y - cy) * Z_pred / fy
        P_cam_pred = np.stack([X_pred, Y_pred, Z_pred], axis=-1)

        R_pred = pred_extrinsic[:3, :3]; t_pred = pred_extrinsic[:3, 3]
        # cam→world for row-vector points: P_world = P_cam @ R_w2c + (-R_w2c.T @ t_w2c)
        # (matches upstream: R_inv = R.T; P_world = P_cam @ R_inv.T + t_inv == P_cam @ R + t_inv)
        P_world_pred = P_cam_pred @ R_pred + (-R_pred.T @ t_pred)

        K_gt_scaled = sample['K'].copy()
        sx = W_pred / gt_depth.shape[1]; sy = H_pred / gt_depth.shape[0]
        K_gt_scaled[0, 0] *= sx; K_gt_scaled[0, 2] *= sx
        K_gt_scaled[1, 1] *= sy; K_gt_scaled[1, 2] *= sy
        fx_gt, fy_gt = K_gt_scaled[0, 0], K_gt_scaled[1, 1]
        cx_gt, cy_gt = K_gt_scaled[0, 2], K_gt_scaled[1, 2]
        Z_gt = gt_depth_resized
        X_gt = (x - cx_gt) * Z_gt / fx_gt
        Y_gt = (y - cy_gt) * Z_gt / fy_gt
        P_cam_gt = np.stack([X_gt, Y_gt, Z_gt], axis=-1)

        R_gt = gt_extrinsic[:3, :3]; t_gt = gt_extrinsic[:3, 3]
        P_world_gt = P_cam_gt @ R_gt + (-R_gt.T @ t_gt)
        P_ref_gt = P_world_gt @ R_ref.T + t_ref
        rot_x_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        P_final_gt = P_ref_gt @ rot_x_180.T

        mask = (Z_pred > 1e-4) & (Z_gt > 1e-4) & np.isfinite(Z_pred) & np.isfinite(Z_gt)
        if pred_confs.ndim == 3:
            mask = mask & (pred_confs[i] > 1e-5)  # same processed-conf filter as pred cloud / GLB
        # Validity mask lookup: prefer sample['validity_mask'] (at GT depth res) — resize to (H_pred, W_pred).
        # Else fall back to vmasks[image_path] (at native image res).
        vm = sample.get('validity_mask') if isinstance(sample, dict) else None
        if vm is None:
            vm = vmasks.get(_path_for(i))
        if vm is not None:
            vm_r = _resize_mask_nearest(vm, H_pred, W_pred)
            mask = mask & vm_r
            n_vm_applied += 1

        pts_pred_flat = P_world_pred[mask]
        pts_gt_flat = P_final_gt[mask]
        if len(pts_pred_flat) > 0:
            if len(pts_pred_flat) > 5000:
                idx = np.random.choice(len(pts_pred_flat), 5000, replace=False)
                pts_pred_flat = pts_pred_flat[idx]
                pts_gt_flat = pts_gt_flat[idx]
            correspondences_pred.append(pts_pred_flat)
            correspondences_gt.append(pts_gt_flat)

    if not correspondences_pred:
        return None, None, None  # caller checks c is None; R/t ignored in that branch
    pts_pred_all = np.concatenate(correspondences_pred, axis=0)
    pts_gt_all = np.concatenate(correspondences_gt, axis=0)
    if max_total_points is not None and len(pts_pred_all) > max_total_points:
        idx = np.random.choice(len(pts_pred_all), max_total_points, replace=False)
        pts_pred_all = pts_pred_all[idx]
        pts_gt_all = pts_gt_all[idx]
    print(f"  Alignment based on {len(pts_pred_all)} corresponding points (validity mask applied to {n_vm_applied}/{S} views).")
    c, R, t = _umeyama_ransac(pts_pred_all.T, pts_gt_all.T, n_iterations=n_iterations)
    return c, R, t


def get_pred_points_masked(pred_data, image_paths, validity_masks_by_path):
    """Build pred cloud from per-view depth with validity mask + depth>0 + processed_conf>1e-5.
    The conf filter is VGGT's analog of the implicit sparsity of MVS GT (which only has depth where it
    can triangulate), so the GT and pred sets end up roughly comparable in density/semantics.
    Same filter as save_prediction_glb and compute_alignment_with_correspondence_masked."""
    pred_depths = pred_data['depth_map'].cpu().squeeze(0).squeeze(-1).numpy()
    _conf_raw = pred_data['depth_conf'].cpu()
    depth_conf_proc = (torch.sigmoid(torch.log(_conf_raw)) * 2 - 1 + 1e-8).squeeze(0).numpy()
    pred_extrinsics = pred_data['extrinsics'].cpu().squeeze(0).float().numpy()
    pred_intrinsics = pred_data['intrinsics'].cpu().squeeze(0).float().numpy()
    S, H_pred, W_pred = pred_depths.shape
    vmasks = validity_masks_by_path or {}
    chunks = []
    for i in range(S):
        pred_depth = pred_depths[i]
        pred_K = pred_intrinsics[i]; pred_extrinsic = pred_extrinsics[i]
        y, x = np.meshgrid(np.arange(H_pred), np.arange(W_pred), indexing='ij')
        fx, fy = pred_K[0, 0], pred_K[1, 1]
        cx, cy = pred_K[0, 2], pred_K[1, 2]
        Z_pred = pred_depth
        X_pred = (x - cx) * Z_pred / fx
        Y_pred = (y - cy) * Z_pred / fy
        P_cam_pred = np.stack([X_pred, Y_pred, Z_pred], axis=-1)
        R_pred = pred_extrinsic[:3, :3]; t_pred = pred_extrinsic[:3, 3]
        # cam→world for row vectors: @ R_w2c (not R_w2c.T) — matches upstream
        P_world_pred = P_cam_pred @ R_pred + (-R_pred.T @ t_pred)
        mask = (Z_pred > 1e-4) & np.isfinite(Z_pred) & (depth_conf_proc[i] > 1e-5)
        if image_paths is not None and i < len(image_paths):
            vm = vmasks.get(osp.normpath(image_paths[i]))
            if vm is not None:
                mask = mask & _resize_mask_nearest(vm, H_pred, W_pred)
        pts = P_world_pred[mask]
        if len(pts) > 0:
            chunks.append(pts.astype(np.float32))
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def discover_scene_features(sam_mask_dir, scene_name):
    """Discover feature names from mask filenames: <stem>_<feature>.npy. Returns sorted unique features."""
    mask_dir = Path(sam_mask_dir) / scene_name / "mask"
    if not mask_dir.exists():
        return []
    features = set()
    for f in mask_dir.glob("*.npy"):
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2:
            features.add(parts[1])
    return sorted(features)


def build_all_tag_patches(image_paths, mask_dir, scene_name, features, Hp, Wp, original_stems=None):
    """Build all_tag_patches[i][feature] = set of patch indices (0-indexed) for image i and feature.
    Same structure as analyze_tag_attention."""
    mask_dir = Path(mask_dir) / scene_name / "mask"
    all_tag_patches = []
    for i, img_path in enumerate(image_paths):
        stem = (original_stems[i] if original_stems and i < len(original_stems) else None) or Path(img_path).stem
        img_tag_patches = {}
        for feature in features:
            npy_path = mask_dir / (stem + f"_{feature}.npy")
            if not npy_path.exists():
                img_tag_patches[feature] = set()
                continue
            arr = np.load(npy_path)
            if arr.ndim == 4:
                arr = arr.sum(axis=(0, 1)) > 0
            elif arr.ndim == 3:
                arr = arr.sum(axis=0) > 0
            img_tag_patches[feature] = map_mask_to_patches(arr.astype(bool), Hp, Wp, image_path=str(img_path))
        all_tag_patches.append(img_tag_patches)
    return all_tag_patches


def _get_image_patch_start(img_idx, patch_start_idx, num_patches_per_img):
    """Token index where patch tokens start for image img_idx."""
    tokens_per_image = patch_start_idx + num_patches_per_img
    return img_idx * tokens_per_image + patch_start_idx


def compute_feature_attention_sum(attn_weights, all_tag_patches, feature, patch_start_idx, num_patches_per_img):
    """Sum of mean attention from tag patches in img_i to tag patches in img_j over all pairs (i,j), i!=j.
    Matches analyze_tag_attention's 'sum' metric per feature."""
    total = 0.0
    num_images = len(all_tag_patches)
    for i in range(num_images):
        for j in range(num_images):
            if i == j:
                continue
            img_i_patches = all_tag_patches[i].get(feature, set())
            img_j_patches = all_tag_patches[j].get(feature, set())
            if len(img_i_patches) == 0 or len(img_j_patches) == 0:
                continue
            img_i_start = _get_image_patch_start(i, patch_start_idx, num_patches_per_img)
            img_j_start = _get_image_patch_start(j, patch_start_idx, num_patches_per_img)
            scores = []
            for p_i in img_i_patches:
                if p_i >= num_patches_per_img:
                    continue
                token_i = img_i_start + p_i
                if token_i >= attn_weights.shape[0]:
                    continue
                for p_j in img_j_patches:
                    if p_j >= num_patches_per_img:
                        continue
                    token_j = img_j_start + p_j
                    if token_j >= attn_weights.shape[1]:
                        continue
                    scores.append(float(attn_weights[token_i, token_j]))
            if scores:
                total += np.mean(scores)
    return total


def compute_per_layer_most_attended_features(model, images, all_tag_patches, features, patch_start_idx, num_patches_per_img, device):
    """Run one forward pass with attention hooks on layers 0-23. For each layer, compute attention sum per feature
    and return best feature per layer. Returns (per_layer_feature: dict[int, str], attention_hooks to remove)."""
    aggregator = model.aggregator
    saved = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inp, out):
            x = inp[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)
            q = q * module.scale
            attn = q @ k.transpose(-2, -1)
            attn_weights = torch.softmax(attn, dim=-1).mean(dim=1).detach().cpu().numpy()[0]
            saved[layer_idx] = attn_weights
        return hook

    for layer in range(24):
        if layer < len(aggregator.global_blocks):
            h = aggregator.global_blocks[layer].attn.register_forward_hook(make_hook(layer))
            hooks.append(h)

    with torch.no_grad():
        _ = aggregator(images)

    per_layer_feature = {}
    for layer in range(24):
        if layer not in saved:
            per_layer_feature[layer] = features[0] if features else "arch"
            continue
        attn = saved[layer]
        best_feat, best_sum = features[0] if features else "arch", -1.0
        for feat in features:
            s = compute_feature_attention_sum(attn, all_tag_patches, feat, patch_start_idx, num_patches_per_img)
            if s > best_sum:
                best_sum = s
                best_feat = feat
        per_layer_feature[layer] = best_feat

    for h in hooks:
        h.remove()
    return per_layer_feature


def get_most_attended_tag(attention_stats_dir, scene_name, default="arch"):
    """Read <attention_stats_dir>/<scene_name>/stats.json and return the tag (key) with largest sum. Masks use stem + '_' + tag + '.npy'."""
    if not attention_stats_dir or not scene_name:
        return default
    path = Path(attention_stats_dir) / scene_name / "stats.json"
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return default
    best_tag, best_sum = default, -1.0
    for tag, stats in data.items():
        if isinstance(stats, dict) and "sum" in stats:
            try:
                s = float(stats["sum"])
                if s > best_sum:
                    best_sum = s
                    best_tag = tag
            except (TypeError, ValueError):
                pass
    return best_tag


def get_topk_attended_tags(attention_stats_dir, scene_name, k, default="arch"):
    """Return up to k tags with the largest 'sum' in stats.json (descending). Falls back to [default]."""
    if not attention_stats_dir or not scene_name:
        return [default]
    path = Path(attention_stats_dir) / scene_name / "stats.json"
    if not path.exists():
        return [default]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return [default]
    scored = []
    for tag, stats in data.items():
        if isinstance(stats, dict) and "sum" in stats:
            try:
                scored.append((float(stats["sum"]), tag))
            except (TypeError, ValueError):
                pass
    if not scored:
        return [default]
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:k]]


def save_aligned_combined_glb(gt_points, pred_points_aligned, output_path, save_n=500000):
    """Save combined GLB with GT (green) and aligned prediction (red), same as evaluate.py. Reused for base and inter."""
    if not TRIMESH_AVAILABLE:
        return
    os.makedirs(osp.dirname(output_path) or ".", exist_ok=True)
    if len(gt_points) > save_n:
        idx = np.random.choice(len(gt_points), save_n, replace=False)
        p_gt = gt_points[idx]
    else:
        p_gt = gt_points
    pc_gt = trimesh.PointCloud(vertices=p_gt)
    pc_gt.colors = np.tile([0, 255, 0, 255], (len(p_gt), 1)).astype(np.uint8)
    if len(pred_points_aligned) > save_n:
        idx = np.random.choice(len(pred_points_aligned), save_n, replace=False)
        p_pred = pred_points_aligned[idx]
    else:
        p_pred = pred_points_aligned
    pc_pred = trimesh.PointCloud(vertices=p_pred)
    pc_pred.colors = np.tile([255, 0, 0, 255], (len(p_pred), 1)).astype(np.uint8)
    scene_comb = trimesh.Scene([pc_gt, pc_pred])
    scene_comb.export(output_path, file_type="glb")
    print(f"  Saved aligned combined GLB to {output_path}")


def _per_pixel_world_and_valid(pred_data, image_paths, validity_masks_by_path=None):
    """Unproject per-view depth to per-pixel world points (S,H,W,3) + per-pixel valid mask (S,H,W),
    using the exact same unprojection + conf>1e-5 (+ validity mask) filter as get_pred_points_masked.
    Keeping the pixel grid lets us build base<->inter correspondences (same pixel = same surface point)."""
    depths = pred_data["depth_map"].cpu().squeeze(0).squeeze(-1).numpy()
    conf_raw = pred_data["depth_conf"].cpu()
    conf = (torch.sigmoid(torch.log(conf_raw)) * 2 - 1 + 1e-8).squeeze(0).numpy()
    ext = pred_data["extrinsics"].cpu().squeeze(0).float().numpy()
    intr = pred_data["intrinsics"].cpu().squeeze(0).float().numpy()
    S, H, W = depths.shape
    world = np.full((S, H, W, 3), np.nan, dtype=np.float32)
    valid = np.zeros((S, H, W), dtype=bool)
    vmasks = validity_masks_by_path or {}
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    for i in range(S):
        Z = depths[i]; K = intr[i]; E = ext[i]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        P_cam = np.stack([(x - cx) * Z / fx, (y - cy) * Z / fy, Z], axis=-1)
        R = E[:3, :3]; t = E[:3, 3]
        world[i] = P_cam @ R + (-R.T @ t)
        m = (Z > 1e-4) & np.isfinite(Z) & (conf[i] > 1e-5)
        if image_paths is not None and i < len(image_paths):
            vm = vmasks.get(image_paths[i])
            if vm is None:
                vm = vmasks.get(osp.normpath(image_paths[i]))
            if vm is not None:
                m = m & _resize_mask_nearest(vm, H, W)
        valid[i] = m & np.isfinite(world[i]).all(axis=-1)
    return world, valid


def save_inter_error_colored_glb(pred_data_base, pred_data_inter, image_paths, output_path,
                                 validity_masks_by_path=None, max_points=500000):
    """Align the intervened prediction to the base prediction (per-pixel-correspondence Sim3, same
    umeyama_ransac as the metric eval), then color every intervened point by its nearest-neighbour
    distance to the base cloud (the accuracy-metric distance). Saves a GLB of the (aligned) intervened
    points colored by error magnitude (turbo: blue=small change, red=large change)."""
    if not TRIMESH_AVAILABLE:
        return
    try:
        from scipy.spatial import cKDTree
        import matplotlib.cm as _cm
        from eval.eval_utils import umeyama_ransac
    except Exception as e:
        print(f"  error-colored GLB skipped (import failed: {e})")
        return

    base_world, base_valid = _per_pixel_world_and_valid(pred_data_base, image_paths, validity_masks_by_path)
    inter_world, inter_valid = _per_pixel_world_and_valid(pred_data_inter, image_paths, validity_masks_by_path)
    base_pts = base_world[base_valid]
    inter_pts = inter_world[inter_valid]
    if len(base_pts) < 100 or len(inter_pts) < 100:
        print("  error-colored GLB skipped (too few points)")
        return

    # Sim3 align inter -> base on pixels valid in BOTH (exact per-pixel correspondence).
    corr = base_valid & inter_valid
    c, R, t = 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    n_corr = int(corr.sum())
    if n_corr >= 100:
        bc = base_world[corr]; ic = inter_world[corr]
        if len(bc) > 30000:
            idx = np.random.choice(len(bc), 30000, replace=False)
            bc, ic = bc[idx], ic[idx]
        try:
            c, R, t = umeyama_ransac(ic.T, bc.T, n_iterations=1000)
            t = np.asarray(t).reshape(3)
            if not np.isfinite(c):
                c, R, t = 1.0, np.eye(3), np.zeros(3)
        except Exception:
            c, R, t = 1.0, np.eye(3), np.zeros(3)
    aligned_inter = (c * (R @ inter_pts.T)).T + t

    # Per-point error = nearest-neighbour distance to the base cloud (accuracy-metric distance).
    dist, _ = cKDTree(base_pts).query(aligned_inter, workers=-1)
    vmax = float(np.percentile(dist, 95)) or 1e-8
    vmax = max(vmax, 1e-8)
    colors = (_cm.turbo(np.clip(dist / vmax, 0.0, 1.0))[:, :3] * 255).astype(np.uint8)

    pts = aligned_inter
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts, colors, dist = pts[idx], colors[idx], dist[idx]
    pc = trimesh.PointCloud(vertices=pts.astype(np.float32),
                            colors=np.concatenate([colors, np.full((len(colors), 1), 255, np.uint8)], axis=1))
    scene = trimesh.Scene([pc])
    # Match pred_base/pred_inter.glb orientation: predictions_to_glb applies apply_scene_alignment
    # (OpenGL conversion + 180 deg about Y + inverse of the first camera). Our aligned_inter points
    # live in the base frame, so apply the same transform using the base extrinsics.
    try:
        from eval.visual_util import apply_scene_alignment
        ext_b = pred_data_base["extrinsics"].cpu().squeeze(0).float().numpy()
        extm = np.zeros((ext_b.shape[0], 4, 4), dtype=np.float64)
        extm[:, :3, :4] = ext_b[:, :3, :4]
        extm[:, 3, 3] = 1.0
        scene = apply_scene_alignment(scene, extm)
    except Exception as e:
        print(f"  (scene alignment for error GLB skipped: {e})")
    os.makedirs(osp.dirname(output_path) or ".", exist_ok=True)
    scene.export(output_path, file_type="glb")
    print(f"  Saved error-colored inter GLB to {output_path} "
          f"(n={len(pts)}, scale c={c:.4f}, median_err={float(np.median(dist)):.6f}, p95(color max)={vmax:.6f})")


def _run_alignment_and_eval_worker(
    gt_points, gt_samples, pred_data_cpu, image_paths, work_dir_str, eval_metrics_path_str,
    first_layer, last_layer, pass_name,
    max_total_points=None,
    n_iterations=1000,
    validity_masks_by_path=None,
):
    """Run alignment + accuracy/completion (same procedure as metric_eval/evaluate.py). Returns (status_str, metrics_dict or None).
    When validity_masks_by_path is provided (or gt_samples carry 'validity_mask'), alignment correspondences
    and the pred point cloud are both restricted to masked-in pixels."""
    try:
        any_sample_vm = any(isinstance(s, dict) and s.get('validity_mask') is not None for s in gt_samples)
        use_masked = bool(validity_masks_by_path) or any_sample_vm
        if use_masked:
            c, R, t = compute_alignment_with_correspondence_masked(
                gt_samples, pred_data_cpu, work_dir_str,
                max_total_points=max_total_points,
                n_iterations=n_iterations,
                image_paths=image_paths,
                validity_masks_by_path=validity_masks_by_path,
            )
        else:
            c, R, t = compute_alignment_with_correspondence(
                gt_samples, pred_data_cpu, work_dir_str,
                max_total_points=max_total_points,
                n_iterations=n_iterations,
                image_paths=image_paths,
            )
        if c is None or not np.isfinite(c):
            return ("alignment_failed", None)
        if use_masked:
            pred_points = get_pred_points_masked(pred_data_cpu, image_paths, validity_masks_by_path)
        else:
            cache_dir = osp.join(work_dir_str, f"temp_{pass_name}_cache")
            pred_points = get_pred_points_clean(pred_data_cpu, image_paths, cache_dir=cache_dir)
        if pred_points is None:
            return ("no_pred_points", None)
        pred_aligned = (c * R @ pred_points.T + t).T
        # Optional ICP refinement on the full clouds after the coarse Sim(3) alignment
        # (copied from jot-jt/extreme-view-3dfm-gen-eval mv_recon/eval.py L118: open3d point-to-point ICP,
        #  init=identity since coarse Sim(3) is already applied; threshold=0.1 for non-DTU).
        if os.environ.get("USE_ICP", "0") == "1":
            import open3d as o3d
            _icp_thr = float(os.environ.get("ICP_THRESHOLD", "0.1"))
            _icp_iter = int(os.environ.get("ICP_MAX_ITER", "0"))  # 0 = open3d default criteria
            _pcd = o3d.geometry.PointCloud(); _pcd.points = o3d.utility.Vector3dVector(np.asarray(pred_aligned, dtype=np.float64))
            _pcd_gt = o3d.geometry.PointCloud(); _pcd_gt.points = o3d.utility.Vector3dVector(np.asarray(gt_points, dtype=np.float64))
            if _icp_iter > 0:
                _reg = o3d.pipelines.registration.registration_icp(
                    _pcd, _pcd_gt, _icp_thr, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=_icp_iter))
            else:
                _reg = o3d.pipelines.registration.registration_icp(
                    _pcd, _pcd_gt, _icp_thr, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
            _T = np.asarray(_reg.transformation)
            pred_aligned = (pred_aligned @ _T[:3, :3].T) + _T[:3, 3]
        acc, acc_med = accuracy(gt_points, pred_aligned)
        comp, comp_med = completion(gt_points, pred_aligned)
        score = (acc + comp) / 2  # mean-based Score (match metric_eval)
        metrics = {}
        if pass_name == "inter" and osp.exists(eval_metrics_path_str):
            with open(eval_metrics_path_str) as f:
                metrics = json.load(f)
        metrics["first_intervention_layer"] = first_layer
        metrics["last_intervention_layer"] = last_layer
        metrics[pass_name] = {
            "Acc-mean": float(acc),
            "Acc-med": float(acc_med),
            "Comp-mean": float(comp),
            "Comp-med": float(comp_med),
            "Score": float(score),
        }
        # Save base/inter aligned to GT combined GLB (reuse same logic as evaluate.py)
        combined_glb_path = osp.join(work_dir_str, f"{pass_name}_aligned_combined.glb")
        save_aligned_combined_glb(gt_points, pred_aligned, combined_glb_path)
        return ("ok", metrics)
    except Exception as e:
        return ("error", str(e))


def _run_alignment_and_eval_worker_put_in_queue(
    result_queue, gt_points, gt_samples, pred_data_cpu, image_paths,
    work_dir_str, eval_metrics_path_str, first_layer, last_layer, pass_name,
    max_total_points=None,
    n_iterations=1000,
    validity_masks_by_path=None,
):
    """Wrapper for Process: run worker and put (status, metrics) in queue so parent can get(timeout=...) and terminate on timeout."""
    status, metrics = _run_alignment_and_eval_worker(
        gt_points, gt_samples, pred_data_cpu, image_paths,
        work_dir_str, eval_metrics_path_str, first_layer, last_layer, pass_name,
        max_total_points=max_total_points,
        n_iterations=n_iterations,
        validity_masks_by_path=validity_masks_by_path,
    )
    result_queue.put((status, metrics))


def get_content_patch_mask(image_paths, processed_H, processed_W, target_size=518, mode="crop"):
    """
    Return (S, Hp*Wp) tensor: 1.0 for content patches, 0.0 for padding.
    Mirrors load_and_preprocess_images so we know which patch rows/cols are padding (white).
    """
    Hp = processed_H // PATCH_SIZE
    Wp = processed_W // PATCH_SIZE
    mask_list = []
    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            if img.mode == "RGBA":
                img = img.convert("RGB")
            else:
                img = img.convert("RGB")
            width, height = img.size
        except Exception:
            mask_list.append(np.ones((Hp * Wp), dtype=np.float32))
            continue
        if mode == "crop":
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
            if new_height > target_size:
                content_h, content_w = target_size, target_size
            else:
                content_h, content_w = new_height, target_size
        else:
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
                content_h, content_w = new_height, target_size
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
                content_h, content_w = target_size, new_width
        # After batch padding to (processed_H, processed_W), content is centered
        pad_top = (processed_H - content_h) // 2
        pad_left = (processed_W - content_w) // 2
        # Valid patch rows: overlap with [pad_top, pad_top+content_h)
        r_min = max(0, (pad_top + 14 - 1) // 14)
        r_max = min(Hp - 1, (pad_top + content_h - 1) // 14)
        c_min = max(0, (pad_left + 14 - 1) // 14)
        c_max = min(Wp - 1, (pad_left + content_w - 1) // 14)
        patch_mask = np.zeros((Hp, Wp), dtype=np.float32)
        patch_mask[r_min : r_max + 1, c_min : c_max + 1] = 1.0
        mask_list.append(patch_mask.flatten())
    return np.stack(mask_list, axis=0)


def _use_pts_500m_paths(image_paths):
    """True if images use pts_500m-style names (e.g. 00_000000.jpg) so we should use pts_500m recon/depth."""
    if not image_paths:
        return False
    basename = osp.basename(image_paths[0])
    return "_" in basename and basename.lower().endswith((".jpg", ".jpeg", ".png"))


def load_scene_info(script_dir: Path, scene_name: str, image_folder_override: str = None):
    """Load scene_info from geocoded CSV. If image_folder_override is None, use original image_folder (for GT)."""
    csv_path = script_dir.parent / "data" / "geocoded_depth_maps_updated.csv"
    if not osp.exists(csv_path):
        raise FileNotFoundError(f"Geocoded CSV not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Category Name"] == scene_name:
                img_folder = image_folder_override if image_folder_override is not None else row["Image Folder"]
                return {
                    "recon_path": row["Recon Path"],
                    "depth_folder": row["Depth Map Folder"],
                    "image_folder": img_folder,
                }
    raise KeyError(f"Scene {scene_name} not found in {csv_path}")


def map_mask_to_patches(mask, Hp, Wp, image_path=None, patch_size=PATCH_SIZE, mode="crop"):
    """Map binary mask [H,W] (in original image coords) to set of patch indices (0-indexed).
    When image_path is provided, applies the same resize-then-crop as load_and_preprocess_images
    so the mask aligns with the processed image before mapping to patches."""
    if mask is None or mask.sum() == 0:
        return set()
    H, W = mask.shape
    target_H, target_W = Hp * patch_size, Wp * patch_size

    if image_path is not None and osp.exists(image_path):
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            width, height = W, H
        # Same transform as load_and_preprocess_images (crop mode)
        target_size = 518
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
        # Resize mask to match image resize
        m_resized = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).convert("L").resize(
                (new_width, new_height), Image.Resampling.NEAREST
            )
        ) > 127
        # Center crop if needed (same as load_and_preprocess_images)
        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            mask = m_resized[start_y : start_y + target_size, :]
        else:
            mask = m_resized
        # If result is smaller than target (e.g. portrait with new_height < 518), pad or leave as-is.
        # Model pads to max dims; for 518x518 crop case we have exactly 518x518.
        if mask.shape[0] != target_H or mask.shape[1] != target_W:
            # Fallback: resize to target (e.g. when new_height < 518 and tensor gets padded)
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8) * 255).convert("L").resize(
                    (target_W, target_H), Image.Resampling.NEAREST
                )
            ) > 127
    elif H != target_H or W != target_W:
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).convert("L").resize(
                (target_W, target_H), Image.Resampling.NEAREST
            )
        ) > 127

    patch_mask = mask.reshape(Hp, patch_size, Wp, patch_size).mean(axis=(1, 3))
    idx = np.where(patch_mask > 0.5)
    return set(r * Wp + c for r, c in zip(idx[0], idx[1]))


def build_arch_token_mask(image_paths, sam_mask_dir, scene_name, processed_H, processed_W, mask_suffix="_arch.npy", original_stems=None, verbose=True):
    """Build (N,) token mask for all images: 1 where token is an arch patch.
    mask_suffix: e.g. '_arch.npy' or '_artificial architecture.npy' for architecture-only SAM output.
    original_stems: optional list of stems (one per image) for mask lookup when dest filenames were uniquified (e.g. 000_foo.jpg -> use stem 'foo' for mask).
    verbose: if False, suppress per-stem print (used when building many masks for per-layer intervention)."""
    Hp, Wp = processed_H // PATCH_SIZE, processed_W // PATCH_SIZE
    num_patches_per_img = Hp * Wp
    tokens_per_image = PATCH_START_IDX + num_patches_per_img
    N = len(image_paths) * tokens_per_image
    mask = torch.zeros(N, dtype=torch.float32)
    masks_found = 0
    for i, img_path in enumerate(image_paths):
        stem = (original_stems[i] if original_stems is not None and i < len(original_stems) else None) or Path(img_path).stem
        npy_path = Path(sam_mask_dir) / scene_name / "mask" / (stem + mask_suffix)
        if verbose:
            print(f"  Checking SAM mask for {stem}: looking for {npy_path}")
        if not npy_path.exists():
            continue
        masks_found += 1
        arr = np.load(npy_path)
        if arr.ndim == 4:
            arr = arr.sum(axis=(0, 1)) > 0
        elif arr.ndim == 3:
            arr = arr.sum(axis=0) > 0
        arch_patches = map_mask_to_patches(arr.astype(bool), Hp, Wp, image_path=img_path)
        if len(arch_patches) == 0:
            print(f"    {stem}: mask found but no patches selected")
        patch_start = i * tokens_per_image + PATCH_START_IDX
        for p in arch_patches:
            if p < num_patches_per_img:
                mask[patch_start + p] = 1.0
    print(f"  SAM masks found for {masks_found}/{len(image_paths)} images")
    return mask


def compute_early_layer_intervention_mask(
    aggregated_tokens_list,
    patch_start_idx: int,
    early_layer_index: int = 4,
    top_fraction: float = 0.25,
    device=None,
    content_mask=None,
):
    """
    Build a token-level intervention mask from early VGGT layers (DINO-like features).
    Tokens with strong cross-image correspondence (high similarity to best match in other views)
    are selected; later layers are encouraged to attend to these.
    Scene-agnostic: no SAM or semantic labels needed.

    Args:
        aggregated_tokens_list: list of [B, S, P, 2C] from model.aggregator(images)
        patch_start_idx: number of special tokens per image (5 for VGGT)
        early_layer_index: which layer output to use (0-indexed; 4 = first 5 blocks)
        top_fraction: fraction of patch tokens to mark as 1 (by correspondence score)
        device: target device for mask (default: same as tokens)
        content_mask: optional (S, num_patches) float, 1=content 0=padding; padding patches are excluded from selection

    Returns:
        mask: (N,) float, N = B*S*P, 1.0 for selected patch tokens, 0 elsewhere
    """
    import torch.nn.functional as F
    early = aggregated_tokens_list[early_layer_index]
    if device is None:
        device = early.device
    B, S, P, C = early.shape
    num_patches = P - patch_start_idx
    if num_patches <= 0:
        return torch.zeros(B * S * P, dtype=torch.float32, device=device)
    # Patch tokens only [B, S, num_patches, C]
    patch_tokens = early[:, :, patch_start_idx:, :].float()
    patch_tokens = F.normalize(patch_tokens, dim=-1)
    # Correspondence score per (batch, image, patch): mean over other images of (max similarity to any patch in that image)
    scores = torch.zeros(B, S, num_patches, dtype=patch_tokens.dtype, device=patch_tokens.device)
    for s in range(S):
        others = [s_prime for s_prime in range(S) if s_prime != s]
        if not others:
            scores[:, s, :] = 1.0
            continue
        # f_s: [B, num_patches, C]
        f_s = patch_tokens[:, s, :, :]
        for s_other in others:
            f_o = patch_tokens[:, s_other, :, :]
            sim = torch.einsum("bpc,bqc->bpq", f_s, f_o)
            max_sim, _ = sim.max(dim=-1)
            scores[:, s, :] = scores[:, s, :] + max_sim
        scores[:, s, :] = scores[:, s, :] / len(others)
    # Exclude padding: padding patches have artificially high correspondence (same white), so zero their scores
    if content_mask is not None:
        cm = torch.as_tensor(content_mask, dtype=scores.dtype, device=scores.device)
        if cm.shape == (S, num_patches):
            scores = scores * cm.unsqueeze(0)
        else:
            cm = cm[:S, :num_patches]
            scores = scores * cm.unsqueeze(0)
    # Flatten to [B, S*num_patches], take top fraction (only among content patches if content_mask was set)
    scores_flat = scores.reshape(B, -1)
    k = max(1, int(scores_flat.numel() * top_fraction))
    threshold = torch.topk(scores_flat.view(-1), k, largest=True).values[-1]
    binary_flat = (scores_flat >= threshold).float()
    # Map to full token sequence: N = B*S*P
    full_mask = torch.zeros(B, S * P, dtype=torch.float32, device=device)
    for s in range(S):
        start = s * P + patch_start_idx
        end = s * P + P
        full_mask[:, start:end] = binary_flat[:, s * num_patches : (s + 1) * num_patches]
    return full_mask.view(-1)


def compute_early_layer_correspondence_scores(
    aggregated_tokens_list,
    patch_start_idx: int,
    early_layer_index: int = 4,
    device=None,
    content_mask=None,
):
    """
    Return (N,) correspondence scores for each token: 0 for non-patch tokens,
    and for patch tokens the mean over other views of max similarity to any patch in that view.
    Used by compute_combined_arch_early_mask to rank SAM architecture patches.
    """
    import torch.nn.functional as F
    early = aggregated_tokens_list[early_layer_index]
    if device is None:
        device = early.device
    B, S, P, C = early.shape
    num_patches = P - patch_start_idx
    if num_patches <= 0:
        return torch.zeros(B * S * P, dtype=torch.float32, device=device)
    patch_tokens = early[:, :, patch_start_idx:, :].float()
    patch_tokens = F.normalize(patch_tokens, dim=-1)
    scores = torch.zeros(B, S, num_patches, dtype=patch_tokens.dtype, device=patch_tokens.device)
    for s in range(S):
        others = [s_prime for s_prime in range(S) if s_prime != s]
        if not others:
            scores[:, s, :] = 1.0
            continue
        f_s = patch_tokens[:, s, :, :]
        for s_other in others:
            f_o = patch_tokens[:, s_other, :, :]
            sim = torch.einsum("bpc,bqc->bpq", f_s, f_o)
            max_sim, _ = sim.max(dim=-1)
            scores[:, s, :] = scores[:, s, :] + max_sim
        scores[:, s, :] = scores[:, s, :] / len(others)
    if content_mask is not None:
        cm = torch.as_tensor(content_mask, dtype=scores.dtype, device=scores.device)
        if cm.shape == (S, num_patches):
            scores = scores * cm.unsqueeze(0)
        else:
            cm = cm[:S, :num_patches]
            scores = scores * cm.unsqueeze(0)
    # Map to full sequence (N,): non-patch positions get 0
    full_scores = torch.zeros(B, S * P, dtype=torch.float32, device=device)
    for s in range(S):
        start = s * P + patch_start_idx
        end = s * P + P
        full_scores[:, start:end] = scores[:, s, :].float()
    return full_scores.view(-1)


def compute_combined_arch_early_mask(
    aggregated_tokens_list,
    arch_token_mask_sam,
    patch_start_idx: int,
    early_layer_index: int = 4,
    top_fraction: float = 0.25,
    device=None,
    content_mask=None,
):
    """
    Combined mask: only SAM "artificial architecture" patches that are also in the
    top fraction by early-layer correspondence. So we encourage attention to
    high-correspondence architecture patches only.
    arch_token_mask_sam: (N,) 1.0 on SAM architecture tokens (e.g. from build_arch_token_mask
                         with SAM_ARCHITECTURE_DIR and mask_suffix="_artificial architecture.npy").
    Returns (N,) float, 1.0 for selected tokens.
    """
    scores = compute_early_layer_correspondence_scores(
        aggregated_tokens_list, patch_start_idx, early_layer_index, device=device, content_mask=content_mask
    )
    if device is None:
        device = scores.device
    arch_sam = arch_token_mask_sam.to(device=device, dtype=torch.float32)
    if arch_sam.shape != scores.shape:
        arch_sam = arch_sam[: scores.numel()].to(device)
    arch_idx = (arch_sam > 0.5).nonzero(as_tuple=True)[0]
    if arch_idx.numel() == 0:
        return torch.zeros_like(arch_sam, device=device)
    arch_scores = scores[arch_idx]
    k = max(1, int(arch_idx.numel() * top_fraction))
    _, topk_pos = torch.topk(arch_scores, k, largest=True)
    combined = torch.zeros_like(arch_sam, device=device)
    combined[arch_idx[topk_pos]] = 1.0
    return combined


def visualize_intervention_patches(
    mask_1d,
    image_paths,
    patch_start_idx: int,
    Hp: int,
    Wp: int,
    patch_size: int,
    output_dir,
    prefix: str = "intervention_patches",
    intervention_tag: str = None,
):
    """
    For each image, overlay which patches are selected (mask=1) and save.
    Also save a montage of all overlays.
    mask_1d: (N,) with N = S * P, P = patch_start_idx + Hp*Wp.
    intervention_tag: if set (e.g. "arch", "dome"), included in filename and figure title so the montage clearly shows which feature was used.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    S = len(image_paths)
    P = patch_start_idx + Hp * Wp
    num_patches = Hp * Wp
    mask_np = mask_1d.cpu().numpy() if hasattr(mask_1d, "cpu") else np.asarray(mask_1d)
    if mask_np.size != S * P:
        mask_np = mask_np[: S * P]
    overlay_h, overlay_w = Hp * patch_size, Wp * patch_size
    # Load preprocessed images so patch grid aligns (same resize+crop as model input)
    try:
        preprocessed = load_and_preprocess_images(image_paths)
        # (N, 3, H, W) tensor in [0,1], convert to (N, H, W, 3) uint8
        preproc_np = (preprocessed.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    except Exception:
        preproc_np = None
    figs = []
    for s in range(S):
        start = s * P + patch_start_idx
        end = s * P + P
        patch_mask = mask_np[start:end].reshape(Hp, Wp).astype(np.float32)
        patch_mask_upsampled = np.repeat(
            np.repeat(patch_mask, patch_size, axis=0), patch_size, axis=1
        )
        if preproc_np is not None and s < len(preproc_np):
            img = preproc_np[s]
        else:
            try:
                img = np.array(Image.open(image_paths[s]).convert("RGB"))
            except Exception:
                img = np.zeros((overlay_h, overlay_w, 3), dtype=np.uint8)
        h, w = img.shape[0], img.shape[1]
        if (h, w) != (overlay_h, overlay_w):
            patch_mask_resized = np.array(
                Image.fromarray((patch_mask_upsampled * 255).astype(np.uint8)).resize(
                    (w, h), Image.Resampling.NEAREST
                )
            ).astype(np.float32) / 255.0
        else:
            patch_mask_resized = patch_mask_upsampled
        overlay = img.copy().astype(np.float32)
        green = np.array([0, 255, 0], dtype=np.float32)
        alpha = 0.5 * np.stack([patch_mask_resized] * 3, axis=-1)  # slightly stronger overlay
        overlay = overlay * (1 - alpha) + green * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        figs.append(overlay)
    n_show = len(figs)
    if n_show > 0:
        cols = min(5, n_show)
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif axes.ndim == 1:
            axes = axes.reshape(1, -1)
        for idx, ax in enumerate(axes.flat):
            if idx < n_show:
                ax.imshow(figs[idx])
                ax.set_title(Path(image_paths[idx]).name[:20], fontsize=8)
            ax.axis("off")
        tag_label = f" ({intervention_tag})" if intervention_tag else ""
        fig.suptitle(f"Intervention patches{tag_label} per image", fontsize=12)
        fig.tight_layout()
        montage_name = f"{prefix}_{intervention_tag}_montage.png" if intervention_tag else f"{prefix}_montage.png"
        montage_path = output_dir / montage_name
        fig.savefig(montage_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved patch montage to {montage_path}")


def ensure_10_images(
    script_dir: Path,
    scene_name_override: str = None,
    csv_script_dir: Path = None,
):
    """Sample NUM_IMAGES: prefer IMAGE_FOLDER env, then CSV scene image folder, else SOURCE_IMAGE_DIR. Copy to images_<N>.
    When scene_name_override and csv_script_dir are set (batch mode), use that scene from CSV only; no SOURCE_IMAGE_DIR fallback.
    When IMAGE_FOLDER env is set, use that folder as source (all images up to NUM_IMAGES).
    """
    images_10_path = script_dir / IMAGES_DIR
    list_path = script_dir / IMAGE_LIST_JSON
    if IMAGE_FOLDER_ENV and osp.exists(IMAGE_FOLDER_ENV):
        all_paths = sorted(glob.glob(osp.join(IMAGE_FOLDER_ENV, "*")))
        all_paths = [p for p in all_paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]
        n_use = min(NUM_IMAGES, len(all_paths)) if all_paths else 0
        if n_use < 1:
            raise RuntimeError(f"IMAGE_FOLDER has {len(all_paths or [])} images, need at least 1")
        selected = all_paths[:n_use]
        images_10_path.mkdir(parents=True, exist_ok=True)
        basenames = [os.path.basename(p) for p in selected]
        for src in selected:
            shutil.copy2(src, images_10_path / os.path.basename(src))
        with open(list_path, "w") as f:
            json.dump({"paths": [str(images_10_path / n) for n in basenames], "basenames": basenames}, f, indent=2)
        print(f"Using {n_use} images from IMAGE_FOLDER -> {images_10_path}")
        return [str(images_10_path / n) for n in basenames]
    use_scene = scene_name_override is not None and csv_script_dir is not None
    scene_name = scene_name_override if use_scene else SCENE_NAME
    # Prefer pts_500m image folder for every scene: <PTS_500M_IMAGE_BASE>/<scene_name>/<scene_name>_images
    pts_500m_folder = PTS_500M_IMAGE_BASE / scene_name / f"{scene_name}_images"
    if osp.exists(pts_500m_folder):
        all_paths = sorted(glob.glob(osp.join(pts_500m_folder, "*")))
        all_paths = [p for p in all_paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]
        if all_paths and len(all_paths) >= NUM_IMAGES:
            indices = np.linspace(0, len(all_paths) - 1, NUM_IMAGES, dtype=int)
            selected = [all_paths[i] for i in indices]
            images_10_path.mkdir(parents=True, exist_ok=True)
            basenames = [os.path.basename(p) for p in selected]
            for src in selected:
                shutil.copy2(src, images_10_path / os.path.basename(src))
            with open(list_path, "w") as f:
                json.dump({"paths": [str(images_10_path / n) for n in basenames], "basenames": basenames}, f, indent=2)
            print(f"Using {NUM_IMAGES} images from pts_500m: {pts_500m_folder}")
            return [str(images_10_path / n) for n in basenames]
        elif all_paths and len(all_paths) >= 1:
            n_use = min(NUM_IMAGES, len(all_paths))
            selected = all_paths[:n_use]
            images_10_path.mkdir(parents=True, exist_ok=True)
            basenames = [os.path.basename(p) for p in selected]
            for src in selected:
                shutil.copy2(src, images_10_path / os.path.basename(src))
            with open(list_path, "w") as f:
                json.dump({"paths": [str(images_10_path / n) for n in basenames], "basenames": basenames}, f, indent=2)
            print(f"Using {n_use} images from pts_500m: {pts_500m_folder}")
            return [str(images_10_path / n) for n in basenames]
    all_paths = None
    try:
        csv_dir = csv_script_dir if use_scene else script_dir
        scene_info = load_scene_info(csv_dir, scene_name, image_folder_override=None)
        orig_folder = scene_info["image_folder"]
        if osp.exists(orig_folder):
            all_paths = sorted(glob.glob(osp.join(orig_folder, "*")))
            all_paths = [p for p in all_paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]
            if len(all_paths) >= NUM_IMAGES:
                print(f"Using original dataset folder for 10 images: {orig_folder}")
    except (FileNotFoundError, KeyError):
        if use_scene:
            raise
        pass
    if not all_paths or len(all_paths) < NUM_IMAGES:
        if use_scene:
            raise RuntimeError(f"Scene {scene_name_override} has {len(all_paths or [])} images, need {NUM_IMAGES}")
        all_paths = sorted(glob.glob(os.path.join(SOURCE_IMAGE_DIR, "*")))
        all_paths = [p for p in all_paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(all_paths) < NUM_IMAGES:
            raise RuntimeError(f"Source has {len(all_paths)} images, need {NUM_IMAGES}")
        print(f"Using pts_500m folder (GT may fail if depth/recon use different names): {SOURCE_IMAGE_DIR}")
    indices = np.linspace(0, len(all_paths) - 1, NUM_IMAGES, dtype=int)
    selected = [all_paths[i] for i in indices]
    images_10_path.mkdir(parents=True, exist_ok=True)
    basenames = [os.path.basename(p) for p in selected]
    for src in selected:
        shutil.copy2(src, images_10_path / os.path.basename(src))
    with open(list_path, "w") as f:
        json.dump({"paths": [str(images_10_path / n) for n in basenames], "basenames": basenames}, f, indent=2)
    print(f"Sampled {NUM_IMAGES} images -> {images_10_path}")
    return [str(images_10_path / n) for n in basenames]


def get_image_paths(
    script_dir: Path,
    scene_name_override: str = None,
    csv_script_dir: Path = None,
):
    """Use the images in images_<N>. Prefer image_list.json (use all paths; batch-prepared list is authoritative).
    Else glob folder and cap at NUM_IMAGES. Only sample from CSV/pts_500m if folder is empty or missing.
    When IMAGE_FOLDER env is set, always re-sample from that folder (ignore existing images dir)."""
    images_10_path = script_dir / IMAGES_DIR
    list_path = script_dir / IMAGE_LIST_JSON
    if not IMAGE_FOLDER_ENV and list_path.exists():
        try:
            with open(list_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            listed = data.get("paths", [])
            if listed:
                return listed
        except (json.JSONDecodeError, OSError):
            pass
    if not IMAGE_FOLDER_ENV and images_10_path.exists():
        all_paths = sorted(glob.glob(osp.join(images_10_path, "*")))
        all_paths = [p for p in all_paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]
        if all_paths:
            selected = all_paths[:NUM_IMAGES] if len(all_paths) >= NUM_IMAGES else all_paths
            with open(list_path, "w") as f:
                json.dump({"paths": selected, "basenames": [os.path.basename(p) for p in selected]}, f, indent=2)
            return selected
    return ensure_10_images(script_dir, scene_name_override=scene_name_override, csv_script_dir=csv_script_dir)


def save_prediction_glb(images, image_paths, depth_map, depth_conf, extrinsics, intrinsics, output_glb, cache_dir="temp_glb_cache", validity_masks_by_path=None):
    """Build point cloud from model outputs and export to GLB.
    When validity_masks_by_path is provided, per-view building masks (resized to pred resolution) zero out
    confidence for masked-out pixels — predictions_to_glb drops anything with conf<=1e-5, so the saved GLB
    contains only building points."""
    pred_depths = depth_map.cpu()
    pred_extrinsics = extrinsics.cpu().float()
    pred_intrinsics = intrinsics.cpu().float()
    point_maps = torch.tensor(unproject_depth_map_to_point_map(
        pred_depths.squeeze(0),
        pred_extrinsics.squeeze(0),
        pred_intrinsics.squeeze(0),
    ))
    depth_conf_processed = torch.sigmoid(torch.log(depth_conf.cpu())) * 2 - 1 + 1e-8
    if validity_masks_by_path and image_paths is not None:
        # Multiply validity mask into the processed conf so predictions_to_glb's conf>1e-5 filter
        # keeps only (validity & conf>1e-5) pixels — same filter as get_pred_points_masked and
        # the masked alignment path.
        conf_np = depth_conf_processed.numpy().squeeze(0).astype(np.float32)  # (S, H, W)
        S, H_p, W_p = conf_np.shape
        n_applied = 0
        for i, p in enumerate(image_paths[:S]):
            vm = validity_masks_by_path.get(p)
            if vm is None:
                vm = validity_masks_by_path.get(osp.normpath(p))
            if vm is None:
                continue  # keep original conf for views without a mask
            vm_r = _resize_mask_nearest(vm, H_p, W_p).astype(np.float32)
            conf_np[i] = conf_np[i] * vm_r
            n_applied += 1
        depth_conf_processed = torch.from_numpy(conf_np).unsqueeze(0)
        if n_applied:
            print(f"  GLB validity mask applied to {n_applied}/{S} views")
    images_np = images.cpu().numpy().squeeze(0).transpose((0, 2, 3, 1))
    preds_for_viz = {
        "world_points_from_depth": point_maps.numpy(),
        "depth_conf": depth_conf_processed.numpy().squeeze(0),
        "images": images_np,
        "extrinsic": pred_extrinsics.squeeze(0).numpy(),
    }
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(os.path.join(cache_dir, "images"), exist_ok=True)
    # Read via np.fromfile + cv2.imdecode (handles non-ASCII paths that cv2.imread chokes on,
    # e.g. U+2019 apostrophe in 'New_Zōjō-ji,_Dec_'21.jpg'), then write under ASCII-only name
    # so downstream predictions_to_glb's sky-mask loop (which uses cv2.imread) just works.
    import cv2 as _cv2
    n_skipped = 0
    for i, img_path in enumerate(image_paths):
        try:
            raw = np.fromfile(img_path, dtype=np.uint8)
            probe = _cv2.imdecode(raw, _cv2.IMREAD_COLOR) if raw.size else None
        except Exception:
            probe = None
        if probe is None:
            print(f"  WARN: skipping unreadable image for GLB sky-mask: {img_path}")
            n_skipped += 1
            continue
        ext = os.path.splitext(img_path)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"
        safe_name = f"{i:03d}{ext}"
        dst = os.path.join(cache_dir, "images", safe_name)
        ok, buf = _cv2.imencode(ext, probe)
        if not ok:
            n_skipped += 1
            continue
        with open(dst, "wb") as _f:
            _f.write(buf.tobytes())
    if n_skipped:
        print(f"  Skipped {n_skipped} unreadable image(s) when building {output_glb}")
    mask_sky_flag = len(os.listdir(os.path.join(cache_dir, "images"))) > 0
    scene = predictions_to_glb(
        preds_for_viz,
        conf_thres=0,
        mask_sky=mask_sky_flag,
        target_dir=cache_dir,
        prediction_mode="Depth Map",
        show_cam=False,
    )
    scene.export(output_glb)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)


def get_pred_points_clean(pred_data, image_paths, cache_dir="temp_glb_cache"):
    """Extract point cloud from pred_data using same pipeline as evaluate.py (predictions_to_glb -> scene vertices)."""
    if not TRIMESH_AVAILABLE:
        return None
    pred_depths = pred_data["depth_map"].cpu()
    pred_extrinsics = pred_data["extrinsics"].cpu().float()
    pred_intrinsics = pred_data["intrinsics"].cpu().float()
    depth_conf = pred_data["depth_conf"].cpu()
    depth_conf_processed = torch.sigmoid(torch.log(depth_conf)) * 2 - 1 + 1e-8
    images_np = pred_data["images"].cpu().numpy().squeeze(0).transpose((0, 2, 3, 1))
    point_maps = torch.tensor(unproject_depth_map_to_point_map(
        pred_depths.squeeze(0),
        pred_extrinsics.squeeze(0),
        pred_intrinsics.squeeze(0),
    ))
    preds_for_viz = {
        "world_points_from_depth": point_maps.numpy(),
        "depth_conf": depth_conf_processed.numpy().squeeze(0),
        "images": images_np,
        "extrinsic": pred_extrinsics.squeeze(0).numpy(),
    }
    if osp.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(osp.join(cache_dir, "images"), exist_ok=True)
    for img_path in image_paths:
        shutil.copy2(img_path, osp.join(cache_dir, "images", os.path.basename(img_path)))
    try:
        scene = predictions_to_glb(
            preds_for_viz,
            conf_thres=0,
            mask_sky=True,
            target_dir=cache_dir,
            prediction_mode="Depth Map",
            show_cam=False,
        )
        vertices_list = []
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.PointCloud):
                vertices_list.append(geom.vertices)
        if vertices_list:
            return np.concatenate(vertices_list, axis=0)
    finally:
        if osp.exists(cache_dir):
            shutil.rmtree(cache_dir)
    return None


def _expand_depth_folders(depth_folders):
    """Expand wildcards in depth folder paths (same as evaluate.py). Returns flat list of existing/concrete paths."""
    folders = [depth_folders] if isinstance(depth_folders, str) else list(depth_folders)
    expanded = []
    for f in folders:
        s = str(f)
        if "*" in s:
            expanded.extend(glob.glob(s))
        else:
            expanded.append(s)
    return expanded


def _find_depth_folders_recursive(candidate_folders):
    """When direct depth paths don't exist, recursively search under each candidate's parent (scene root) for any dir named 'depth'. Returns list of existing paths (shallowest first)."""
    seen = set()
    out = []
    for p in candidate_folders:
        path = Path(p)
        # Scene root = parent of the candidate depth path (e.g. .../SceneName/depth -> .../SceneName)
        scene_root = path.parent
        if not scene_root.exists():
            continue
        # Recursively find any directory named 'depth' under scene root
        for depth_dir in scene_root.rglob("depth"):
            if depth_dir.is_dir():
                try:
                    res = depth_dir.resolve()
                    key = str(res)
                    if key not in seen:
                        seen.add(key)
                        out.append(key)
                except (OSError, RuntimeError):
                    pass
    # Prefer shallower paths (e.g. .../Scene/depth before .../Scene/foo/depth)
    out.sort(key=lambda x: (x.count(osp.sep), x))
    return out


def generate_gt_by_unproject(
    image_paths,
    image_folder: str,
    depth_folders,
    recon_path: str,
    output_glb_path: str,
    verbose: bool = True,
    save_cameras: bool = True,
    camera_scale_factor: float = 1.0,
    camera_color: tuple = None,
    validity_masks_by_path: dict = None,
):
    """
    Generate GT by loading per-image camera pose + depth and unprojecting.
    Follows evaluate.py: rel_path from image_folder, resolve_depth_path(depth_folder, rel_path) with wildcard expansion.
    depth_folders: path (str) or list of paths (tried in order); paths may contain * and are glob-expanded.
    save_cameras: if True, include camera frustum meshes in the GLB; if False, point cloud only.
    camera_scale_factor: multiplier for frustum size (e.g. 0.05 for very small).
    camera_color: RGB tuple for frustum color (e.g. (255, 0, 0) for red).
    validity_masks_by_path: optional {image_path: bool ndarray (H_img,W_img)}. When present for a view, only pixels
      inside the mask contribute points to the fused GT, and the mask (resized to depth res) is stored on the
      gt_samples_list entry as 'validity_mask' so alignment can reuse it.
    """
    if not osp.exists(recon_path):
        if verbose:
            print(f"  recon_path not found: {recon_path}")
        return None, []
    folders = _expand_depth_folders(depth_folders)
    if not any(osp.exists(f) for f in folders):
        # Recursively search under each candidate's parent (scene root) for any dir named 'depth'
        folders = _find_depth_folders_recursive(folders)
    if not any(osp.exists(f) for f in folders):
        if verbose:
            print(f"  no depth_folder found in: {folders}")
        return None, []

    try:
        cameras, images = read_model_cameras_images(recon_path)
    except Exception as e:
        if verbose:
            print(f"  Failed to load COLMAP: {e}")
        return None, []

    name_to_id = {img.name: iid for iid, img in images.items()}

    # Reference frame: first image (alphabetically by our list)
    sorted_paths = sorted(image_paths)
    ref_path = sorted_paths[0]
    ref_basename = osp.basename(ref_path)
    ref_basenames_to_try = [ref_basename]
    if "_" in ref_basename:
        ref_basenames_to_try.append(ref_basename.split("_", 1)[1])
    ref_colmap_id = None
    for bn in ref_basenames_to_try:
        for cname, iid in name_to_id.items():
            if osp.basename(cname) == bn:
                ref_colmap_id = iid
                break
        if ref_colmap_id is not None:
            break
    if ref_colmap_id is None:
        R_ref, t_ref = np.eye(3), np.zeros(3)
    else:
        R_ref = qvec2rotmat(images[ref_colmap_id].qvec)
        t_ref = images[ref_colmap_id].tvec

    all_points_list = []
    all_colors_list = []
    gt_samples_list = []

    for image_path in sorted_paths:
        basename = osp.basename(image_path)
        if image_folder and image_folder in image_path:
            rel_path = osp.relpath(image_path, image_folder)
        else:
            rel_path = None
        # Match COLMAP by basename or, for XX_YYYYYY.jpg, by YYYYYY.jpg (dgpp naming)
        basenames_to_try = [basename]
        if "_" in basename:
            basenames_to_try.append(basename.split("_", 1)[1])
        colmap_match_name = None
        for bn in basenames_to_try:
            for colmap_img_name in name_to_id.keys():
                if osp.basename(colmap_img_name) == bn:
                    colmap_match_name = colmap_img_name
                    if rel_path is None:
                        rel_path = colmap_img_name
                    break
            if colmap_match_name is not None:
                break
        if colmap_match_name is None:
            if verbose:
                print(f"  No COLMAP match for {basename}, skip")
            continue
        if rel_path is None:
            rel_path = basename

        depth_path = None
        # When image is XX_YYYYYY.jpg, dgpp depth folder has YYYYYY.jpg.npy; evaluate.py tries basename/stem + .npy/.npz
        rel_paths_to_try = [rel_path, basename]
        if "_" in basename:
            after_underscore = basename.split("_", 1)[1]
            rel_paths_to_try.append(after_underscore)
        for depth_folder in folders:
            if not osp.exists(depth_folder):
                continue
            for rp in rel_paths_to_try:
                depth_path = resolve_depth_path(depth_folder, rp, had_wildcard=False)
                if depth_path is not None:
                    break
            if depth_path is not None:
                break
        if not depth_path or not osp.exists(depth_path):
            if verbose:
                print(f"  No depth for {basename} (rel_path={rel_path}), skip")
            continue

        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            height, width = img_array.shape[0], img_array.shape[1]
        except Exception as e:
            if verbose:
                print(f"  Failed to load image {image_path}: {e}")
            continue

        try:
            depthmap = load_depth_map(depth_path, (height, width))
        except Exception as e:
            if verbose:
                print(f"  Failed to load depth {depth_path}: {e}")
            continue

        colmap_id = name_to_id.get(colmap_match_name) if colmap_match_name else None
        if colmap_id is None:
            if verbose:
                print(f"  No COLMAP pose for {basename}, skip")
            continue

        qvec = images[colmap_id].qvec
        tvec = images[colmap_id].tvec
        R = qvec2rotmat(qvec)
        extr = np.concatenate([R, tvec[:, None]], axis=1)
        cam_id = images[colmap_id].camera_id
        cam = cameras[cam_id]
        if cam.model != "PINHOLE":
            continue
        fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
        K = np.eye(3, dtype=np.float32)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fy, cx, cy

        depth_input = depthmap[np.newaxis, ..., np.newaxis]
        intrinsic_input = K[np.newaxis, :, :]
        extrinsic_input = extr[np.newaxis, :, :]
        pointmap = unproject_depth_map_to_point_map(
            depth_input,
            extrinsic_input,
            intrinsic_input,
        )[0]

        valid_mask = depthmap > 1e-4
        # Validity mask (per-image building mask): resize to depth resolution, intersect with depth-valid mask.
        sample_vm_resized = None
        if validity_masks_by_path:
            vm_raw = validity_masks_by_path.get(image_path)
            if vm_raw is None:
                vm_raw = validity_masks_by_path.get(osp.normpath(image_path))
            if vm_raw is not None:
                sample_vm_resized = _resize_mask_nearest(vm_raw, depthmap.shape[0], depthmap.shape[1])
                valid_mask = valid_mask & sample_vm_resized
        valid_pts = pointmap[valid_mask]
        valid_colors = img_array[valid_mask]
        valid_mask_pts = np.isfinite(valid_pts).all(axis=1) & (np.linalg.norm(valid_pts, axis=1) > 1e-4)
        valid_pts = valid_pts[valid_mask_pts]
        valid_colors = valid_colors[valid_mask_pts]
        if len(valid_pts) == 0:
            continue
        all_points_list.append(valid_pts)
        all_colors_list.append(valid_colors)
        sample = {
            "image_path": image_path,
            "depth_path": depth_path,
            "depth_map": depthmap,
            "K": K,
            "extrinsic": extr,
            "R_ref": R_ref,
            "t_ref": t_ref,
        }
        if sample_vm_resized is not None:
            sample["validity_mask"] = sample_vm_resized
        gt_samples_list.append(sample)
        if verbose:
            print(f"  Unprojected {basename}: {len(valid_pts)} points")

    if not all_points_list:
        return None, []

    fused = np.concatenate(all_points_list, axis=0)
    fused_colors = np.concatenate(all_colors_list, axis=0)
    fused = fused @ R_ref.T + t_ref
    rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    fused = fused @ rot_x.T

    # Prepare cameras for frustum visualization (same as evaluate.py generate_gt_points)
    transformed_cameras = []
    T_w2ref = np.eye(4)
    T_w2ref[:3, :3] = R_ref
    T_w2ref[:3, 3] = t_ref
    T_ref2out = np.eye(4)
    T_ref2out[:3, :3] = rot_x
    T_w2out = T_ref2out @ T_w2ref
    for s in gt_samples_list:
        ext = s["extrinsic"]
        if ext.shape == (3, 4):
            ext = np.vstack([ext, [0, 0, 0, 1]])
        c2w = np.linalg.inv(ext)
        c2w_out = T_w2out @ c2w
        transformed_cameras.append({"cam_to_world": c2w_out})

    os.makedirs(osp.dirname(output_glb_path) or ".", exist_ok=True)
    pointcloud_to_glb(
        fused, fused_colors, output_glb_path,
        cameras=transformed_cameras if save_cameras else None,
        camera_scale_factor=camera_scale_factor,
        camera_color=camera_color,
    )
    return fused, gt_samples_list


def main(scene_name=None, work_dir=None, model=None):
    """Run intervention for one scene. When model is provided, it is reused (no reload). Returns the model for reuse in batch."""
    script_dir = Path(__file__).resolve().parent
    if work_dir is None:
        work_dir = script_dir
    if scene_name is None:
        scene_name = SCENE_NAME
    work_dir = Path(work_dir)
    in_batch = work_dir != script_dir
    model_provided = model is not None
    if in_batch and torch.cuda.is_available() and not model_provided:
        gc.collect()
        torch.cuda.empty_cache()
    image_paths = get_image_paths(
        work_dir,
        scene_name_override=scene_name if in_batch else None,
        csv_script_dir=script_dir if in_batch else None,
    )
    print(f"Using {len(image_paths)} images from {work_dir / IMAGES_DIR}")
    original_stems = None
    list_path = work_dir / IMAGE_LIST_JSON
    if list_path.exists():
        try:
            with open(list_path, "r", encoding="utf-8") as f:
                list_data = json.load(f)
            if list_data.get("original_stems") and len(list_data["original_stems"]) == len(image_paths):
                original_stems = list_data["original_stems"]
        except (json.JSONDecodeError, OSError):
            pass

    # Load per-image building validity masks once (keyed by image_path). When enabled, these masks restrict
    # GT fusion, RANSAC alignment correspondences, and pred-cloud evaluation to masked-in (building) pixels.
    validity_masks_by_path = {}
    if USE_VALIDITY_MASKS and VALIDITY_MASKS_BASE:
        validity_masks_by_path = load_validity_masks(scene_name, image_paths)
        n_loaded = len(validity_masks_by_path)
        print(f"Validity masks: {n_loaded}/{len(image_paths)} views have a mask from {VALIDITY_MASKS_BASE}")

    gt_points, gt_samples = None, []
    # Use this scene's pts_500m folder for recon/depth when images are from pts_500m (IMAGE_FOLDER or PTS_500M_IMAGE_BASE/scene).
    pts_500m_base = Path(IMAGE_FOLDER_ENV).parent if IMAGE_FOLDER_ENV else (PTS_500M_IMAGE_BASE / scene_name)
    pts_500m_recon_candidates = [
        pts_500m_base / "dense" / "0" / "sparse",
        pts_500m_base / "sparse",
    ]
    pts_500m_depth_dir = pts_500m_base / "depth"
    # ---- 1. GT fusing: per-image pose + depth -> fused point cloud, save {scene_name}_gt.glb ----
    gt_glb_path = str(work_dir / f"{scene_name}_gt.glb")
    try:
        scene_info_path = work_dir / "scene_info.json"
        if scene_info_path.exists():
            with open(scene_info_path, "r", encoding="utf-8") as f:
                scene_info = json.load(f)
            image_folder = scene_info.get("image_folder", str(work_dir / IMAGES_DIR))
            recon_path = scene_info["recon_path"]
            depth_candidates = [
                scene_info["depth_folder"],
                str(pts_500m_depth_dir),
                str(work_dir / "depth"),
            ]
            print("  Using scene_info.json for GT (recon + depth)")
        else:
            scene_info = load_scene_info(script_dir, scene_name, image_folder_override=None)
            image_folder = str(work_dir / IMAGES_DIR)
            use_pts = _use_pts_500m_paths(image_paths)
            if use_pts:
                recon_path = None
                for p in pts_500m_recon_candidates:
                    if osp.exists(p):
                        recon_path = str(p)
                        break
                if recon_path is None:
                    recon_path = scene_info["recon_path"]
                    print("  pts_500m recon not found, using CSV recon (poses may not match pts_500m image names)")
                else:
                    print("  Using pts_500m recon for pose:", recon_path)
            else:
                recon_path = scene_info["recon_path"]
            depth_candidates = [
                scene_info["depth_folder"],
                str(pts_500m_depth_dir),
                str(work_dir / "depth"),
            ]
        result = generate_gt_by_unproject(
            image_paths,
            image_folder,
            depth_candidates,
            recon_path,
            gt_glb_path,
            verbose=True,
            validity_masks_by_path=validity_masks_by_path,
        )
        if result is not None:
            gt_points, gt_samples = result
        if gt_points is not None:
            print("Saved GT GLB to", gt_glb_path)
        else:
            gt_samples = []
            pts_gt = Path(SOURCE_IMAGE_DIR).parent / f"{scene_name}_gt.glb"
            if pts_gt.exists():
                shutil.copy2(pts_gt, gt_glb_path)
                print("GT had no points; copied precomputed GT from", pts_gt, "->", gt_glb_path)
            else:
                print("GT generation returned None; no fallback GT found")
    except Exception as e:
        import traceback
        print("GT generation failed:", e)
        traceback.print_exc()
        gt_points, gt_samples = None, []
        pts_gt = Path(SOURCE_IMAGE_DIR).parent / f"{scene_name}_gt.glb"
        if pts_gt.exists():
            shutil.copy2(pts_gt, gt_glb_path)
            print("Copied precomputed GT from", pts_gt, "->", gt_glb_path)

    if gt_samples:
        gt_samples.sort(key=lambda x: x["image_path"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    try:
        if model is None:
            print("Loading VGGT-1B...")
            model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
            model.eval()
        else:
            model = model.to(device)
            model.eval()

        images = load_and_preprocess_images(image_paths).to(device).unsqueeze(0)
        _, processed_H, processed_W = images.shape[0], images.shape[-2], images.shape[-1]
        if len(image_paths) > 10 and device == "cuda":
            print("  Note: Many images may cause OOM in intervention (O(N^2)); use NUM_IMAGES=10 or SKIP_BASELINE_PASS=1 if needed.")

        content_mask = None
        # if (USE_EARLY_LAYER_MASK or USE_COMBINED_ARCH_EARLY_MASK) and EXCLUDE_PADDING_FROM_EARLY_MASK:
        #     content_mask = get_content_patch_mask(image_paths, processed_H, processed_W, target_size=518, mode="crop")
        #     print(f"  Content patch mask: excluding padding from early-layer selection (EXCLUDE_PADDING_FROM_EARLY_MASK=1)")

        first_intervention_layer = int(os.environ.get("FIRST_INTERVENTION_LAYER", "12"))
        last_intervention_layer = int(os.environ.get("LAST_INTERVENTION_LAYER", "15"))
        eval_metrics_path = work_dir / f"eval_metrics_{first_intervention_layer}_{last_intervention_layer}.json"

        arch_token_mask = None
        use_metric_eval_base = bool(METRIC_EVAL_BASE)
        if not SKIP_BASELINE_PASS and not use_metric_eval_base:
            # ---- 2. Base prediction -> pred_base.glb + eval ----
            print("Pass 1: baseline (no intervention)...")
            if PROBE and torch.cuda.is_available():
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); _probe_t = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast(dtype=dtype, device_type="cuda"):
                aggregated_tokens_list, ps_idx = model.aggregator(images)
                # if USE_COMBINED_ARCH_EARLY_MASK:
                #     arch_sam = build_arch_token_mask(
                #         image_paths, str(SAM_ARCHITECTURE_DIR), scene_name, processed_H, processed_W,
                #         mask_suffix="_artificial architecture.npy",
                #     )
                #     arch_token_mask = compute_combined_arch_early_mask(
                #         aggregated_tokens_list, arch_sam, ps_idx,
                #         early_layer_index=EARLY_LAYER_INDEX,
                #         top_fraction=EARLY_LAYER_TOP_FRACTION,
                #         device=device,
                #         content_mask=content_mask,
                #     )
                #     n_sam = int(arch_sam.sum().item())
                #     n_combined = int(arch_token_mask.sum().item())
                #     print(f"  Combined (SAM arch + early-layer): {n_combined} tokens (top {EARLY_LAYER_TOP_FRACTION*100:.0f}% of {n_sam} SAM arch by correspondence)")
                # elif USE_EARLY_LAYER_MASK:
                #     arch_token_mask = compute_early_layer_intervention_mask(
                #         aggregated_tokens_list, ps_idx,
                #         early_layer_index=EARLY_LAYER_INDEX,
                #         top_fraction=EARLY_LAYER_TOP_FRACTION,
                #         device=device,
                #         content_mask=content_mask,
                #     )
                #     n_early = int(arch_token_mask.sum().item())
                #     print(f"  Early-layer intervention mask: {n_early} tokens (top {EARLY_LAYER_TOP_FRACTION*100:.0f}% by correspondence)")
                depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

            if PROBE and torch.cuda.is_available():
                torch.cuda.synchronize(); _probe_base_gb = torch.cuda.max_memory_allocated() / 1e9; _probe_base_s = time.perf_counter() - _probe_t

            pred_data_base = {
                "images": images,
                "depth_map": depth_map,
                "depth_conf": depth_conf,
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
            }
            save_prediction_glb(
                images, image_paths, depth_map, depth_conf, extrinsics, intrinsics,
                str(work_dir / "pred_base.glb"),
                cache_dir=str(work_dir / "temp_glb_cache"),
                validity_masks_by_path=validity_masks_by_path,
            )
            print("Saved pred_base.glb")
            # Save raw tensors for per-pixel multi-step composition (opt-in via SAVE_RAW_TENSORS=1)
            if os.environ.get("SAVE_RAW_TENSORS", "").strip() == "1":
                np.savez_compressed(
                    str(work_dir / "base_raw.npz"),
                    depth_map=depth_map.detach().cpu().half().numpy(),
                    depth_conf=depth_conf.detach().cpu().half().numpy(),
                    extrinsics=extrinsics.detach().cpu().float().numpy(),
                    intrinsics=intrinsics.detach().cpu().float().numpy(),
                )
                print("  Saved base_raw.npz (depth/conf fp16, ext/intr fp32)")
            if not SKIP_EVAL and gt_samples and gt_points is not None and len(gt_points) >= 100:
                pred_data_base_cpu = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in pred_data_base.items()}
                alignment_max_pts = ALIGNMENT_MAX_POINTS if ALIGNMENT_MAX_POINTS > 0 else None
                if EVAL_TIMEOUT_SEC <= 0:
                    status, metrics = _run_alignment_and_eval_worker(
                        gt_points, gt_samples, pred_data_base_cpu, image_paths,
                        str(work_dir), str(eval_metrics_path),
                        first_intervention_layer, last_intervention_layer, "base",
                        max_total_points=alignment_max_pts,
                        n_iterations=ALIGNMENT_N_ITERATIONS,
                        validity_masks_by_path=validity_masks_by_path,
                    )
                else:
                    result_queue = multiprocessing.Queue()
                    process = multiprocessing.Process(
                        target=_run_alignment_and_eval_worker_put_in_queue,
                        args=(
                            result_queue, gt_points, gt_samples, pred_data_base_cpu, image_paths,
                            str(work_dir), str(eval_metrics_path),
                            first_intervention_layer, last_intervention_layer, "base",
                            alignment_max_pts,
                            ALIGNMENT_N_ITERATIONS,
                            validity_masks_by_path,
                        ),
                    )
                    process.start()
                    try:
                        status, metrics = result_queue.get(timeout=EVAL_TIMEOUT_SEC)
                    except QueueEmpty:
                        process.terminate()
                        process.join(timeout=5)
                        status, metrics = "timeout", None
                        print("  Base eval: timed out (alignment/evaluation took too long)")
                    else:
                        process.join(timeout=5)
                if status == "ok" and metrics:
                    print(f"  Base eval: Acc-med={metrics['base']['Acc-med']:.4f}, Comp-med={metrics['base']['Comp-med']:.4f}, Score={metrics['base']['Score']:.4f}")
                    with open(eval_metrics_path, "w") as f:
                        json.dump(metrics, f, indent=2)
                elif status == "alignment_failed":
                    print("  Base eval: alignment failed")
                elif status == "no_pred_points":
                    print("  Base eval: could not extract pred points")
                elif status == "error":
                    print(f"  Base eval: failed ({metrics})")
            elif SKIP_EVAL:
                print("  Skipping base evaluation (SKIP_EVAL=1)")
            # Free Pass 1 outputs; keep images on GPU for fast second pass.
            del depth_map, depth_conf, extrinsics, intrinsics
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        # ---- Pass 2: with intervention -> pred_inter.glb ----
        intervention_tag = None
        per_layer_feature = None  # dict[layer, feature] when per_layer_intervention
        sam_mask_dir = os.environ.get("SAM_MASK_DIR", SAM_MASK_DIR)
        Hp, Wp = processed_H // PATCH_SIZE, processed_W // PATCH_SIZE
        num_patches_per_img = Hp * Wp
        tokens_per_image = PATCH_START_IDX + num_patches_per_img
        num_images = len(image_paths)
        # Read at runtime so batch script can set PER_LAYER_INTERVENTION per pass (e.g. output_best_inter vs output_per_layer)
        per_layer_intervention = os.environ.get("PER_LAYER_INTERVENTION", "").strip().lower() in ("1", "true", "yes")

        if per_layer_intervention:
            # Per-layer: compute most-attended feature from each layer's attention, intervene with per-layer features
            first_intervention_layer = 0
            last_intervention_layer = 23
            eval_metrics_path = work_dir / f"eval_metrics_{first_intervention_layer}_{last_intervention_layer}.json"
            features = discover_scene_features(sam_mask_dir, scene_name)
            if not features:
                print("  PER_LAYER_INTERVENTION: no features found in mask dir, falling back to single-feature (arch)")
                features = ["arch"]
            print(f"  Per-layer intervention: discovered features {features}")
            all_tag_patches = build_all_tag_patches(image_paths, sam_mask_dir, scene_name, features, Hp, Wp, original_stems)
            print("  Computing per-layer most-attended features (pre-pass with attention hooks on layers 0-23)...")
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            per_layer_feature = compute_per_layer_most_attended_features(
                model, images, all_tag_patches, features, PATCH_START_IDX, num_patches_per_img, device
            )
            # Build mask per feature, then per-layer mask
            feature_masks = {}
            for feat in set(per_layer_feature.values()):
                feature_masks[feat] = build_arch_token_mask(
                    image_paths, sam_mask_dir, scene_name, processed_H, processed_W,
                    mask_suffix=f"_{feat}.npy", original_stems=original_stems, verbose=False
                )
            per_layer_arch_token_mask = {
                layer: feature_masks[per_layer_feature[layer]]
                for layer in range(first_intervention_layer, last_intervention_layer + 1)
                if per_layer_feature.get(layer) in feature_masks
            }
            arch_token_mask = per_layer_arch_token_mask
            loss_valid_mask = None  # per-layer uses different masks
            intervention_feature_data = {"per_layer": True, **{str(k): v for k, v in per_layer_feature.items()}}
            intervention_feature_path = work_dir / "intervention_feature.json"
            with open(intervention_feature_path, "w") as f:
                json.dump(intervention_feature_data, f, indent=2)
            print(f"  Saved per-layer intervention features to {intervention_feature_path}")
            for layer in range(first_intervention_layer, last_intervention_layer + 1):
                print(f"    Layer {layer}: {per_layer_feature.get(layer, '?')}")
            # Visualize first layer's patches
            if per_layer_arch_token_mask:
                first_layer_mask = per_layer_arch_token_mask.get(first_intervention_layer)
                if first_layer_mask is not None:
                    visualize_intervention_patches(
                        first_layer_mask,
                        image_paths,
                        patch_start_idx=PATCH_START_IDX,
                        Hp=Hp,
                        Wp=Wp,
                        patch_size=PATCH_SIZE,
                        output_dir=work_dir,
                        prefix="intervention_patches",
                        intervention_tag=f"per_layer_L{first_intervention_layer}_{per_layer_feature.get(first_intervention_layer, '?')}",
                    )
            n_intervention = sum(int(m.sum().item()) for m in per_layer_arch_token_mask.values())  # total across layers
        elif arch_token_mask is None:
            # SAM-only: single feature (top-1) from ATTENTION_STATS_DIR, or union of top-k, or default _arch.npy
            if ATTENTION_STATS_DIR and TOP_K_ELEMENTS > 1:
                tags = get_topk_attended_tags(ATTENTION_STATS_DIR, scene_name, TOP_K_ELEMENTS, default="arch")
                intervention_tag = "top%d" % len(tags)
                print(f"  Top-{TOP_K_ELEMENTS} attended tags: {tags}")
                arch_token_mask = None
                for _t in tags:
                    _m = build_arch_token_mask(image_paths, sam_mask_dir, scene_name, processed_H, processed_W,
                                               mask_suffix=f"_{_t}.npy", original_stems=original_stems, verbose=False)
                    if _m is not None and int(_m.sum().item()) > 0:
                        arch_token_mask = _m if arch_token_mask is None else torch.maximum(arch_token_mask, _m)
                with open(work_dir / "intervention_feature.json", "w") as f:
                    json.dump({"per_layer": False, "intervention_tag": intervention_tag, "top_k": TOP_K_ELEMENTS, "tags": tags}, f, indent=2)
                print(f"Arch token mask (SAM, top-{TOP_K_ELEMENTS} union): {int(arch_token_mask.sum().item()) if arch_token_mask is not None else 0} tokens")
            else:
                if ATTENTION_STATS_DIR:
                    arch_tag = get_most_attended_tag(ATTENTION_STATS_DIR, scene_name, default="arch")
                    mask_suffix = f"_{arch_tag}.npy"
                    intervention_tag = arch_tag
                    print(f"  Most-attended arch tag from stats: {arch_tag} -> mask suffix {mask_suffix}")
                else:
                    mask_suffix = "_arch.npy"
                    intervention_tag = "arch"
                intervention_feature_path = work_dir / "intervention_feature.json"
                with open(intervention_feature_path, "w") as f:
                    json.dump({"per_layer": False, "intervention_tag": intervention_tag, "mask_suffix": mask_suffix}, f, indent=2)
                print(f"  Saved intervention feature to {intervention_feature_path}")
                arch_token_mask = build_arch_token_mask(image_paths, sam_mask_dir, scene_name, processed_H, processed_W, mask_suffix=mask_suffix, original_stems=original_stems)
            print(f"Arch token mask (SAM): {int(arch_token_mask.sum().item())} arch tokens")
            loss_valid_mask = None
            if arch_token_mask is not None and num_images > 0:
                loss_valid_mask = arch_token_mask.clone()
                n_valid_imgs = 0
                for i in range(num_images):
                    start = i * tokens_per_image + PATCH_START_IDX
                    end = start + num_patches_per_img
                    if arch_token_mask[start:end].sum() < 1:
                        loss_valid_mask[start:end] = 0
                    else:
                        n_valid_imgs += 1
                if n_valid_imgs < num_images:
                    print(f"  Loss: using {n_valid_imgs}/{num_images} images with segmentation masks (skipping {num_images - n_valid_imgs} without masks)")
            visualize_intervention_patches(
                arch_token_mask,
                image_paths,
                patch_start_idx=PATCH_START_IDX,
                Hp=Hp,
                Wp=Wp,
                patch_size=PATCH_SIZE,
                output_dir=work_dir,
                prefix="intervention_patches",
                intervention_tag=intervention_tag,
            )
            n_intervention = int(arch_token_mask.sum().item())
        else:
            n_intervention = int(arch_token_mask.sum().item())

        print(f"Intervention token mask: {n_intervention} tokens")
        _step_size = float(os.environ.get("INTERVENTION_STEP_SIZE", "50000"))
        _use_sum = os.environ.get("INTERVENTION_USE_SUM", "").strip().lower() in ("1", "true", "yes")
        if _use_sum:
            print("Intervention: using sum (A^l(e^*)) instead of mean")
        _alpha_env = os.environ.get("INTERVENTION_ADAPTIVE_ALPHA", "").strip()
        _adaptive_alpha = float(_alpha_env) if _alpha_env else None
        if _adaptive_alpha is not None:
            print(f"Intervention: adaptive step mode (α={_adaptive_alpha}); step_size env ignored")
        _max_steps = int(os.environ.get("INTERVENTION_MAX_STEPS", "1"))
        _attn_thresh = float(os.environ.get("INTERVENTION_ATTN_THRESH", "0.1"))
        if _max_steps > 1:
            print(f"Intervention: multi-step mode (max_steps={_max_steps}, attn_improve_thresh={_attn_thresh})")
        def _compute_depth_consistency(dm, ext, intr_mat, mask_tokens, n_pairs=3, n_views=5, ds=4):
            """Cross-view depth consistency on masked patches (memory-lean version).
            Downsamples depth by ds (default 4x), uses only n_views (default 5) sampled views,
            and n_pairs (default 3) random view pairs. Returns (masked_error, global_error).
            Higher = baseline is worse = more room for intervention."""
            import random as _rand
            _rand.seed(42)
            dm_np = dm.cpu().float().numpy()
            if dm_np.shape[-1] == 1:
                dm_np = dm_np[..., 0]
            dm_np = dm_np.reshape(-1, dm_np.shape[-2], dm_np.shape[-1])  # → (S, H, W)
            S_full, H_full, W_full = dm_np.shape
            ext_np = ext.cpu().squeeze(0).float().numpy()
            ext_np = ext_np[:, :3, :]                                    # (S, 3, 4)
            intr_np = intr_mat.cpu().squeeze(0).float().numpy()          # (S, 3, 3)
            # Sample a subset of views to limit memory
            view_ids = list(range(S_full))
            if S_full > n_views:
                view_ids = _rand.sample(view_ids, n_views)
            # Downsample depth for memory efficiency
            dm_ds = dm_np[view_ids, ::ds, ::ds]                          # (n_views, H', W')
            H, W = dm_ds.shape[1], dm_ds.shape[2]
            # Adjust intrinsics for downsampling
            intr_ds = intr_np[view_ids].copy()
            intr_ds[:, 0, :] /= ds
            intr_ds[:, 1, :] /= ds
            ext_ds = ext_np[view_ids]
            # Build pixel mask at downsampled resolution from token mask
            _Hp, _Wp = H_full // PATCH_SIZE, W_full // PATCH_SIZE
            _tmask = mask_tokens.cpu().numpy().astype(bool) if hasattr(mask_tokens, 'cpu') else np.asarray(mask_tokens, dtype=bool)
            pix_mask = np.zeros((len(view_ids), H, W), dtype=bool)
            for _vi, _si in enumerate(view_ids):
                _start = PATCH_START_IDX + _si * _Hp * _Wp
                _end = _start + _Hp * _Wp
                if _end > len(_tmask):
                    break
                _patch_mask = _tmask[_start:_end].reshape(_Hp, _Wp)
                _full = np.repeat(np.repeat(_patch_mask, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1)[:H_full, :W_full]
                pix_mask[_vi] = _full[::ds, ::ds][:H, :W]
            # Unproject only the sampled views at downsampled resolution
            dm_ds_4d = dm_ds[..., None]                                  # (n_views, H', W', 1)
            world_pts = unproject_depth_map_to_point_map(dm_ds_4d, ext_ds, intr_ds)  # (n_views, H', W', 3)
            # Sample pairs from the subset
            n_v = len(view_ids)
            pairs = [(a, b) for a in range(n_v) for b in range(a+1, n_v)]
            if len(pairs) > n_pairs:
                pairs = _rand.sample(pairs, n_pairs)
            masked_errs, global_errs = [], []
            for a, b in pairs:
                pts = world_pts[a].reshape(-1, 3)
                ones = np.ones((pts.shape[0], 1), dtype=np.float32)
                pts_h = np.concatenate([pts, ones], axis=1)
                pts_cam = (ext_ds[b] @ pts_h.T).T[:, :3]
                z_proj = pts_cam[:, 2]
                pts_pix = (intr_ds[b] @ pts_cam.T).T
                u = pts_pix[:, 0] / (pts_pix[:, 2] + 1e-8)
                v = pts_pix[:, 1] / (pts_pix[:, 2] + 1e-8)
                valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z_proj > 0.01)
                if valid.sum() < 50:
                    continue
                ui = np.clip(u[valid].astype(int), 0, W - 1)
                vi = np.clip(v[valid].astype(int), 0, H - 1)
                z_actual = dm_ds[b, vi, ui]
                rel_err = np.abs(z_proj[valid] - z_actual) / (np.abs(z_actual) + 1e-8)
                global_errs.append(float(np.median(rel_err)))
                # Masked subset: only pixels that correspond to masked patches in view a
                pm_flat = pix_mask[a].reshape(-1)[valid]
                if pm_flat.sum() > 10:
                    masked_errs.append(float(np.median(rel_err[pm_flat])))
            g_err = float(np.median(global_errs)) if global_errs else 0.0
            m_err = float(np.median(masked_errs)) if masked_errs else g_err
            return m_err, g_err

        # --- Determine α from baseline depth consistency (single pass, no retry) ---
        _use_reproj_alpha = os.environ.get("INTERVENTION_REPROJ_ALPHA", "").strip().lower() in ("1", "true", "yes")
        _alpha_min = float(os.environ.get("INTERVENTION_ALPHA_MIN", "0.05"))
        _alpha_max = float(os.environ.get("INTERVENTION_ALPHA_MAX", "0.4"))

        if _use_reproj_alpha and _adaptive_alpha is not None and not SKIP_BASELINE_PASS and not use_metric_eval_base:
            # Compute baseline depth consistency on masked patches
            _base_dm = pred_data_base["depth_map"]
            _base_ext = pred_data_base["extrinsics"]
            _base_intr = pred_data_base["intrinsics"]
            masked_err, global_err = _compute_depth_consistency(
                _base_dm, _base_ext, _base_intr, arch_token_mask, n_pairs=5
            )
            # Inverted map: high ratio → hard scene → conservative (low α).
            # Low ratio → easy scene → more room to push (high α).
            ratio = masked_err / (global_err + 1e-8) if global_err > 1e-8 else 1.0
            # ratio=0.5 → α_max, ratio=2.0 → α_min, clipped.
            t = (ratio - 0.5) / (2.0 - 0.5)
            t = max(0.0, min(1.0, t))
            alpha_eff = _alpha_max - t * (_alpha_max - _alpha_min)
            print(f"  Depth consistency: masked_err={masked_err:.4f}, global_err={global_err:.4f}, ratio={ratio:.3f} → α={alpha_eff:.4f} (range [{_alpha_min}, {_alpha_max}])")
        elif _adaptive_alpha is not None:
            alpha_eff = _adaptive_alpha
        else:
            alpha_eff = None

        _loss_type = os.environ.get("INTERVENTION_LOSS_TYPE", "arch_attn").strip()

        # Build per-tag masks for block-diagonal localized intervention.
        # Active when INTERVENTION_LOSS_TYPE=multi_self_coherence. Each tag is an independent
        # group whose intra-group attention gets boosted — so non-tagged regions are left alone
        # AND tagged regions don't starve each other via softmax.
        group_token_masks = None
        if _loss_type == "multi_self_coherence" and not per_layer_intervention:
            all_feats = discover_scene_features(sam_mask_dir, scene_name)
            if all_feats:
                group_token_masks = {}
                for feat in all_feats:
                    m = build_arch_token_mask(
                        image_paths, sam_mask_dir, scene_name, processed_H, processed_W,
                        mask_suffix=f"_{feat}.npy", original_stems=original_stems, verbose=False,
                    )
                    if int(m.sum().item()) > 0:
                        group_token_masks[feat] = m
                print(f"  multi_self_coherence: {len(group_token_masks)} groups "
                      f"[{', '.join(f'{t}={int(m.sum().item())}' for t, m in group_token_masks.items())}]")
                # Union mask replaces arch_token_mask so measurement / visualization stays meaningful.
                if group_token_masks:
                    union = torch.zeros_like(next(iter(group_token_masks.values())))
                    for m in group_token_masks.values():
                        union = torch.maximum(union, m)
                    arch_token_mask = union
                    intervention_tag = intervention_tag or "multi"
                    loss_valid_mask = union.clone()

        print(f"Pass 2: with intervention (layers {first_intervention_layer}–{last_intervention_layer}, α={alpha_eff}, loss={_loss_type})...")
        intervention = VGGTAttentionIntervention(
            model,
            arch_mask=None,
            arch_token_mask=arch_token_mask,
            loss_valid_mask=loss_valid_mask,
            first_intervention_layer=first_intervention_layer,
            last_intervention_layer=last_intervention_layer,
            step_size=_step_size,
            attention_label=intervention_tag or "per_layer",
            layer_feature_labels=per_layer_feature if per_layer_intervention else None,
            use_sum=_use_sum,
            adaptive_alpha=alpha_eff,
            max_steps=_max_steps,
            attn_improve_thresh=_attn_thresh,
            loss_type=_loss_type,
            group_token_masks=group_token_masks,
        )
        if PROBE and torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); _probe_ti = time.perf_counter()
        intervention.register_intervention()
        with torch.no_grad(), torch.amp.autocast(dtype=dtype, device_type="cuda"):
            aggregated_tokens_list, ps_idx = model.aggregator(images)
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        intervention.remove_intervention()
        if PROBE and torch.cuda.is_available():
            torch.cuda.synchronize(); _probe_inter_gb = torch.cuda.max_memory_allocated() / 1e9; _probe_inter_s = time.perf_counter() - _probe_ti
            _bg = round(_probe_base_gb, 3) if "_probe_base_gb" in locals() else None
            _bs = round(_probe_base_s, 3) if "_probe_base_s" in locals() else None
            print("PROBE_RESULT " + json.dumps({"scene": scene_name, "num_images": int(os.environ.get("NUM_IMAGES", "50")),
                  "base_gb": _bg, "base_s": _bs, "inter_gb": round(_probe_inter_gb, 3), "inter_s": round(_probe_inter_s, 3)}), flush=True)
        del aggregated_tokens_list
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # Delta attention vs layer index graph
        if getattr(intervention, "attention_deltas", None):
            deltas = intervention.attention_deltas
            layers = [d[0] for d in deltas]
            delta_vals = [d[3] for d in deltas]
            fig, ax = plt.subplots()
            ax.plot(layers, delta_vals, "o-", color="C0")
            feat_label = ("per_layer" if per_layer_intervention else intervention_tag) or "intervention"
            ax.set_xlabel("Layer index")
            ax.set_ylabel(f"Δ {feat_label} attention")
            ax.set_title(f"{feat_label.capitalize()} attention delta vs layer (intervention {first_intervention_layer}–{last_intervention_layer})")
            ax.axhline(y=0, color="gray", linestyle="--")
            fig.tight_layout()
            delta_plot_path = work_dir / f"{feat_label}_attention_delta_{first_intervention_layer}_{last_intervention_layer}.png"
            fig.savefig(delta_plot_path, dpi=150)
            plt.close(fig)
            print(f"  Saved delta plot to {delta_plot_path}")

        pred_data_inter = {
            "images": images,
            "depth_map": depth_map,
            "depth_conf": depth_conf,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
        }
        # Save raw tensors for per-pixel multi-step composition (opt-in via SAVE_RAW_TENSORS=1)
        if os.environ.get("SAVE_RAW_TENSORS", "").strip() == "1":
            np.savez_compressed(
                str(work_dir / "inter_raw.npz"),
                depth_map=depth_map.detach().cpu().half().numpy(),
                depth_conf=depth_conf.detach().cpu().half().numpy(),
                extrinsics=extrinsics.detach().cpu().float().numpy(),
                intrinsics=intrinsics.detach().cpu().float().numpy(),
                arch_token_mask=(arch_token_mask.detach().cpu().numpy().astype(np.uint8)
                                 if (arch_token_mask is not None and not isinstance(arch_token_mask, dict))
                                 else np.zeros(1, dtype=np.uint8)),
            )
            print("  Saved inter_raw.npz (depth/conf fp16, mask uint8)")
        save_prediction_glb(
            images, image_paths, depth_map, depth_conf, extrinsics, intrinsics,
            str(work_dir / "pred_inter.glb"),
            cache_dir=str(work_dir / "temp_glb_cache"),
            validity_masks_by_path=validity_masks_by_path,
        )
        print("Saved pred_inter.glb")
        # Error-colored intervened cloud: align inter->base and color each intervened point by its
        # nearest-neighbour distance to the base prediction (how much the intervention moved the recon).
        if not SKIP_BASELINE_PASS and not use_metric_eval_base:
            try:
                save_inter_error_colored_glb(
                    pred_data_base, pred_data_inter, image_paths,
                    str(work_dir / "inter_error_colored.glb"),
                    validity_masks_by_path=validity_masks_by_path,
                )
            except Exception as e:
                print(f"  error-colored GLB failed: {e}")
        else:
            print("  error-colored GLB skipped (no base pass in this process)")
        if not SKIP_EVAL and gt_samples and gt_points is not None and len(gt_points) >= 100:
            pred_data_inter_cpu = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in pred_data_inter.items()}
            alignment_max_pts = ALIGNMENT_MAX_POINTS if ALIGNMENT_MAX_POINTS > 0 else None
            if EVAL_TIMEOUT_SEC <= 0:
                status, metrics = _run_alignment_and_eval_worker(
                    gt_points, gt_samples, pred_data_inter_cpu, image_paths,
                    str(work_dir), str(eval_metrics_path),
                    first_intervention_layer, last_intervention_layer, "inter",
                    max_total_points=alignment_max_pts,
                    n_iterations=ALIGNMENT_N_ITERATIONS,
                    validity_masks_by_path=validity_masks_by_path,
                )
            else:
                result_queue = multiprocessing.Queue()
                process = multiprocessing.Process(
                    target=_run_alignment_and_eval_worker_put_in_queue,
                    args=(
                        result_queue, gt_points, gt_samples, pred_data_inter_cpu, image_paths,
                        str(work_dir), str(eval_metrics_path),
                        first_intervention_layer, last_intervention_layer, "inter",
                        alignment_max_pts,
                        ALIGNMENT_N_ITERATIONS,
                        validity_masks_by_path,
                    ),
                )
                process.start()
                try:
                    status, metrics = result_queue.get(timeout=EVAL_TIMEOUT_SEC)
                except QueueEmpty:
                    process.terminate()
                    process.join(timeout=5)
                    status, metrics = "timeout", None
                    print("  Inter eval: timed out (alignment/evaluation took too long)")
                else:
                    process.join(timeout=5)
            if status == "ok" and metrics:
                if use_metric_eval_base and "base" not in metrics:
                    base_path = Path(METRIC_EVAL_BASE) / scene_name / "metrics.json"
                    if base_path.exists():
                        try:
                            with open(base_path, "r", encoding="utf-8") as f:
                                base_data = json.load(f)
                            metrics["base"] = {
                                "Acc-mean": float(base_data.get("Acc-mean", 0)),
                                "Acc-med": float(base_data.get("Acc-med", 0)),
                                "Comp-mean": float(base_data.get("Comp-mean", 0)),
                                "Comp-med": float(base_data.get("Comp-med", 0)),
                                "Score": float(base_data.get("Score", 0)),
                            }
                            print(f"  Base: copied from {base_path}")
                        except Exception as e:
                            print(f"  Base: failed to load from metric_eval: {e}")
                    else:
                        print(f"  Base: metric_eval not found at {base_path}")
                print(f"  Inter eval: Acc-med={metrics['inter']['Acc-med']:.4f}, Comp-med={metrics['inter']['Comp-med']:.4f}, Score={metrics['inter']['Score']:.4f}")
                if intervention_tag is not None:
                    metrics["intervention_tag"] = intervention_tag
                with open(eval_metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)
            elif status == "alignment_failed":
                print("  Inter eval: alignment failed")
            elif status == "no_pred_points":
                print("  Inter eval: could not extract pred points")
            elif status == "error":
                print(f"  Inter eval: failed ({metrics})")
        elif SKIP_EVAL:
            print("  Skipping inter evaluation (SKIP_EVAL=1)")
        print("Done.")
        if in_batch and device == "cuda":
            try:
                del images
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()
        return model
    except torch.cuda.OutOfMemoryError as e:
        if in_batch and torch.cuda.is_available() and not model_provided:
            try:
                del model
            except NameError:
                pass
            try:
                del images
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()
        print("CUDA out of memory (OOM). Try: SKIP_BASELINE_PASS=1 and/or NUM_IMAGES=10 or lower (intervention uses O(N^2) in token count).")
        raise

if __name__ == "__main__":
    main()
