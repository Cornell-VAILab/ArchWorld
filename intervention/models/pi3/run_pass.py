# --- repo bootstrap: make the sibling `eval` package importable ---------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE)))
# eval/ modules import each other flatly, so that directory goes on the path too
for _p in (_REPO_ROOT, _os.path.join(_REPO_ROOT, "eval"), _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# -----------------------------------------------------------------------------
import csv
import gc
import glob
import json
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PI3_ROOT = SCRIPT_DIR.parent
if str(PI3_ROOT) not in sys.path:
    sys.path.insert(0, str(PI3_ROOT))

VGGT_ROOT = Path(os.environ.get("VGGT_ROOT", ""))  # only needed if vggt is not installed
if str(VGGT_ROOT) not in sys.path and str(VGGT_ROOT) not in ("", "."):
    sys.path.insert(0, str(VGGT_ROOT))

from pi3.models.pi3 import Pi3
from eval.eval_utils import accuracy, completion, umeyama_ransac
from vggt.utils.load_fn import load_and_preprocess_images
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    from .loss import Pi3AttentionIntervention
except ImportError:
    from loss import Pi3AttentionIntervention


PATCH_SIZE = 14
PATCH_START_IDX = 5
NUM_IMAGES = int(os.environ.get("NUM_IMAGES", "50"))
IMAGES_DIR = f"images_{NUM_IMAGES}"
IMAGE_LIST_JSON = "image_list.json"
SCENE_NAME = os.environ.get("SCENE_NAME", "").strip()
SAM_MASK_DIR = os.environ.get("SAM_MASK_DIR", "").strip()
ATTENTION_STATS_DIR = os.environ.get("ATTENTION_STATS_DIR", "").strip()
SKIP_BASELINE_PASS = os.environ.get("SKIP_BASELINE_PASS", "").strip() == "1"
SKIP_EVAL = os.environ.get("SKIP_EVAL", "").strip() == "1"
PROBE = os.environ.get("PROBE", "").strip() == "1"
import time as _time_probe
# Sky-seg ONNX model used to drop sky points from the eval cloud (matches teammate's evaluate.py / DA3 port).
SKYSEG_ONNX = os.environ.get("SKYSEG_ONNX", _os.path.join(_REPO_ROOT, "intervention", "assets", "skyseg.onnx")).strip()
_SKYSEG_SESSION = None
# Filtered metric eval: restrict GT fusion, alignment, and eval cloud to building (validity) mask pixels.
USE_VALIDITY_MASKS = os.environ.get("USE_VALIDITY_MASKS", "0").strip().lower() in ("1", "true", "yes")


def _resize_mask_nearest(mask_bool, target_h, target_w):
    return np.array(
        Image.fromarray(mask_bool.astype(np.uint8) * 255).convert("L").resize(
            (target_w, target_h), Image.Resampling.NEAREST
        )
    ) > 127


def _import_vggt_run_pass():
    """The VGGT pass owns the shared ground-truth and validity-mask code."""
    import importlib
    return importlib.import_module("models.vggt.run_pass")

def _get_validity_masks(scene_name, image_paths):
    """Per-image building validity masks {image_path: bool ndarray}; {} unless USE_VALIDITY_MASKS."""
    if not USE_VALIDITY_MASKS:
        return {}
    try:
        vggt_run_pass = _import_vggt_run_pass()
        return vggt_run_pass.load_validity_masks(scene_name, image_paths)
    except Exception as exc:
        print(f"Validity masks unavailable: {exc}")
        return {}


def _sanitize_scene_dir(name: str) -> str:
    import re

    s = re.sub(r"[^\w\-., ()]", "_", name)
    return s.strip("_") or "scene"


def map_mask_to_patches(mask, Hp, Wp, image_path=None):
    if mask is None or np.sum(mask) <= 0:
        return set()
    mask = mask.astype(np.uint8)
    target_h, target_w = Hp * PATCH_SIZE, Wp * PATCH_SIZE
    if image_path and osp.exists(image_path):
        with Image.open(image_path) as img:
            width, height = img.size
        new_width = 518
        new_height = round(height * (new_width / max(1, width)) / 14) * 14
        mask = np.array(
            Image.fromarray(mask * 255).convert("L").resize((new_width, new_height), Image.Resampling.NEAREST)
        ) > 127
        if new_height > 518:
            start_y = (new_height - 518) // 2
            mask = mask[start_y : start_y + 518, :]
        if mask.shape != (target_h, target_w):
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8) * 255)
                .convert("L")
                .resize((target_w, target_h), Image.Resampling.NEAREST)
            ) > 127
    elif mask.shape != (target_h, target_w):
        mask = np.array(
            Image.fromarray(mask * 255).convert("L").resize((target_w, target_h), Image.Resampling.NEAREST)
        ) > 127
    patch_mask = mask.reshape(Hp, PATCH_SIZE, Wp, PATCH_SIZE).mean(axis=(1, 3))
    idx = np.where(patch_mask > 0.5)
    return set(r * Wp + c for r, c in zip(idx[0], idx[1]))


def build_arch_token_mask(image_paths, sam_mask_dir, scene_name, processed_h, processed_w, mask_suffix="_arch.npy", original_stems=None):
    Hp, Wp = processed_h // PATCH_SIZE, processed_w // PATCH_SIZE
    num_patches_per_img = Hp * Wp
    tokens_per_image = PATCH_START_IDX + num_patches_per_img
    mask = torch.zeros(len(image_paths) * tokens_per_image, dtype=torch.float32)
    found = 0
    for i, img_path in enumerate(image_paths):
        stem = (original_stems[i] if original_stems is not None and i < len(original_stems) else None) or Path(img_path).stem
        npy = Path(sam_mask_dir) / scene_name / "mask" / f"{stem}{mask_suffix}"
        if not npy.exists():
            continue
        found += 1
        arr = np.load(npy)
        if arr.ndim == 4:
            arr = arr.sum(axis=(0, 1)) > 0
        elif arr.ndim == 3:
            arr = arr.sum(axis=0) > 0
        patches = map_mask_to_patches(arr.astype(bool), Hp, Wp, image_path=img_path)
        start = i * tokens_per_image + PATCH_START_IDX
        for p in patches:
            if p < num_patches_per_img:
                mask[start + p] = 1.0
    print(f"  SAM masks found for {found}/{len(image_paths)} images")
    return mask


def get_most_attended_tag(attention_stats_dir, scene_name, default="arch"):
    path = Path(attention_stats_dir) / scene_name / "stats.json"
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    best, best_sum = default, -1.0
    for tag, stats in data.items():
        if isinstance(stats, dict) and "sum" in stats:
            try:
                val = float(stats["sum"])
            except (TypeError, ValueError):
                continue
            if val > best_sum:
                best_sum = val
                best = tag
    return best


TOP_K_ELEMENTS = int(os.environ.get("TOP_K_ELEMENTS", "1"))   # >1: steer the union of the top-k most-attended arch elements instead of a single anchor


def get_topk_attended_tags(attention_stats_dir, scene_name, k, default="arch"):
    """Return up to k tags with the largest 'sum' in stats.json (descending). Falls back to [default]."""
    path = Path(attention_stats_dir) / scene_name / "stats.json"
    if not path.exists():
        return [default]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [default]
    scored = []
    for tag, stats in data.items():
        if isinstance(stats, dict) and "sum" in stats:
            try:
                scored.append((float(stats["sum"]), tag))
            except (TypeError, ValueError):
                continue
    if not scored:
        return [default]
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:k]]


def get_image_paths(work_dir: Path):
    list_path = work_dir / IMAGE_LIST_JSON
    if list_path.exists():
        data = json.loads(list_path.read_text(encoding="utf-8"))
        if data.get("paths"):
            return data["paths"]
    images_dir = work_dir / IMAGES_DIR
    if images_dir.is_dir():
        paths = sorted(str(p) for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if paths:
            return paths
    raise RuntimeError(f"No images found in {work_dir}")


def _get_original_stems(work_dir: Path, image_paths):
    list_path = work_dir / IMAGE_LIST_JSON
    if not list_path.exists():
        return None
    try:
        data = json.loads(list_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    stems = data.get("original_stems")
    if stems and len(stems) == len(image_paths):
        return stems
    return None


def _extract_pred_points_and_colors(res, images):
    points = res["points"][0].detach().cpu().numpy()  # [S,H,W,3]
    conf_raw = res["conf"][0, ..., 0].detach().cpu()
    conf = torch.sigmoid(conf_raw).numpy()
    colors = images.detach().cpu().permute(0, 2, 3, 1).numpy()

    pts_flat = points.reshape(-1, 3)
    conf_flat = conf.reshape(-1)
    col_flat = colors.reshape(-1, 3)
    if len(conf_flat) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    th = np.percentile(conf_flat, 50.0)
    keep = np.isfinite(pts_flat).all(axis=1) & np.isfinite(conf_flat) & (conf_flat >= th) & (conf_flat > 1e-5)
    return pts_flat[keep].astype(np.float32), col_flat[keep].astype(np.float32)


def _get_skyseg_session():
    """The sky-removal model used when building the evaluation cloud.

    The published numbers for this model were produced with sky pixels removed.
    Without the ONNX file the evaluation still runs, but on clouds that include
    sky, and the numbers will not match the paper. See the README.
    """
    global _SKYSEG_SESSION
    # An empty SKYSEG_ONNX is how run.py --no-sky-filter asks for the filter to be
    # skipped; only a path that was given and is missing is an error.
    if not SKYSEG_ONNX:
        return None
    if _SKYSEG_SESSION is None:
        if not _os.path.exists(SKYSEG_ONNX):
            raise FileNotFoundError(
                f"sky segmentation model not found at {SKYSEG_ONNX!r}. "
                f"run.py fetches it automatically; otherwise download it from "
                f"https://huggingface.co/JianyuanWang/skyseg and set SKYSEG_ONNX, "
                f"or pass --no-sky-filter to evaluate without it "
                f"(results will not match the published numbers).")
        import onnxruntime

        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = 4
        _SKYSEG_SESSION = onnxruntime.InferenceSession(SKYSEG_ONNX, sess_options=so)
    return _SKYSEG_SESSION


def _run_skyseg(session, input_size, image_bgr):
    """Replicates teammate visual_util.run_skyseg: returns a 0-255 sky score map (uint8)."""
    import cv2

    resized = cv2.resize(image_bgr.copy(), dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1).reshape(-1, 3, input_size[0], input_size[1]).astype("float32")
    iname = session.get_inputs()[0].name
    oname = session.get_outputs()[0].name
    res = np.array(session.run([oname], {iname: x})).squeeze()
    mn, mx = float(np.min(res)), float(np.max(res))
    res = (res - mn) / (mx - mn + 1e-8) * 255.0
    return res.astype("uint8")


def _pi3_eval_maps(res):
    """Match teammate run_inference (pi3): bilinear-upsample points & raw conf to 518x518."""
    pred_pts = res["points"].squeeze(0)  # (N, h, w, 3)
    pred_pts = F.interpolate(
        pred_pts.permute(0, 3, 1, 2), (518, 518), mode="bilinear", align_corners=False, antialias=True
    ).permute(0, 2, 3, 1)
    pred_conf = res["conf"].squeeze(0)  # (N, H, W, 1)
    pred_conf = F.interpolate(
        pred_conf.permute(0, 3, 1, 2), (518, 518), mode="bilinear", align_corners=False, antialias=True
    ).squeeze(1)  # (N, 518, 518)
    return pred_pts.detach().cpu().numpy().astype(np.float32), pred_conf.detach().cpu().float().numpy()


def _pi3_per_pixel_valid(point_maps, conf, images, image_paths=None, validity_masks_by_path=None):
    """Per-pixel keep mask (n,h,w) + color maps (n,h,w,3) for Pi3 point maps.
    Filter = raw conf > 1e-4 + sky dropped (ONNX skyseg) + (optional) building validity mask."""
    import cv2
    n, h, w, _ = point_maps.shape
    conf_proc = conf.astype(np.float32).copy()
    imgs_rgb = images.detach().cpu().permute(0, 2, 3, 1).numpy()  # (S, H0, W0, 3) float [0,1]
    col_maps = np.stack([cv2.resize(imgs_rgb[i], (w, h), interpolation=cv2.INTER_LINEAR) for i in range(n)], axis=0)
    session = _get_skyseg_session()
    if session is not None:
        for i in range(n):
            frame_bgr = cv2.cvtColor((imgs_rgb[i] * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            sky_score = _run_skyseg(session, [320, 320], frame_bgr)
            sky_score = cv2.resize(sky_score, (w, h), interpolation=cv2.INTER_LINEAR)
            conf_proc[i][sky_score >= 20] = -1.0
    keep = (conf_proc > 1e-4) & np.isfinite(point_maps).all(axis=-1)
    vmasks = validity_masks_by_path or {}
    if vmasks and image_paths is not None:
        for i in range(min(n, len(image_paths))):
            vm = vmasks.get(image_paths[i])
            if vm is None:
                vm = vmasks.get(osp.normpath(image_paths[i]))
            if vm is not None:
                keep[i] = keep[i] & _resize_mask_nearest(vm, h, w)
    return keep, col_maps


def _build_eval_cloud_pi3(point_maps, conf, images, image_paths=None, validity_masks_by_path=None):
    """Pi3 eval cloud: raw conf>1e-4 + sky dropped + (optional) building validity mask.
    Returns (points (P,3), colors (P,3) uint8) — the exact set used for evaluation."""
    keep, col_maps = _pi3_per_pixel_valid(point_maps, conf, images, image_paths, validity_masks_by_path)
    pts = point_maps[keep].reshape(-1, 3).astype(np.float32)
    cols = (col_maps[keep].reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
    if len(pts) > 500000:
        idx = np.random.choice(len(pts), 500000, replace=False)
        pts = pts[idx]
        cols = cols[idx]
    return pts, cols


def save_inter_error_colored_glb_pi3(base_map, base_conf, inter_map, inter_conf, images, image_paths,
                                     output_path, validity_masks_by_path=None, max_points=500000):
    """Align intervened Pi3 prediction to base (per-pixel-correspondence Sim3), color each intervened
    point by nearest-neighbour distance to the base cloud (turbo). Pi3 point maps are world points;
    pred_*.glb are raw-world, so this matches their frame."""
    if not TRIMESH_AVAILABLE:
        return
    try:
        from scipy.spatial import cKDTree
        import matplotlib.cm as _cm
    except Exception as e:
        print(f"  error-colored GLB skipped (import failed: {e})")
        return
    bv, _ = _pi3_per_pixel_valid(base_map, base_conf, images, image_paths, validity_masks_by_path)
    iv, _ = _pi3_per_pixel_valid(inter_map, inter_conf, images, image_paths, validity_masks_by_path)
    base_pts = base_map[bv]; inter_pts = inter_map[iv]
    if len(base_pts) < 100 or len(inter_pts) < 100:
        print("  error-colored GLB skipped (too few points)")
        return
    corr = bv & iv
    c, R, t = 1.0, np.eye(3), np.zeros(3)
    if int(corr.sum()) >= 100:
        bc = base_map[corr]; ic = inter_map[corr]
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
    dist, _ = cKDTree(base_pts).query(aligned_inter, workers=-1)
    vmax = max(float(np.percentile(dist, 95)), 1e-8)
    colors = (_cm.turbo(np.clip(dist / vmax, 0.0, 1.0))[:, :3] * 255).astype(np.uint8)
    pts = aligned_inter
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts, colors, dist = pts[idx], colors[idx], dist[idx]
    pc = trimesh.PointCloud(vertices=pts.astype(np.float32),
                            colors=np.concatenate([colors, np.full((len(colors), 1), 255, np.uint8)], axis=1))
    os.makedirs(osp.dirname(output_path) or ".", exist_ok=True)
    trimesh.Scene([pc]).export(output_path, file_type="glb")
    print(f"  Saved error-colored inter GLB to {output_path} "
          f"(n={len(pts)}, scale c={c:.4f}, median_err={float(np.median(dist)):.6f}, p95(color max)={vmax:.6f})")


def save_aligned_combined_glb(gt_points, pred_points_aligned, output_path, save_n=500000):
    """GT (green) + aligned prediction (red) combined GLB, identical to VGGT intervene/evaluate.py."""
    if not TRIMESH_AVAILABLE:
        return
    os.makedirs(osp.dirname(output_path) or ".", exist_ok=True)
    if len(gt_points) > save_n:
        gt_points = gt_points[np.random.choice(len(gt_points), save_n, replace=False)]
    pc_gt = trimesh.PointCloud(vertices=gt_points)
    pc_gt.colors = np.tile([0, 255, 0, 255], (len(gt_points), 1)).astype(np.uint8)
    if len(pred_points_aligned) > save_n:
        pred_points_aligned = pred_points_aligned[np.random.choice(len(pred_points_aligned), save_n, replace=False)]
    pc_pred = trimesh.PointCloud(vertices=pred_points_aligned)
    pc_pred.colors = np.tile([255, 0, 0, 255], (len(pred_points_aligned), 1)).astype(np.uint8)
    trimesh.Scene([pc_gt, pc_pred]).export(output_path, file_type="glb")
    print(f"  Saved aligned combined GLB to {output_path}")


def pointcloud_to_glb(points: np.ndarray, colors: np.ndarray | None, output_path: str, cameras=None):
    if not TRIMESH_AVAILABLE:
        print("trimesh not available; skipping GLB export")
        return
    scene = trimesh.Scene()
    pc = trimesh.PointCloud(vertices=points if len(points) else np.zeros((1, 3), dtype=np.float32))
    if colors is not None and len(colors) == len(points):
        c = colors
        if c.max() <= 1.0:
            c = (c * 255).astype(np.uint8)
        else:
            c = c.astype(np.uint8)
        pc.colors = c
    scene.add_geometry(pc)
    # Camera frustums are optional; skip to keep this dependency-light in pi3 env.
    out_dir = osp.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    scene.export(output_path, file_type="glb")


def save_prediction_glb_pi3(res, images, output_glb: Path):
    pts, cols = _extract_pred_points_and_colors(res, images)
    cams = []
    if "camera_poses" in res:
        c2w = res["camera_poses"][0].detach().cpu().numpy()
        for i in range(c2w.shape[0]):
            cams.append({"cam_to_world": c2w[i]})
    pointcloud_to_glb(pts, cols, str(output_glb), cameras=cams)
    return pts


def compute_alignment_with_correspondence_pi3(
    gt_samples,
    pred_points_map,
    pred_conf_map,
    image_paths=None,
    max_total_points=None,
    n_iterations=1000,
):
    import cv2 as _cv2

    S, H_pred, W_pred, _ = pred_points_map.shape
    if image_paths is not None and len(image_paths) == S:
        gt_by_path = {osp.normpath(s["image_path"]): s for s in gt_samples}

        def _get_sample(i):
            return gt_by_path.get(osp.normpath(image_paths[i]))
    else:
        def _get_sample(i):
            return gt_samples[i] if i < len(gt_samples) else None

    correspondences_gt, correspondences_pred = [], []
    for i in range(S):
        sample = _get_sample(i)
        if sample is None:
            continue
        gt_depth = sample["depth_map"]
        gt_extrinsic = sample["extrinsic"]
        R_ref = sample["R_ref"]
        t_ref = sample["t_ref"]
        P_world_pred = pred_points_map[i]
        pred_conf = pred_conf_map[i]

        gt_depth_resized = _cv2.resize(gt_depth, (W_pred, H_pred), interpolation=_cv2.INTER_NEAREST)
        y, x = np.meshgrid(np.arange(H_pred), np.arange(W_pred), indexing="ij")
        K_gt_scaled = sample["K"].copy()
        sx = W_pred / gt_depth.shape[1]
        sy = H_pred / gt_depth.shape[0]
        K_gt_scaled[0, 0] *= sx
        K_gt_scaled[0, 2] *= sx
        K_gt_scaled[1, 1] *= sy
        K_gt_scaled[1, 2] *= sy
        fx_gt, fy_gt = K_gt_scaled[0, 0], K_gt_scaled[1, 1]
        cx_gt, cy_gt = K_gt_scaled[0, 2], K_gt_scaled[1, 2]
        Z_gt = gt_depth_resized
        X_gt = (x - cx_gt) * Z_gt / fx_gt
        Y_gt = (y - cy_gt) * Z_gt / fy_gt
        P_cam_gt = np.stack([X_gt, Y_gt, Z_gt], axis=-1)

        R_gt = gt_extrinsic[:3, :3]
        t_gt = gt_extrinsic[:3, 3]
        P_world_gt = P_cam_gt @ R_gt + (-R_gt.T @ t_gt)
        P_ref_gt = P_world_gt @ R_ref.T + t_ref
        rot_x_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        P_final_gt = P_ref_gt @ rot_x_180.T

        mask = (Z_gt > 1e-4) & np.isfinite(Z_gt) & np.isfinite(pred_conf) & (pred_conf > 1e-4)
        mask = mask & np.isfinite(P_world_pred).all(axis=-1)
        # Filtered eval: restrict alignment correspondences to building (validity) mask pixels.
        vm = sample.get("validity_mask") if isinstance(sample, dict) else None
        if vm is not None:
            mask = mask & _resize_mask_nearest(vm, H_pred, W_pred)
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
        return None, None, None
    pts_pred_all = np.concatenate(correspondences_pred, axis=0)
    pts_gt_all = np.concatenate(correspondences_gt, axis=0)
    if max_total_points is not None and len(pts_pred_all) > max_total_points:
        idx = np.random.choice(len(pts_pred_all), max_total_points, replace=False)
        pts_pred_all = pts_pred_all[idx]
        pts_gt_all = pts_gt_all[idx]
    print(f"  Alignment based on {len(pts_pred_all)} corresponding points.")
    c, R, t = umeyama_ransac(pts_pred_all.T, pts_gt_all.T, n_iterations=n_iterations)
    return c, R, t


def align_and_eval(gt_points, gt_samples, pred_points_map, pred_conf_map, pred_points_flat, image_paths):
    if gt_points is None or len(gt_points) < 100 or pred_points_flat is None or len(pred_points_flat) < 100:
        return None, None
    c, R, t = compute_alignment_with_correspondence_pi3(
        gt_samples=gt_samples,
        pred_points_map=pred_points_map,
        pred_conf_map=pred_conf_map,
        image_paths=image_paths,
        max_total_points=None,
        n_iterations=int(os.environ.get("ALIGNMENT_N_ITERATIONS", "1000")),
    )
    if c is None:
        return None, None
    aligned = (c * (R @ pred_points_flat.T) + t).T
    # Optional ICP refinement on full clouds after coarse Sim(3) (copied from jot-jt eval.py L118:
    # open3d point-to-point ICP, init=identity, threshold=0.1 for non-DTU).
    if os.environ.get("USE_ICP", "0") == "1":
        import open3d as o3d
        _icp_thr = float(os.environ.get("ICP_THRESHOLD", "0.1"))
        _icp_iter = int(os.environ.get("ICP_MAX_ITER", "0"))
        _pcd = o3d.geometry.PointCloud(); _pcd.points = o3d.utility.Vector3dVector(np.asarray(aligned, dtype=np.float64))
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
        aligned = (aligned @ _T[:3, :3].T) + _T[:3, 3]
    acc_mean, acc_med = accuracy(gt_points, aligned)
    comp_mean, comp_med = completion(gt_points, aligned)
    metrics = {
        "Acc-mean": float(acc_mean),
        "Acc-med": float(acc_med),
        "Comp-mean": float(comp_mean),
        "Comp-med": float(comp_med),
        "Acc_mean": float(acc_mean),
        "Acc_med": float(acc_med),
        "Comp_mean": float(comp_mean),
        "Comp_med": float(comp_med),
        "Score": float(0.5 * (acc_med + comp_med)),
    }
    return metrics, aligned


def save_metrics(eval_metrics_path: Path, key: str, metrics_dict: dict):
    if eval_metrics_path.exists():
        try:
            data = json.loads(eval_metrics_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}
    data[key] = metrics_dict
    eval_metrics_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_gt(scene_name: str, work_dir: Path, image_paths, validity_masks_by_path=None):
    """
    Load GT points using the same robust unprojection routine used by VGGT intervene pipeline.
    This handles copied image names like 000_<orig>.jpg by matching both full basename and suffix after underscore.
    When validity_masks_by_path is given, GT fusion is restricted to building-mask pixels and each
    gt_sample carries 'validity_mask' for the masked alignment.
    """
    try:
        vggt_run_pass = _import_vggt_run_pass()
        generate_gt_by_unproject = getattr(vggt_run_pass, "generate_gt_by_unproject")
    except Exception as exc:
        print(f"Skipping GT/eval because ECCV2026.intervene.run_pass import failed: {exc}")
        return None, []
    scene_info_path = work_dir / "scene_info.json"
    if not scene_info_path.exists():
        print(f"Skipping GT/eval: scene_info.json missing in {work_dir}")
        return None, []
    scene_info = json.loads(scene_info_path.read_text(encoding="utf-8"))
    image_folder = str(work_dir / IMAGES_DIR)
    recon_path = scene_info.get("recon_path", "")
    depth_candidates = [scene_info.get("depth_folder", "")]
    result = generate_gt_by_unproject(
        image_paths=image_paths,
        image_folder=image_folder,
        depth_folders=depth_candidates,
        recon_path=recon_path,
        output_glb_path=str(work_dir / f"{scene_name}_gt.glb"),
        verbose=True,
        validity_masks_by_path=validity_masks_by_path or {},
    )
    if result is None:
        return None, []
    gt_points, gt_samples = result
    return gt_points, gt_samples


def _forward_pi3(model, images, device):
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if device == "cuda":
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            return model(images[None])
    with torch.no_grad():
        return model(images[None])


def main(scene_name=None, work_dir=None, model=None):
    scene_name = (scene_name or SCENE_NAME or "").strip()
    if not scene_name:
        raise RuntimeError("scene_name is required.")
    work_dir = Path(work_dir) if work_dir is not None else SCRIPT_DIR
    work_dir.mkdir(parents=True, exist_ok=True)

    image_paths = get_image_paths(work_dir)
    original_stems = _get_original_stems(work_dir, image_paths)
    print(f"Using {len(image_paths)} images from {work_dir}")

    images = load_and_preprocess_images(image_paths)
    processed_h, processed_w = images.shape[-2:]

    validity_masks_by_path = _get_validity_masks(scene_name, image_paths)
    if validity_masks_by_path:
        print(f"Validity masks: {len(validity_masks_by_path)}/{len(image_paths)} views (filtered eval)")

    gt_points, gt_samples = (None, []) if SKIP_EVAL else load_gt(scene_name, work_dir, image_paths, validity_masks_by_path)
    if gt_points is not None:
        print(f"Loaded GT points: {len(gt_points)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        print("Loading Pi3...")
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    else:
        model = model.to(device).eval()
    images = images.to(device)

    first_layer = int(os.environ.get("FIRST_INTERVENTION_LAYER", "1"))
    last_layer = int(os.environ.get("LAST_INTERVENTION_LAYER", "35"))
    eval_metrics_path = work_dir / f"eval_metrics_{first_layer}_{last_layer}.json"

    if not SKIP_BASELINE_PASS:
        print("Pass 1: baseline (no intervention)")
        if PROBE and torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); _probe_t = _time_probe.perf_counter()
        res_base = _forward_pi3(model, images, device)
        if PROBE and torch.cuda.is_available():
            torch.cuda.synchronize(); _probe_base_gb = torch.cuda.max_memory_allocated() / 1e9; _probe_base_s = _time_probe.perf_counter() - _probe_t
        pred_map_base, pred_conf_base = _pi3_eval_maps(res_base)
        eval_pts_base, eval_cols_base = _build_eval_cloud_pi3(pred_map_base, pred_conf_base, images, image_paths, validity_masks_by_path)
        pointcloud_to_glb(eval_pts_base, eval_cols_base, str(work_dir / "pred_base.glb"))
        if not SKIP_EVAL and gt_points is not None:
            metrics_base, aligned_base = align_and_eval(
                gt_points, gt_samples, pred_map_base, pred_conf_base, eval_pts_base, image_paths
            )
            if metrics_base is not None:
                save_metrics(eval_metrics_path, "base", metrics_base)
                if aligned_base is not None:
                    save_aligned_combined_glb(gt_points, aligned_base, str(work_dir / "base_aligned_combined.glb"))
                print(f"  Base eval: Score={metrics_base['Score']:.4f}")
        del res_base
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    attn_stats_dir = os.environ.get("ATTENTION_STATS_DIR", ATTENTION_STATS_DIR).strip()
    if attn_stats_dir and TOP_K_ELEMENTS > 1:
        # Steer the union of the top-k most-attended architectural elements instead of a single anchor.
        tags = get_topk_attended_tags(attn_stats_dir, scene_name, TOP_K_ELEMENTS, default="arch")
        arch_tag = "top%d" % len(tags)
        print(f"  Top-{TOP_K_ELEMENTS} attended tags: {tags}")
        arch_token_mask = None
        for _t in tags:
            _m = build_arch_token_mask(
                image_paths=image_paths,
                sam_mask_dir=SAM_MASK_DIR,
                scene_name=scene_name,
                processed_h=processed_h,
                processed_w=processed_w,
                mask_suffix=f"_{_t}.npy",
                original_stems=original_stems,
            )
            if _m is not None and int(_m.sum().item()) > 0:
                arch_token_mask = _m if arch_token_mask is None else torch.maximum(arch_token_mask, _m)
        if arch_token_mask is None:
            arch_token_mask = build_arch_token_mask(
                image_paths=image_paths,
                sam_mask_dir=SAM_MASK_DIR,
                scene_name=scene_name,
                processed_h=processed_h,
                processed_w=processed_w,
                mask_suffix="_arch.npy",
                original_stems=original_stems,
            )
        (work_dir / "intervention_feature.json").write_text(
            json.dumps({"per_layer": False, "intervention_tag": arch_tag,
                        "top_k": TOP_K_ELEMENTS, "tags": tags}, indent=2),
            encoding="utf-8",
        )
    else:
        if attn_stats_dir:
            arch_tag = get_most_attended_tag(attn_stats_dir, scene_name, default="arch")
            mask_suffix = f"_{arch_tag}.npy"
            print(f"Using most-attended tag: {arch_tag}")
        else:
            arch_tag = "arch"
            mask_suffix = "_arch.npy"

        arch_token_mask = build_arch_token_mask(
            image_paths=image_paths,
            sam_mask_dir=SAM_MASK_DIR,
            scene_name=scene_name,
            processed_h=processed_h,
            processed_w=processed_w,
            mask_suffix=mask_suffix,
            original_stems=original_stems,
        )
        (work_dir / "intervention_feature.json").write_text(
            json.dumps({"per_layer": False, "intervention_tag": arch_tag, "mask_suffix": mask_suffix}, indent=2),
            encoding="utf-8",
        )
    print(f"Intervention token mask: {int(arch_token_mask.sum().item())}")

    step_size = float(os.environ.get("INTERVENTION_STEP_SIZE", "1000"))
    print(f"Pass 2: with intervention (layers {first_layer}-{last_layer}, step_size={step_size})")
    intervention = Pi3AttentionIntervention(
        model=model,
        arch_token_mask=arch_token_mask,
        first_intervention_layer=first_layer,
        last_intervention_layer=last_layer,
        step_size=step_size,
        attention_label=arch_tag,
        global_layers_only=True,
    )
    if PROBE and torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); _probe_ti = _time_probe.perf_counter()
    intervention.register_intervention()
    try:
        res_inter = _forward_pi3(model, images, device)
    finally:
        intervention.remove_intervention()
    if PROBE and torch.cuda.is_available():
        torch.cuda.synchronize(); _probe_inter_gb = torch.cuda.max_memory_allocated() / 1e9; _probe_inter_s = _time_probe.perf_counter() - _probe_ti
        _bg = round(_probe_base_gb, 3) if "_probe_base_gb" in locals() else None
        _bs = round(_probe_base_s, 3) if "_probe_base_s" in locals() else None
        print("PROBE_RESULT " + json.dumps({"scene": scene_name, "num_images": int(os.environ.get("NUM_IMAGES", "50")),
              "base_gb": _bg, "base_s": _bs, "inter_gb": round(_probe_inter_gb, 3), "inter_s": round(_probe_inter_s, 3)}), flush=True)

    pred_map_inter, pred_conf_inter = _pi3_eval_maps(res_inter)
    eval_pts_inter, eval_cols_inter = _build_eval_cloud_pi3(pred_map_inter, pred_conf_inter, images, image_paths, validity_masks_by_path)
    pointcloud_to_glb(eval_pts_inter, eval_cols_inter, str(work_dir / "pred_inter.glb"))
    if not SKIP_EVAL and gt_points is not None:
        metrics_inter, aligned_inter = align_and_eval(
            gt_points, gt_samples, pred_map_inter, pred_conf_inter, eval_pts_inter, image_paths
        )
        if metrics_inter is not None:
            save_metrics(eval_metrics_path, "inter", metrics_inter)
            if aligned_inter is not None:
                save_aligned_combined_glb(gt_points, aligned_inter, str(work_dir / "inter_aligned_combined.glb"))
            print(f"  Inter eval: Score={metrics_inter['Score']:.4f}")
    # Error-colored intervened cloud: align inter->base, color by NN distance to base.
    if not SKIP_BASELINE_PASS and "pred_map_base" in dir():
        try:
            save_inter_error_colored_glb_pi3(
                pred_map_base, pred_conf_base, pred_map_inter, pred_conf_inter, images, image_paths,
                str(work_dir / "inter_error_colored.glb"),
                validity_masks_by_path=validity_masks_by_path,
            )
        except Exception as e:
            print(f"  error-colored GLB failed: {e}")
    del res_inter
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print("Done.")
    return model


if __name__ == "__main__":
    main()
