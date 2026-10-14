# --- repo bootstrap: make the sibling `eval` package importable ---------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE)))
# eval/ modules import each other flatly, so that directory goes on the path too
for _p in (_REPO_ROOT, _os.path.join(_REPO_ROOT, "eval"), _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# -----------------------------------------------------------------------------
import gc
import json
import os
import os.path as osp
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DA3_ROOT = SCRIPT_DIR.parent
SRC_ROOT = DA3_ROOT / "src"
for _p in (DA3_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

VGGT_ROOT = Path(os.environ.get("VGGT_ROOT", ""))  # only needed if vggt is not installed
if str(VGGT_ROOT) not in sys.path and str(VGGT_ROOT) not in ("", "."):
    sys.path.insert(0, str(VGGT_ROOT))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.export.glb import export_to_glb
from eval.eval_utils import accuracy, completion, umeyama_ransac
from vggt.utils.load_fn import load_and_preprocess_images

try:
    from .loss import DA3AttentionIntervention, get_da3_backbone
except ImportError:
    from loss import DA3AttentionIntervention, get_da3_backbone


PATCH_SIZE = 14
PATCH_START_IDX = 1
NUM_IMAGES = int(os.environ.get("NUM_IMAGES", "50"))
IMAGES_DIR = f"images_{NUM_IMAGES}"
IMAGE_LIST_JSON = "image_list.json"
SCENE_NAME = os.environ.get("SCENE_NAME", "").strip()
SAM_MASK_DIR = os.environ.get("SAM_MASK_DIR", "").strip()
ATTENTION_STATS_DIR = os.environ.get("ATTENTION_STATS_DIR", "").strip()
MODEL_DIR = os.environ.get("MODEL_DIR", "depth-anything/DA3-LARGE-1.1").strip()
PROCESS_RES = int(os.environ.get("PROCESS_RES", "518"))
PROCESS_RES_METHOD = os.environ.get("PROCESS_RES_METHOD", "upper_bound_resize").strip()
REF_VIEW_STRATEGY = os.environ.get("REF_VIEW_STRATEGY", "first").strip()
SKIP_BASELINE_PASS = os.environ.get("SKIP_BASELINE_PASS", "").strip() == "1"
SKIP_EVAL = os.environ.get("SKIP_EVAL", "").strip() == "1"
PROBE = os.environ.get("PROBE", "").strip() == "1"
import time as _time_probe
USE_RAY_POSE = os.environ.get("USE_RAY_POSE", "").strip().lower() in ("1", "true", "yes")
DA3_EVAL_CONF_PERCENTILE = float(os.environ.get("DA3_EVAL_CONF_PERCENTILE", "40"))
# Sky-seg ONNX model used to drop sky points from the eval cloud (matches teammate's evaluate.py).
SKYSEG_ONNX = os.environ.get("SKYSEG_ONNX", _os.path.join(_REPO_ROOT, "intervention", "assets", "skyseg.onnx")).strip()
# Confidence percentile used to drop low-confidence (non-sky) eval points (teammate uses 60 -> keep top 40%).
DA3_EVAL_SKY_CONF_PERCENTILE = float(os.environ.get("DA3_EVAL_SKY_CONF_PERCENTILE", "60"))
_SKYSEG_SESSION = None


def _sanitize_scene_dir(name: str) -> str:
    import re

    s = re.sub(r"[^\w\-., ()]", "_", name)
    return s.strip("_") or "scene"


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


def _resize_mask_like_da3(mask: np.ndarray, image_path: str, process_res: int, process_res_method: str):
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255).convert("L")
    with Image.open(image_path) as img:
        w, h = img.size
    if process_res_method in ("upper_bound_resize", "upper_bound_crop"):
        scale = process_res / float(max(w, h))
    elif process_res_method in ("lower_bound_resize", "lower_bound_crop"):
        scale = process_res / float(min(w, h))
    else:
        raise ValueError(f"Unsupported process_res_method: {process_res_method}")
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    mask_img = mask_img.resize((new_w, new_h), Image.Resampling.NEAREST)
    if process_res_method.endswith("resize"):
        def nearest_multiple(x: int, p: int) -> int:
            down = (x // p) * p
            up = down + p
            return up if abs(up - x) <= abs(x - down) else down

        final_w = max(1, nearest_multiple(new_w, PATCH_SIZE))
        final_h = max(1, nearest_multiple(new_h, PATCH_SIZE))
        if (final_w, final_h) != (new_w, new_h):
            mask_img = mask_img.resize((final_w, final_h), Image.Resampling.NEAREST)
    else:
        final_w = (new_w // PATCH_SIZE) * PATCH_SIZE
        final_h = (new_h // PATCH_SIZE) * PATCH_SIZE
        left = (new_w - final_w) // 2
        top = (new_h - final_h) // 2
        mask_img = mask_img.crop((left, top, left + final_w, top + final_h))
    return np.asarray(mask_img) > 127


def map_mask_to_patches(mask, hp, wp, image_path=None):
    if mask is None or np.sum(mask) <= 0:
        return set()
    mask = mask.astype(bool)
    target_h, target_w = hp * PATCH_SIZE, wp * PATCH_SIZE
    if image_path and osp.exists(image_path):
        mask = _resize_mask_like_da3(mask, image_path, PROCESS_RES, PROCESS_RES_METHOD)
    if mask.shape != (target_h, target_w):
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255)
            .convert("L")
            .resize((target_w, target_h), Image.Resampling.NEAREST)
        ) > 127
    patch_mask = mask.reshape(hp, PATCH_SIZE, wp, PATCH_SIZE).mean(axis=(1, 3))
    idx = np.where(patch_mask > 0.5)
    return set(r * wp + c for r, c in zip(idx[0], idx[1]))


def build_arch_token_mask(image_paths, sam_mask_dir, scene_name, processed_h, processed_w, mask_suffix="_arch.npy", original_stems=None):
    hp, wp = processed_h // PATCH_SIZE, processed_w // PATCH_SIZE
    num_patches_per_img = hp * wp
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
        patches = map_mask_to_patches(arr.astype(bool), hp, wp, image_path=img_path)
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


def _import_vggt_run_pass():
    """The VGGT pass owns the shared ground-truth and validity-mask code."""
    import importlib
    return importlib.import_module("models.vggt.run_pass")

def load_validity_masks(scene_name: str, image_paths):
    if os.environ.get("USE_VALIDITY_MASKS", "1").strip().lower() not in ("1", "true", "yes"):
        return {}
    try:
        vggt_run_pass = _import_vggt_run_pass()
        return vggt_run_pass.load_validity_masks(scene_name, image_paths)
    except Exception as exc:
        print(f"Validity masks unavailable: {exc}")
        return {}


def load_gt(scene_name: str, work_dir: Path, image_paths, validity_masks_by_path=None):
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
    result = generate_gt_by_unproject(
        image_paths=image_paths,
        image_folder=str(work_dir / IMAGES_DIR),
        depth_folders=[scene_info.get("depth_folder", "")],
        recon_path=scene_info.get("recon_path", ""),
        output_glb_path=str(work_dir / f"{scene_name}_gt.glb"),
        verbose=True,
        validity_masks_by_path=validity_masks_by_path or {},
    )
    if result is None:
        return None, []
    return result


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        out = np.eye(4, dtype=ext.dtype)
        out[:3, :4] = ext
        return out
    raise ValueError(f"Unexpected extrinsic shape: {ext.shape}")


def _resize_mask_nearest(mask_bool, target_h, target_w):
    return np.array(
        Image.fromarray(mask_bool.astype(np.uint8) * 255)
        .convert("L")
        .resize((target_w, target_h), Image.Resampling.NEAREST)
    ) > 127


def _prediction_to_points(prediction: Prediction, image_paths=None, validity_masks_by_path=None, conf_percentile=DA3_EVAL_CONF_PERCENTILE):
    depth = prediction.depth
    conf = prediction.conf if prediction.conf is not None else np.ones_like(depth)
    intrinsics = prediction.intrinsics
    extrinsics = prediction.extrinsics
    images = prediction.processed_images
    n, h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pix = np.stack([xs, ys, np.ones_like(xs)], axis=-1).reshape(-1, 3)
    conf_thr = np.percentile(conf[np.isfinite(conf)], conf_percentile) if np.isfinite(conf).any() else -np.inf
    pts_maps, conf_maps, pts_flat, cols_flat = [], [], [], []
    vmasks = validity_masks_by_path or {}
    for i in range(n):
        d = depth[i]
        k_inv = np.linalg.inv(intrinsics[i])
        c2w = np.linalg.inv(_as_homogeneous44(extrinsics[i]))
        valid = np.isfinite(d) & (d > 1e-4) & np.isfinite(conf[i]) & (conf[i] >= conf_thr)
        if image_paths is not None and i < len(image_paths):
            vm = vmasks.get(osp.normpath(image_paths[i]))
            if vm is None:
                vm = vmasks.get(image_paths[i])
            if vm is not None:
                valid &= _resize_mask_nearest(vm, h, w)
        rays = k_inv @ pix.T
        cam = rays * d.reshape(-1)[None, :]
        cam_h = np.vstack([cam, np.ones((1, cam.shape[1]))])
        world = (c2w @ cam_h)[:3].T.reshape(h, w, 3).astype(np.float32)
        pts_maps.append(world)
        conf_maps.append(conf[i])
        keep = valid & np.isfinite(world).all(axis=-1)
        if keep.any():
            pts_flat.append(world[keep])
            if images is not None:
                cols_flat.append(images[i][keep].astype(np.uint8))
    pts_flat = np.concatenate(pts_flat, axis=0) if pts_flat else np.zeros((0, 3), dtype=np.float32)
    cols_flat = np.concatenate(cols_flat, axis=0) if cols_flat else np.zeros((0, 3), dtype=np.uint8)
    return np.stack(pts_maps), np.stack(conf_maps), pts_flat, cols_flat


def save_prediction_glb_da3(prediction: Prediction, output_glb: Path, image_paths=None, validity_masks_by_path=None):
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output_glb.parent)) as tmp_dir:
        export_to_glb(
            prediction,
            tmp_dir,
            num_max_points=int(os.environ.get("DA3_GLB_MAX_POINTS", "1000000")),
            conf_thresh_percentile=float(os.environ.get("DA3_GLB_CONF_PERCENTILE", "40")),
            show_cameras=os.environ.get("DA3_SHOW_CAMERAS", "1").strip().lower() in ("1", "true", "yes"),
            export_depth_vis=False,
        )
        scene_glb = Path(tmp_dir) / "scene.glb"
        if scene_glb.exists():
            os.replace(scene_glb, output_glb)
    return _prediction_to_points(prediction, image_paths, validity_masks_by_path)[2]


def compute_alignment_with_correspondence_da3(
    gt_samples,
    pred_points_map,
    pred_conf_map,
    image_paths=None,
    max_total_points=None,
    n_iterations=1000,
):
    import cv2 as _cv2

    s_count, h_pred, w_pred, _ = pred_points_map.shape
    if image_paths is not None and len(image_paths) == s_count:
        gt_by_path = {osp.normpath(s["image_path"]): s for s in gt_samples}

        def _get_sample(idx):
            return gt_by_path.get(osp.normpath(image_paths[idx]))
    else:
        def _get_sample(idx):
            return gt_samples[idx] if idx < len(gt_samples) else None

    correspondences_gt, correspondences_pred = [], []
    for i in range(s_count):
        sample = _get_sample(i)
        if sample is None:
            continue
        gt_depth = sample["depth_map"]
        gt_extrinsic = sample["extrinsic"]
        r_ref = sample["R_ref"]
        t_ref = sample["t_ref"]
        pred_conf = pred_conf_map[i]
        gt_depth_resized = _cv2.resize(gt_depth, (w_pred, h_pred), interpolation=_cv2.INTER_NEAREST)
        y, x = np.meshgrid(np.arange(h_pred), np.arange(w_pred), indexing="ij")
        k_gt_scaled = sample["K"].copy()
        sx = w_pred / gt_depth.shape[1]
        sy = h_pred / gt_depth.shape[0]
        k_gt_scaled[0, 0] *= sx
        k_gt_scaled[0, 2] *= sx
        k_gt_scaled[1, 1] *= sy
        k_gt_scaled[1, 2] *= sy
        fx_gt, fy_gt = k_gt_scaled[0, 0], k_gt_scaled[1, 1]
        cx_gt, cy_gt = k_gt_scaled[0, 2], k_gt_scaled[1, 2]
        z_gt = gt_depth_resized
        x_gt = (x - cx_gt) * z_gt / fx_gt
        y_gt = (y - cy_gt) * z_gt / fy_gt
        p_cam_gt = np.stack([x_gt, y_gt, z_gt], axis=-1)
        r_gt = gt_extrinsic[:3, :3]
        t_gt = gt_extrinsic[:3, 3]
        p_world_gt = p_cam_gt @ r_gt + (-r_gt.T @ t_gt)
        p_ref_gt = p_world_gt @ r_ref.T + t_ref
        rot_x_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        p_final_gt = p_ref_gt @ rot_x_180.T
        mask = (z_gt > 1e-4) & np.isfinite(z_gt) & np.isfinite(pred_conf)
        mask &= np.isfinite(pred_points_map[i]).all(axis=-1)
        vm = sample.get("validity_mask") if isinstance(sample, dict) else None
        if vm is not None:
            mask &= _resize_mask_nearest(vm, h_pred, w_pred)
        pts_pred_flat = pred_points_map[i][mask]
        pts_gt_flat = p_final_gt[mask]
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
    c, r, t = umeyama_ransac(pts_pred_all.T, pts_gt_all.T, n_iterations=n_iterations)
    return c, r, t


def align_and_eval(gt_points, gt_samples, pred_points_map, pred_conf_map, pred_points_flat, image_paths):
    if gt_points is None or len(gt_points) < 100 or pred_points_flat is None or len(pred_points_flat) < 100:
        return None, None
    c, r, t = compute_alignment_with_correspondence_da3(
        gt_samples=gt_samples,
        pred_points_map=pred_points_map,
        pred_conf_map=pred_conf_map,
        image_paths=image_paths,
        max_total_points=None,
        n_iterations=int(os.environ.get("ALIGNMENT_N_ITERATIONS", "1000")),
    )
    if c is None:
        return None, None
    aligned = (c * (r @ pred_points_flat.T) + t).T
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


def _preprocess(model: DepthAnything3, image_paths):
    # Match teammate evaluate.py: load_and_preprocess_images -> uint8 list -> model._preprocess_inputs.
    imgs01 = load_and_preprocess_images(image_paths)  # (S, 3, H, W) float [0, 1]
    image_list = [(im.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for im in imgs01]
    imgs_cpu, extrinsics, intrinsics = model._preprocess_inputs(
        image_list,
        None,
        None,
        PROCESS_RES,
        PROCESS_RES_METHOD,
    )
    imgs, ex_t, in_t = model._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)
    return imgs_cpu, imgs, ex_t, in_t


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


def _da3_per_pixel_world_valid(prediction: Prediction, image_paths=None, validity_masks_by_path=None):
    """Per-pixel world points (n,h,w,3), valid mask (n,h,w), and colors (n,h,w,3) for DA3.
    Filter = drop sky (ONNX skyseg) + low-conf (top-(100-pct)% kept) + (optional) building validity mask.
    Keeping the pixel grid lets us build base<->inter correspondences for the error-colored GLB."""
    import cv2

    depth = prediction.depth  # (n, h, w)
    conf = prediction.conf if prediction.conf is not None else np.ones_like(depth)
    intrinsics = prediction.intrinsics
    extrinsics = prediction.extrinsics
    images = prediction.processed_images  # (n, h, w, 3) uint8 RGB
    n, h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pix = np.stack([xs, ys, np.ones_like(xs)], axis=-1).reshape(-1, 3)
    vmasks = validity_masks_by_path or {}

    # DA3 conf is >= 1 for surfaces; <=0 means invalid/sky -> mark -1 (teammate convention).
    conf_proc = conf.astype(np.float32).copy()
    conf_proc[conf_proc <= 0] = -1.0

    # Drop sky via ONNX sky segmentation, per frame, mapped to conf resolution.
    session = _get_skyseg_session()
    if session is not None:
        for i in range(n):
            frame_bgr = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
            sky_score = _run_skyseg(session, [320, 320], frame_bgr)
            sky_score = cv2.resize(sky_score, (w, h), interpolation=cv2.INTER_LINEAR)
            conf_proc[i][sky_score >= 20] = -1.0

    # Drop low-confidence non-sky points by percentile over positive values.
    pos = conf_proc[conf_proc > 0]
    if len(pos) > 0:
        cutoff = np.percentile(pos, DA3_EVAL_SKY_CONF_PERCENTILE)
        conf_proc[(conf_proc > 0) & (conf_proc < cutoff)] = -1.0

    world_all = np.full((n, h, w, 3), np.nan, dtype=np.float32)
    valid_all = np.zeros((n, h, w), dtype=bool)
    for i in range(n):
        d = depth[i]
        k_inv = np.linalg.inv(intrinsics[i])
        c2w = np.linalg.inv(_as_homogeneous44(extrinsics[i]))
        cam = (k_inv @ pix.T) * d.reshape(-1)[None, :]
        cam_h = np.vstack([cam, np.ones((1, cam.shape[1]))])
        world = (c2w @ cam_h)[:3].T.reshape(h, w, 3).astype(np.float32)
        world_all[i] = world
        keep = (conf_proc[i] >= 0) & np.isfinite(world).all(axis=-1) & np.isfinite(d) & (d > 1e-4)
        if image_paths is not None and i < len(image_paths):
            vm = vmasks.get(image_paths[i])
            if vm is None:
                vm = vmasks.get(osp.normpath(image_paths[i]))
            if vm is not None:
                keep = keep & _resize_mask_nearest(vm, h, w)
        valid_all[i] = keep
    return world_all, valid_all, images


def _build_eval_cloud_da3(prediction: Prediction, image_paths=None, validity_masks_by_path=None):
    """Eval (Chamfer) cloud: sky + low-conf dropped, restricted to building validity-mask pixels when
    validity_masks_by_path is given. Returns (pts (P,3) float32, cols (P,3) uint8)."""
    world, valid, images = _da3_per_pixel_world_valid(prediction, image_paths, validity_masks_by_path)
    n = world.shape[0]
    pts_flat = [world[i][valid[i]] for i in range(n) if valid[i].any()]
    cols_flat = [images[i][valid[i]].astype(np.uint8) for i in range(n) if valid[i].any()]
    pts = np.concatenate(pts_flat, axis=0) if pts_flat else np.zeros((0, 3), dtype=np.float32)
    cols = np.concatenate(cols_flat, axis=0) if cols_flat else np.zeros((0, 3), dtype=np.uint8)
    if len(pts) > 500000:
        idx = np.random.choice(len(pts), 500000, replace=False)
        pts = pts[idx]
        cols = cols[idx]
    return pts, cols


def save_inter_error_colored_glb_da3(pred_base, pred_inter, image_paths, output_path,
                                     validity_masks_by_path=None, max_points=500000):
    """Align intervened DA3 prediction to base (per-pixel-correspondence Sim3), color each intervened
    point by its nearest-neighbour distance to the base cloud (turbo). Same frame as pred_*.glb (raw world)."""
    try:
        import trimesh
        from scipy.spatial import cKDTree
        import matplotlib.cm as _cm
        from eval.eval_utils import umeyama_ransac
    except Exception as e:
        print(f"  error-colored GLB skipped (import failed: {e})")
        return
    bw, bv, _ = _da3_per_pixel_world_valid(pred_base, image_paths, validity_masks_by_path)
    iw, iv, _ = _da3_per_pixel_world_valid(pred_inter, image_paths, validity_masks_by_path)
    base_pts = bw[bv]; inter_pts = iw[iv]
    if len(base_pts) < 100 or len(inter_pts) < 100:
        print("  error-colored GLB skipped (too few points)")
        return
    corr = bv & iv
    c, R, t = 1.0, np.eye(3), np.zeros(3)
    if int(corr.sum()) >= 100:
        bc = bw[corr]; ic = iw[corr]
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


def _save_points_glb(points, colors, output_path):
    """Save a point-cloud GLB (eval points + colors)."""
    try:
        import trimesh
    except ImportError:
        return
    os.makedirs(osp.dirname(output_path) or ".", exist_ok=True)
    verts = points if len(points) else np.zeros((1, 3), dtype=np.float32)
    pc = trimesh.PointCloud(vertices=verts)
    if colors is not None and len(colors) == len(points) and len(points) > 0:
        pc.colors = colors.astype(np.uint8)
    trimesh.Scene([pc]).export(output_path, file_type="glb")


def save_aligned_combined_glb(gt_points, pred_points_aligned, output_path, save_n=500000):
    """GT (green) + aligned prediction (red) combined GLB, identical to VGGT intervene/evaluate.py."""
    try:
        import trimesh
    except ImportError:
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


def _forward_da3(model: DepthAnything3, imgs, ex_t, in_t):
    device = imgs.device
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    net = model.model
    if device.type == "cuda":
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            return net(
                imgs,
                ex_t,
                in_t,
                export_feat_layers=[],
                infer_gs=False,
                use_ray_pose=USE_RAY_POSE,
                ref_view_strategy=REF_VIEW_STRATEGY,
            )
    with torch.no_grad():
        return net(
            imgs,
            ex_t,
            in_t,
            export_feat_layers=[],
            infer_gs=False,
            use_ray_pose=USE_RAY_POSE,
            ref_view_strategy=REF_VIEW_STRATEGY,
        )


def _to_prediction(model: DepthAnything3, raw_output, imgs_cpu):
    prediction = model._convert_to_prediction(raw_output)
    return model._add_processed_images(prediction, imgs_cpu)


def _default_layer_range(model):
    backbone = get_da3_backbone(model)
    alt_start = int(getattr(backbone, "alt_start", -1))
    n_layers = len(backbone.blocks)
    if alt_start < 0:
        return 0, n_layers - 1
    first = alt_start if alt_start % 2 == 1 else alt_start + 1
    return first, n_layers - 1


def main(scene_name=None, work_dir=None, model=None):
    scene_name = (scene_name or SCENE_NAME or "").strip()
    if not scene_name:
        raise RuntimeError("scene_name is required.")
    work_dir = Path(work_dir) if work_dir is not None else SCRIPT_DIR
    work_dir.mkdir(parents=True, exist_ok=True)
    image_paths = get_image_paths(work_dir)
    original_stems = _get_original_stems(work_dir, image_paths)
    print(f"Using {len(image_paths)} images from {work_dir}")

    device = torch.device(os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    if model is None:
        print(f"Loading DA3 model: {MODEL_DIR}")
        model = DepthAnything3.from_pretrained(MODEL_DIR).to(device).eval()
    else:
        model = model.to(device).eval()

    default_first, default_last = _default_layer_range(model)
    first_layer = int(os.environ.get("FIRST_INTERVENTION_LAYER", str(default_first)))
    last_layer = int(os.environ.get("LAST_INTERVENTION_LAYER", str(default_last)))
    eval_metrics_path = work_dir / f"eval_metrics_{first_layer}_{last_layer}.json"

    imgs_cpu, imgs, ex_t, in_t = _preprocess(model, image_paths)
    processed_h, processed_w = imgs_cpu.shape[-2:]
    validity_masks_by_path = load_validity_masks(scene_name, image_paths)
    if validity_masks_by_path:
        print(f"Validity masks: {len(validity_masks_by_path)}/{len(image_paths)} views")

    gt_points, gt_samples = (None, []) if SKIP_EVAL else load_gt(scene_name, work_dir, image_paths, validity_masks_by_path)
    if gt_points is not None:
        print(f"Loaded GT points: {len(gt_points)}")

    if not SKIP_BASELINE_PASS:
        print("Pass 1: baseline (no intervention)")
        if PROBE:
            if torch.cuda.is_available():
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            _probe_t = _time_probe.perf_counter()
        raw_base = _forward_da3(model, imgs, ex_t, in_t)
        if PROBE:
            if torch.cuda.is_available():
                torch.cuda.synchronize(); _probe_base_gb = torch.cuda.max_memory_allocated() / 1e9
            else:
                _probe_base_gb = -1.0
            _probe_base_s = _time_probe.perf_counter() - _probe_t
        pred_base = _to_prediction(model, raw_base, imgs_cpu)
        eval_pts_base, eval_cols_base = _build_eval_cloud_da3(pred_base, image_paths, validity_masks_by_path)
        _save_points_glb(eval_pts_base, eval_cols_base, str(work_dir / "pred_base.glb"))
        if not SKIP_EVAL and gt_points is not None:
            pred_map_base, pred_conf_base, _, _ = _prediction_to_points(
                pred_base, image_paths, validity_masks_by_path
            )
            metrics_base, aligned_base = align_and_eval(
                gt_points, gt_samples, pred_map_base, pred_conf_base, eval_pts_base, image_paths
            )
            if metrics_base is not None:
                save_metrics(eval_metrics_path, "base", metrics_base)
                if aligned_base is not None:
                    save_aligned_combined_glb(gt_points, aligned_base, str(work_dir / "base_aligned_combined.glb"))
                print(f"  Base eval: Score={metrics_base['Score']:.4f}")
        # keep pred_base (CPU numpy) for the error-colored GLB; free only the GPU raw output
        del raw_base
        gc.collect()
        if device.type == "cuda":
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
    intervention = DA3AttentionIntervention(
        model=model,
        arch_token_mask=arch_token_mask,
        first_intervention_layer=first_layer,
        last_intervention_layer=last_layer,
        step_size=step_size,
        attention_label=arch_tag,
        global_layers_only=True,
    )
    if PROBE:
        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        _probe_ti = _time_probe.perf_counter()
    intervention.register_intervention()
    try:
        raw_inter = _forward_da3(model, imgs, ex_t, in_t)
    finally:
        intervention.remove_intervention()
    if PROBE:
        _cuda = torch.cuda.is_available()
        if _cuda:
            torch.cuda.synchronize(); _probe_inter_gb = torch.cuda.max_memory_allocated() / 1e9
        else:
            _probe_inter_gb = -1.0
        _probe_inter_s = _time_probe.perf_counter() - _probe_ti
        _bg = round(_probe_base_gb, 3) if "_probe_base_gb" in locals() else None
        _bs = round(_probe_base_s, 3) if "_probe_base_s" in locals() else None
        print("PROBE_RESULT " + json.dumps({"scene": scene_name, "num_images": int(os.environ.get("NUM_IMAGES", "50")),
              "cuda": bool(_cuda), "base_gb": _bg, "base_s": _bs, "inter_gb": round(_probe_inter_gb, 3), "inter_s": round(_probe_inter_s, 3)}), flush=True)

    pred_inter = _to_prediction(model, raw_inter, imgs_cpu)
    eval_pts_inter, eval_cols_inter = _build_eval_cloud_da3(pred_inter, image_paths, validity_masks_by_path)
    _save_points_glb(eval_pts_inter, eval_cols_inter, str(work_dir / "pred_inter.glb"))
    if not SKIP_EVAL and gt_points is not None:
        pred_map_inter, pred_conf_inter, _, _ = _prediction_to_points(
            pred_inter, image_paths, validity_masks_by_path
        )
        metrics_inter, aligned_inter = align_and_eval(
            gt_points, gt_samples, pred_map_inter, pred_conf_inter, eval_pts_inter, image_paths
        )
        if metrics_inter is not None:
            save_metrics(eval_metrics_path, "inter", metrics_inter)
            if aligned_inter is not None:
                save_aligned_combined_glb(gt_points, aligned_inter, str(work_dir / "inter_aligned_combined.glb"))
            print(f"  Inter eval: Score={metrics_inter['Score']:.4f}")
    # Error-colored intervened cloud: align inter->base, color by NN distance to base.
    if not SKIP_BASELINE_PASS and "pred_base" in dir():
        try:
            save_inter_error_colored_glb_da3(
                pred_base, pred_inter, image_paths,
                str(work_dir / "inter_error_colored.glb"),
                validity_masks_by_path=validity_masks_by_path,
            )
        except Exception as e:
            print(f"  error-colored GLB failed: {e}")
        try:
            del pred_base
        except NameError:
            pass
    del raw_inter, pred_inter
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Done.")
    return model


if __name__ == "__main__":
    main()
