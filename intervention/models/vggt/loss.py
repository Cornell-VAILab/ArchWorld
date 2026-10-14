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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.checkpoint import checkpoint

def _default_loss_scale():
    return float(os.environ.get("INTERVENTION_LOSS_SCALE", "1e5"))

class VGGTAttentionIntervention:
    def __init__(self, model, arch_mask=None, arch_token_mask=None, loss_valid_mask=None, first_intervention_layer=10, last_intervention_layer=23, step_size=1.0, attention_label=None, loss_scale=None, record_grad_norms=False, layer_feature_labels=None, use_sum=False, adaptive_alpha=None, max_steps=1, attn_improve_thresh=0.1, loss_type="arch_attn", group_token_masks=None):
        self.model = model
        self.arch_mask = arch_mask
        self.arch_token_mask = arch_token_mask  # (N,) tensor or dict[layer_idx, Tensor] for per-layer features
        # Per-group masks for loss_type="multi_self_coherence": dict[tag, (N,) tensor] of disjoint or overlapping
        # semantic masks (statue, column, dome, ...). Each group's intra-group attention is boosted independently,
        # so non-tagged tokens are untouched and tagged groups don't compete via softmax.
        self.group_token_masks = group_token_masks
        self.loss_valid_mask = loss_valid_mask  # optional (N,): 1 only for arch tokens in images that have at least one arch token; used to skip images without masks in loss
        self.attention_label = attention_label if attention_label else "arch"  # for logging: "dome", "arch", "door", etc.
        self.layer_feature_labels = layer_feature_labels  # optional dict[layer_idx, str] for per-layer logging
        self.first_intervention_layer = first_intervention_layer
        self.last_intervention_layer = last_intervention_layer
        self.step_size = step_size
        self.loss_scale = float(loss_scale) if loss_scale is not None else _default_loss_scale()
        self.intervention_hooks = []
        self.record_grad_norms = record_grad_norms  # for suggest_step_size.py: record grad norm per layer
        self.use_sum = use_sum  # if True: maximize A^l(e^*) = sum (paper Eq.); else: maximize mean (default, more stable)
        # Adaptive step: when set, ignore step_size; choose step so ||delta||_masked = alpha * ||x||_masked per layer.
        # One pass, self-tunes per scene/layer via activation norms -- replaces the step_size sweep.
        self.adaptive_alpha = float(adaptive_alpha) if adaptive_alpha is not None else None
        # Loss type: which objective drives the gradient direction.
        #   arch_attn (default): maximize attention from ALL tokens → arch tokens (original)
        #   self_coherence: maximize attention from ARCH tokens → ARCH tokens (cluster arch group)
        #   bottom_k: maximize attention to the LEAST-attended arch tokens (focus where needed)
        #   contextual: maximize attention from NON-ARCH tokens → ARCH tokens (make context notice arch)
        self.loss_type = loss_type
        # Multi-step: repeat gradient ascent per layer up to max_steps. Stop early if attention improvement
        # drops below attn_improve_thresh * first_step_improvement, or if attention decreases (revert to best).
        self.max_steps = max(1, int(max_steps))
        self.attn_improve_thresh = float(attn_improve_thresh)
    
    def compute_arch_loss(self, attention_map, arch_mask_tokens):
        """
        Compute loss to maximize attention to arch regions
        """
        B, num_heads, N, _ = attention_map.shape
        
        # Convert mask to float
        mask_float = arch_mask_tokens.float().to(attention_map.device)
        
        # Create 2D mask
        mask_2d = mask_float.unsqueeze(0) * mask_float.unsqueeze(1)
        
        # Apply mask
        masked_attn = attention_map * mask_2d.unsqueeze(0).unsqueeze(0)
        
        # Sum and normalize
        arch_attention = masked_attn.sum(dim=-1).mean(dim=(0, 1))
        
        # Normalize by number of arch tokens
        num_arch_tokens = mask_float.sum()
        if num_arch_tokens > 0:
            arch_attention = arch_attention / num_arch_tokens
        
        # Use negative mean with constant scale (scale amplifies gradient)
        loss = self.loss_scale * (-arch_attention.mean())
        
        return loss
        
    def pixel_mask_to_token_mask(self, pixel_mask, patch_size=14):
        """Convert pixel-level mask to token-level mask"""
        H, W = pixel_mask.shape
        if H != W:
            pixel_mask = F.pad(pixel_mask, (0, 0, (W - H) // 2, (W - H) // 2), value=0)
        token_w = W // patch_size
        pixel_mask_resized = pixel_mask.reshape(token_w, patch_size, token_w, patch_size)
        token_mask = F.pad(pixel_mask_resized.max(dim=3)[0].max(dim=1)[0].flatten(), (5, 0))
        return token_mask.flatten()
    
    def intervene_forward_pre_hook(self, module, input, layer_idx=None):
        """
        PRE-hook that modifies input before attention. All on GPU.
        Uses gradient checkpointing and in-place updates to minimize VRAM.
        layer_idx: optional layer number for logging (e.g. 5 for "Layer 5 arch attention ...").
        """
        x = input[0]
        dev = x.device
        B, N, C = x.shape

        if dev.type == "cuda":
            torch.cuda.empty_cache()

        qkv_weight = module.qkv.weight.detach()
        qkv_bias = module.qkv.bias.detach() if module.qkv.bias is not None else None
        scale = module.scale

        # Per-layer mask: arch_token_mask can be dict[layer_idx, Tensor] for per-layer features
        if self.arch_token_mask is not None:
            if isinstance(self.arch_token_mask, dict) and layer_idx is not None:
                arch_mask_tokens = self.arch_token_mask.get(layer_idx)
                if arch_mask_tokens is None:
                    return input
                arch_mask_tokens = arch_mask_tokens.to(dev).float()
            else:
                arch_mask_tokens = self.arch_token_mask.to(dev).float()
            if arch_mask_tokens.dim() > 1:
                arch_mask_tokens = arch_mask_tokens.view(-1)
            arch_mask_tokens = arch_mask_tokens[:N]
        else:
            arch_mask_tokens = self.pixel_mask_to_token_mask(
                self.arch_mask.cpu()
            ).to(dev)

        if arch_mask_tokens.sum() < 1:
            return input

        mask_float = arch_mask_tokens.float()
        num_heads, head_dim = module.num_heads, module.head_dim
        loss_scale = self.loss_scale

        def _forward_to_loss(x_i):
            qkv = x_i @ qkv_weight.T
            if qkv_bias is not None:
                qkv = qkv + qkv_bias
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            num_arch = mask_float.sum()
            # v_custom: attn @ v_custom gives per-query sum of attention to arch keys
            v_custom = mask_float.view(1, 1, N, 1).to(q.dtype).to(q.device).expand(B, num_heads, N, head_dim)
            attn_out = F.scaled_dot_product_attention(q, k, v_custom, scale=scale)
            # attn_out[:,:,i,0] = sum_{j in arch} attn[i,j]  (how much query i attends to arch keys)
            arch_before = (attn_out[:, :, :, 0].sum() / (num_arch ** 2 + 1e-8)).item()

            if self.loss_type == "self_coherence":
                # Maximize arch→arch attention only: only count queries that ARE arch tokens
                # This makes arch tokens cluster together without perturbing non-arch
                query_mask = mask_float.view(1, 1, N)  # (1, 1, N) — which queries are arch
                arch_to_arch = (attn_out[:, :, :, 0] * query_mask).sum()
                if num_arch > 0:
                    arch_to_arch = arch_to_arch / (num_arch ** 2)
                loss_val = loss_scale * (-arch_to_arch)

            elif self.loss_type == "contextual":
                # Maximize non-arch→arch attention: make context tokens notice architecture
                nonarch_mask = (1.0 - mask_float).view(1, 1, N)
                num_nonarch = nonarch_mask.sum() + 1e-8
                nonarch_to_arch = (attn_out[:, :, :, 0] * nonarch_mask).sum()
                nonarch_to_arch = nonarch_to_arch / (num_nonarch * num_arch + 1e-8)
                loss_val = loss_scale * (-nonarch_to_arch)

            elif self.loss_type == "bottom_k":
                # Maximize attention to the LEAST-attended arch tokens (bottom 25%)
                # Per-token total incoming arch attention: sum over heads and batch
                per_query_attn = attn_out[:, :, :, 0].mean(dim=(0, 1))  # (N,)
                # Only look at arch queries
                arch_idx = mask_float.bool()
                arch_vals = per_query_attn[arch_idx]  # attention received by each arch token
                k = max(1, int(0.25 * arch_vals.shape[0]))
                bottom_vals, _ = torch.topk(arch_vals, k, largest=False)
                loss_val = loss_scale * (-bottom_vals.mean())

            elif self.loss_type == "attend_excite":
                # Attend & Excite (Chefer et al., adapted): push up the weakest-linked query-to-arch attention.
                # attn_out[:,:,i,0] = sum_{j in arch} attn(i,j) = "how much query i is aware of arch".
                # Loss = -min_i of this — ensures NO query is blind to architecture.
                # (True A&E uses min-over-keys of max-over-queries, but per-key info needs D_v = num_arch,
                #  which breaks flash-attention and OOMs at N=7680. This is A&E-in-spirit on the query side.)
                per_query_arch = attn_out[:, :, :, 0].mean(dim=(0, 1))  # (N,) — avg over B, H
                # Only consider NON-arch queries (arch queries naturally have high self-attn; focus on others)
                nonarch_mask_b = (1.0 - mask_float).bool()
                if nonarch_mask_b.any():
                    nonarch_vals = per_query_arch[nonarch_mask_b]
                    # Soft-min via negative log-sum-exp (differentiable, smoother than hard min)
                    temperature = 20.0  # higher = closer to hard min
                    soft_min = -torch.logsumexp(-temperature * nonarch_vals, dim=0) / temperature
                    loss_val = loss_scale * (-soft_min)
                else:
                    loss_val = torch.zeros((), device=q.device, dtype=q.dtype, requires_grad=True)

            elif self.loss_type == "multi_self_coherence":
                # Block-diagonal localized intervention: for each semantic group g with mask m_g,
                # maximize attention from g's query tokens to g's key tokens, summed over groups.
                # Because each group's queries are a disjoint (or lightly overlapping) subset, the
                # softmax budget is not stolen between groups — statue and dome both consolidate.
                groups = self.group_token_masks
                if not groups:
                    # Fall back to single-mask self_coherence if no group dict supplied
                    query_mask = mask_float.view(1, 1, N)
                    arch_to_arch = (attn_out[:, :, :, 0] * query_mask).sum()
                    if num_arch > 0:
                        arch_to_arch = arch_to_arch / (num_arch ** 2)
                    loss_val = loss_scale * (-arch_to_arch)
                else:
                    total = torch.zeros((), device=q.device, dtype=q.dtype)
                    n_active = 0
                    for _tag, m_g in groups.items():
                        m_g = m_g.to(q.device).float().view(-1)[:N]
                        num_g = m_g.sum()
                        if num_g < 1:
                            continue
                        v_g = m_g.view(1, 1, N, 1).to(q.dtype).expand(B, num_heads, N, head_dim)
                        ao_g = F.scaled_dot_product_attention(q, k, v_g, scale=scale)
                        # ao_g[:,:,i,0] = sum_{j in g} attn(i,j). Keep only queries i that ARE in group g.
                        q_in_g = m_g.view(1, 1, N)
                        g_to_g = (ao_g[:, :, :, 0] * q_in_g).sum() / (num_g ** 2 + 1e-8)
                        total = total + g_to_g
                        n_active += 1
                    # Normalize by number of active groups so summed loss magnitude stays O(1),
                    # letting step sizes remain comparable to the single-tag self_coherence case.
                    if n_active > 0:
                        total = total / n_active
                    loss_val = loss_scale * (-total)

            else:  # "arch_attn" (original default)
                if self.use_sum:
                    if self.loss_valid_mask is not None:
                        valid_mask = self.loss_valid_mask.to(attn_out.device).float().view(1, 1, -1)[:, :, : attn_out.shape[2]]
                        arch_attention_scalar = (attn_out[:, :, :, 0] * valid_mask).sum()
                    else:
                        arch_attention_scalar = attn_out[:, :, :, 0].sum()
                    loss_val = loss_scale * (-arch_attention_scalar)
                else:
                    arch_attention = attn_out[:, :, :, 0].mean(dim=(0, 1))
                    if num_arch > 0:
                        arch_attention = arch_attention / num_arch
                    if self.loss_valid_mask is not None:
                        valid_mask = self.loss_valid_mask.to(arch_attention.device).float()[: arch_attention.shape[0]]
                        valid_sum = valid_mask.sum() + 1e-8
                        loss_val = loss_scale * (-(arch_attention * valid_mask).sum() / valid_sum)
                    else:
                        loss_val = loss_scale * (-arch_attention.mean())

            return loss_val, arch_before

        def _measure_attn(x_t):
            """Cheap arch-attention measurement (forward through QKV only, no grad)."""
            with torch.no_grad():
                qkv_t = x_t @ qkv_weight.T
                if qkv_bias is not None:
                    qkv_t = qkv_t + qkv_bias
                qkv_t = qkv_t.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                q_t, k_t, v_t = qkv_t.unbind(0)
                del qkv_t
                v_c = mask_float.view(1, 1, N, 1).to(q_t.dtype).to(q_t.device).expand(B, num_heads, N, head_dim)
                ao = F.scaled_dot_product_attention(q_t, k_t, v_c, scale=scale)
                val = (ao[:, :, :, 0].sum() / (mask_float.sum() ** 2 + 1e-8)).item()
                del q_t, k_t, v_t, ao, v_c
            return val

        def _grad_step(x_t):
            """One gradient-ascent step. Returns (updated x, step_eff used, grad_norm)."""
            x_t = x_t.detach().requires_grad_(True)
            loss, _ = checkpoint(_forward_to_loss, x_t, use_reentrant=True)
            loss.backward()
            g = x_t.grad
            del loss
            g_norm = float(g.norm().item()) if g is not None else 0.0
            with torch.no_grad():
                if self.adaptive_alpha is not None:
                    mask_bool = mask_float.bool()
                    if mask_bool.any():
                        x_m_norm = x_t.detach()[:, mask_bool, :].norm()
                        g_m_norm = g[:, mask_bool, :].norm().clamp(min=1e-8)
                        s = self.adaptive_alpha * (x_m_norm / g_m_norm)
                    else:
                        s = torch.tensor(0.0, device=g.device)
                    x_t.sub_(s * g)
                else:
                    s = self.step_size
                    x_t.sub_(s * g)
            del g
            x_t.grad = None
            return x_t, float(s) if isinstance(s, (int, float)) else float(s.item()), g_norm

        # --- Multi-step gradient ascent with attention-based early stopping ---
        arch_attn_before_val = _measure_attn(x.detach())
        best_x = x.detach().clone()
        best_attn = arch_attn_before_val
        first_step_gain = None
        x_cur = x.detach()
        n_steps_taken = 0

        for step_i in range(self.max_steps):
            with torch.enable_grad():
                x_cur, step_eff, grad_norm = _grad_step(x_cur)
            # Record grad norm (first step only, for suggest_step_size.py)
            if step_i == 0 and layer_idx is not None and getattr(self, "record_grad_norms", False):
                if not hasattr(self, "grad_norms"):
                    self.grad_norms = []
                self.grad_norms.append((layer_idx, grad_norm))

            attn_now = _measure_attn(x_cur)
            gain = attn_now - best_attn
            n_steps_taken = step_i + 1

            if self.max_steps > 1 and layer_idx is not None:
                print(f"  [step {step_i+1}/{self.max_steps}] layer {layer_idx}: attn={attn_now:.6f} (Δ{gain:+.6f}), step_eff={step_eff:.4g}")

            if attn_now > best_attn:
                best_attn = attn_now
                best_x = x_cur.detach().clone()
                if first_step_gain is None:
                    first_step_gain = gain
                elif gain < self.attn_improve_thresh * first_step_gain:
                    # Diminishing returns — stop
                    break
            else:
                # Attention decreased — revert to best and stop
                x_cur = best_x
                break

        x_intervene = best_x
        arch_attn_after_val = best_attn

        delta = arch_attn_after_val - arch_attn_before_val
        # For per-layer: use layer-specific label if arch_token_mask is dict with layer_feature metadata
        label = self.attention_label
        if isinstance(getattr(self, "layer_feature_labels", None), dict):
            label = self.layer_feature_labels.get(layer_idx, label)
        steps_note = f" [{n_steps_taken} steps]" if self.max_steps > 1 else ""
        if layer_idx is not None:
            # Record for every hooked layer, not only in per-layer-feature mode: run_pass.py's
            # delta plot reads this list, and da3/pi3 both record on `layer_idx is not None`.
            self.attention_deltas.append((layer_idx, arch_attn_before_val, arch_attn_after_val, delta))
            print(f"Layer {layer_idx} {label} attention {arch_attn_before_val:.6f} -> {arch_attn_after_val:.6f} (Δ{delta:.6f}){steps_note}")
        else:
            print(f"{label.capitalize()} attention {arch_attn_before_val:.6f} -> {arch_attn_after_val:.6f} (Δ{delta:.6f}){steps_note}")
        input[0].data.copy_(x_intervene)
        del x_intervene
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return input

    def _make_hook(self, layer_idx):
        """Return a hook that calls intervene_forward_pre_hook with this layer index."""
        def hook(module, input):
            return self.intervene_forward_pre_hook(module, input, layer_idx=layer_idx)
        return hook

    def register_intervention(self):
        """Register the intervention hook at the specified layers (first_intervention_layer to last_intervention_layer, inclusive)."""
        self.attention_deltas = []
        if self.record_grad_norms:
            self.grad_norms = []
        for layer in range(self.first_intervention_layer, self.last_intervention_layer + 1):
            target_block = self.model.aggregator.global_blocks[layer]
            self.intervention_hooks.append(target_block.attn.register_forward_pre_hook(
                self._make_hook(layer)
            ))
            print(f"Registered intervention hook at layer {layer}")
    
    def remove_intervention(self):
        """Remove the intervention hook"""
        for hook in self.intervention_hooks:
            hook.remove()
        self.intervention_hooks = []


def compute_arch_attention_stats(attention_weights, arch_mask, patch_size=14):
    """Compute statistics about attention to arch regions"""
    if attention_weights is None:
        return None

    H, W = arch_mask.shape
    if H != W:
        arch_mask = F.pad(arch_mask, (0, 0, (W - H) // 2, (W - H) // 2), value=0)
    token_w = W // patch_size

    arch_mask_resized = arch_mask.reshape(token_w, patch_size, token_w, patch_size)
    token_mask = arch_mask_resized.max(dim=3)[0].max(dim=1)[0].flatten()
    
    attn = attention_weights[0].mean(dim=0)[5::]
    arch_attention = attn[token_mask.bool().cpu()].mean()
    non_arch_attention = attn[~token_mask.bool().cpu()].mean()
    
    return {
        "arch_attention_mean": arch_attention.item(),
        "non_arch_attention_mean": non_arch_attention.item(),
        "attention_ratio": (arch_attention / non_arch_attention).item()
    }