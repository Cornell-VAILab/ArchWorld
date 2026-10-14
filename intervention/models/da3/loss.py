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
import torch.nn.functional as F


def _default_loss_scale() -> float:
    return float(os.environ.get("INTERVENTION_LOSS_SCALE", "1e5"))


def get_da3_anyview_net(model):
    """Return the DA3 any-view branch, handling both wrapped and nested models."""
    net = getattr(model, "model", model)
    return getattr(net, "da3", net)


def get_da3_backbone(model):
    net = get_da3_anyview_net(model)
    return net.backbone.pretrained


def get_da3_global_layers(model, first_layer: int | None = None, last_layer: int | None = None) -> list[int]:
    """Global DA3 layers are odd DINO blocks at or after alt_start."""
    backbone = get_da3_backbone(model)
    alt_start = int(getattr(backbone, "alt_start", -1))
    n_layers = len(backbone.blocks)
    start = 0 if first_layer is None else max(0, int(first_layer))
    end = n_layers - 1 if last_layer is None else min(int(last_layer), n_layers - 1)
    if alt_start < 0:
        return []
    return [idx for idx in range(start, end + 1) if idx >= alt_start and idx % 2 == 1]


class DA3AttentionIntervention:
    """
    Attention intervention for DA3 DINOv2 global attention blocks.

    DA3 alternates local and global blocks after ``alt_start``. The global blocks
    flatten all views into one token sequence, matching the SAM mask layout.
    """

    def __init__(
        self,
        model,
        arch_token_mask,
        first_intervention_layer: int | None = None,
        last_intervention_layer: int | None = None,
        step_size: float = 1000.0,
        loss_scale: float | None = None,
        attention_label: str = "arch",
        global_layers_only: bool = True,
    ):
        self.model = model
        self.backbone = get_da3_backbone(model)
        self.arch_token_mask = arch_token_mask
        self.first_intervention_layer = first_intervention_layer
        self.last_intervention_layer = last_intervention_layer
        self.step_size = float(step_size)
        self.loss_scale = float(loss_scale) if loss_scale is not None else _default_loss_scale()
        self.attention_label = attention_label
        self.global_layers_only = global_layers_only
        self.full_query_objective = os.environ.get("INTERVENTION_FULL_QUERY_OBJECTIVE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        self.intervention_hooks = []
        self.attention_deltas = []

    def _qkv(self, module, x, pos=None):
        bsz, n_tokens, dim = x.shape
        qkv = module.qkv(x).reshape(bsz, n_tokens, 3, module.num_heads, dim // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = module.q_norm(q), module.k_norm(k)
        if getattr(module, "rope", None) is not None and pos is not None:
            q = module.rope(q, pos)
            k = module.rope(k, pos)
        return q, k, v

    def _attention_score(self, module, x, mask, pos=None):
        q, k, v = self._qkv(module, x, pos=pos)
        query_idx = torch.nonzero(mask > 0.5, as_tuple=False).view(-1)
        if query_idx.numel() == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        if self.full_query_objective:
            q_sel = q
            n_query = q.shape[2]
        else:
            q_sel = q.index_select(2, query_idx)
            n_query = query_idx.numel()
        v_custom = mask.view(1, 1, -1, 1).to(v.dtype).expand(q.shape[0], module.num_heads, q.shape[2], q.shape[-1]).contiguous()
        attn_out = F.scaled_dot_product_attention(q_sel.contiguous().to(v.dtype), k.contiguous().to(v.dtype), v_custom)
        num_arch = mask.sum().clamp(min=1.0)
        return attn_out[:, :, :, 0].sum() / (n_query * num_arch + 1e-8)

    def _measure_arch_attention(self, module, x, mask, pos=None):
        with torch.no_grad():
            return float(self._attention_score(module, x, mask, pos=pos).item())

    def intervene_forward_pre_hook(self, module, args, kwargs, layer_idx=None):
        x = args[0]
        if x.ndim != 3:
            return args, kwargs

        mask = self.arch_token_mask.to(x.device).float().view(-1)
        if x.shape[1] != mask.shape[0]:
            return args, kwargs
        if mask.sum() < 1:
            return args, kwargs

        pos = kwargs.get("pos", None)
        attn_before = self._measure_arch_attention(module, x.detach(), mask, pos=pos)
        with torch.enable_grad():
            x_grad = x.detach().requires_grad_(True)
            arch_attention = self._attention_score(module, x_grad, mask, pos=pos)
            loss = -self.loss_scale * arch_attention
            loss.backward()
            grad = x_grad.grad
            if grad is None or not torch.isfinite(grad).all():
                return args, kwargs
            with torch.no_grad():
                x_new = x_grad - self.step_size * grad
                if not torch.isfinite(x_new).all():
                    return args, kwargs

        attn_after = self._measure_arch_attention(module, x_new.detach(), mask, pos=pos)
        delta = attn_after - attn_before
        if layer_idx is not None:
            self.attention_deltas.append((layer_idx, attn_before, attn_after, delta))
            print(
                f"Layer {layer_idx} {self.attention_label} attention "
                f"{attn_before:.6f} -> {attn_after:.6f} (Δ{delta:.6f})"
            )
        return (x_new.detach(),), kwargs

    def _make_hook(self, layer_idx):
        def hook(module, args, kwargs):
            return self.intervene_forward_pre_hook(module, args, kwargs, layer_idx=layer_idx)

        return hook

    def register_intervention(self):
        self.attention_deltas = []
        if self.global_layers_only:
            layers = get_da3_global_layers(
                self.model,
                self.first_intervention_layer,
                self.last_intervention_layer,
            )
        else:
            n_layers = len(self.backbone.blocks)
            start = 0 if self.first_intervention_layer is None else max(0, int(self.first_intervention_layer))
            end = n_layers - 1 if self.last_intervention_layer is None else min(int(self.last_intervention_layer), n_layers - 1)
            layers = list(range(start, end + 1))
        if not layers:
            raise RuntimeError("No DA3 intervention layers selected. Check model alt_start/layer range.")
        for layer in layers:
            block = self.backbone.blocks[layer]
            handle = block.attn.register_forward_pre_hook(self._make_hook(layer), with_kwargs=True)
            self.intervention_hooks.append(handle)
            print(f"Registered DA3 intervention hook at DINO block {layer}")

    def remove_intervention(self):
        for hook in self.intervention_hooks:
            hook.remove()
        self.intervention_hooks = []
