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


class Pi3AttentionIntervention:
    """
    Attention intervention for Pi3 decoder attention layers.

    Notes:
    - Pi3 alternates frame/global decoder streams. A global token mask (across all images)
      only matches odd/global layers, so hooks are only registered there by default.
    - Hook runs as pre-hook on attention modules and directly edits hidden states.
    """

    def __init__(
        self,
        model,
        arch_token_mask,
        first_intervention_layer: int = 1,
        last_intervention_layer: int = 35,
        step_size: float = 1000.0,
        loss_scale: float | None = None,
        attention_label: str = "arch",
        global_layers_only: bool = True,
    ):
        self.model = model
        self.arch_token_mask = arch_token_mask
        self.first_intervention_layer = int(first_intervention_layer)
        self.last_intervention_layer = int(last_intervention_layer)
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

    def _measure_arch_attention(self, module, x, mask, xpos=None):
        with torch.no_grad():
            qkv = module.qkv(x).reshape(x.shape[0], x.shape[1], 3, module.num_heads, x.shape[2] // module.num_heads).transpose(1, 3)
            q, k, v = [qkv[:, :, i] for i in range(3)]
            q, k = module.q_norm(q).to(v.dtype), module.k_norm(k).to(v.dtype)
            if getattr(module, "rope", None) is not None and xpos is not None:
                q = module.rope(q, xpos)
                k = module.rope(k, xpos)
            q = q * module.scale
            q_idx = torch.nonzero(mask > 0.5, as_tuple=False).view(-1)
            if q_idx.numel() == 0:
                return 0.0
            if self.full_query_objective:
                q_sel = q
                n_query = q.shape[2]
            else:
                q_sel = q.index_select(2, q_idx)
                n_query = q_idx.numel()
            v_custom = mask.view(1, 1, -1, 1).to(v.dtype).expand(q.shape[0], module.num_heads, q.shape[2], q.shape[-1]).contiguous()
            attn_out = F.scaled_dot_product_attention(q_sel.contiguous().to(v.dtype), k.contiguous().to(v.dtype), v_custom)
            num_arch = mask.sum().clamp(min=1.0)
            val = (attn_out[:, :, :, 0].sum() / (n_query * num_arch + 1e-8)).item()
        return val

    def intervene_forward_pre_hook(self, module, args, kwargs, layer_idx=None):
        x = args[0]
        if x.ndim != 3:
            return args, kwargs

        dev = x.device
        full_mask = self.arch_token_mask.to(dev).float().view(-1)
        if x.shape[1] != full_mask.shape[0]:
            # Token layout mismatch (usually frame layers) -> skip safely.
            return args, kwargs
        if full_mask.sum() < 1:
            return args, kwargs

        xpos = kwargs.get("xpos", None)

        def _forward_to_loss(x_i):
            qkv = module.qkv(x_i).reshape(x_i.shape[0], x_i.shape[1], 3, module.num_heads, x_i.shape[2] // module.num_heads).transpose(1, 3)
            q, k, v = [qkv[:, :, i] for i in range(3)]
            q, k = module.q_norm(q).to(v.dtype), module.k_norm(k).to(v.dtype)
            if getattr(module, "rope", None) is not None and xpos is not None:
                q = module.rope(q, xpos)
                k = module.rope(k, xpos)
            q = q * module.scale

            mask_float = full_mask
            q_idx = torch.nonzero(mask_float > 0.5, as_tuple=False).view(-1)
            if q_idx.numel() == 0:
                return torch.zeros((), device=q.device, dtype=q.dtype, requires_grad=True)
            if self.full_query_objective:
                q_sel = q
                n_query = q.shape[2]
            else:
                q_sel = q.index_select(2, q_idx)
                n_query = q_idx.numel()
            v_custom = mask_float.view(1, 1, -1, 1).to(v.dtype).expand(q.shape[0], module.num_heads, q.shape[2], q.shape[-1]).contiguous()
            attn_out = F.scaled_dot_product_attention(q_sel.contiguous().to(v.dtype), k.contiguous().to(v.dtype), v_custom)
            num_arch = mask_float.sum().clamp(min=1.0)
            arch_attention = attn_out[:, :, :, 0].sum() / (n_query * num_arch + 1e-8)
            return -self.loss_scale * arch_attention

        attn_before = self._measure_arch_attention(module, x.detach(), full_mask, xpos=xpos)
        with torch.enable_grad():
            x_grad = x.detach().requires_grad_(True)
            loss = _forward_to_loss(x_grad)
            loss.backward()
            g = x_grad.grad
            if g is None:
                return args, kwargs
            with torch.no_grad():
                x_new = x_grad - self.step_size * g
        attn_after = self._measure_arch_attention(module, x_new.detach(), full_mask, xpos=xpos)
        delta = attn_after - attn_before
        if layer_idx is not None:
            self.attention_deltas.append((layer_idx, attn_before, attn_after, delta))
            print(f"Layer {layer_idx} {self.attention_label} attention {attn_before:.6f} -> {attn_after:.6f} (Δ{delta:.6f})")
        return (x_new.detach(),), kwargs

    def _make_hook(self, layer_idx):
        def hook(module, args, kwargs):
            return self.intervene_forward_pre_hook(module, args, kwargs, layer_idx=layer_idx)

        return hook

    def register_intervention(self):
        self.attention_deltas = []
        n_layers = len(self.model.decoder)
        start = max(0, self.first_intervention_layer)
        end = min(self.last_intervention_layer, n_layers - 1)
        for layer in range(start, end + 1):
            if self.global_layers_only and layer % 2 == 0:
                continue
            block = self.model.decoder[layer]
            h = block.attn.register_forward_pre_hook(self._make_hook(layer), with_kwargs=True)
            self.intervention_hooks.append(h)
            print(f"Registered Pi3 intervention hook at decoder layer {layer}")

    def remove_intervention(self):
        for hook in self.intervention_hooks:
            hook.remove()
        self.intervention_hooks = []
