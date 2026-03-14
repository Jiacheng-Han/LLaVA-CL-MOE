# -*- coding: utf-8 -*-
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.lora import LoraConfig, LoraLayer, LoraModel

try:
    from peft.tuners.lora import Linear8bitLt, Linear4bit
    import bitsandbytes as bnb
    is_bnb_available = True
except ImportError:
    is_bnb_available = False


@dataclass
class MOELoraConfig(LoraConfig):
    """
    Configuration class for MoE LoRA.
    """

    task_embedding_dim: int = field(default=64)
    expert_num: int = field(default=4)
    router_temperature: float = field(default=1.0)

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = "MOE_LORA_CL"


class MOELoraModel(LoraModel):
    """
    Wrapper that swaps target Linear layers to MOELoraLinear.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

    def _create_new_module(self, lora_config, adapter_name, target, **kwargs):
        if isinstance(target, nn.Linear):  # 针对每个线性层进行处理替换
            return MOELoraLinear(
                base_layer=target,
                adapter_name=adapter_name,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                router_temperature=getattr(lora_config, "router_temperature", 1.0),
            )
        raise ValueError(f"Target module {target} is not supported. Only nn.Linear is supported.")

def normalize_moe_lora_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """
    Normalize legacy MoE LoRA router keys to current format.

    Current format:
      - *.lora_router.weight: [K, d]
      - *.lora_router.bias:   [K]

    Legacy compatible format:
      - *.lora_routers.{k}.weight: [1, d] / [d]
      - *.lora_routers.{k}.bias:   [1] / scalar

    Unsupported legacy router keys (e.g., multi-layer head like
    lora_routers.{k}.0.weight) are dropped intentionally.
    """

    if state_dict is None:
        return state_dict, {"legacy_router_keys_dropped": 0, "legacy_router_groups_converted": 0}

    normalized: Dict[str, torch.Tensor] = {}
    processed_legacy_keys = set()
    linear_legacy_keys = set()
    legacy_router_keys_seen = 0
    converted_groups = 0

    # prefix -> {"weight": {idx: tensor}, "bias": {idx: tensor}}
    legacy_linear_router_groups: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {}

    for key, value in state_dict.items():
        if ".lora_routers." not in key:
            normalized[key] = value
            continue
        legacy_router_keys_seen += 1

        prefix, suffix = key.split(".lora_routers.", 1)
        suffix_parts = suffix.split(".")
        if len(suffix_parts) == 2 and suffix_parts[0].isdigit() and suffix_parts[1] in ("weight", "bias"):
            expert_idx = int(suffix_parts[0])
            wb = suffix_parts[1]
            if prefix not in legacy_linear_router_groups:
                legacy_linear_router_groups[prefix] = {"weight": {}, "bias": {}}
            legacy_linear_router_groups[prefix][wb][expert_idx] = value
            processed_legacy_keys.add(key)
            linear_legacy_keys.add(key)
            continue

        # Unsupported legacy key style; keep it dropped on purpose.
        processed_legacy_keys.add(key)

    for prefix, group in legacy_linear_router_groups.items():
        if not group["weight"]:
            continue

        sorted_indices = sorted(group["weight"].keys())
        weight_rows = []
        bias_rows = []
        can_use_bias = len(group["bias"]) == len(sorted_indices)

        for idx in sorted_indices:
            w = group["weight"][idx]
            if w.dim() == 2 and w.shape[0] == 1:
                w = w.squeeze(0)
            elif w.dim() == 1:
                w = w
            else:
                # Invalid shape for linear-head router conversion, skip this group.
                weight_rows = []
                bias_rows = []
                break

            weight_rows.append(w)

            if can_use_bias:
                b = group["bias"][idx]
                if b.dim() == 1 and b.numel() == 1:
                    b = b.reshape(())
                elif b.dim() == 0:
                    b = b
                else:
                    can_use_bias = False
                bias_rows.append(b)

        if not weight_rows:
            continue

        normalized[f"{prefix}.lora_router.weight"] = torch.stack(weight_rows, dim=0)
        if can_use_bias and len(bias_rows) == len(weight_rows):
            normalized[f"{prefix}.lora_router.bias"] = torch.stack([b.reshape(1) for b in bias_rows], dim=0).reshape(-1)
        converted_groups += 1

    dropped_legacy = legacy_router_keys_seen - len(linear_legacy_keys)
    return normalized, {
        "legacy_router_keys_seen": legacy_router_keys_seen,
        "legacy_router_keys_dropped": dropped_legacy,
        "legacy_router_groups_converted": converted_groups,
    }


class MOELoraLayer(LoraLayer):
    """
    Core data structure for experts + matrix router.
    Router form: softmax(x @ Wmix), where Wmix in R^(d x K).
    """

    def __init__(
        self,
        base_layer: nn.Module,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        try:
            LoraLayer.__init__(self, base_layer=base_layer)
        except Exception:
            LoraLayer.__init__(self)
            self.base_layer = base_layer

        self.r_val = r
        self.lora_alpha_val = lora_alpha
        self.scaling_val = self.lora_alpha_val / self.r_val if self.r_val > 0 else 0.0

        self.in_features = in_features
        self.out_features = out_features
        self.lora_dropout_layer = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        self.lora_A_experts = nn.ModuleList([])
        self.lora_B_experts = nn.ModuleList([])

        # 矩阵路由: Linear(d, K), expanded to K+1 on new task.
        self.lora_router: Optional[nn.Linear] = None

        # Keep grad-mask hooks to freeze old router columns while training the new one.
        self._router_weight_hook = None
        self._router_bias_hook = None

        self.current_expert_num = 0
        self.saved_router_logits = None

    @property
    def lora_routers(self):
        """
        Backward-compatible alias for old external calls (e.g., builder device cast).
        """
        return self.lora_router

    def _remove_router_hooks(self):
        if self._router_weight_hook is not None:
            self._router_weight_hook.remove()
            self._router_weight_hook = None
        if self._router_bias_hook is not None:
            self._router_bias_hook.remove()
            self._router_bias_hook = None

    def _new_router(self, out_dim: int, device: torch.device, dtype: torch.dtype) -> nn.Linear:
        router = nn.Linear(self.in_features, out_dim, bias=True)
        nn.init.normal_(router.weight, std=0.02)
        nn.init.zeros_(router.bias)
        router = router.to(device=device, dtype=dtype)
        return router

    def _set_trainable_router_column(self, col_idx: int):
        # register hook，只让 col_idx 列梯度更新
        if self.lora_router is None:
            return

        self._remove_router_hooks()
        for param in self.lora_router.parameters():
            param.requires_grad = True
            param.grad = None

        def weight_hook(grad):
            mask = torch.zeros_like(grad)
            mask[col_idx, :] = 1.0
            return grad * mask

        def bias_hook(grad):
            mask = torch.zeros_like(grad)
            mask[col_idx] = 1.0
            return grad * mask

        self._router_weight_hook = self.lora_router.weight.register_hook(weight_hook)
        self._router_bias_hook = self.lora_router.bias.register_hook(bias_hook)

    def _expand_router(self):
        if self.lora_router is None:
            base_weight = self.base_layer.weight
            self.lora_router = self._new_router(
                out_dim=1,
                device=base_weight.device,
                dtype=base_weight.dtype,
            )
            return

        old_router = self.lora_router
        old_k = old_router.out_features
        new_router = self._new_router(
            out_dim=old_k + 1,
            device=old_router.weight.device,
            dtype=old_router.weight.dtype,
        )
        with torch.no_grad():
            new_router.weight[:old_k].copy_(old_router.weight)
            new_router.bias[:old_k].copy_(old_router.bias)
        self.lora_router = new_router

    def add_new_task_expert(self):
        # Freeze all old experts.
        for p in self.lora_A_experts.parameters():
            p.requires_grad = False
            p.grad = None
        for p in self.lora_B_experts.parameters():
            p.requires_grad = False
            p.grad = None

        # Add one new expert for the incoming task.
        new_A = nn.Linear(self.in_features, self.r_val, bias=False)
        new_B = nn.Linear(self.r_val, self.out_features, bias=False)
        nn.init.kaiming_uniform_(new_A.weight, a=math.sqrt(5))
        nn.init.zeros_(new_B.weight)

        if hasattr(self.base_layer, "weight"):
            target_device = self.base_layer.weight.device
            target_dtype = self.base_layer.weight.dtype
            new_A = new_A.to(device=target_device, dtype=target_dtype)
            new_B = new_B.to(device=target_device, dtype=target_dtype)

        self.lora_A_experts.append(new_A)
        self.lora_B_experts.append(new_B)
        self.current_expert_num += 1

        # Expand matrix router from K to K+1, then only train the new column.
        self._expand_router()
        self._set_trainable_router_column(self.current_expert_num - 1)


class MOELoraLinear(nn.Module, MOELoraLayer):
    """
    Replaced linear layer that applies:
    base_layer(x) + scaling * sum_k softmax(router(x))_k * expert_k(x)
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        router_temperature: float = 1.0,
        **kwargs,
    ):
        nn.Module.__init__(self)
        MOELoraLayer.__init__(
            self,
            base_layer=base_layer,
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self._active_adapter = adapter_name
        self.router_temperature = router_temperature
        self.add_new_task_expert()

    def update_layer(self, *args, **kwargs):
        # Keep compatibility with PEFT internal calls.
        pass

    def forward(self, x: torch.Tensor, *args, **kwargs):
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)

        if self.current_expert_num == 0 or self.r_val == 0 or self.lora_router is None:
            return result.to(previous_dtype)

        x_lora = x.to(self.lora_A_experts[0].weight.dtype)
        x_dropped = self.lora_dropout_layer(x_lora)

        # Router: use sample-level routing by pooling token states first.
        # [B, T, D] -> [B, D], then obtain one shared expert mixture per sample.
        if x_lora.dim() == 3:
            router_input = x_lora.mean(dim=1)
        else:
            router_input = x_lora

        router_logits = self.lora_router(router_input)  # [B, K] or [*, K]
        self.saved_router_logits = router_logits
        router_weights = F.softmax(router_logits / self.router_temperature, dim=-1)

        # force_expert = 1 # 强制使用指定个专家（仅用于测试）
        # if not self.training:
        #     if 0 <= force_expert < self.current_expert_num:
        #         forced = torch.zeros_like(router_weights)
        #         forced[..., force_expert] = 1.0
        #         router_weights = forced

        # 选择top-1专家
        # if not self.training:
        #     # top-1 expert selection at inference
        #     top1_idx = router_weights.argmax(dim=-1)   # [B] or [...]
        #     one_hot = torch.zeros_like(router_weights)
        #     one_hot.scatter_(-1, top1_idx.unsqueeze(-1), 1.0)
        #     router_weights = one_hot

        # 查看权重
        # if not self.training:
        #     if not hasattr(self, "router_stat"):
        #         self.router_stat = torch.zeros(router_weights.shape[-1], device=router_weights.device)
        #         self.router_count = 0

        #     self.router_stat += router_weights.mean(dim=0).detach()
        #     self.router_count += 1

        # Experts: compute token-level expert outputs, but mix them with sample-level weights.
        expert_outs = []
        for i in range(self.current_expert_num):
            expert_out = self.lora_B_experts[i](self.lora_A_experts[i](x_dropped))
            expert_outs.append(expert_out)

        expert_outs = torch.stack(expert_outs, dim=-2)
        if expert_outs.dim() == 4:
            # [B, T, K, O] * [B, 1, K, 1]
            router_weights = router_weights.unsqueeze(1)
        lora_out = (expert_outs * router_weights.unsqueeze(-1)).sum(dim=-2)

        result = result + (lora_out * self.scaling_val).to(previous_dtype)

        # if not self.training and self.router_count % 50 == 0: # 每50个batch打印一次平均权重
        #     avg = (self.router_stat / self.router_count).detach().cpu()
        #     print(f"[Router Debug] avg weights: {avg}")

        return result