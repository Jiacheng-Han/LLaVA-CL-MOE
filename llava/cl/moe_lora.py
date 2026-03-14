# -*- encoding: utf-8 -*-
import math
import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.lora import LoraConfig, LoraLayer, LoraModel


@dataclass
class MOELoraConfig(LoraConfig):
    """
    Configuration class for MOE LoRA.
    """
    task_embedding_dim: int = field(default=64)   # 当前版本未使用，先保留
    expert_num: int = field(default=4)            # 当前版本不做硬限制，先保留
    router_temperature: float = field(default=1.0)

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = "MOE_LORA_CL"


class MOELoraModel(LoraModel):
    """
    模型构建与替换类
    """
    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

    def _create_new_module(self, lora_config, adapter_name, target, **kwargs):
        if isinstance(target, nn.Linear):
            return MOELoraLinear(
                base_layer=target,
                adapter_name=adapter_name,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                router_temperature=getattr(lora_config, "router_temperature", 1.0),
            )
        else:
            raise ValueError(
                f"Target module {target} is not supported. Only nn.Linear is supported."
            )


class MOELoraLayer(LoraLayer):
    """
    管理多个 LoRA expert + 线性 router
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

        if lora_dropout > 0.0:
            self.lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout_layer = nn.Identity()

        # 多专家 LoRA
        self.lora_A_experts = nn.ModuleList([])
        self.lora_B_experts = nn.ModuleList([])

        # 关键改动：
        # 1) router 保存“历史所有已固定专家”的路由，形状 Linear(in_features, num_fixed_experts)
        # 2) new_router 只给当前新加专家临时训练，形状 Linear(in_features, 1)
        self.router = None
        self.new_router = None

        self.current_expert_num = 0

    @property
    def num_experts(self):
        return self.current_expert_num

    def _make_single_router(self, device=None, dtype=None):
        layer = nn.Linear(self.in_features, 1, bias=True)
        nn.init.normal_(layer.weight, std=0.02)
        nn.init.zeros_(layer.bias)
        if device is not None:
            layer = layer.to(device=device, dtype=dtype)
        return layer

    def _make_multi_router(self, out_dim: int, device=None, dtype=None):
        layer = nn.Linear(self.in_features, out_dim, bias=True)
        nn.init.normal_(layer.weight, std=0.02)
        nn.init.zeros_(layer.bias)
        if device is not None:
            layer = layer.to(device=device, dtype=dtype)
        return layer

    def add_new_task_expert(self):
        """
        新任务到来时：
        1. 冻结旧 expert
        2. 新增一个 expert
        3. 如果这是第一个 expert，则创建 router = Linear(in_features, 1)
        4. 如果不是第一个 expert，则创建 new_router = Linear(in_features, 1)
           等任务训练结束后再 merge 到 router 里
        """
        # 冻结旧专家
        for p in self.lora_A_experts.parameters():
            p.requires_grad = False
        for p in self.lora_B_experts.parameters():
            p.requires_grad = False

        # 冻结旧 router
        if self.router is not None:
            for p in self.router.parameters():
                p.requires_grad = False

        # 清理旧 new_router（正常流程下不会残留；防御式处理）
        if self.new_router is not None:
            for p in self.new_router.parameters():
                p.requires_grad = False

        # 新增当前任务 expert
        new_A = nn.Linear(self.in_features, self.r_val, bias=False)
        new_B = nn.Linear(self.r_val, self.out_features, bias=False)

        nn.init.kaiming_uniform_(new_A.weight, a=math.sqrt(5))
        nn.init.zeros_(new_B.weight)

        # 对齐 device / dtype
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype
        new_A = new_A.to(device=device, dtype=dtype)
        new_B = new_B.to(device=device, dtype=dtype)

        self.lora_A_experts.append(new_A)
        self.lora_B_experts.append(new_B)

        # router 逻辑
        if self.current_expert_num == 0:
            # 第一个 expert：直接建总 router（1维）
            self.router = self._make_multi_router(1, device=device, dtype=dtype)
            self.new_router = None
        else:
            # 后续 expert：只为当前 expert 建一个临时 new_router
            self.new_router = self._make_single_router(device=device, dtype=dtype)

        self.current_expert_num += 1

    def fix_router(self):
        """
        任务训练结束后，把 new_router 合并到 router 中：
        old: Linear(in_features, N-1)
        new: Linear(in_features, 1)
        => merged: Linear(in_features, N)
        """
        if self.new_router is None:
            return

        if self.router is None:
            raise RuntimeError("router is None while new_router exists, which is invalid.")

        device = self.router.weight.device
        dtype = self.router.weight.dtype

        old_out_dim = self.router.out_features
        merged_router = nn.Linear(self.in_features, old_out_dim + 1, bias=True).to(
            device=device, dtype=dtype
        )

        with torch.no_grad():
            merged_router.weight[:old_out_dim].copy_(self.router.weight.data)
            merged_router.bias[:old_out_dim].copy_(self.router.bias.data)

            merged_router.weight[old_out_dim:old_out_dim + 1].copy_(self.new_router.weight.data)
            merged_router.bias[old_out_dim:old_out_dim + 1].copy_(self.new_router.bias.data)

        self.router = merged_router
        self.new_router = None

    def freeze_experts(self):
        for p in self.lora_A_experts.parameters():
            p.requires_grad = False
            p.grad = None
        for p in self.lora_B_experts.parameters():
            p.requires_grad = False
            p.grad = None

    def freeze_router(self):
        if self.router is not None:
            for p in self.router.parameters():
                p.requires_grad = False
                p.grad = None
        if self.new_router is not None:
            for p in self.new_router.parameters():
                p.requires_grad = False
                p.grad = None

    def end_of_task_training(self):
        """
        1. 如果有 new_router，先 merge
        2. 冻结所有 expert
        3. 冻结 router
        """
        if self.new_router is not None:
            self.fix_router()

        self.freeze_experts()
        self.freeze_router()


class MOELoraLinear(nn.Module, MOELoraLayer):
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

        # 初始加载时创建第一个 expert + router
        self.add_new_task_expert()

    def update_layer(self, *args, **kwargs):
        # 拦截 PEFT 默认 update_layer 行为
        pass

    def _compute_router_logits(self, x_router: torch.Tensor):
        """
        - 历史专家：直接走已合并好的 self.router
        - 当前新专家：如果存在 self.new_router，则再拼一个 1 维 logits
        """
        if self.router is None:
            raise RuntimeError("router has not been initialized.")

        logits = self.router(x_router)  # [..., num_fixed_experts]

        if self.new_router is not None:
            new_logits = self.new_router(x_router)  # [..., 1]
            logits = torch.cat([logits, new_logits], dim=-1)

        return logits

    def forward(self, x: torch.Tensor, *args, **kwargs):
        previous_dtype = x.dtype

        # base layer 输出
        result = self.base_layer(x, *args, **kwargs)

        if self.current_expert_num == 0 or self.r_val == 0:
            return result.to(previous_dtype)

        expert_dtype = self.lora_A_experts[0].weight.dtype
        x_lora = x.to(expert_dtype)

        # dropout
        x_dropped = self.lora_dropout_layer(x_lora)

        # # 只有一个 expert 时，不走 router softmax，直接使用该 expert
        # if self.current_expert_num == 1 and self.new_router is None:
        #     self.saved_router_logits = None
        #     lora_out = self._compute_single_expert(x_dropped, 0)
        #     result = result + (lora_out * self.scaling_val).to(previous_dtype)
        #     return result

        # router 输入：
        # 不对每个 token 单独过各自 router，
        # 而是先做 token 维聚合，再一次性得到 expert logits（sample-level）
        
        # 如果 x 是 [B, T, C]，则取 mean(dim=1) => [B, C]
        # 如果 x 是 [B, C]，则直接用
        if x_lora.dim() == 3:
            x_router = x_lora.mean(dim=1)
        elif x_lora.dim() == 2:
            x_router = x_lora
        else:
            raise ValueError(f"Unsupported input shape for router: {x_lora.shape}")

        router_logits = self._compute_router_logits(x_router)
        self.saved_router_logits = router_logits

        router_weights = F.softmax(
            router_logits / self.router_temperature,
            dim=-1
        ).to(expert_dtype)  # [B, E]

        # 累加各 expert 输出
        lora_out = torch.zeros_like(result, dtype=expert_dtype)

        for i in range(self.current_expert_num):
            if x_lora.dim() == 3:
                weight_i = router_weights[:, i].view(-1, 1, 1)
            else:
                weight_i = router_weights[:, i].view(-1, 1)

            # 权重很小时跳过
            if torch.all(weight_i < 1e-4):
                continue

            expert_out = self.lora_B_experts[i](
                self.lora_A_experts[i](x_dropped)
            )
            lora_out = lora_out + expert_out * weight_i

        result = result + (lora_out * self.scaling_val).to(previous_dtype)
        return result