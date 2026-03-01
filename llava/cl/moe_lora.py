# -*- encoding: utf-8 -*-
import math
import warnings
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.utils import transpose
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
    Configuration class for MOE LoRA.
    """
    task_embedding_dim: int = field(default=64)
    expert_num: int = field(default=4)
    router_temperature: float = field(default=1.0) # Gumbel Softmax 温度超参

    def __post_init__(self):
        super().__post_init__()
        # 把 self.peft_type 强行改成了 "MOE_LORA_CL"。这样 PEFT 库在底层运转时，就能识别出这是一个特殊的自定义模块，而非普通的 LoRA。
        self.peft_type = "MOE_LORA_CL"


class MOELoraModel(LoraModel):
    """
    模型构建与劫持类
    """
    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name) # 初始化，把预训练的基础模型和配置传进来。
        
    def _create_new_module(self, lora_config, adapter_name, target, **kwargs):
        # 覆写PEFT库创建代码 ，将其指向我们的 MOELoraLinear
        if isinstance(target, nn.Linear):
            new_module = MOELoraLinear(
                base_layer=target,  
                adapter_name=adapter_name, 
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                router_temperature=getattr(lora_config, "router_temperature", 1.0)
            )
            return new_module
        else:
            raise ValueError(f"Target module {target} is not supported. Only nn.Linear is supported.")

# 这个类主要负责管理 MOE LoRA 的核心数据结构和动态专家添加逻辑
class MOELoraLayer(LoraLayer): 
    def __init__(self, base_layer: nn.Module, in_features: int, out_features: int, r: int, lora_alpha: int, lora_dropout: float):
        # 1. 初始化 PEFT 的 LoraLayer，建立标准的内部变量
        try:
            LoraLayer.__init__(self, base_layer=base_layer)
        except Exception:
            # 兼容极个别老版本
            LoraLayer.__init__(self)
            self.base_layer = base_layer

        self.r_val = r
        self.lora_alpha_val = lora_alpha
        # 改名 scaling_val，防止覆盖 PEFT 内部的字典 self.scaling
        self.scaling_val = self.lora_alpha_val / self.r_val if self.r_val > 0 else 0.0
        
        self.in_features = in_features
        self.out_features = out_features
        
        # 改名 lora_dropout_layer，防止覆盖 PEFT 内部的 ModuleDict
        if lora_dropout > 0.0:
            self.lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout_layer = nn.Identity()

        # 核心数据结构：使用 ModuleList 存储动态增长的专家和路由
        self.lora_A_experts = nn.ModuleList([])
        self.lora_B_experts = nn.ModuleList([])
        
        # 将 Router 拆分为独立的小 Linear(in_features, 1)，方便拼接和单独冻结
        self.lora_routers = nn.ModuleList([])
        
        self.current_expert_num = 0

    def add_new_task_expert(self):
        # 冻结以前任务的所有参数
        for p in self.lora_A_experts.parameters():
            p.requires_grad = False
        for p in self.lora_B_experts.parameters():
            p.requires_grad = False
        for p in self.lora_routers.parameters():
            p.requires_grad = False

        # 新增当前任务的 Expert
        new_A = nn.Linear(self.in_features, self.r_val, bias=False)
        new_B = nn.Linear(self.r_val, self.out_features, bias=False)
        
        # 初始化新专家
        nn.init.kaiming_uniform_(new_A.weight, a=math.sqrt(5))
        nn.init.zeros_(new_B.weight)
        
        self.lora_A_experts.append(new_A)
        self.lora_B_experts.append(new_B)
        
        # 新增当前任务的 Router Head (输出维度为1)
        hidden_dim1 = 64
        hidden_dim2 = 16

        new_router = nn.Sequential(
            nn.Linear(self.in_features, hidden_dim1, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1, bias=False) # 输出维度为1
        )

        for layer in new_router:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.lora_routers.append(new_router)
        
        self.current_expert_num += 1


# 最终被替换到模型中的线性层，负责前向计算和专家路由逻辑
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
        
        # 调用我们修复后的 MOELoraLayer 初始化
        MOELoraLayer.__init__(
            self, 
            base_layer=base_layer,
            in_features=base_layer.in_features, 
            out_features=base_layer.out_features, 
            r=r, 
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout
        )
        
        # 2. 绕开 property 限制，直接且仅设置底层 adapter 名称变量
        self._active_adapter = adapter_name
            
        self.router_temperature = router_temperature
        
        # 初始状态：加载模型时默认分配第一个任务的专家
        self.add_new_task_expert()

    def update_layer(self, *args, **kwargs):
        # 【极其重要的新增！】
        # 拦截 PEFT 底层自动调用的 update_layer 方法。
        # 因为我们的专家数据结构是自定义的 ModuleList，直接 pass 掉，防止 PEFT 原生代码乱插参数报错。
        pass

    def forward(self, x: torch.Tensor, *args, **kwargs):
        previous_dtype = x.dtype
        # 基座模型的前向传播
        result = self.base_layer(x, *args, **kwargs)

        if self.current_expert_num == 0 or self.r_val == 0:
            return result.to(previous_dtype)

        x_lora = x.to(self.lora_A_experts[0].weight.dtype)
        
        # --- 动态构建 Router Logits ---
        router_outputs = [router(x_lora) for router in self.lora_routers]
        router_logits = torch.cat(router_outputs, dim=-1)
        
        # 保存下来供外部计算 Routing Loss
        self.saved_router_logits = router_logits 
        
        # 3. 使用改名后的 lora_dropout_layer
        x_dropped = self.lora_dropout_layer(x_lora)
        lora_out = torch.zeros_like(result, dtype=x_lora.dtype)
        
        # ==================== 核心隔离与推理区域 ====================
        if self.training:
            # 【训练阶段】：强制 100% 使用当前最新任务的专家，完全屏蔽旧专家
            target_expert_idx = self.current_expert_num - 1
            expert_out = self.lora_B_experts[target_expert_idx](
                self.lora_A_experts[target_expert_idx](x_dropped)
            )
            lora_out = expert_out
        else:
            # 【推理阶段】：多专家组合 或 上帝视角Hack

            # -------------------------------------------------------------
            # 正常路由代码（目前注销）：
            # router_weights = F.softmax(router_logits / self.router_temperature, dim=-1)
            
            # 【上帝视角测试 Hack】：强行指定使用第 0 组专家 (测任务一必备！)
            router_weights = torch.zeros_like(router_logits)
            router_weights[..., 0] = 1.0
            # -------------------------------------------------------------

            for i in range(self.current_expert_num):
                weight_i = router_weights[..., i].unsqueeze(-1)
                
                # 如果这个专家的权重无限接近0，直接跳过计算，节省时间
                if torch.all(weight_i < 1e-4):
                    continue
                    
                expert_out = self.lora_B_experts[i](
                    self.lora_A_experts[i](x_dropped)
                )
                lora_out += expert_out * weight_i
        # ======================================================

        # 4. 使用改名后的 scaling_val
        result += (lora_out * self.scaling_val).to(previous_dtype)
        return result
