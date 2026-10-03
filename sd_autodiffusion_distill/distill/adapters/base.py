# sd_autodiffusion_distill/distill/adapters/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

Condition = Union[torch.Tensor, Dict[str, Any], Tuple[Any, ...]]

class BaseAdapter(ABC):
    """
    Adapter wraps a denoiser (UNet / DiT / MMDiT ...) and provides:
      - set_policy(policy): apply architecture/compute policy (layer drop, gating, etc.)
      - prepare_condition(cond, uncond, device): produce model-ready condition batch for CFG
      - forward(x, t, condition): run denoiser forward
      - to_model_time(u_schedule): convert normalized time u to model-specific time representation
      - from_model_time(model_time): convert model-specific time back to normalized time u
    """

    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def set_policy(self, policy: Any) -> None:
        ...

    @abstractmethod
    def prepare_condition(
        self,
        cond: Condition,
        uncond: Condition,
        device: torch.device
    ) -> Condition:
        """
        Convert (cond, uncond) into the final condition fed into model.forward
        for classifier-free guidance batching (usually 2B).
        For SD1.x UNet: return Tensor [2B, 77, C]
        For SD3/DiT: could be dict/tuple, etc.
        """
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Condition) -> torch.Tensor:
        """
        Must accept condition returned by prepare_condition().
        t should be in model-specific time representation (e.g., discrete index for SDv1.4,
        continuous noise parameter for SDv3).
        """
        ...

    @abstractmethod
    def to_model_time(self, u_schedule: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized time u (in [0,1]) to model-specific time representation.
        
        Args:
            u_schedule: Normalized time tensor in [0,1], shape (K,) or (B,)
        
        Returns:
            Model-specific time tensor (e.g., discrete indices for SDv1.4,
            continuous sigma/t for SDv3)
        
        Examples:
            - SDv1.4: discrete indices in [0, 999]
            - SDv3: continuous noise parameters (sigma, logSNR, etc.)
        """
        ...

    @abstractmethod
    def from_model_time(self, model_time: torch.Tensor) -> torch.Tensor:
        """
        Convert model-specific time representation back to normalized time u in [0,1].
        
        Args:
            model_time: Model-specific time tensor
        
        Returns:
            Normalized time tensor in [0,1]
        """
        ...

