import sys
import os
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn

# --- Path Setup ---
current_file = os.path.abspath(__file__)
distill_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
project_root = os.path.dirname(distill_dir)
stable_diffusion_path = os.path.join(project_root, "examples", "Stable Diffusion")
if stable_diffusion_path not in sys.path:
    sys.path.insert(0, stable_diffusion_path)

from ldm.modules.diffusionmodules.openaimodel import UNetModel, ResBlock, AttentionBlock

try:
    from ldm.modules.diffusionmodules.openaimodel import TimestepBlock
except Exception:
    raise ImportError(
        "Cannot import TimestepBlock from ldm.modules.diffusionmodules.openaimodel. "
        "Your openaimodel.py must expose TimestepBlock for ResBlock wrapping."
    )

# Optional: SpatialTransformer (cross-attn lives here in SDv1.x)
try:
    from ldm.modules.attention import SpatialTransformer  # type: ignore
    SPATIAL_AVAILABLE = True
except Exception:
    SpatialTransformer = None  # type: ignore
    SPATIAL_AVAILABLE = False

from .base import BaseAdapter
from ..nas.search_space import SegmentPolicy


class ControllableResBlock(TimestepBlock):
    """
    ResBlock can change channels. We do NOT skip it.
    But we CAN gate its EFFECT in a shape-safe way:
      x_skip = original.skip_connection(x) (matches channels)
      y      = original(x, emb)
      out    = x_skip + gate_scale * (y - x_skip)
    """
    def __init__(self, original: ResBlock, block_id: int, stage_id: int):
        super().__init__()
        self.original = original
        self.block_id = block_id
        self.stage_id = stage_id
        self.gate_scale: float = 1.0  # stage_gate_scale[stage_id]

    def forward(self, x, emb):
        y = self.original(x, emb)

        # ResBlock in SD typically has skip_connection for channel alignment
        skip = getattr(self.original, "skip_connection", None)
        if callable(skip):
            x_skip = skip(x)
        else:
            # Fallback: if no skip_connection attr, use x (may be wrong if channels changed)
            # This should rarely happen in SD openaimodel ResBlock.
            x_skip = x

        g = float(self.gate_scale)
        if g >= 1.0:
            return y
        if g <= 0.0:
            return x_skip
        return x_skip + (y - x_skip) * g


class ControllableAttentionBlock(nn.Module):
    """
    AttentionBlock does not change channels -> safe to skip by identity.
    Also supports continuous gating:
      out = x + gate_scale*(attn(x) - x)
    """
    def __init__(self, original: AttentionBlock, block_id: int, stage_id: int):
        super().__init__()
        self.original = original
        self.block_id = block_id
        self.stage_id = stage_id
        self.enabled: bool = True
        self.gate_scale: float = 1.0  # stage_gate_scale[stage_id]

    def forward(self, x):
        if not self.enabled:
            return x
        y = self.original(x)
        g = float(self.gate_scale)
        if g >= 1.0:
            return y
        if g <= 0.0:
            return x
        return x + (y - x) * g


class ControllableSpatialTransformer(nn.Module):
    """
    SpatialTransformer is cross-attn in SDv1.x; safe to skip (shape preserved).
    Adds:
      - enabled (mask)
      - gate_scale: out = x + gate*(y-x)
      - cond_gain: scales context for THIS stage: context' = cond_gain * context
    """
    def __init__(self, original: "SpatialTransformer", block_id: int, stage_id: int):
        super().__init__()
        self.original = original
        self.block_id = block_id
        self.stage_id = stage_id
        self.enabled: bool = True
        self.gate_scale: float = 1.0     # stage_gate_scale[stage_id]
        self.cond_gain: float = 1.0      # stage_cond_gain[stage_id]

    def forward(self, x, context=None):
        if not self.enabled:
            return x

        ctx = context
        cg = float(self.cond_gain)
        if ctx is not None and cg != 1.0:
            ctx = ctx * cg

        y = self.original(x, context=ctx)
        g = float(self.gate_scale)
        if g >= 1.0:
            return y
        if g <= 0.0:
            return x
        return x + (y - x) * g


class UNetAdapter(BaseAdapter):
    """
    Adapter for LDM/Stable Diffusion UNetModel.
    We wrap:
      - ResBlock            (NOT skippable, but gated via skip_connection blend)
      - AttentionBlock      (skippable + gated)
      - SpatialTransformer  (skippable + gated + stage_cond_gain on context) [if available]
    """

    def __init__(self, model: UNetModel):
        self.controllable_blocks: List[nn.Module] = []
        self.num_stages: int = 0
        super().__init__(model)

        if len(self.controllable_blocks) == 0:
            self.parse_model()

        # Indices that are safe to skip (mask applies here)
        self._skippable_indices = [
            i for i, b in enumerate(self.controllable_blocks)
            if isinstance(b, (ControllableAttentionBlock, ControllableSpatialTransformer))
        ]

        # Helpful: count cross-attn wrappers
        n_attn = sum(isinstance(b, ControllableAttentionBlock) for b in self.controllable_blocks)
        n_xattn = sum(isinstance(b, ControllableSpatialTransformer) for b in self.controllable_blocks)

        print(f"[UNetAdapter] Wrapped blocks: total={self.total_layers}, attn={n_attn}, spatial_xattn={n_xattn}, stages={self.total_stages}")
        print(f"[UNetAdapter] Skippable indices: {len(self._skippable_indices)}/{self.total_layers}")

    @property
    def total_layers(self) -> int:
        return len(self.controllable_blocks)

    @property
    def total_stages(self) -> int:
        return int(self.num_stages)

    def _input_stage(self, i: int, n: int) -> int:
        if n <= 1:
            return 0
        if i < n / 3:
            return 0
        if i < 2 * n / 3:
            return 1
        return 2

    def _output_stage(self, i: int, n: int) -> int:
        if n <= 1:
            return 4
        if i < n / 3:
            return 4
        if i < 2 * n / 3:
            return 5
        return 6

    def parse_model(self) -> Dict[str, Any]:
        self.controllable_blocks = []
        block_id_counter = 0

        def wrap_in_container(container: nn.Module, stage_id: int):
            nonlocal block_id_counter

            # SD uses TimestepEmbedSequential (indexable like nn.Sequential)
            if hasattr(container, "__len__") and hasattr(container, "__getitem__"):
                for j in range(len(container)):
                    m = container[j]
                    if isinstance(m, ResBlock):
                        w = ControllableResBlock(m, block_id_counter, stage_id)
                        container[j] = w
                        self.controllable_blocks.append(w)
                        block_id_counter += 1
                    elif isinstance(m, AttentionBlock):
                        w = ControllableAttentionBlock(m, block_id_counter, stage_id)
                        container[j] = w
                        self.controllable_blocks.append(w)
                        block_id_counter += 1
                    elif SPATIAL_AVAILABLE and SpatialTransformer is not None and isinstance(m, SpatialTransformer):
                        w = ControllableSpatialTransformer(m, block_id_counter, stage_id)
                        container[j] = w
                        self.controllable_blocks.append(w)
                        block_id_counter += 1
                return

            # Fallback replace by named children
            for name, m in list(container.named_children()):
                if isinstance(m, ResBlock):
                    w = ControllableResBlock(m, block_id_counter, stage_id)
                    setattr(container, name, w)
                    self.controllable_blocks.append(w)
                    block_id_counter += 1
                elif isinstance(m, AttentionBlock):
                    w = ControllableAttentionBlock(m, block_id_counter, stage_id)
                    setattr(container, name, w)
                    self.controllable_blocks.append(w)
                    block_id_counter += 1
                elif SPATIAL_AVAILABLE and SpatialTransformer is not None and isinstance(m, SpatialTransformer):
                    w = ControllableSpatialTransformer(m, block_id_counter, stage_id)
                    setattr(container, name, w)
                    self.controllable_blocks.append(w)
                    block_id_counter += 1

        # input blocks
        n_in = len(self.model.input_blocks)
        for i in range(n_in):
            wrap_in_container(self.model.input_blocks[i], stage_id=self._input_stage(i, n_in))

        # middle block
        wrap_in_container(self.model.middle_block, stage_id=3)

        # output blocks
        n_out = len(self.model.output_blocks)
        for i in range(n_out):
            wrap_in_container(self.model.output_blocks[i], stage_id=self._output_stage(i, n_out))

        self.num_stages = (max(b.stage_id for b in self.controllable_blocks) + 1) if self.controllable_blocks else 0

        if self.total_layers == 0:
            raise RuntimeError("[UNetAdapter] Parsed 0 controllable blocks. ResBlock/AttentionBlock/SpatialTransformer not found.")

        return {"controllable_blocks": self.controllable_blocks, "num_stages": self.num_stages}

    def set_policy(self, policy: SegmentPolicy):
        """
        Apply:
          - layer_mask: controls enabled/disabled for SKIPPABLE blocks (AttentionBlock / SpatialTransformer)
            Supports two formats:
              * len(mask)==total_layers: mask indexed by all wrapped blocks (legacy)
              * len(mask)==len(skippable): mask only for skippable blocks (recommended)
          - stage_gate_scale: per-stage continuous gating for ALL wrapped blocks
          - stage_cond_gain: per-stage context scaling for SpatialTransformer only
        """
        if self.total_layers <= 0:
            raise RuntimeError("[UNetAdapter] total_layers <= 0; parse_model likely failed.")

        # -----------------------
        # 1) stage vectors (repair to ones if missing/mismatch)
        # -----------------------
        total_stages = max(1, self.total_stages)

        stage_gate = getattr(policy, "stage_gate_scale", None)
        if stage_gate is None or len(stage_gate) != total_stages:
            stage_gate = tuple([1.0] * total_stages)

        stage_cond = getattr(policy, "stage_cond_gain", None)
        if stage_cond is None or len(stage_cond) != total_stages:
            stage_cond = tuple([1.0] * total_stages)

        # -----------------------
        # 2) mask handling (two supported lengths)
        # -----------------------
        lm = getattr(policy, "layer_mask", None)
        if lm is None or len(lm) == 0:
            # default: all enabled
            # we will treat as skippable-only mask
            lm = tuple([True] * len(self._skippable_indices))

        if len(lm) == self.total_layers:
            # legacy full-length mask: map directly by block index
            def is_enabled(block_index: int) -> bool:
                return bool(lm[block_index])
        elif len(lm) == len(self._skippable_indices):
            # recommended mask: only for skippable blocks in the same order as _skippable_indices
            sk_map = {idx: bool(lm[k]) for k, idx in enumerate(self._skippable_indices)}
            def is_enabled(block_index: int) -> bool:
                return bool(sk_map.get(block_index, True))  # non-skippable defaults True
        else:
            raise ValueError(
                f"[UNetAdapter] Policy mask length ({len(lm)}) must be either "
                f"total_layers ({self.total_layers}) or skippable_layers ({len(self._skippable_indices)})."
            )

        # -----------------------
        # 3) apply to blocks
        # -----------------------
        for i, block in enumerate(self.controllable_blocks):
            sid = int(getattr(block, "stage_id", 0))
            sid = max(0, min(sid, total_stages - 1))

            g = float(stage_gate[sid])
            c = float(stage_cond[sid])

            # continuous gating to everyone (ResBlock also gets gate_scale, though not skippable)
            if hasattr(block, "gate_scale"):
                block.gate_scale = g

            # per-stage cond gain only meaningful for cross-attn blocks
            if hasattr(block, "cond_gain"):
                block.cond_gain = c

            # mask only affects skippable blocks (AttentionBlock / SpatialTransformer)
            if isinstance(block, (ControllableAttentionBlock, ControllableSpatialTransformer)):
                block.enabled = is_enabled(i)

    def prepare_condition(self, cond: torch.Tensor, uncond: torch.Tensor, device: torch.device) -> torch.Tensor:
        cond = cond.to(device)
        uncond = uncond.to(device)
        return torch.cat([uncond, cond], dim=0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x, t, context=context, **kwargs)

    # --- time helpers you already have (kept) ---
    def to_model_time(self, u_schedule: torch.Tensor, N: int = 1000) -> torch.Tensor:
        u_schedule = torch.clamp(u_schedule, 0.0, 1.0)
        idx = torch.round(u_schedule * (N - 1)).long()
        idx = torch.clamp(idx, 0, N - 1)
        return idx

    def from_model_time(self, model_time: torch.Tensor, N: int = 1000) -> torch.Tensor:
        model_time = torch.clamp(model_time.long(), 0, N - 1)
        u = model_time.float() / (N - 1)
        return u

    def sanitize_indices_desc(self, idx: torch.Tensor) -> torch.Tensor:
        unique = torch.unique(idx)
        sorted_desc = torch.sort(unique, descending=True).values
        return sorted_desc
