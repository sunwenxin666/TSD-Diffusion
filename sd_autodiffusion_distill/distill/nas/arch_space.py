
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Callable

import numpy as np


@dataclass(frozen=True)
class UNetArchConfig:
    """Static UNet architecture edits relative to the supplied SD U-Net.

    Attributes:
        attn_ds: Attention placement expressed as downsample scales.
                 Example: (4, 2, 1) means attention at 16x16, 32x32, 64x64 feature maps.
                 See module docstring for resolution mapping details.
        num_res_blocks: Global depth - number of ResBlocks per stage (same for all stages).
                       Typical values: 1, 2, 3 (teacher usually uses 2).
                       Note: This is a simplified global setting. In practice, depth requirements
                       may vary by time-step (early steps need depth>=2, late steps can use depth>=1).
    """

    # Attention placement expressed as downsample scales (ds): e.g., (4, 2, 1)
    # Resolution mapping: ds=1→64x64, ds=2→32x32, ds=4→16x16, ds=8→8x8
    attn_ds: Tuple[int, ...]

    # Global depth: number of ResBlocks per stage (same for all stages).
    # Typical values: 1, 2, 3 (teacher usually uses 2)
    # Limitation: This is a global setting. Ideal would be time-step dependent depth.
    num_res_blocks: int

    def validate(self, num_stages: int = None) -> None:
        """Validate architecture config."""
        if int(self.num_res_blocks) <= 0:
            raise ValueError(f"num_res_blocks must be positive, got {self.num_res_blocks}")
        if any(int(ds) <= 0 for ds in self.attn_ds):
            raise ValueError(f"attn_ds must be positive, got {self.attn_ds}")

    def key(self) -> str:
        return str((tuple(int(x) for x in self.attn_ds), int(self.num_res_blocks)))


def enumerate_candidate_archs(
    teacher_num_res_blocks: int,
    rng: Optional[np.random.RandomState] = None,
    max_candidates: int = 24,
) -> List[UNetArchConfig]:
    """Generate architecture candidates based on DiC-validated insights.

    Candidate patterns are designed based on theoretical insights:
    - High-res (64x64, ds=1): May be removable (conv handles local texture)
    - Mid-res (32x32, ds=2): Balanced importance
    - Low-res (16x16, ds=4): Critical for semantics
    - Bottleneck (8x8, ds=8): Almost indispensable
    
    Args:
        teacher_num_res_blocks: Teacher's global num_res_blocks (int)
        rng: Random number generator
        max_candidates: Maximum number of candidates to return
    """

    if rng is None:
        rng = np.random.RandomState(0)

    # 1) Attention resolution patterns (DiC-aligned design)
    # Patterns ordered by theoretical priority and DiC insights:
    attn_patterns = [
        # === Conservative baseline ===
        (4, 2, 1),  # Original SD v1.4: all resolutions (16x16, 32x32, 64x64)
                   # Highest quality baseline, always included first
        
        # === DiC Insight 1: High-res (64x64) may be removable ===
        (4, 2),     # Remove high-res (64x64), keep mid+low (32x32, 16x16)
                   # Test: Can conv handle local texture without attention?
        
        # === DiC Insight 2: Low-res (16x16) is critical ===
        (4,),       # Only low-res (16x16) - most critical layer
                   # Extreme test: Can we rely solely on bottleneck semantics?
        
        (4, 1),     # Low-res + high-res, skip mid-res
                   # Test: Is mid-res (32x32) truly necessary?
        
        # === Balanced patterns ===
        (2, 1),     # Mid-res + high-res, skip low-res
                   # Test: Can mid-res compensate for missing low-res?
        
        (2, 4),     # Mid-res + low-res, skip high-res
                   # Balanced: structure + semantics, no local details
        
        # === Extreme tests (may fail but informative) ===
        (2,),       # Only mid-res (32x32) - extreme simplification
        (1,),       # Only high-res (64x64) - likely insufficient
        (8,),       # Only bottleneck (8x8) - if exists, test global-only
        tuple(),    # No attention - extreme test (likely to fail)
    ]

    # 2) Global depth candidates (simplified: same depth for all stages)
    # Teacher usually uses 2. We search in {1, 2, 3}
    # 
    # NOTE: This is a simplified global setting. Ideal would be time-step dependent:
    #   - Early steps (high noise, t>0.7): depth >= 2 required (structure formation)
    #   - Middle steps (0.7 > t > 0.25): depth >= 1 acceptable (refinement)
    #   - Late steps (low noise, t<0.25): depth >= 1 acceptable (detail polishing)
    # 
    # Current implementation uses global depth as a compromise for training-free NAS.
    teacher_global_depth = int(teacher_num_res_blocks)
    depth_candidates = [1, 2, 3]
    
    # Ensure teacher's depth is included
    if teacher_global_depth not in depth_candidates:
        depth_candidates.append(teacher_global_depth)
        depth_candidates.sort()

    # Cross product + cap
    cands: List[UNetArchConfig] = []
    for ap in attn_patterns:
        for depth in depth_candidates:
            cands.append(UNetArchConfig(attn_ds=tuple(ap), num_res_blocks=int(depth)))

    # Ensure conservative baseline (4, 2, 1) with depth=2 is always included
    conservative_config = UNetArchConfig(attn_ds=(4, 2, 1), num_res_blocks=2)
    if conservative_config not in cands:
        cands.insert(0, conservative_config)
    else:
        # Move it to the front to ensure it's always evaluated
        cands.remove(conservative_config)
        cands.insert(0, conservative_config)

    # Shuffle deterministically (but keep conservative config at front)
    if len(cands) > 1:
        rng.shuffle(cands[1:])
    cands = cands[: int(max_candidates)]

    # Validate
    for c in cands:
        c.validate()
    return cands


def analyze_attention_importance(
    baseline_arch: UNetArchConfig,
    evaluator_fn,  # Callable[[UNetArchConfig], Dict[str, float]]
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Analyze importance of each attention resolution layer.
    
    This function tests removing each resolution layer from the baseline architecture
    and evaluates the impact, providing insights into which layers are critical.
    
    Args:
        baseline_arch: Baseline architecture (typically original SD v1.4 pattern)
        evaluator_fn: Function that takes UNetArchConfig and returns metrics dict
                     with keys like 'fid', 'latency', etc.
        verbose: Whether to print analysis results
    
    Returns:
        Dictionary mapping resolution to removal impact metrics.
        Example: {
            'ds_1': {'fid_delta': 2.5, 'latency_delta': -0.1, ...},
            'ds_2': {'fid_delta': 1.2, 'latency_delta': -0.05, ...},
            ...
        }
    
    Example usage:
        >>> baseline = UNetArchConfig(attn_ds=(4, 2, 1), num_res_blocks=2)
        >>> def eval_fn(arch):
        ...     # Your evaluation logic here
        ...     return {'fid': 15.2, 'latency': 0.5}
        >>> results = analyze_attention_importance(baseline, eval_fn)
        >>> # Results show which layers can be safely removed
    """
    if not baseline_arch.attn_ds:
        if verbose:
            print("[AttentionAnalysis] Baseline has no attention layers, skipping analysis.")
        return {}
    
    baseline_metrics = evaluator_fn(baseline_arch)
    baseline_fid = baseline_metrics.get('fid', float('inf'))
    baseline_lat = baseline_metrics.get('latency', float('inf'))
    
    results = {}
    
    if verbose:
        print(f"\n[AttentionAnalysis] Baseline: {baseline_arch.key()}")
        print(f"  FID: {baseline_fid:.3f}, Latency: {baseline_lat:.3f}")
        print(f"  Testing removal of each resolution layer...")
    
    # Test removing each resolution layer
    for ds_to_remove in sorted(set(baseline_arch.attn_ds)):
        test_pattern = tuple(ds for ds in baseline_arch.attn_ds if ds != ds_to_remove)
        test_arch = UNetArchConfig(
            attn_ds=test_pattern,
            num_res_blocks=baseline_arch.num_res_blocks
        )
        
        try:
            test_metrics = evaluator_fn(test_arch)
            test_fid = test_metrics.get('fid', float('inf'))
            test_lat = test_metrics.get('latency', float('inf'))
            
            fid_delta = test_fid - baseline_fid
            lat_delta = test_lat - baseline_lat
            
            # Map ds to resolution for readability
            resolution_map = {1: "64x64", 2: "32x32", 4: "16x16", 8: "8x8"}
            res_str = resolution_map.get(ds_to_remove, f"{ds_to_remove}x{ds_to_remove}")
            
            results[f"ds_{ds_to_remove}"] = {
                'resolution': res_str,
                'fid_delta': fid_delta,
                'latency_delta': lat_delta,
                'fid_ratio': test_fid / baseline_fid if baseline_fid > 0 else float('inf'),
                'test_fid': test_fid,
                'test_latency': test_lat,
            }
            
            if verbose:
                impact = "CRITICAL" if fid_delta > 2.0 else "MODERATE" if fid_delta > 0.5 else "MINOR"
                print(f"  Remove ds={ds_to_remove} ({res_str}): "
                      f"FID Δ={fid_delta:+.3f}, Lat Δ={lat_delta:+.3f} [{impact}]")
        
        except Exception as e:
            if verbose:
                print(f"  Remove ds={ds_to_remove}: FAILED - {e}")
            results[f"ds_{ds_to_remove}"] = {'error': str(e)}
    
    if verbose:
        print("\n[AttentionAnalysis] Summary:")
        print("  Layers ordered by importance (largest FID delta = most critical):")
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if 'fid_delta' in v],
            key=lambda x: x[1].get('fid_delta', 0),
            reverse=True
        )
        for key, val in sorted_results:
            res = val.get('resolution', '?')
            delta = val.get('fid_delta', 0)
            print(f"    {res} (ds={key.split('_')[1]}): ΔFID={delta:+.3f}")
    
    return results


def get_resolution_for_ds(downsample_factor: int, latent_size: int = 64) -> Tuple[int, int]:
    """Get feature map resolution for a given downsample factor.
    
    Args:
        downsample_factor: Attention downsample factor (1, 2, 4, 8, ...)
        latent_size: Latent space size (default 64 for 512x512 images)
    
    Returns:
        (height, width) tuple of feature map resolution
    """
    feature_size = latent_size // downsample_factor
    return (feature_size, feature_size)


def format_arch_description(arch: UNetArchConfig, latent_size: int = 64) -> str:
    """Format architecture config as human-readable description.
    
    Args:
        arch: Architecture configuration
        latent_size: Latent space size (default 64 for 512x512 images)
    
    Returns:
        Formatted string describing the architecture
    """
    if not arch.attn_ds:
        attn_desc = "No attention layers"
    else:
        resolutions = [get_resolution_for_ds(ds, latent_size) for ds in sorted(arch.attn_ds)]
        res_strs = [f"{h}x{w}" for h, w in resolutions]
        attn_desc = f"Attention at: {', '.join(res_strs)} (ds={sorted(arch.attn_ds)})"
    
    return f"Depth={arch.num_res_blocks}, {attn_desc}"
