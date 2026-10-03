#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import sys
import os

# 添加上级目录到 sys.path，使同一包内的模块可以被导入
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from arch_space import Genotype, SamplerPolicy, enumerate_candidate_archs, make_dense_policy
from evaluator_with_timesteps import Evaluator
from searcher import EvolutionarySearcher


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _require_healthy_cuda(device: str, use_fp16: bool) -> None:
    if str(device).lower() != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Configured device=cuda, but torch.cuda.is_available() is False. "
            "Abort NAS search to avoid silently falling back to CPU."
        )
    if torch.cuda.device_count() <= 0:
        raise RuntimeError("Configured device=cuda, but no CUDA devices are visible.")
    if not torch.backends.cudnn.is_available():
        raise RuntimeError(
            "CUDA is visible but cuDNN is unavailable. Abort NAS search because this environment is unhealthy."
        )

    dev = torch.device("cuda:0")
    try:
        x = torch.randn(2, 3, 8, 8, device=dev)
        w = torch.randn(4, 3, 3, 3, device=dev)
        y = torch.nn.functional.conv2d(x, w, padding=1)
        if use_fp16:
            with torch.amp.autocast("cuda", enabled=True):
                _ = torch.nn.functional.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
        _ = float(y.mean().item())
    except Exception as exc:
        raise RuntimeError(
            "CUDA/cuDNN health check failed during a simple GPU op. "
            "Abort NAS search instead of running for hours on a broken stack."
        ) from exc

    dev_name = torch.cuda.get_device_name(0)
    cudnn_ver = torch.backends.cudnn.version()
    print(
        f"[Startup] CUDA healthy: device={dev_name}, cudnn={cudnn_ver}, fp16={bool(use_fp16)}",
        flush=True,
    )


def _summarize_metrics(metrics):
    fid = float(metrics.get("fid", float("inf")))
    lat = float(metrics.get("latency_ms_per_img", float("inf")))
    std = float(metrics.get("std", metrics.get("fid_std", float("inf"))))
    return f"fid={fid:.4f}, lat_ms={lat:.4f}, std={std:.4f}"


def _policy_is_nontrivial(policy) -> bool:
    skip_blocks = tuple(getattr(policy, "skip_blocks", tuple()) or tuple())
    layer_gates = tuple(getattr(policy, "layer_gates", tuple()) or tuple())
    class_gain = float(getattr(policy, "class_gain", 1.0))
    return bool(skip_blocks) or bool(layer_gates) or abs(class_gain - 1.0) > 1e-6


def _resolve_diffusion_num_steps(config_num_steps: int, fixed_timesteps) -> int:
    resolved = int(config_num_steps)
    if fixed_timesteps:
        min_required = max(int(t) for t in fixed_timesteps) + 1
        if resolved < min_required:
            print(
                "[Config][Warn] num_steps looks like the number of sampled timesteps, "
                f"but fixed_timesteps require at least {min_required}. "
                f"Auto-correcting diffusion num_steps from {resolved} to {min_required}.",
                flush=True,
            )
            resolved = min_required
    return resolved


def _validate_fixed_timesteps_or_die(fixed_timesteps, *, num_steps: int):
    ts = [int(x) for x in fixed_timesteps]
    if len(ts) == 0:
        raise ValueError("timesteps_npy_path produced an empty timestep list.")
    if len(ts) != len(set(ts)):
        raise ValueError(
            f"Invalid fixed_timesteps with duplicates: {ts}. "
            "Your example [947, 500, 500, 137] is invalid because one denoising update is collapsed. "
            "Regenerate Stage0 timesteps or use make_lsun_bedroom_safe_timesteps.py."
        )
    if any(t < 0 or t >= int(num_steps) for t in ts):
        raise ValueError(f"fixed_timesteps out of range for num_steps={num_steps}: {ts}")
    if any(ts[i] <= ts[i + 1] for i in range(len(ts) - 1)):
        raise ValueError(
            f"fixed_timesteps must be strictly descending high->low, got {ts}. "
            "For 4-step LSUN, a safe initial value is [848, 588, 361, 126]."
        )
    return ts


def get_peak_gpu_memory() -> float:
    """获取当前GPU的峰值内存使用量（MB）"""
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
    return 0.0


def _metric_peak_gpu_mb(metrics) -> float:
    if not metrics:
        return 0.0
    gb_alloc = float(metrics.get("peak_gpu_allocated_gb", 0.0) or 0.0)
    gb_reserved = float(metrics.get("peak_gpu_reserved_gb", 0.0) or 0.0)
    return max(gb_alloc, gb_reserved) * 1024.0


def main():
    parser = argparse.ArgumentParser(description="Two-stage NAS for guided-diffusion ADM")
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--config", type=str, default=str(script_dir / "config_formal.yaml"))

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = str(config.get("device", "cuda"))
    use_fp16 = bool(config.get("fp16", False))
    _require_healthy_cuda(device=device, use_fp16=use_fp16)

    # 初始化搜索成本统计
    search_cost_stats = {
        "total_search_time_hours": 0.0,
        "total_search_time_seconds": 0.0,
        "total_evaluated_candidates": 0,
        "stage0_evaluated_candidates": 0,
        "stage1_evaluated_candidates": 0,
        "stage0_time_seconds": 0.0,
        "stage1_time_seconds": 0.0,
        "peak_gpu_memory_mb": 0.0,
        "seed": int(config.get("seed", 0)),
    }

    outdir = Path(config["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "args.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    timesteps_npy_path = config.get("timesteps_npy_path")
    if not timesteps_npy_path:
        raise ValueError("timesteps_npy_path is required for Stage1 fixed-timestep NAS.")
    fixed_timesteps = [int(x) for x in np.load(timesteps_npy_path).tolist()]
    fixed_timesteps = _validate_fixed_timesteps_or_die(
        fixed_timesteps,
        num_steps=int(config.get("num_steps", 1000)),
    )
    np.save(outdir / "best_timesteps.npy", np.array(fixed_timesteps, dtype=np.int64))

    diffusion_num_steps = _resolve_diffusion_num_steps(
        int(config.get("num_steps", 1000)),
        fixed_timesteps,
    )

    base_eval_kwargs = dict(
        ckpt_path=str(config["ckpt_path"]),
        image_size=int(config.get("image_size", 64)),
        num_steps=diffusion_num_steps,
        device=device,
        fp16=use_fp16,
        noise_schedule=str(config.get("noise_schedule", "linear")),
        classifier_path=config.get("classifier_path"),
        classifier_scale=float(config.get("classifier_scale", 0.0)),
        classifier_depth=int(config.get("classifier_depth", 4)),
        classifier_use_fp16=bool(config.get("classifier_use_fp16", False)),
        require_project_fid=bool(config.get("require_project_fid", False)),
        ref_mu=config.get("ref_mu"),
        ref_sigma=config.get("ref_sigma"),
        batch_size=int(config.get("batch_size", 16)),
        num_samples=int(config.get("num_samples", 2000)),
        seed=int(config.get("seed", 0)),
        max_resblocks_hint=int(config.get("max_resblocks", 256)),
        outdir=str(outdir),
        latents_shape=config.get("latents_shape"),
        calculate_is=bool(config.get("calculate_is", False)),
        calculate_precision_recall=bool(config.get("calculate_precision_recall", False)),
        calculate_kid=bool(config.get("calculate_kid", False)),
        is_splits=int(config.get("is_splits", 10)),
        pr_k=int(config.get("pr_k", 3)),
        kid_num_subsets=int(config.get("kid_num_subsets", 100)),
        kid_max_subset_size=int(config.get("kid_max_subset_size", 1000)),
        real_images_path=config.get("real_images_path"),
        eval_dataset=str(config.get("eval_dataset", "imagenet64")),
        ref_stats_pkl=config.get("ref_stats_pkl"),
        fixed_timesteps=fixed_timesteps,
        mo_eval_repeats=int(config.get("mo_eval_repeats", 1)),
        use_median_fid=bool(config.get("use_median_fid", True)),
        use_median_latency=bool(config.get("use_median_latency", True)),
        class_sampling_mode=str(config.get("class_sampling_mode", "uniform")),
        searchable_block_mode=str(config.get("searchable_block_mode", "attention_only")),
        protected_attention_resolutions=config.get("protected_attention_resolutions"),
        protect_middle_attention=bool(config.get("protect_middle_attention", True)),
        protected_block_types=config.get("protected_block_types"),
    )

    teacher_num_res_blocks = int(config.get("teacher_num_res_blocks", 3))
    arch_candidates = enumerate_candidate_archs(
        teacher_num_res_blocks=teacher_num_res_blocks,
        rng=np.random.RandomState(int(config.get("seed", 0))),
        max_candidates=int(config.get("arch_max_candidates", 1)),
        allow_depth_search=bool(config.get("allow_depth_search", False)),
        depth_candidates=config.get("arch_depth_candidates", None),
        attention_patterns=config.get("arch_attention_patterns", None),
    )

    stage0_records = []
    best_arch = None
    best_metrics = None
    num_policy_steps = len(fixed_timesteps)
    total_start = time.time()
    stage0_start = time.time()

    print(
        "[Stage0] start "
        f"candidates={len(arch_candidates)}, num_samples={base_eval_kwargs['num_samples']}, "
        f"batch_size={base_eval_kwargs['batch_size']}, repeats={base_eval_kwargs['mo_eval_repeats']}, "
        f"diffusion_num_steps={diffusion_num_steps}, sampled_steps={len(fixed_timesteps)}",
        flush=True,
    )

    for idx, arch in enumerate(arch_candidates, start=1):
        stage0_item_start = time.time()
        print(f"[Stage0] evaluating {idx}/{len(arch_candidates)} arch={arch.to_dict()}", flush=True)
        evaluator = Evaluator(arch_cfg=arch, **base_eval_kwargs)
        dense_g = Genotype(
            step_policies=tuple(make_dense_policy() for _ in range(num_policy_steps)),
            sampler_policy=SamplerPolicy(ddim_eta=0.0),
            fitness=-1.0,
            metrics={},
        )
        g_eval = evaluator.evaluate_genotype(dense_g)
        metrics = dict(g_eval.metrics or {})
        stage0_records.append({"arch": arch.to_dict(), "metrics": metrics})
        elapsed = time.time() - stage0_item_start
        print(
            f"[Stage0] done {idx}/{len(arch_candidates)} in {_fmt_seconds(elapsed)} "
            f"{_summarize_metrics(metrics)}",
            flush=True,
        )

        # 更新搜索成本统计
        search_cost_stats["stage0_evaluated_candidates"] += 1
        search_cost_stats["total_evaluated_candidates"] += 1
        search_cost_stats["peak_gpu_memory_mb"] = max(
            search_cost_stats["peak_gpu_memory_mb"],
            get_peak_gpu_memory(),
            _metric_peak_gpu_mb(metrics),
        )

        fid_v = float(metrics.get("fid", float("inf")))
        lat_v = float(metrics.get("latency_ms_per_img", float("inf")))
        del evaluator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not (math.isfinite(fid_v) and math.isfinite(lat_v)):
            print(f"[Stage0] ignoring invalid candidate metrics: {metrics}", flush=True)
            continue
        if best_metrics is None:
            best_arch = arch
            best_metrics = metrics
        else:
            best_fid = float(best_metrics.get("fid", float("inf")))
            best_lat = float(best_metrics.get("latency_ms_per_img", float("inf")))
            if (fid_v < best_fid) or (fid_v == best_fid and lat_v < best_lat):
                best_arch = arch
                best_metrics = metrics

    if best_arch is None:
        raise RuntimeError("Stage0 failed to produce a valid architecture.")

    # 记录Stage0耗时
    search_cost_stats["stage0_time_seconds"] = time.time() - stage0_start

    (outdir / "stage0_results.json").write_text(json.dumps(stage0_records, indent=2), encoding="utf-8")
    (outdir / "best_arch.json").write_text(
        json.dumps({"arch": best_arch.to_dict(), "metrics": best_metrics}, indent=2),
        encoding="utf-8",
    )
    with open(outdir / "best_arch.pkl", "wb") as f:
        pickle.dump(best_arch, f)

    print(
        f"[Stage0] complete in {_fmt_seconds(time.time() - stage0_start)} "
        f"best_arch={best_arch.to_dict()} {_summarize_metrics(best_metrics)}",
        flush=True,
    )

    if bool(config.get("static_only", False)):
        elapsed = time.time() - total_start
        search_cost_stats.update(
            total_search_time_seconds=elapsed,
            total_search_time_hours=elapsed / 3600.0,
            static_only=True,
            selection_rule="minimum FID; latency breaks exact FID ties",
        )
        (outdir / "search_cost_report.json").write_text(
            json.dumps(search_cost_stats, indent=2), encoding="utf-8"
        )
        print("[T+A] static architecture saved; dynamic-policy search disabled.", flush=True)
        return

    evaluator = Evaluator(arch_cfg=best_arch, **base_eval_kwargs)
    dense_baseline = Genotype(
        step_policies=tuple(make_dense_policy() for _ in range(num_policy_steps)),
        sampler_policy=SamplerPolicy(ddim_eta=0.0),
        fitness=-1.0,
        metrics=dict(best_metrics or {}),
    )
    quality_baseline_mode = str(config.get("quality_baseline_mode", "stage0_best_dense")).lower()
    quality_fid_baseline = None
    if quality_baseline_mode == "stage0_best_dense" and best_metrics is not None:
        quality_fid_baseline = float(best_metrics.get("fid", float("inf")))
    elif quality_baseline_mode in {"none", ""}:
        quality_fid_baseline = None
    else:
        raise ValueError(
            f"Unsupported quality_baseline_mode={quality_baseline_mode!r}. "
            "Expected 'stage0_best_dense' or 'none'."
        )
    quality_fid_delta = float(config.get("quality_fid_delta", float("inf")))
    quality_fid_limit = (
        float("inf")
        if quality_fid_baseline is None
        else float(quality_fid_baseline) + float(quality_fid_delta)
    )
    print(
        "[Stage1] searchable_blocks="
        f"{len(evaluator.searchable_block_ids)}/{evaluator.total_blocks} "
        f"mode={base_eval_kwargs['searchable_block_mode']} "
        f"ids={list(evaluator.searchable_block_ids)}",
        flush=True,
    )
    print(
        "[Stage1] quality_guard "
        f"baseline_mode={quality_baseline_mode}, "
        f"baseline_fid={quality_fid_baseline}, "
        f"delta={quality_fid_delta}, "
        f"limit={quality_fid_limit}, "
        f"protected_last_n_steps={int(config.get('protected_last_n_steps', 0))}, "
        f"policy_search_mode={str(config.get('policy_search_mode', 'hybrid')).lower()}",
        flush=True,
    )
    searcher = EvolutionarySearcher(
        search_config={
            "population_num": int(config.get("population", 12)),
            "select_num": int(config.get("select_num", 4)),
            "mutation_num": int(config.get("mutation_num", 4)),
            "crossover_num": int(config.get("crossover_num", 4)),
            "m_prob": float(config.get("mutation_prob", 0.35)),
            "skip_max_ratio": float(config.get("skip_max_ratio", 0.35)),
            "fid_max": float(config.get("fid_max", float("inf"))),
            "std_max": float(config.get("std_max", float("inf"))),
            "std_penalty_weight": float(config.get("std_penalty_weight", 0.0)),
            "is_min": config.get("is_min", None),
            "is_penalty_weight": float(config.get("is_penalty_weight", 0.0)),
            "precision_min": config.get("precision_min", None),
            "recall_min": config.get("recall_min", None),
            "pr_penalty_weight": float(config.get("pr_penalty_weight", 0.0)),
            "kid_max": config.get("kid_max", None),
            "kid_penalty_weight": float(config.get("kid_penalty_weight", 0.0)),
            "step_policy_init": dict(config.get("step_policy_init", {})),
            "step_policy_mutation": dict(config.get("step_policy_mutation", {})),
            "quality_fid_baseline": quality_fid_baseline,
            "quality_fid_delta": quality_fid_delta,
            "quality_constraint_strict": bool(config.get("quality_constraint_strict", True)),
            "protected_last_n_steps": int(config.get("protected_last_n_steps", 0)),
            "policy_search_mode": str(config.get("policy_search_mode", "hybrid")).lower(),
            "final_selection_priority": str(config.get("final_selection_priority", "fid_first")).lower(),
            "search_artifact_dir": str(outdir),
        },
        total_blocks=int(evaluator.total_blocks),
        searchable_block_ids=evaluator.searchable_block_ids,
    )

    population = searcher.init_population(num_steps=num_policy_steps)
    nonempty_policies = [
        sum(1 for p in g.step_policies if _policy_is_nontrivial(p))
        for g in population
    ]
    num_generations = int(config.get("generations", 10))
    include_initial_population_evaluation = bool(
        config.get("include_initial_population_evaluation", False)
    )
    stage1_history = []
    stage1_start = time.time()
    evaluation_rounds = num_generations + int(include_initial_population_evaluation)
    total_stage1_evals = max(1, evaluation_rounds * len(population))
    completed_stage1_evals = 0
    all_stage1_evaluated = []

    print(
        "[Stage1] start "
        f"generations={num_generations}, population={len(population)}, "
        f"evaluate_initial_population={include_initial_population_evaluation}, "
        f"num_samples={base_eval_kwargs['num_samples']}, repeats={base_eval_kwargs['mo_eval_repeats']}, "
        f"init_nonempty_step_policies={nonempty_policies}",
        flush=True,
    )

    for round_idx in range(evaluation_rounds):
        is_initial_population = include_initial_population_evaluation and round_idx == 0
        generation = round_idx if include_initial_population_evaluation else round_idx + 1
        phase_name = "initial_population" if is_initial_population else f"generation {generation}/{num_generations}"
        gen_start = time.time()
        print(f"[Stage1] {phase_name} start", flush=True)
        evaluated_population = []
        for cand_idx, g in enumerate(population, start=1):
            item_start = time.time()
            print(
                f"[Stage1] {phase_name} candidate {cand_idx}/{len(population)} start",
                flush=True,
            )
            g_eval = evaluator.evaluate_genotype(g)
            evaluated_population.append(g_eval)
            all_stage1_evaluated.append(g_eval)
            completed_stage1_evals += 1
            
            # 更新搜索成本统计
            search_cost_stats["stage1_evaluated_candidates"] += 1
            search_cost_stats["total_evaluated_candidates"] += 1
            search_cost_stats["peak_gpu_memory_mb"] = max(
                search_cost_stats["peak_gpu_memory_mb"],
                get_peak_gpu_memory(),
                _metric_peak_gpu_mb(g_eval.metrics),
            )
            
            elapsed = time.time() - item_start
            avg_eval = (time.time() - stage1_start) / max(1, completed_stage1_evals)
            remain = max(0.0, (total_stage1_evals - completed_stage1_evals) * avg_eval)
            print(
                f"[Stage1] {phase_name} candidate {cand_idx}/{len(population)} "
                f"done in {_fmt_seconds(elapsed)} {_summarize_metrics(g_eval.metrics or {})} "
                f"stage1_eta={_fmt_seconds(remain)}",
                flush=True,
            )
        stage1_history.append(
            {
                "phase": "initial_population" if is_initial_population else "evolutionary_generation",
                "generation": int(generation) if not is_initial_population else 0,
                "population": [g.to_dict() for g in evaluated_population],
                "quality_feasible_size": int(len(searcher.quality_feasible_population(evaluated_population))),
                "quality_fid_limit": float(searcher.quality_fid_limit()),
            }
        )
        (outdir / "stage1_history.json").write_text(json.dumps(stage1_history, indent=2), encoding="utf-8")
        if round_idx < evaluation_rounds - 1:
            searcher.evolve(evaluated_population)
            pareto_index = 0 if is_initial_population else generation
            with open(outdir / f"pareto_gen_{pareto_index}.pkl", "wb") as f:
                pickle.dump(searcher.top_k, f)
            population = searcher.population
        best_gen = min(
            evaluated_population,
            key=lambda item: float((item.metrics or {}).get("fid", float("inf"))),
        )
        print(
            f"[Stage1] {phase_name} complete in {_fmt_seconds(time.time() - gen_start)} "
            f"best={_summarize_metrics(best_gen.metrics or {})}",
            flush=True,
        )

    candidate_pool = list(all_stage1_evaluated) if all_stage1_evaluated else list(evaluated_population)
    g_star, final_selection_report = searcher.select_final_candidate(
        candidate_pool,
        fallback=dense_baseline,
    )
    searcher.dump_search_artifacts(
        str(outdir),
        final_population=candidate_pool,
        selected=g_star,
        selection_report=final_selection_report,
    )

    with open(outdir / "g_star.pkl", "wb") as f:
        pickle.dump(g_star, f)
    (outdir / "g_star.json").write_text(json.dumps(g_star.to_dict(), indent=2), encoding="utf-8")
    (outdir / "final_selection_report.json").write_text(
        json.dumps(final_selection_report, indent=2),
        encoding="utf-8",
    )

    # 记录完整的搜索成本统计
    total_time_seconds = time.time() - total_start
    stage1_time_seconds = time.time() - stage1_start
    final_metrics = dict(g_star.metrics or {})

    search_cost_stats["total_search_time_seconds"] = total_time_seconds
    search_cost_stats["total_search_time_hours"] = total_time_seconds / 3600.0
    search_cost_stats["stage1_time_seconds"] = stage1_time_seconds
    search_cost_stats["peak_gpu_memory_mb"] = max(
        search_cost_stats["peak_gpu_memory_mb"],
        get_peak_gpu_memory(),
        float(final_selection_report.get("search_peak_gpu_allocated_gb", 0.0) or 0.0) * 1024.0,
        float(final_selection_report.get("search_peak_gpu_reserved_gb", 0.0) or 0.0) * 1024.0,
        _metric_peak_gpu_mb(final_metrics),
    )

    # 生成完整的搜索报告（包含论文需要的6个指标）
    convergence_tail = dict(searcher.convergence_history[-1]) if getattr(searcher, "convergence_history", None) else {}
    search_report = {
        # 论文需要的6个核心指标
        "final_fid": float(final_metrics.get("fid", float("inf"))),
        "final_latency_ms_per_img": float(final_metrics.get("latency_ms_per_img", float("inf"))),
        "total_search_time_hours": search_cost_stats["total_search_time_hours"],
        "total_evaluated_candidates": search_cost_stats["total_evaluated_candidates"],
        "peak_gpu_memory_mb": search_cost_stats["peak_gpu_memory_mb"],
        "peak_gpu_memory_gb": search_cost_stats["peak_gpu_memory_mb"] / 1024.0,
        "search_stability": {
            "seed": search_cost_stats["seed"],
            "note": "Run with 3 different seeds to get mean ± std",
        },
        
        # 详细的搜索过程信息
        "detailed_stats": search_cost_stats,
        "convergence_last": convergence_tail,
        "final_selection_report": final_selection_report,
        "final_fid_std": float(final_metrics.get("fid_std", float("inf"))),
        "final_flops_g": float(final_metrics.get("flops_g", float("inf"))),
        
        # Stage0详细信息
        "stage0": {
            "evaluated_candidates": search_cost_stats["stage0_evaluated_candidates"],
            "time_seconds": search_cost_stats["stage0_time_seconds"],
            "best_fid": float(best_metrics.get("fid", float("inf"))) if best_metrics else float("inf"),
        },
        
        # Stage1详细信息
        "stage1": {
            "evaluated_candidates": search_cost_stats["stage1_evaluated_candidates"],
            "evolutionary_candidate_evaluations": search_cost_stats["stage1_evaluated_candidates"],
            "time_seconds": search_cost_stats["stage1_time_seconds"],
            "num_generations": num_generations,
            "population_size": len(population),
            "initial_population_evaluated": include_initial_population_evaluation,
            "expected_candidate_count": total_stage1_evals,
            "convergence_history_files": {
                "jsonl": str(outdir / "convergence_history.jsonl"),
                "csv": str(outdir / "convergence_history.csv"),
            },
        },
    }
    
    # 保存搜索报告
    (outdir / "search_cost_report.json").write_text(
        json.dumps(search_report, indent=2),
        encoding="utf-8",
    )
    
    print(
        f"[Done] total={_fmt_seconds(time.time() - total_start)} "
        f"stage1={_fmt_seconds(time.time() - stage1_start)} "
        f"g_star={_summarize_metrics(g_star.metrics or {})}",
        flush=True,
    )
    print("best_arch:", best_arch.to_dict())
    print("best_arch_metrics:", best_metrics)
    print("g_star_metrics:", g_star.metrics)
    print("final_selection_report:", final_selection_report)
    
    # 打印搜索成本摘要
    print("\n" + "="*80)
    print("SEARCH COST SUMMARY (for paper)")
    print("="*80)
    print(f"Final FID: {search_report['final_fid']:.4f}")
    print(f"Final latency: {search_report['final_latency_ms_per_img']:.4f} ms/img")
    print(f"Total search time: {search_report['total_search_time_hours']:.4f} hours")
    print(f"Total evaluated candidates: {search_report['total_evaluated_candidates']}")
    print(f"Peak GPU memory: {search_report['peak_gpu_memory_mb']:.2f} MB")
    print(f"Search stability: Seed {search_cost_stats['seed']} (run 3 seeds for mean ± std)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
