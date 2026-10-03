import os
import pickle
import time
from datetime import datetime
from typing import Tuple, Any
from contextlib import contextmanager
import sys

import numpy as np
from tqdm import tqdm

from .adapters.unet import UNetAdapter
from .nas.searcher import EvolutionSearcher
from .nas.improved_evaluator import ImprovedEvaluator
from .nas.search_space import (
    Genotype,
    SamplerPolicy,
    make_default_schedule,
    make_all_true_segment_policy,
)
from .nas.arch_space import enumerate_candidate_archs, UNetArchConfig

from .teacher import load_teacher_pipeline
from .config import Config as GlobalConfig

from .utils.prompts import load_prompts_from_json, DEFAULT_PROMPTS_JSON


# -----------------------------
# Utils: silence noisy outputs
# -----------------------------
@contextmanager
def suppress_output(enabled: bool = True):
    """
    Suppress BOTH stdout and stderr.
    Many solvers / libs print to stderr, so suppressing stdout alone is insufficient.
    """
    if not enabled:
        yield
        return

    saved_out = sys.stdout
    saved_err = sys.stderr
    try:
        with open(os.devnull, "w") as fnull:
            sys.stdout = fnull
            sys.stderr = fnull
            yield
    finally:
        sys.stdout = saved_out
        sys.stderr = saved_err


def run_search():
    # Where to save results
    output_dir = getattr(GlobalConfig, "output_dir", "./checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    # Result TXT log (append mode)
    result_txt_path = os.path.join(output_dir, "nas_results.txt")

    def append_result_line(line: str):
        with open(result_txt_path, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
            f.flush()

    # Header
    append_result_line("=" * 120)
    append_result_line(f"Run start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    append_result_line(
        f"model_id={getattr(GlobalConfig, 'model_id', None)} device={getattr(GlobalConfig, 'device', None)}"
    )
    append_result_line("=" * 120)
    append_result_line(
        "timestamp\tstage\tgen\tidx\tpopulation_size\tfid\tlatency\tstd\tclip_similarity\ttext_penalty\tfitness"
    )

    print("--- Stage 1: Setup for NAS Search ---")

    # Silence teacher load warnings if desired
    with suppress_output(bool(getattr(GlobalConfig, "suppress_solver_logs", True))):
        teacher_pipeline = load_teacher_pipeline(
            GlobalConfig.model_id,
            device=GlobalConfig.device,
            use_fp16=getattr(GlobalConfig, "use_fp16", True),
        )

    if not hasattr(teacher_pipeline, "encode_text") or teacher_pipeline.encode_text is None:
        raise AttributeError(
            "TeacherPipeline must expose encode_text(prompts)->Tensor-like. "
            "Please ensure teacher.py sets self.encode_text."
        )

    # ------------------------------------------------------------
    # Prompts: move everything to config
    # ------------------------------------------------------------
    prompts = list(getattr(GlobalConfig, "prompts", []) or [])
    if not prompts:
        prompts_path = getattr(GlobalConfig, "prompts_path", None) or DEFAULT_PROMPTS_JSON
        prompts = load_prompts_from_json(prompts_path)

    # FID reference
    fid_ref_mu_path = getattr(GlobalConfig, "fid_ref_mu_path")
    fid_ref_sigma_path = getattr(GlobalConfig, "fid_ref_sigma_path")

    # shared knobs
    latents_shape = getattr(GlobalConfig, "latents_shape", (2, 4, 64, 64))

    # MO eval knobs
    mo_eval_repeats = int(getattr(GlobalConfig, "mo_eval_repeats", 1))
    mo_seed_base = int(getattr(GlobalConfig, "mo_seed_base", 12345))
    latency_include_fid_time = bool(getattr(GlobalConfig, "latency_include_fid_time", False))

    # evaluator knobs
    fixed_batch_size = int(getattr(GlobalConfig, "fixed_batch_size", 1))
    num_warmup_runs = int(getattr(GlobalConfig, "num_warmup_runs", 1))
    num_latency_measurements = int(getattr(GlobalConfig, "num_latency_measurements", 2))
    use_median_latency = bool(getattr(GlobalConfig, "use_median_latency", True))
    use_median_fid = bool(getattr(GlobalConfig, "use_median_fid", True))
    target_num_images = getattr(GlobalConfig, "target_num_images", None)

    enable_text_alignment = bool(getattr(GlobalConfig, "enable_text_alignment", False))
    clip_similarity_threshold = float(getattr(GlobalConfig, "clip_similarity_threshold", 0.15))
    clip_similarity_penalty_weight = float(getattr(GlobalConfig, "clip_similarity_penalty_weight", 0.0))
    clip_model_name = getattr(GlobalConfig, "clip_model_name", "ViT-L/14")

    # optional sampler type
    sampler_type = getattr(GlobalConfig, "sampler_type", "acdms")

    # ------------------------------------------------------------
    # Given discrete timesteps (from config npy)
    # NOTE: timesteps themselves do NOT participate in search.
    # We search per-step policies aligned to these fixed timesteps.
    # ------------------------------------------------------------
    ns = teacher_pipeline.noise_schedule
    t0 = 1.0 / float(ns.total_N)
    tT = float(ns.T)

    ts_path = getattr(GlobalConfig, "timesteps_npy_path", None)
    if ts_path is None:
        raise ValueError("Config.timesteps_npy_path is required for step-wise NAS.")
    timesteps_arr = np.load(ts_path).reshape(-1).astype(np.float32)

    max_steps = getattr(GlobalConfig, "timesteps_max_steps", None)
    if max_steps is not None:
        timesteps_arr = timesteps_arr[: int(max_steps)]

    # Convert indices/u -> actual sampler timesteps in [t0,tT]
    if float(np.max(timesteps_arr)) > 1.5:
        # treat as discrete indices (0..999 or similar)
        u = np.clip(timesteps_arr / 999.0, 0.0, 1.0)
        timesteps_arr = t0 + u * (tT - t0)
    else:
        # treat as u in [0,1] if it looks like u; else treat as already-in-t
        if float(np.max(timesteps_arr)) <= 1.0 + 1e-3 and float(np.min(timesteps_arr)) >= -1e-3:
            u = np.clip(timesteps_arr, 0.0, 1.0)
            timesteps_arr = t0 + u * (tT - t0)
        else:
            # already in [t0,tT]
            timesteps_arr = np.clip(timesteps_arr, t0, tT)

    if bool(getattr(GlobalConfig, "timesteps_force_descending", True)):
        timesteps_arr = np.sort(timesteps_arr)[::-1]

    eval_config = {
        "noise_schedule": teacher_pipeline.noise_schedule,
        "vae": teacher_pipeline.vae,

        "encode_text": teacher_pipeline.encode_text,
        "text_encoder": getattr(teacher_pipeline, "text_encoder", None),
        "teacher_pipeline": teacher_pipeline,

        "prompts": prompts,
        "latents_shape": latents_shape,
        "fid_ref_stats_path": (fid_ref_mu_path, fid_ref_sigma_path),
        "eval_batch_size": int(getattr(GlobalConfig, "eval_batch_size", 4)),
        "cfg_scale": float(getattr(GlobalConfig, "cfg_scale", 4.5)),

        # fixed timesteps (float32, descending)
        "timesteps": timesteps_arr,

        "mo_eval_repeats": mo_eval_repeats,
        "mo_seed_base": mo_seed_base,
        "latency_include_fid_time": latency_include_fid_time,

        "fixed_batch_size": fixed_batch_size,
        "num_warmup_runs": num_warmup_runs,
        "num_latency_measurements": num_latency_measurements,
        "use_median_latency": use_median_latency,

        "use_median_fid": use_median_fid,
        "target_num_images": target_num_images,

        "enable_text_alignment": enable_text_alignment,
        "clip_similarity_threshold": clip_similarity_threshold,
        "clip_similarity_penalty_weight": clip_similarity_penalty_weight,
        "clip_model_name": clip_model_name,
        "clip_local_dir": getattr(GlobalConfig, "clip_model_path", ""),

        "sampler_type": sampler_type,

        # NEW: fixed timesteps for the evaluator/sampler
        "timesteps": timesteps_arr,
    }

    print("\n--- Stage 2: Running Multi-Objective NAS (Stage0 arch + Stage1 schedule/policy) ---")

    # ============================================================
    # Stage0: training-free architecture search (DiC-aligned)
    #   Innovations:
    #     A) attention placement by downsample scale
    #     B) global depth (num_res_blocks as int)
    # ============================================================

    def _extract_teacher_num_res_blocks(unet) -> int:
        nrb = getattr(unet, "num_res_blocks", None)
        if isinstance(nrb, (list, tuple)):
            return int(nrb[0]) if len(nrb) > 0 else 2
        if isinstance(nrb, int):
            return int(nrb)
        return 2

    def _build_unet_from_teacher(teacher_unet, arch: UNetArchConfig):
        import inspect
        from ldm.modules.diffusionmodules.openaimodel import UNetModel

        sig = inspect.signature(UNetModel.__init__)
        params = [p for p in sig.parameters.values() if p.name != "self"]

        overrides = {
            "attention_resolutions": tuple(int(x) for x in arch.attn_ds),
            "num_res_blocks": int(arch.num_res_blocks),
        }

        kwargs = {}
        for p in params:
            name = p.name
            if name in overrides:
                kwargs[name] = overrides[name]
                continue
            if hasattr(teacher_unet, name):
                kwargs[name] = getattr(teacher_unet, name)
                continue

            alt_map = {
                "in_channels": "in_channels",
                "out_channels": "out_channels",
                "model_channels": "model_channels",
                "channel_mult": "channel_mult",
                "dropout": "dropout",
                "num_heads": "num_heads",
                "num_head_channels": "num_head_channels",
                "use_checkpoint": "use_checkpoint",
                "use_scale_shift_norm": "use_scale_shift_norm",
                "resblock_updown": "resblock_updown",
                "use_new_attention_order": "use_new_attention_order",
            }
            alt = alt_map.get(name, None)
            if alt is not None and hasattr(teacher_unet, alt):
                kwargs[name] = getattr(teacher_unet, alt)
                continue

            if p.default is not inspect._empty:
                continue

            raise KeyError(
                f"[ArchSearch] Cannot infer required UNetModel arg '{name}' from teacher_unet. "
                "Please extend _build_unet_from_teacher mapping."
            )

        return UNetModel(**kwargs)

    def _load_partial_weights_shape_match(student_unet, teacher_unet) -> int:
        s_sd = student_unet.state_dict()
        t_sd = teacher_unet.state_dict()
        matched = 0
        for k, v in t_sd.items():
            if k in s_sd and s_sd[k].shape == v.shape:
                s_sd[k] = v
                matched += 1
        student_unet.load_state_dict(s_sd, strict=False)
        return matched

    def _patch_sample_return_prompts(evaluator: ImprovedEvaluator):
        # ImprovedEvaluator does not have _sample method, this patch is not needed
        pass

    def stage0_search_architecture(num_steps: int) -> UNetArchConfig:
        teacher_unet = teacher_pipeline.unet
        teacher_nrb = _extract_teacher_num_res_blocks(teacher_unet)

        arch_candidates = enumerate_candidate_archs(
            teacher_num_res_blocks=teacher_nrb,
            rng=np.random.RandomState(int(getattr(GlobalConfig, "seed", 42))),
            max_candidates=int(getattr(GlobalConfig, "arch_max_candidates", 8)),
        )
        print(f"[ArchSearch] Candidates: {len(arch_candidates)}")

        best_arch = None
        best_metrics = None

        pbar = tqdm(arch_candidates, desc="Stage0 ArchSearch", dynamic_ncols=True)
        for arch in pbar:
            cand_unet = _build_unet_from_teacher(teacher_unet, arch)
            # GPU修复：将候选UNet移动到正确的设备和数据类型
            device = teacher_pipeline.device
            dtype = teacher_pipeline.dtype
            cand_unet = cand_unet.to(device=device, dtype=dtype)
            matched = _load_partial_weights_shape_match(cand_unet, teacher_unet)

            orig_unet = teacher_pipeline.unet
            teacher_pipeline.unet = cand_unet
            try:
                # wrap/parse 这些日志也很吵，一并静音
                with suppress_output(bool(getattr(GlobalConfig, "suppress_solver_logs", True))):
                    cand_adapter = UNetAdapter(teacher_pipeline.unet)

                    base_policy = make_all_true_segment_policy(
                        cand_adapter.total_layers, cand_adapter.total_stages
                    )
                    base_g = Genotype(
                        u_schedule=make_default_schedule(num_steps=num_steps, N=1000),
                        segment_policies=(base_policy,),
                        segment_boundaries=tuple(),
                        sampler_policy=SamplerPolicy(),
                        fitness=-1.0,
                        metrics={},
                    )

                    arch_eval_config = dict(eval_config)
                    arch_eval_config["total_layers"] = cand_adapter.total_layers
                    arch_eval_config["total_stages"] = cand_adapter.total_stages
                    arch_eval_config["teacher_pipeline"] = teacher_pipeline

                    cand_evaluator = ImprovedEvaluator(cand_adapter, arch_eval_config)
                    _patch_sample_return_prompts(cand_evaluator)

                    g_eval = cand_evaluator.evaluate_genotype(base_g)

                m = g_eval.metrics or {}
                fid_v = float(m.get("fid", float("inf")))
                lat_v = float(m.get("latency", float("inf")))

                if best_metrics is None:
                    best_fid = float("inf")
                    best_lat = float("inf")
                else:
                    best_fid = float(best_metrics.get("fid", float("inf")))
                    best_lat = float(best_metrics.get("latency", float("inf")))

                pbar.set_postfix(
                    cur=f"{arch.key()}",
                    fid=f"{fid_v:.3f}",
                    lat=f"{lat_v:.3f}",
                    best_fid=f"{best_fid:.3f}",
                    matched=matched,
                )

                if best_arch is None:
                    best_arch, best_metrics = arch, m
                else:
                    if (fid_v < best_fid) or (fid_v == best_fid and lat_v < best_lat):
                        best_arch, best_metrics = arch, m

                tqdm.write(f"[ArchSearch] arch={arch.key()} matched={matched} fid={fid_v:.3f} lat={lat_v:.3f}")

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                append_result_line(
                    f"{ts}\tstage0\t0\t0\t{len(arch_candidates)}\t{fid_v}\t{lat_v}\t0\t0\t0\t0"
                )

            finally:
                teacher_pipeline.unet = orig_unet

        if best_arch is None:
            raise RuntimeError("[ArchSearch] No valid architecture candidate evaluated.")

        print(f"[ArchSearch] Winner: {best_arch.key()} metrics={best_metrics}")
        return best_arch

    # Stage0 run
    enable_arch_search = bool(getattr(GlobalConfig, "enable_arch_search", True))
    # num_steps is defined by fixed timesteps
    num_steps = int(len(timesteps_arr))

    best_arch = None
    if enable_arch_search:
        best_arch = stage0_search_architecture(num_steps=num_steps)

    # ============================================================
    # Stage1: schedule/policy/sampler search on the selected best_arch
    # IMPORTANT: build a FRESH student unet from best_arch (avoid wrapped modules)
    # ============================================================

    # Keep original SD v1.4 teacher unet as weight donor
    teacher_unet = teacher_pipeline.unet

    if best_arch is not None:
        student_unet = _build_unet_from_teacher(teacher_unet, best_arch)
        # GPU修复：将学生UNet移动到正确的设备和数据类型
        device = teacher_pipeline.device
        dtype = teacher_pipeline.dtype
        student_unet = student_unet.to(device=device, dtype=dtype)
        _load_partial_weights_shape_match(student_unet, teacher_unet)
        teacher_pipeline.unet = student_unet  # Stage1 operates on student arch
        print(f"[Stage1] Using student_unet built from best_arch={best_arch.key()}")

    # adapter init can be noisy; optionally silence
    with suppress_output(bool(getattr(GlobalConfig, "suppress_solver_logs", True))):
        adapter = UNetAdapter(teacher_pipeline.unet)

    search_config = {
        "policy_mode": "step",
        "population_num": int(getattr(GlobalConfig, "population_num", 4)),
        "select_num": int(getattr(GlobalConfig, "select_num", 2)),
        "mutation_num": int(getattr(GlobalConfig, "mutation_num", 1)),
        "crossover_num": int(getattr(GlobalConfig, "crossover_num", 1)),
        "m_prob": float(getattr(GlobalConfig, "m_prob", 0.1)),

        # step-policy search knobs (all in config)
        "step_policy_init": dict(getattr(GlobalConfig, "step_policy_init", {})),
        "step_policy_mutation": dict(getattr(GlobalConfig, "step_policy_mutation", {})),
    }
    adapter_info = {"total_layers": adapter.total_layers, "total_stages": adapter.total_stages}
    searcher = EvolutionSearcher(search_config, adapter_info)

    eval_config_local = dict(eval_config)
    eval_config_local["total_layers"] = adapter.total_layers
    eval_config_local["total_stages"] = adapter.total_stages
    eval_config_local["teacher_pipeline"] = teacher_pipeline

    evaluator = ImprovedEvaluator(adapter, eval_config_local)
    _patch_sample_return_prompts(evaluator)
    print("[Stage1] ImprovedEvaluator configured successfully.")

    num_generations = int(getattr(GlobalConfig, "num_generations", 2))
    num_segments = 0  # ignored in step mode

    print(f"[Stage1] Starting NAS: generations={num_generations}, steps={num_steps}")
    population = searcher.init_population(num_steps=num_steps, num_segments=num_segments)
    evaluated_population = None

    gen_pbar = tqdm(range(num_generations), desc="Stage1 NAS Generations", dynamic_ncols=True)
    for gen in gen_pbar:
        gen_start = time.time()
        tqdm.write(f"\n--- Generation {gen + 1}/{num_generations} ---")
        evaluated_population = []

        pop_pbar = tqdm(
            enumerate(population),
            total=len(population),
            desc=f"Evaluating Pop (Gen {gen+1}/{num_generations})",
            dynamic_ncols=True,
        )

        for idx, g in pop_pbar:
            t0 = time.time()

            with suppress_output(bool(getattr(GlobalConfig, "suppress_solver_logs", True))):
                g_eval = evaluator.evaluate_genotype(g)

            evaluated_population.append(g_eval)
            dt = time.time() - t0

            m = g_eval.metrics or {}
            fid_v = m.get("fid", None)
            lat_v = m.get("latency", None)
            std_v = m.get("std", None)
            clip_v = m.get("clip_similarity", None)
            pen_v = m.get("text_alignment_penalty", None)
            fit_v = getattr(g_eval, "fitness", None)

            pop_pbar.set_postfix(
                fid=f"{float(fid_v):.3f}" if fid_v is not None else "na",
                lat=f"{float(lat_v):.3f}" if lat_v is not None else "na",
            )

            tqdm.write(
                f"[Gen {gen+1} | {idx+1}/{len(population)}] "
                f"fid={fid_v} lat={lat_v} std={std_v} clip={clip_v} penalty={pen_v} fitness={fit_v} "
                f"(elapsed={dt:.2f}s)"
            )

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_result_line(
                f"{ts}\tstage1\t{gen+1}\t{idx+1}\t{len(population)}\t"
                f"{fid_v}\t{lat_v}\t{std_v}\t{clip_v}\t{pen_v}\t{fit_v}"
            )

        searcher.evolve(evaluated_population)

        pareto_path = os.path.join(output_dir, f"pareto_gen_{gen+1}.pkl")
        with open(pareto_path, "wb") as f:
            pickle.dump(searcher.top_k, f)
        tqdm.write(f"[MO] Pareto elites saved to {pareto_path}")

        population = searcher.population
        gen_pbar.set_postfix(last_gen_sec=f"{time.time() - gen_start:.1f}s")

    elites = searcher.top_k if searcher.top_k else []
    if len(elites) > 0:
        g_star = sorted(elites, key=lambda g: float((g.metrics or {}).get("fid", float("inf"))))[0]
    else:
        if evaluated_population is None:
            raise RuntimeError("No evaluated_population available to select g_star.")
        g_star = sorted(evaluated_population, key=lambda g: float((g.metrics or {}).get("fid", float("inf"))))[0]

    print("\n--- NAS Finished ---")
    print(f"Chosen g_star (min fid among elites): {g_star}")
    if best_arch is not None:
        print(f"Chosen best_arch (training-free): {best_arch.key()}")

    # Save outputs
    g_star_path = os.path.join(output_dir, "g_star.pkl")
    with open(g_star_path, "wb") as f:
        pickle.dump(g_star, f)
    print(f"g_star saved to {g_star_path}")

    if best_arch is not None:
        arch_path = os.path.join(output_dir, "best_arch.pkl")
        with open(arch_path, "wb") as f:
            pickle.dump(best_arch, f)
        print(f"best_arch saved to {arch_path}")

    pareto_final_path = os.path.join(output_dir, "pareto_front.pkl")
    with open(pareto_final_path, "wb") as f:
        pickle.dump(elites, f)
    print(f"Pareto front saved to {pareto_final_path}")

    # distillation bundle (option-2 recommended)
    distill_target_path = os.path.join(output_dir, "distill_target.pkl")
    distill_target = {
        "best_arch": best_arch,
        "g_star": g_star,
        "u_schedule": g_star.u_schedule,
        "sampler_policy": g_star.sampler_policy,
        "segment_boundaries": g_star.segment_boundaries,
        "segment_policies": g_star.segment_policies,
        "metrics": g_star.metrics,
        "model_id": getattr(GlobalConfig, "model_id", None),
        "cfg_scale": float(getattr(GlobalConfig, "cfg_scale", 4.5)),
        "sampler_type": sampler_type,
    }
    with open(distill_target_path, "wb") as f:
        pickle.dump(distill_target, f)
    print(f"distill_target saved to {distill_target_path}")

    save_cfg_snapshot = bool(getattr(GlobalConfig, "save_config_snapshot", False))
    if save_cfg_snapshot:
        cfg_snapshot_path = os.path.join(output_dir, "run_config_snapshot.pkl")
        snapshot = {}
        for k in dir(GlobalConfig):
            if k.startswith("_"):
                continue
            try:
                snapshot[k] = getattr(GlobalConfig, k)
            except Exception:
                pass
        with open(cfg_snapshot_path, "wb") as f:
            pickle.dump(snapshot, f)
        print(f"config snapshot saved to {cfg_snapshot_path}")

    append_result_line(f"Run end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    append_result_line("=" * 120)
    print(f"[LOG] All per-genotype results appended to: {result_txt_path}")


if __name__ == "__main__":
    run_search()
