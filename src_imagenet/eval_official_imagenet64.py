import argparse
import json
import os
import shlex
import subprocess
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVALUATOR = os.path.join(PROJECT_ROOT, "examples", "guided_diffusion", "evaluations", "evaluator.py")
PROBE_MARKER = "__OFFICIAL_EVAL_TF_PROBE__="


def _format_cmd(cmd):
    return " ".join(shlex.quote(part) for part in cmd)


def _probe_tensorflow_runtime(python_exe: str):
    probe_code = f"""
import json
import sys

try:
    import tensorflow as tf
    build_info = tf.sysconfig.get_build_info()
    result = {{
        "ok": True,
        "python": sys.executable,
        "tf_version": getattr(tf, "__version__", None),
        "is_cuda_build": bool(build_info.get("is_cuda_build", False)),
        "cuda_version": build_info.get("cuda_version"),
        "cudnn_version": build_info.get("cudnn_version"),
        "gpu_devices": [dev.name for dev in tf.config.list_physical_devices("GPU")],
    }}
except Exception as exc:
    result = {{
        "ok": False,
        "python": sys.executable,
        "error": repr(exc),
    }}

print({PROBE_MARKER!r} + json.dumps(result, ensure_ascii=True))
"""
    proc = subprocess.run(
        [python_exe, "-c", probe_code],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    combined = []
    if proc.stdout:
        combined.extend(proc.stdout.splitlines())
    if proc.stderr:
        combined.extend(proc.stderr.splitlines())

    payload = None
    for line in reversed(combined):
        if line.startswith(PROBE_MARKER):
            payload = line[len(PROBE_MARKER) :]
            break

    if payload is None:
        details = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            "Failed to inspect the TensorFlow evaluator runtime. "
            f"python={python_exe!r} returncode={proc.returncode}. {details}"
        )

    result = json.loads(payload)
    result["returncode"] = proc.returncode
    result["stdout"] = proc.stdout
    result["stderr"] = proc.stderr
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate samples against the official guided-diffusion ImageNet64 reference batch."
    )
    parser.add_argument("--ref_batch", required=True, help="Path to VIRTUAL_imagenet64_labeled.npz")
    parser.add_argument("--sample_batch", required=True, help="Path to generated samples .npz")
    parser.add_argument(
        "--evaluator_python",
        default=os.environ.get("OFFICIAL_EVAL_PYTHON") or sys.executable,
        help="Python executable used to run the TensorFlow official evaluator.",
    )
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--require_gpu",
        action="store_true",
        help="Refuse to continue when no GPU is visible to the TensorFlow evaluator.",
    )
    parser.add_argument(
        "--calculate_latency",
        type=bool,
        default=False,
        help="Whether to calculate latency",
    )
    args = parser.parse_args()

    ref_batch = os.path.abspath(args.ref_batch)
    sample_batch = os.path.abspath(args.sample_batch)
    evaluator_python = args.evaluator_python

    if not os.path.isfile(ref_batch):
        raise FileNotFoundError(f"Reference batch not found: {ref_batch}")
    if not os.path.isfile(sample_batch):
        raise FileNotFoundError(f"Sample batch not found: {sample_batch}")
    if not os.path.isfile(EVALUATOR):
        raise FileNotFoundError(f"Evaluator script not found: {EVALUATOR}")

    runtime = _probe_tensorflow_runtime(evaluator_python)
    gpu_devices = runtime.get("gpu_devices") or []

    print("[src_imagenet] running official evaluator")
    print(f"[src_imagenet] ref_batch={ref_batch}")
    print(f"[src_imagenet] sample_batch={sample_batch}")
    print(f"[src_imagenet] evaluator_python={evaluator_python}")
    print(
        "[src_imagenet] tensorflow="
        f"{runtime.get('tf_version')} cuda_build={runtime.get('cuda_version')} "
        f"cudnn_build={runtime.get('cudnn_version')} gpu_devices={gpu_devices}"
    )
    if not runtime.get("ok", False):
        print(f"[src_imagenet] TensorFlow runtime probe failed: {runtime.get('error')}")
        raise SystemExit(2)
    if not gpu_devices:
        print("[src_imagenet] No GPU is visible to the TensorFlow evaluator runtime.")
        if args.require_gpu and not args.allow_cpu:
            print("[src_imagenet] Refusing to continue on CPU because --require_gpu was set.")
            raise SystemExit(2)
        print("[src_imagenet] Continuing on CPU.")

    cmd = [evaluator_python, EVALUATOR, ref_batch, sample_batch, "--calculate_latency", str(args.calculate_latency)]
    print(f"[src_imagenet] cmd={_format_cmd(cmd)}")
    raise SystemExit(subprocess.call(cmd, cwd=PROJECT_ROOT))


if __name__ == "__main__":
    main()
