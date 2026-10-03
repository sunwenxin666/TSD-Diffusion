"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
# from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE). 
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def setup_dist(distributed=None, backend=None, init_method="env://"):
    """
    Setup a distributed process group.
    If no distributed environment variables are present, fall back to
    single-process execution by default.
    """
    if is_dist_avail_and_initialized():
        return
    if distributed is None:
        distributed = all(
            key in os.environ for key in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
        )
    if not distributed:
        return
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

    # comm = MPI.COMM_WORLD
    backend = backend or ("gloo" if not th.cuda.is_available() else "nccl")
    if "MASTER_ADDR" not in os.environ:
        if backend == "gloo":
            hostname = "localhost"
        else:
            hostname = socket.gethostbyname(socket.getfqdn())
        os.environ["MASTER_ADDR"] = hostname
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(_find_free_port())
    dist.init_process_group(backend=backend, init_method=init_method)


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        rank = dist.get_rank() if is_dist_avail_and_initialized() else 0
        return th.device(f"cuda:{rank % GPUS_PER_NODE}")
        # return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()

    return th.load(io.BytesIO(data), **kwargs)
    # chunk_size = 2 ** 30  # MPI has a relatively small size limit
    # if MPI.COMM_WORLD.Get_rank() == 0:
    #     with bf.BlobFile(path, "rb") as f:
    #         data = f.read()
    #     num_chunks = len(data) // chunk_size
    #     if len(data) % chunk_size:
    #         num_chunks += 1
    #     MPI.COMM_WORLD.bcast(num_chunks)
    #     for i in range(0, len(data), chunk_size):
    #         MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    # else:
    #     num_chunks = MPI.COMM_WORLD.bcast(None)
    #     data = bytes()
    #     for _ in range(num_chunks):
    #         data += MPI.COMM_WORLD.bcast(None)

    # return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if not is_dist_avail_and_initialized():
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
