"""Utilities for distributed training across multiple GPUs."""

import os
from functools import lru_cache

import torch
import torch.distributed as dist


def get_rank() -> int:
    """Get the rank of the current process.

    Returns:
        Process rank (0 if not in distributed mode).
    """
    if dist.is_initialized():
        return dist.get_rank()

    # Check environment variables (set by torchrun/SLURM)
    for var in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if var in os.environ:
            return int(os.environ[var])

    return 0


def get_local_rank() -> int:
    """Get the local rank of the current process on this node.

    Returns:
        Local rank (0 if not in distributed mode).
    """
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])

    return get_rank()


def get_world_size() -> int:
    """Get the total number of processes.

    Returns:
        World size (1 if not in distributed mode).
    """
    if dist.is_initialized():
        return dist.get_world_size()

    # Check environment variables
    for var in ("WORLD_SIZE", "SLURM_NTASKS"):
        if var in os.environ:
            return int(os.environ[var])

    return 1


def is_main_process() -> bool:
    """Check if this is the main (rank 0) process.

    Use this to guard logging, checkpointing, and other
    operations that should only happen once.

    Returns:
        True if rank is 0.
    """
    return get_rank() == 0


def is_distributed() -> bool:
    """Check if running in distributed mode.

    Returns:
        True if world_size > 1.
    """
    return get_world_size() > 1


def setup_distributed(backend: str = "nccl") -> None:
    """Initialize the distributed process group.

    Should be called at the start of training when using multiple GPUs.
    Uses environment variables set by torchrun or SLURM.

    Args:
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU).
    """
    if dist.is_initialized():
        return

    if not is_distributed():
        return

    # Initialize process group
    dist.init_process_group(backend=backend)

    # Set device for this process
    local_rank = get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def cleanup_distributed() -> None:
    """Clean up the distributed process group.

    Should be called at the end of training.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronize all processes.

    Blocks until all processes reach this point.
    No-op if not in distributed mode.
    """
    if dist.is_initialized():
        dist.barrier()


def broadcast_object(obj: object, src: int = 0) -> object:
    """Broadcast a Python object from src rank to all other ranks.

    Args:
        obj: Object to broadcast (only needs to be set on src rank).
        src: Source rank.

    Returns:
        The broadcasted object on all ranks.
    """
    if not dist.is_initialized():
        return obj

    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce a tensor by averaging across all processes.

    Args:
        tensor: Tensor to reduce.

    Returns:
        Reduced tensor (averaged across all processes).
    """
    if not dist.is_initialized():
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / get_world_size()
    return tensor


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    """Get the appropriate device for this process.

    In distributed mode, returns the device for the local rank.
    Otherwise, returns CUDA if available, else CPU.

    Returns:
        torch.device for this process.
    """
    if torch.cuda.is_available():
        if is_distributed():
            return torch.device(f"cuda:{get_local_rank()}")
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
