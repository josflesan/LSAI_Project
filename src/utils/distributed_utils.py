import os
import torch
import torch.distributed as dist

def init_distributed():
    """
    Initialise the distributed environment.
    Assumes that the environment variables RANK, LOCAL_RANK, and WORLD_SIZE are set.
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    node_id = os.environ.get("SLURM_NODEID", "N/A")

    # Set the current device for this process
    torch.cuda.set_device(local_rank)

    # Initialise the process group with NCCL backend (requires NVIDIA GPUs)
    dist.init_process_group(backend="nccl")

    print(f"[Distributed Init] Rank {rank} initialized on {node_id} on GPU {local_rank}.")
    dist.barrier()
    if rank == 0:
        print(f"[Rank {rank}] All ranks ready!")
    
    return rank, local_rank, world_size

def compare_tensors(tensor1, tensor2, tol=1e-5, prefix=""):
    """
    Simple comparison of two tensors, printing basic difference statistics.

    Parameters:
        tensor1 (torch.Tensor): First tensor to compare
        tensor2 (torch.Tensor): Second tensor to compare
        tol (float): Tolerance for considering values as close
    """

    # Calculate differences
    abs_diff = (tensor1 - tensor2).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # Simple check if tensors are close
    is_close = torch.allclose(tensor1, tensor2, rtol=tol, atol=tol)

    # Print brief comparison summary
    rank = dist.get_rank() if dist.is_initialized() else 0
    prefix = f"[{prefix}]" if prefix else ""
    print(f"{prefix}[Rank {rank}] Tensors match: {is_close} | Max diff: {max_diff:.6e} | Mean diff: {mean_diff:.6e}", flush=True)
