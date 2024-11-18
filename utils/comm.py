import torch
import torch.distributed as dist
import os

def get_gpu_name(local_rank=None):
    local_rank = local_rank if local_rank is not None else get_local_rank()
    gpu_name = torch.cuda.get_device_name(local_rank).lower().split(" ")[-1]
    return gpu_name

def dist_init():
    rank = get_global_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    gpu_name = get_gpu_name()
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Rank {rank} local rank {local_rank} world size {world_size} {gpu_name}")

    return world_size

def is_local_leader():
    """Return True if the current process is the local leader."""
    return get_local_rank() == 0

def get_global_rank():
    return int(os.environ["RANK"])


def get_local_world_size():
    gpus_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
    return gpus_per_node


def get_world_size():
    world_size = int(os.environ["WORLD_SIZE"])
    return world_size


def get_local_rank():
    return int(os.environ["LOCAL_RANK"])