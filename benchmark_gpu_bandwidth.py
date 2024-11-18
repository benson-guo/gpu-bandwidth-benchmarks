import torch
import torch.distributed as dist
import os
import contextlib
from utils.profile import extract_kernel_runtime, get_profiler_context, extract_memcpy_runtime
from utils.comm import dist_init, get_local_rank, get_global_rank, get_world_size, is_local_leader
import argparse

# Measure the send/receive bandwidth between two GPUs
def measure_bandwidth(rank, size, device, rank1, rank2, warmup_iterations=10, iterations=50, parallel=5):
    is_sender = rank == rank1
    is_receiver = rank == rank2
    if not is_sender and not is_receiver:
        tensors = []
    else:
        tensors = [torch.randn(size, device=device) for _ in range(parallel)]
    torch.cuda.synchronize()
    for _ in range(warmup_iterations):
        works = []
        for tensor in tensors:
            if is_sender:
                work = dist.isend(tensor, dst=rank2)
            elif is_receiver:
                work = dist.irecv(tensor, src=rank1)
            works.append(work)
        for work in works:
            work.wait()
    if is_sender:
        profiler_ctx = get_profiler_context()
    else:
        profiler_ctx = contextlib.nullcontext()
    dist.barrier()
    torch.cuda.synchronize()
    with profiler_ctx:
        for _ in range(iterations):
            works = []
            for tensor in tensors:
                if is_sender:
                    work = dist.isend(tensor, dst=rank2)
                elif is_receiver:
                    work = dist.irecv(tensor, src=rank1)
                works.append(work)
            for work in works:
                work.wait()
        dist.barrier()
        torch.cuda.synchronize()
    if not is_sender:
        return 0.0
    avg_comm_time = extract_kernel_runtime(num_iterations=iterations * parallel)
    print(f"Average communication time: {avg_comm_time:.2f} ms")
    if avg_comm_time == 0:
        print("Error: No communication kernels recorded", flush=True)
        bandwidth = 0.0
    else:
        bandwidth = size * tensor.element_size() / 1e9 / (avg_comm_time / 1000)  # in Gb/s
        print(f"Bandwidth: {bandwidth:.2f} Gb/s", flush=True)
    return bandwidth

def run_bandwidth_test(args, size):
    world_size = get_world_size()
    rank = get_global_rank()
    local_rank = get_local_rank()
    local_leader = is_local_leader()
    device = torch.device(f"cuda:{local_rank}")
    bandwidths = [[0 for _ in range(world_size)] for _ in range(world_size)]
    for i in range(world_size):
        for j in range(world_size):
            if i != j:
                if local_leader:
                    print(f"Running send/recv bandwidth test for size {size} between rank {i} and rank {j}")
                bandwidth = measure_bandwidth(rank, size, device, i, j, parallel=args.parallel)
                # all reduce bandwidth
                b_tensor = torch.tensor([bandwidth], device=device)
                dist.all_reduce(b_tensor)
                bandwidths[i][j] = b_tensor[0].item()
            # Sync between sends and receives
            dist.barrier()
    if local_leader:
        print("Bandwidths (Gb/s):")
        print("  ", end="")
        for j in range(world_size):
            print(f"{j:>8}", end="")
        print()
        for i in range(world_size):
            print(f"{i:>2} ", end="")
            for j in range(world_size):
                print(f"{bandwidths[i][j]:>8.2f}", end="")
            print(flush=True)
    dist.barrier()

def pcie_test(rank, size, device, rank1, warmup_iterations=10, iterations=10, cpu_to_gpu=True, pin_memory=True, all_gpus=False):
    gpu_tensor = torch.randn(size, device=device)
    cpu_tensor = torch.randn(size, pin_memory=pin_memory)
    torch.cuda.synchronize()

    is_leader = rank == rank1 or all_gpus
    for _ in range(warmup_iterations):
        if is_leader:
            if cpu_to_gpu:
                _ = cpu_tensor.to(device)
            else:
                cpu_tensor.copy_(gpu_tensor)
    out_dir = os.path.join(os.path.expanduser("~"), "profiler_tensorboard", f"gpu{rank}")
    if is_leader:
        profiler_ctx = get_profiler_context(out_dir=out_dir)
    else:
        profiler_ctx = contextlib.nullcontext()
    dist.barrier()
    torch.cuda.synchronize()
    with profiler_ctx:
        for _ in range(iterations):
            if is_leader:
                if cpu_to_gpu:
                    _ = cpu_tensor.to(device)
                else:
                    cpu_tensor.copy_(gpu_tensor)
        dist.barrier()
        torch.cuda.synchronize()
    if not is_leader:
        return 0.0
    avg_comm_time = extract_memcpy_runtime(num_iterations=iterations, trace_dir=out_dir)
    print(f"Average communication time (rank {rank}): {avg_comm_time:.2f} ms")
    if avg_comm_time == 0:
        print("Error: No communication kernels recorded", flush=True)
        bandwidth = 0.0
    else:
        bandwidth = size * cpu_tensor.element_size() / 1e9 / (avg_comm_time / 1000)  # in Gb/s
        print(f"Bandwidth: {bandwidth:.2f} GB/s", flush=True)
    return bandwidth

def run_pcie_test(size, cpu_to_gpu=True, pin_memory=True):
    world_size = get_world_size()
    rank = get_global_rank()
    local_rank = get_local_rank()
    local_leader = is_local_leader()
    device = torch.device(f"cuda:{local_rank}")
    bandwidths = [0.0 for _ in range(world_size)]
    for i in range(world_size):
        if local_leader:
            print(f"Running pcie bandwidth test for size {size} on rank {i}")
        bandwidth = pcie_test(rank, size, device, i, cpu_to_gpu=cpu_to_gpu, pin_memory=pin_memory)
        b_tensor = torch.tensor([bandwidth], device=device)
        dist.all_reduce(b_tensor)
        bandwidths[i] = b_tensor[0].item()
        # Sync between sends and receives
        dist.barrier()
    if local_leader:
        print("PCIE Bandwidths (GB/s):")
        for j in range(world_size):
            print(f"{j:>8}", end="")
        print()
        for i in range(world_size):
            print(f"{bandwidths[i]:>8.2f}", end="")
        print()
    dist.barrier()

def run_pcie_test_all(size, cpu_to_gpu=True, pin_memory=True):
    rank = get_global_rank()
    local_rank = get_local_rank()
    local_leader = is_local_leader()
    device = torch.device(f"cuda:{local_rank}")
    bandwidths = [0.0]
    bandwidth = pcie_test(rank, size, device, 0, cpu_to_gpu=cpu_to_gpu, pin_memory=pin_memory, all_gpus=True)
    b_tensor = torch.tensor([bandwidth], device=device)
    dist.all_reduce(b_tensor)
    bandwidths[0] = b_tensor[0].item()
    # Sync between sends and receives
    dist.barrier()
    if local_leader:
        print(f"Concurrent PCIE Bandwidth (GB/s): {bandwidths[0]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=5, help="number of parallel communication")
    parser.add_argument("--send_msg_size", type=int, default=10 ** 7, help="size of send/recv message")
    parser.add_argument("--pcie_msg_size", type=int, default=10 ** 8, help="size of send/recv message")
    args = parser.parse_args()
    dist_init()
    local_leader = is_local_leader()
    for size in [args.send_msg_size]:
        if local_leader:
            print(f"Running bandwidth test for size {size}")
        # Run bandwidth test between all pairs of GPUs
        run_bandwidth_test(args, size)
    for size in [args.pcie_msg_size]:
        if local_leader:
            print(f"Running pcie test for size {size}, cpu_to_gpu=True, pin_memory=True, all_gpus=True")
        run_pcie_test_all(size, cpu_to_gpu=True, pin_memory=True)
        if local_leader:
            print(f"Running pcie test for size {size}, cpu_to_gpu=True, pin_memory=True")
        run_pcie_test(size, cpu_to_gpu=True, pin_memory=True)
        if local_leader:
            print(f"Running pcie test for size {size}, cpu_to_gpu=False, pin_memory=True")
        run_pcie_test(size, cpu_to_gpu=False, pin_memory=True)
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
                