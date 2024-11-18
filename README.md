# Running Benchmarks

## Requirements
Tested with PyTorch 2.4.0, NCCL 2.20.5

Run the following command on each machine
```sh
python3 -m torch.distributed.run --nproc_per_node=<num_gpus_per_node> --nnodes=<num_nodes> --node_rank=<node_rank> --master_addr=<addr> -m benchmark_gpu_bandwidth
```
