import json
import os
import torch
import glob

def extract_kernel_runtime(num_iterations=1, trace_dir=None):
    """
    Extracts the average duration of kernel operations from a given trace file.
    """
    file_path = get_profiler_trace_path(trace_dir=trace_dir)
    with open(file_path, "r") as file:
        profiling_data = json.load(file)

    # Filter out "Memcpy DtoH (Device -> Pinned)" events and calculate durations
    event_count = {}
    total_duration = 0
    for event in profiling_data["traceEvents"]:
        if event.get("cat") != "kernel":
            continue
        event_name = event.get("name")
        if event_name not in event_count:
            event_count[event_name] = 0
        event_count[event_name] += 1
        total_duration += (
            event.get("dur", 0) / 1000
        )  # Convert from microseconds to milliseconds

    # Calculate average duration if there are any durations collected
    avg_duration = total_duration / num_iterations

    return avg_duration

def get_profiler_trace_path(trace_dir=None):
    if trace_dir is None:
        log_dir = os.path.join(os.path.expanduser("~"), "profiler_tensorboard/*")
    else:
        log_dir = os.path.join(trace_dir, "*")
    files = glob.glob(log_dir)
    # assume we are fetching most recent trace
    profiler_trace = max(files, key=os.path.getmtime)

    return profiler_trace

def get_profiler_context(out_dir=None, detailed_trace=False):
    if out_dir is None:
        out_dir = os.path.join(os.path.expanduser("~"), "profiler_tensorboard")
    os.makedirs(out_dir, exist_ok=True)
    handler = torch.profiler.tensorboard_trace_handler(out_dir)
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    return torch.profiler.profile(
        activities=activities,
        on_trace_ready=handler,
        profile_memory=detailed_trace,
        record_shapes=detailed_trace,
        with_stack=detailed_trace,
        with_modules=detailed_trace,
    )

def extract_memcpy_runtime(num_iterations=1, trace_dir=None):
    """
    Extracts the average duration of kernel operations from a given trace file.
    """
    file_path = get_profiler_trace_path(trace_dir=trace_dir)
    with open(file_path, "r") as file:
        profiling_data = json.load(file)

    # Filter out "Memcpy DtoH (Device -> Pinned)" events and calculate durations
    event_count = {}
    total_duration = 0
    for event in profiling_data["traceEvents"]:
        if event.get("cat") != "gpu_memcpy":
            continue
        event_name = event.get("name")
        if event_name not in event_count:
            event_count[event_name] = 0
        event_count[event_name] += 1
        total_duration += (
            event.get("dur", 0) / 1000
        )  # Convert from microseconds to milliseconds

    # Calculate average duration if there are any durations collected
    avg_duration = total_duration / num_iterations

    return avg_duration
