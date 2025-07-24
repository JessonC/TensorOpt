# profiler.py
import torch

def profile_model(model, input):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        model(input)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
