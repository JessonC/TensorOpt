# mapping/benchmark.py
import torch, time, psutil, os
from pathlib import Path
from dataset import MIMODataset
from torch.utils.data import DataLoader
from mapping.baseline import ls_channel_estimation_baseline, zf_detection_baseline
from mapping.optimized import ls_channel_estimation_opt, zf_detection_opt
from utils.metrics import mse

device = torch.device("cuda")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATCH = 1024                 # 大 batch 展现批量优势
DATA = PROJECT_ROOT / "data" / "mimo_data.npz" # 你的 NPZ 数据

loader = DataLoader(MIMODataset(DATA), batch_size=BATCH)

@torch.no_grad()
def run_once(ls_fn, zf_fn):
    Y, X, _ = next(iter(loader))
    Y, X = Y.to(device), X.to(device)

    torch.cuda.synchronize()
    t0 = time.time()

    H_est = ls_fn(Y, X)          # [B, Nr, Nt]
    s_hat = zf_fn(H_est, Y)      # [B, Nt]

    torch.cuda.synchronize()
    dt = (time.time() - t0)*1000   # 毫秒
    err = mse(s_hat, X).item()

    mem = torch.cuda.max_memory_allocated()/1024**2  # MB
    torch.cuda.reset_peak_memory_stats()
    return dt, mem, err

print("Running baseline ...")
t_base, mem_base, err_base = run_once(ls_channel_estimation_baseline, zf_detection_baseline)
print("Running optimized ...")
t_opt,  mem_opt,  err_opt  = run_once(ls_channel_estimation_opt,      zf_detection_opt)

print("\n=== 结果对比 (batch={} 每次) ===".format(BATCH))
print(f"Baseline   : {t_base:7.2f} ms,  {mem_base:6.1f} MB,  MSE={err_base:.3e}")
print(f"Optimized  : {t_opt:7.2f} ms,  {mem_opt:6.1f} MB,  MSE={err_opt :.3e}")
speedup = t_base / t_opt
print(f"\n⚡ Speed‑up : ×{speedup:.1f}")
