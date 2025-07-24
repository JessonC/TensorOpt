# mapping/optimized.py
import torch
from torch.cuda.amp import autocast

@torch.no_grad()
def ls_channel_estimation_opt(Y, X):
    """
    完全向量化 + Half 精度。
    Y: [B, Nr]  X: [B, Nt]
    公式: Ĥ = (Y x* ) / |x|²   —— 同时算完 batch 里所有样本
    """
    with autocast():                             # AMP 半精度
        num = Y.unsqueeze(2) * X.conj().unsqueeze(1)       # [B, Nr, Nt]
        den = (X.real**2 + X.imag**2).unsqueeze(1)         # [B, 1, Nt]
        H_est = num / den
    return H_est

@torch.no_grad()
def zf_detection_opt(H_est, Y):
    """
    Batch 版 ZF:   ŝ = pinv(H)  · y
    torch.linalg.pinv 支持 batch；利用 Tensor Core.
    """
    with autocast():
        W_zf = torch.linalg.pinv(H_est)           # [B, Nt, Nr]
        s_hat = (W_zf @ Y.unsqueeze(-1)).squeeze(-1)  # [B, Nt]
    return s_hat
