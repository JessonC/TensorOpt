# models/classical.py
import torch

def ls_channel_estimation(Y, X):
    """
    输入:
      Y: [B, Nr]  complex  （含导频的接收向量）
      X: [B, Nt]  complex  （对应发送导频）
    输出:
      H_ls: [B, Nr, Nt] complex
    """
    # 分子: y * conj(x) 的外积
    H_num = Y.unsqueeze(2) * X.conj().unsqueeze(1)         # [B, Nr, Nt]
    # 分母: |x|^2
    H_den = (X.real**2 + X.imag**2).unsqueeze(1)           # [B, 1, Nt]
    H_ls  = H_num / H_den                                  # [B, Nr, Nt]
    return H_ls

def zf_detection(H_est, Y):
    """
    ZF: s_hat = (H^H H)^{-1} H^H y
    H_est : [B, Nr, Nt] complex
    Y     : [B, Nr]     complex
    返回   : [B, Nt]     complex
    """
    H_h = H_est.conj().transpose(-1, -2)                   # [B, Nt, Nr]
    # 计算 (H^H H)^{-1} H^H
    W_zf = torch.linalg.pinv(H_est)                        # [B, Nt, Nr]
    s_hat = (W_zf @ Y.unsqueeze(-1)).squeeze(-1)           # [B, Nt]
    return s_hat
