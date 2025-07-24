# mapping/baseline.py
import torch

@torch.no_grad()
def ls_channel_estimation_baseline(Y, X):
    """
    Y: [B, Nr]  X: [B, Nt]  (complex)
    逐样本 for‑loop，显式 pinv —– 最低效但数值标准。
    """
    B, Nr = Y.shape
    Nt = X.shape[1]
    H_est = torch.zeros((B, Nr, Nt), dtype=Y.dtype, device=Y.device)
    for b in range(B):
        x_pinv = torch.linalg.pinv(X[b].unsqueeze(1))  # [Nt,1] 的伪逆
        H_est[b] = Y[b].unsqueeze(1) @ x_pinv          # [Nr, Nt]
    return H_est  # [B, Nr, Nt]

@torch.no_grad()
def zf_detection_baseline(H_est, Y):
    """
    逐样本 pinv –– 仍然是 for‑loop。
    """
    B = Y.shape[0]
    Nt = H_est.shape[2]
    s_hat = torch.zeros((B, Nt), dtype=Y.dtype, device=Y.device)
    for b in range(B):
        W = torch.linalg.pinv(H_est[b])  # [Nt, Nr]
        s_hat[b] = (W @ Y[b]).squeeze()
    return s_hat                         # [B, Nt]
