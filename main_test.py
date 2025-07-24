# main_test.py
import torch
from dataset import MIMODataset
from torch.utils.data import DataLoader
from models.mimo_model import MIMOJointNet
from models.classical import ls_channel_estimation, zf_detection
from utils.metrics import mse, evm
from utils.profiler import profile_model

device = torch.device("cuda")
dataset = MIMODataset("data/mimo_data.npz")
loader = DataLoader(dataset, batch_size=128)

model = MIMOJointNet(Nt=4, Nr=8).to(device)
model.load_state_dict(torch.load("models/mimo_model.pth"))
model.eval()

with torch.no_grad():
    total_mse_ai = 0
    total_mse_classic = 0
    for Y, X, H in loader:
        Y, X, H = Y.to(device), X.to(device), H.to(device)

        # AI方法
        pred_X_ai = model(Y)
        total_mse_ai += mse(pred_X_ai, X).item()

        # 经典方法
        H_est = ls_channel_estimation(Y, X)
        pred_X_classic = zf_detection(H_est, Y)
        total_mse_classic += mse(pred_X_classic, X).item()

    print(f"AI Method MSE: {total_mse_ai/len(loader):.4f}")
    print(f"Classic Method MSE: {total_mse_classic/len(loader):.4f}")

# Profiler分析
Y_sample, _, _ = next(iter(loader))
profile_model(model, Y_sample.to(device))
