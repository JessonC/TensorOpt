# main_train.py
import torch, os
from torch.utils.data import DataLoader
from dataset import MIMODataset
from models.mimo_model import MIMOJointNet
from utils.metrics import mse

device = torch.device("cuda")
dataset = MIMODataset("data/mimo_data.npz")
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = MIMOJointNet(Nt=4, Nr=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    total_loss = 0
    for Y, X, _ in loader:
        Y, X = Y.to(device), X.to(device)
        pred_X = model(Y)
        loss = mse(pred_X, X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, MSE Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "models/mimo_model.pth")
