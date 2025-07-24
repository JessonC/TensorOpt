# generate_data.py
import numpy as np
import os
os.makedirs("data", exist_ok=True)
def generate_mimo_data(Nt, Nr, num_samples, SNR_dB):
    X = (np.random.randn(num_samples, Nt) + 1j*np.random.randn(num_samples, Nt))/np.sqrt(2)
    H = (np.random.randn(num_samples, Nr, Nt) + 1j*np.random.randn(num_samples, Nr, Nt))/np.sqrt(2)
    noise_sigma = 10 ** (-SNR_dB / 20)
    Y = np.einsum('bnm,bm->bn', H, X) + noise_sigma*(np.random.randn(num_samples, Nr)+1j*np.random.randn(num_samples, Nr))/np.sqrt(2)
    return X, H, Y

if __name__ == "__main__":
    X, H, Y = generate_mimo_data(4, 8, 10000, 20)
    np.savez("data/mimo_data.npz", X=X, H=H, Y=Y)
