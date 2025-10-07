import torch, os
ROOT = r"C:\Users\btlim\OneDrive\Desktop\brain_to_text"
probs = torch.load(os.path.join(ROOT, "Outputs", "phenome_probs.pt"))  # (B, T, C)
print("probs shape:", tuple(probs.shape))
print("min/max:", probs.min().item(), probs.max().item())
# 可存成 .npy 給語言模組同學
import numpy as np
np.save(os.path.join(ROOT, "Outputs", "phenome_probs.npy"), probs.numpy())
print("saved:", os.path.join(ROOT, "Outputs", "phenome_probs.npy"))
