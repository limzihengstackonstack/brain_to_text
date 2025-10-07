# train_baseline.py
# ------------------
# A minimal, runnable baseline:
# - uses synthetic (fake) data to verify the whole training pipeline
# - LSTM -> Linear -> softmax over phoneme classes
# - saves best model and a sample of phoneme probabilities

import os, math, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===== Hyperparams (keep small so it runs everywhere) =====
NUM_CLASSES   = 40      # pretend 40 phoneme classes
SEQ_LEN       = 200     # length of each sequence chunk
TRAIN_SAMPLES = 800
VAL_SAMPLES   = 200
BATCH_SIZE    = 16
HIDDEN_DIM    = 128
EPOCHS        = 3
LR            = 1e-3
SEED          = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Paths =====
ROOT = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.path.join(ROOT, "Outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ===== Reproducibility =====
def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 在檔案頂部新增：
import os, numpy as np
from torch.utils.data import Dataset, DataLoader
from real_loader import load_signals_from_zip, load_labels_pkl, make_windows

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")
ZIP_PATH = os.path.join(DATA_DIR, "t15_copyTask_neuralData.zip")
PKL_PATH = os.path.join(DATA_DIR, "t15_copyTask.pkl")

class BrainDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()  # (N, T, C)
        self.y = torch.from_numpy(y).long()   # (N, T)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.y[i]           # (T, C), (T,)

def build_loaders_real(win=200, stride=200, val_ratio=0.2, batch_size=16):
    X_tc = load_signals_from_zip(ZIP_PATH)   # (T, C)

    # 先嘗試從 pkl 讀標籤；失敗就先用暫時標籤（全 0）讓流程跑通
    try:
        y_t  = load_labels_pkl(PKL_PATH)     # (T,)
        if len(y_t) < len(X_tc):
            # 長度不足先截齊；多於則截到 T
            y_t = y_t[:len(X_tc)]
        elif len(y_t) > len(X_tc):
            y_t = y_t[:len(X_tc)]
    except Exception as e:
        print("⚠️ 無法從 PKL 讀到逐時間標籤，先用暫時標籤（全 0）。原因：", e)
        y_t = np.zeros((X_tc.shape[0],), dtype=np.int64)

    X, y = make_windows(X_tc, y_t, win=win, stride=stride)  # (N,T,C), (N,T)
    N = X.shape[0]; n_val = max(1, int(N * val_ratio))
    Xtr, ytr = X[:-n_val], y[:-n_val]
    Xva, yva = X[-n_val:], y[-n_val:]
    tr = DataLoader(BrainDS(Xtr, ytr), batch_size=batch_size, shuffle=True)
    va = DataLoader(BrainDS(Xva, yva), batch_size=batch_size)
    return tr, va


# ===== Model =====
class BrainToPhonemeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,     # ← 這裡要等於特徵維度 C（例如 512）
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        assert x.dim() == 3, f"expect (B,T,C), got {x.shape}"
        h, _ = self.lstm(x)
        return self.fc(h)



# ===== Train & Eval =====
def train_one_epoch(model, loader, opt, crit):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        logits = model(xb)  # (B,T,C)
        loss = crit(logits.view(-1, NUM_CLASSES), yb.view(-1))
        loss.backward()
        opt.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    total, correct, tokens = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = crit(logits.view(-1, NUM_CLASSES), yb.view(-1))
        total += loss.item() * xb.size(0)
        pred = logits.argmax(-1)
        correct += (pred == yb).sum().item()
        tokens += yb.numel()
    return total / len(loader.dataset), correct / tokens

def main():
    print(f"Device: {DEVICE}")
    set_seed(SEED)
    train_loader, val_loader = build_loaders_real()

    # 由一個 batch 自動偵測特徵維度 C
    xb, yb = next(iter(train_loader))
    FEAT_DIM = xb.shape[-1]  # 例如 512
    model = BrainToPhonemeLSTM(input_dim=FEAT_DIM,
                               hidden_dim=HIDDEN_DIM,
                               num_classes=NUM_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    hist = {"tr_loss": [], "va_loss": [], "va_acc": []}

    best = math.inf
    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, crit)
        va_loss, va_acc = evaluate(model, val_loader, crit)
        print(f"[Epoch {ep}] train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={va_acc:.4f}")

        hist["tr_loss"].append(tr_loss)
        hist["va_loss"].append(va_loss)
        hist["va_acc"].append(va_acc)
        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "baseline_best.pt"))

    # export a small batch of phoneme probabilities for the language teammate
    model.eval()
    xb, _ = next(iter(val_loader))
    xb = xb.to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(xb), dim=-1).cpu()   # (B,T,C)
    torch.save(probs, os.path.join(OUT_DIR, "phenome_probs.pt"))
    print("Saved:", os.path.join(OUT_DIR, "baseline_best.pt"),
          "and", os.path.join(OUT_DIR, "phenome_probs.pt"))

    print("Saved:", os.path.join(OUT_DIR, "baseline_best.pt"),
          "and", os.path.join(OUT_DIR, "phenome_probs.pt"))

    # --- NEW: 存訓練紀錄 + 畫圖 ---
    import json
    import matplotlib.pyplot as plt

    with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    plt.figure()
    plt.plot(hist["tr_loss"], label="Train loss")
    plt.plot(hist["va_loss"], label="Val loss")
    plt.legend()
    plt.xlabel("Epoch");
    plt.ylabel("Loss")
    plt.title("Baseline Training Curve")
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=150)
    plt.close()
    print("✅ Saved training curve:", os.path.join(OUT_DIR, "loss_curve.png"))

if __name__ == "__main__":
    main()


