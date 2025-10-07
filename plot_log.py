import json
import matplotlib.pyplot as plt
import os

# === 修改成你的專案路徑 ===
ROOT = r"C:\Users\btlim\OneDrive\Desktop\brain_to_text"
OUT_DIR = os.path.join(ROOT, "Outputs")

# === 讀取訓練紀錄 ===
history_path = os.path.join(OUT_DIR, "history.json")

if not os.path.exists(history_path):
    print("❌ 找不到 history.json，請先執行 train_baseline.py 產生紀錄")
else:
    with open(history_path, "r") as f:
        hist = json.load(f)

    print("✅ 成功讀取 history.json：")
    print("train_loss =", hist["tr_loss"])
    print("val_loss   =", hist["va_loss"])
    print("val_acc    =", hist["va_acc"])

    # === 繪製曲線圖 ===
    plt.figure()
    plt.plot(hist["tr_loss"], label="Train loss")
    plt.plot(hist["va_loss"], label="Val loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline Training Curve")
    plt.show()  # 顯示圖，不存檔
