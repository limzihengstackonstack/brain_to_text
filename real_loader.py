# real_loader.py
import os, zipfile, pickle, numpy as np
from scipy.io import loadmat

def _pick_best_key_from_mat(mat):
    cand = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        try:
            arr = np.asarray(v)
            if arr.ndim >= 2 and arr.size > 0:
                cand.append((arr.size, k))
        except Exception:
            pass
    if not cand:
        raise ValueError("在 .mat 內找不到合適的數據 key")
    cand.sort(reverse=True)
    return cand[0][1]

def load_signals_from_zip(zip_path):
    """回傳 X_tc: (T, C) float32"""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"找不到 zip 檔：{zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
        print("📦 ZIP 內檔案（前 10 個）:", names[:10])

        # 先優先挑 hdf5 的 train 檔，再退而求其次
        pick = None
        for nm in names:
            if nm.lower().endswith(".hdf5") and "data_train" in nm.lower():
                pick = nm;
                break
        if pick is None:
            for ext in [".hdf5", ".mat", ".npy", ".npz", ".csv", ".txt"]:
                for nm in names:
                    if nm.lower().endswith(ext):
                        pick = nm;
                        break
                if pick: break
        if pick is None:
            raise ValueError("ZIP 內沒有支援的副檔名")

        out_dir = os.path.dirname(zip_path)
        # 👉 保留 ZIP 內的子資料夾層級，避免路徑錯誤
        out_path = os.path.normpath(os.path.join(out_dir, pick))
        if not os.path.exists(out_path):
            z.extract(pick, out_dir)

    ext = os.path.splitext(out_path)[1].lower()
    print(f"✅ 選用檔案：{pick}（{ext}） → {out_path}")

    # === 根據副檔名讀取資料 ===
    if ext in ['.zip']:
        # TODO: 這裡處理 ZIP 的邏輯（如果有的話）
        pass

    elif ext in ['.hdf5', '.h5']:
        import h5py

        def pick_largest_2d_from_group(g):
            best_name, best_size, best_shape = None, -1, None
            stack = [g]
            while stack:
                gg = stack.pop()
                for name, obj in gg.items():
                    if isinstance(obj, h5py.Group):
                        stack.append(obj)
                    elif isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                        size = int(np.prod(obj.shape))
                        if size > best_size:
                            best_name, best_size, best_shape = obj.name, size, obj.shape
            return best_name, best_shape  # 可能為 (None, None)

        with h5py.File(out_path, "r") as f:
            top_keys = list(f.keys())
            print("📂 HDF5 內部 keys:", top_keys[:20])

            arrays = []
            chosen = []
            for tk in top_keys:
                obj = f[tk]
                if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                    arr = np.array(obj).astype(np.float32)
                    src_name = obj.name
                elif isinstance(obj, h5py.Group):
                    name, shp = pick_largest_2d_from_group(obj)
                    if name is None:
                        print(f"  ⚠️ {tk}: 找不到 2D dataset，跳過")
                        continue
                    arr = np.array(f[name]).astype(np.float32)
                    src_name = name
                else:
                    print(f"  ⚠️ {tk}: 不是 group 或 2D dataset，跳過")
                    continue

                arr = np.squeeze(arr)
                if arr.ndim != 2:
                    print(f"  ⚠️ {tk}: shape={arr.shape} 不是 2D，跳過")
                    continue

                # 期望 (T, C)，若像 (C, T) 就轉置（通常 T >> C）
                if arr.shape[0] <= arr.shape[1]:
                    arr = arr.T

                arrays.append(arr)
                chosen.append((tk, src_name, arr.shape))
                print(f"  ✅ {tk}: 使用 {src_name} shape={arr.shape}")

                if not arrays:
                    raise ValueError("HDF5 內層沒找到可用 2D 資料集")

                # --- 統一通道數（這裡貼方案2）---
                channels = [a.shape[1] for a in arrays]
                from collections import Counter
                target_C = Counter(channels).most_common(1)[0][0]

                kept = [a.astype(np.float32) for a in arrays if a.shape[1] == target_C]
                dropped = len(arrays) - len(kept)
                arrays = kept
                print(f"🔧 保留 C={target_C} 的 trial={len(arrays)}，丟棄 {dropped} 個通道數不同的 trial")

                # --- 串接所有 trial ---
                X = np.concatenate(arrays, axis=0)
                print("📐 串接後 X shape:", X.shape)

            return X  # (T, C)


def load_labels_pkl(pkl_path):
    """從 pkl 中取出 per-timestep 的音素 id 序列 y_t: (T,) int64"""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"找不到 pkl 檔：{pkl_path}")
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        # 依據你們 pkl 的實際 keys 調整這裡的選單
        for key in ["phoneme_ids", "phonemes", "labels", "y"]:
            if key in obj:
                y_t = np.asarray(obj[key]).astype(np.int64).squeeze()
                print(f"✅ 使用 pkl key：{key}；y_t shape={y_t.shape}")
                return y_t
        raise KeyError(f"在 pkl 裡找不到音素序列的 key，實際 keys={list(obj.keys())}")
    else:
        # pkl 不是 dict 的情況：嘗試轉成一維
        y_t = np.asarray(obj).astype(np.int64).squeeze()
        if y_t.ndim != 1:
            raise ValueError(f"pkl 內容不是 1 維，shape={y_t.shape}")
        print(f"✅ 直接使用 pkl 陣列；y_t shape={y_t.shape}")
        return y_t

def make_windows(X_tc, y_t, win=200, stride=200):
    """把長序列切成視窗，回傳 (N,T,C) 與 (N,T)"""
    T, C = X_tc.shape
    if len(y_t) < T:
        raise ValueError(f"標籤長度 {len(y_t)} < 資料長度 {T}，請確認對齊")
    y_t = y_t[:T]
    xs, ys = [], []
    for s in range(0, T - win + 1, stride):
        e = s + win
        xs.append(X_tc[s:e]); ys.append(y_t[s:e])
    X = np.stack(xs).astype(np.float32)
    y = np.stack(ys).astype(np.int64)
    print("📦 切窗後 shapes:", X.shape, y.shape)
    return X, y
