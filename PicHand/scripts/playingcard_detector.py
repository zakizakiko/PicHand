"""
detect_and_save_cards.py
----------------------------------------
Playing-card-detection (YOLOv5) 用
Mac + Anaconda環境で安全に動く推論スクリプト

📦 機能:
- YOLOv5モデルでカードを検出
- Matplotlibで可視化（OpenCVウィンドウを使用しない）
- 検出結果をJSONファイルとして保存
"""

import torch
import cv2
import json
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# --- ユーザー設定 ---
IMAGE_PATH = "hand.jpg"           # 📸 推論したい画像を指定
MODEL_PATH = "card_detector.pt"   # 🧠 geaxgxモデルの重みファイル

# --- 出力設定 ---
OUTPUT_JSON = "detected_cards.json"
CONF_THRESHOLD = 0.25  # 検出信頼度の閾値（0〜1）

# --- モデル読み込み ---
print("🔹 Loading YOLOv5 model...")
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH)
model.conf = CONF_THRESHOLD
print("✅ Model loaded.")

# --- 画像推論 ---
print(f"🔹 Running inference on {IMAGE_PATH} ...")
results = model(IMAGE_PATH)

# --- 検出結果のDataFrameを取得 ---
df = results.pandas().xyxy[0]
if len(df) == 0:
    print("⚠️ No cards detected.")
else:
    print(f"✅ Detected {len(df)} objects.")
    print(df[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]])

# --- 可視化（Matplotlibで表示）---
print("🔹 Displaying result...")
result_img = results.render()[0]
result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
plt.imshow(result_img)
plt.axis("off")
plt.title("Detected Playing Cards")
plt.show()

# --- JSON保存 ---
print("🔹 Saving JSON...")
detections = []
for _, row in df.iterrows():
    detections.append({
        "label": row["name"],
        "confidence": float(row["confidence"]),
        "bbox": {
            "xmin": float(row["xmin"]),
            "ymin": float(row["ymin"]),
            "xmax": float(row["xmax"]),
            "ymax": float(row["ymax"])
        }
    })

output = {
    "image": Path(IMAGE_PATH).name,
    "detections": detections,
    "timestamp": datetime.now().isoformat(timespec="seconds")
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"✅ Results saved to {OUTPUT_JSON}")
