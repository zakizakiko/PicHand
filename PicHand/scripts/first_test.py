from ultralytics import YOLO
import cv2

# --- 1. 学習済みYOLOモデルをロード ---
# YOLOv8nは軽量・高速な汎用モデル（物体検出）
model = YOLO("yolov8n.pt")  # もしくは "yolov8s.pt" など

# --- 2. 推論 ---
# 例: 手札画像ファイルを指定（ユーザーの画像ファイル名に変更）
image_path = "/Users/nana/PicHand/data_pic/sample_1.png"  # <= あなたの画像パスに変更
results = model(image_path)

# --- 3. 検出結果の可視化 ---
annotated = results[0].plot()  # 検出枠を描画
cv2.imshow("Detected Cards (Generic)", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- 4. 検出されたバウンディングボックス情報の表示 ---
for box in results[0].boxes.xyxy:
    x1, y1, x2, y2 = box.tolist()
    print(f"Detected object: ({x1:.0f},{y1:.0f}) - ({x2:.0f},{y2:.0f})")
