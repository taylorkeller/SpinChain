import cv2
import numpy as np
import torch
import time
from collections import deque
import os
import blockchain_logger  # must define record_match(...)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/spinchain-yolo/weights/best.pt')
model.conf = 0.1
model.iou = 0.05
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.9)
cap.set(39, 0)   # Disable autofocus
cap.set(28, 30)  # Manual focus

class ROIStabilityTracker:
    def __init__(self, class_id):
        self.class_id = class_id
        self.last_status = "Moving"
        self.last_pos = None
        self.last_gray = None
        self.stability_score = float('inf')
        self.score_history = deque(maxlen=5)
        self.stable_count = 0
        self.pos_history = deque(maxlen=12)
        self.stopped_frame_index = None

    def update(self, frame, center, frame_idx):
        self.pos_history.append(center)
        smoothed = np.mean(self.pos_history, axis=0).astype(int)
        x, y = smoothed
        half = 60
        roi = frame[max(y - half, 0):min(y + half, frame.shape[0]),
                    max(x - half, 0):min(x + half, frame.shape[1])]
        if roi.shape[0] < 32 or roi.shape[1] < 32:
            return "Moving"

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if self.last_gray is not None and self.last_gray.shape == gray.shape:
            flow = cv2.calcOpticalFlowFarneback(self.last_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            self.score_history.append(np.mean(mag))
            self.stability_score = np.mean(self.score_history)
        else:
            self.stability_score = float('inf')

        self.last_gray = gray.copy()

        if self.last_status == "Stopped":
            dx = abs(center[0] - self.last_pos[0])
            dy = abs(center[1] - self.last_pos[1])
            if dx < 15 and dy < 15:
                return self.last_status
            else:
                self.last_status = "Moving"

        if self.stability_score < 0.30:
            self.stable_count += 1
            if self.stable_count >= 15:
                if self.last_status != "Stopped":
                    self.stopped_frame_index = frame_idx
                self.last_status = "Stopped"
        else:
            self.stable_count = 0
            self.last_status = "Moving"

        return self.last_status

trackers = {}
stop_by_class = {}
last_seen_frame = {}
next_id = 0
frame_index = 0
AUTO_SHUTDOWN_FRAMES = 180
MAX_MISSING_FRAMES = 180
shutdown_triggered = False
shutdown_start_frame = None

def match_tracker(center, cls, threshold=80):
    best, dist = None, float('inf')
    for tid, t in trackers.items():
        if t.class_id == cls and t.last_pos is not None:
            d = np.linalg.norm(np.array(t.last_pos) - np.array(center))
            if d < threshold and d < dist:
                best, dist = tid, d
    return best

# 🛠 Store actual stopped class IDs in order
final_stopped_ids = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_index += 1
    current_ids = set()
    results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cls_id = int(cls)
        center = (cx, cy)

        matched = match_tracker(center, cls_id)
        if matched is not None:
            tracker = trackers[matched]
        else:
            tracker = ROIStabilityTracker(cls_id)
            trackers[next_id] = tracker
            matched = next_id
            next_id += 1

        tracker.last_pos = center
        status = tracker.update(frame, center, frame_index)
        current_ids.add(matched)
        last_seen_frame[cls_id] = frame_index

        if status == "Stopped" and cls_id not in stop_by_class:
            stop_by_class[cls_id] = frame_index
            final_stopped_ids.append(cls_id)
            print(f"[Frame {frame_index}] ❌ {model.names[cls_id]} has stopped.")

    # Check for lost Beyblades
    for cid, last_seen in last_seen_frame.items():
        if cid not in stop_by_class and frame_index - last_seen > MAX_MISSING_FRAMES:
            stop_by_class[cid] = frame_index
            final_stopped_ids.append(cid)
            print(f"[Frame {frame_index}] ❌ {model.names[cid]} lost for 3s — declared stopped.")

    if not shutdown_triggered and len(stop_by_class) > 0:
        shutdown_triggered = True
        shutdown_start_frame = frame_index

    if shutdown_triggered and frame_index - shutdown_start_frame >= AUTO_SHUTDOWN_FRAMES:
        print("\n🛑 Auto-shutdown triggered.")
        break

    for tid in current_ids:
        t = trackers[tid]
        label = f"{model.names[t.class_id]}: {t.last_status} (Δ={t.stability_score:.2f})"
        color = (0, 255, 0) if t.last_status == "Moving" else (0, 0, 255)
        if t.last_pos:
            cv2.circle(frame, t.last_pos, 20, color, 2)
            cv2.putText(frame, label, (t.last_pos[0] - 70, t.last_pos[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("ROI Stability Beyblade Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 📋 Print final stop order
print("\n📋 Stop Order:")
for i, (cls_id, frame_idx) in enumerate(stop_by_class.items()):
    print(f"{i+1}. {model.names[cls_id]} stopped at frame {frame_idx}")

# 🧠 Determine match result
if len(final_stopped_ids) >= 2:
    id1, id2 = final_stopped_ids[:2]
    name1 = model.names[id1]
    name2 = model.names[id2]

    if id1 in stop_by_class and id2 in stop_by_class:
        winner = name2 if stop_by_class[id1] < stop_by_class[id2] else name1
        loser = name1 if winner == name2 else name2
        tie = "No"
    elif id1 in stop_by_class:
        winner, loser, tie = name2, name1, "No"
    elif id2 in stop_by_class:
        winner, loser, tie = name1, name2, "No"
    else:
        winner = loser = "None"
        tie = "Yes"

    print(f"\n🏁 Submitting result: {name1} vs {name2} | Winner: {winner} | Loser: {loser} | Tie: {tie}")
    tx_hash = blockchain_logger.record_match(name1, name2, winner, loser, tie)
    print(f"✅ Submitted match to blockchain: {tx_hash}")
