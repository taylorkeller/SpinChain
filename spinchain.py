import cv2
import numpy as np
import torch
import time
from collections import deque

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/spinchain-yolo/weights/best.pt')
model.conf = 0.1
model.iou = 0.05
model.to('cuda')

# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.9)
# Turn off autofocus (device dependent)
cap.set(39, 0)  # 39 is CV_CAP_PROP_AUTOFOCUS
cap.set(28, 30)  # 28 is CV_CAP_PROP_FOCUS, set to ~30 (adjust based on your camera)

# ROI tracker using optical flow + stability score
class ROIStabilityTracker:
    def __init__(self, class_id):
        self.last_status = "Moving"
        self.last_roi = None
        self.last_gray = None
        self.stability_score = float('inf')
        self.score_history = deque(maxlen=5)
        self.stable_count = 0
        self.last_pos = None
        self.class_id = class_id
        self.pos_history = deque(maxlen=12)
        self.stopped_frame_index = None  # For stop tracking

    def update(self, full_frame, current_center, current_frame_index):
        self.pos_history.append(current_center)
        smoothed_pos = np.mean(self.pos_history, axis=0).astype(int)
        x_center, y_center = smoothed_pos
        half_size = 60

        y1 = max(y_center - half_size, 0)
        y2 = min(y_center + half_size, full_frame.shape[0])
        x1 = max(x_center - half_size, 0)
        x2 = min(x_center + half_size, full_frame.shape[1])

        roi = full_frame[y1:y2, x1:x2]
        if roi.shape[0] < 32 or roi.shape[1] < 32:
            return "Moving"

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if self.last_gray is not None and self.last_gray.shape == gray.shape:
            flow = cv2.calcOpticalFlowFarneback(self.last_gray, gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_movement = np.mean(mag)
            self.score_history.append(mean_movement)
            self.stability_score = np.mean(self.score_history)
        else:
            self.stability_score = float('inf')

        self.last_gray = gray.copy()

        if self.last_status == "Stopped":
            dx = abs(current_center[0] - self.last_pos[0])
            dy = abs(current_center[1] - self.last_pos[1])
            if dx < 15 and dy < 15:
                return self.last_status
            else:
                self.last_status = "Moving"

        if self.stability_score < 0.30:
            self.stable_count += 1
            if self.stable_count >= 15:
                if self.last_status != "Stopped":
                    self.stopped_frame_index = current_frame_index
                self.last_status = "Stopped"
        else:
            self.stable_count = 0
            self.last_status = "Moving"

        return self.last_status

trackers = {}
next_id = 0
stop_by_class = {}
frame_index = 0

def match_existing_trackers(center, cls, threshold=80):
    best_match = None
    best_dist = float('inf')
    for tid, tracker in trackers.items():
        if tracker.last_pos is not None and tracker.class_id == cls:
            last_pos = tracker.last_pos
            dist = ((last_pos[0] - center[0])**2 + (last_pos[1] - center[1])**2)**0.5
            if dist < threshold and dist < best_dist:
                best_dist = dist
                best_match = tid
    return best_match

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    current_ids = set()
    frame_index += 1

    for i, (*xyxy, conf, cls) in enumerate(results.xyxy[0]):
        x1, y1, x2, y2 = map(int, xyxy)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        center = (cx, cy)

        matched_id = match_existing_trackers(center, int(cls))
        if matched_id is not None:
            trackers[matched_id].last_pos = center
            status = trackers[matched_id].update(frame, center, frame_index)
            current_ids.add(matched_id)
        else:
            trackers[next_id] = ROIStabilityTracker(int(cls))
            trackers[next_id].last_pos = center
            trackers[next_id].update(frame, center, frame_index)
            current_ids.add(next_id)
            next_id += 1

    # Conflict filtering
    min_distance_apart = 80
    active_trackers = [(tid, trackers[tid].last_pos) for tid in current_ids]

    for i in range(len(active_trackers)):
        tid1, pos1 = active_trackers[i]
        for j in range(i + 1, len(active_trackers)):
            tid2, pos2 = active_trackers[j]
            if pos1 is None or pos2 is None:
                continue
            dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
            if dist < min_distance_apart:
                if trackers[tid1].stability_score > trackers[tid2].stability_score:
                    current_ids.discard(tid2)
                else:
                    current_ids.discard(tid1)

    if len(current_ids) > 0 and len(current_ids) < len(trackers):
        cv2.putText(frame, "⚠ Conflict detected - filtering overlap",
                    (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    for tid in current_ids:
        tracker = trackers[tid]
        status = tracker.last_status
        pos = tracker.last_pos
        score = tracker.stability_score
        name = model.names[tracker.class_id]
        label = f"{name}: {status} (Δ={score:.2f})"
        cv2.circle(frame, pos, 20, (0, 255, 0) if status == "Moving" else (0, 0, 255), 2)
        cv2.putText(frame, label, (pos[0] - 70, pos[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if status == "Stopped" and tracker.class_id not in stop_by_class:
            stop_by_class[tracker.class_id] = frame_index
            print(f"[Frame {frame_index}] ❌ Beyblade ({name}) has stopped.")

    cv2.imshow("ROI Stability Beyblade Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print stop order by class
print("\n📋 Final Stop Order by Class:")
for i, (cls_id, frame_idx) in enumerate(stop_by_class.items()):
    name = model.names[cls_id]
    print(f"{i+1}. {name} stopped at frame {frame_idx}")
