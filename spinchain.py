import cv2
import numpy as np
import torch
import time
from collections import deque

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/spinchain-yolo/weights/best.pt')
model.conf = 0.1
model.iou = 0.2
model.to('cuda')

# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

# ROI stability tracker (background subtraction)
class ROIStabilityTracker:
    def __init__(self, class_id):
        self.last_status = "Moving"
        self.last_roi = None
        self.stability_score = 0
        self.score_history = deque(maxlen=5)
        self.stable_count = 0
        self.last_pos = None
        self.class_id = class_id
        self.pos_history = deque(maxlen=8)

    def update(self, full_frame, current_center):
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

        if self.last_roi is None or self.last_roi.shape != gray.shape:
            self.last_roi = gray.copy()
            self.last_status = "Moving"
            return self.last_status

        diff = cv2.absdiff(cv2.GaussianBlur(self.last_roi, (3, 3), 0), gray)
        score = np.mean(diff)
        self.score_history.append(score)
        self.stability_score = np.mean(self.score_history)
        self.last_roi = gray.copy()

        if self.stability_score < 5.0:
            self.stable_count += 1
            if self.stable_count >= 3:
                self.last_status = "Stopped"
        else:
            self.stable_count = 0
            self.last_status = "Moving"

        return self.last_status

trackers = {}
next_id = 0

def match_existing_trackers(center, threshold=50):
    for tid, tracker in trackers.items():
        if tracker.last_pos is not None:
            last_pos = tracker.last_pos
            dist = ((last_pos[0] - center[0])**2 + (last_pos[1] - center[1])**2)**0.5
            if dist < threshold:
                return tid
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    current_ids = set()

    for i, (*xyxy, conf, cls) in enumerate(results.xyxy[0]):
        x1, y1, x2, y2 = map(int, xyxy)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        center = (cx, cy)

        matched_id = match_existing_trackers(center)
        if matched_id is not None:
            trackers[matched_id].last_pos = center
            status = trackers[matched_id].update(frame, center)
            current_ids.add(matched_id)
        else:
            trackers[next_id] = ROIStabilityTracker(int(cls))
            trackers[next_id].last_pos = center
            trackers[next_id].update(frame, center)
            current_ids.add(next_id)
            next_id += 1

    for tid in current_ids:
        tracker = trackers[tid]
        status = tracker.last_status
        pos = tracker.last_pos
        score = tracker.stability_score
        label = f"Bey {tid} ({model.names[tracker.class_id]}): {status} (Δ={score:.1f})"
        cv2.circle(frame, pos, 20, (0, 255, 0) if status == "Moving" else (0, 0, 255), 2)
        cv2.putText(frame, label, (pos[0] - 70, pos[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("ROI Stability Beyblade Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()