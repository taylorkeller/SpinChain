import cv2
import numpy as np
import torch
import time
import os
import hashlib
import secrets
from collections import deque
from dotenv import load_dotenv
from eth_utils import keccak
from eth_abi import encode
from eth_abi.packed import encode_packed
from blockchain_logger import verify_shared_secret_hash
from blockchain_logger import contract
from blockchain_logger import request_challenge, record_match_with_hmac
import qrcode
from pyzbar.pyzbar import decode
import threading

# === Load .env ===
load_dotenv()

# === Settings ===
MIN_SEEN_FRAMES = 100
AUTO_SHUTDOWN_FRAMES = 180
MAX_MISSING_FRAMES = 180
MODEL_VERSION = "v1.0"
MODEL_PATH = "yolov5/runs/train/spinchain-yolo/weights/best.pt"

PRIVATE_KEY = os.getenv("PRIVATE_KEY")
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")

# === Load YOLOv5 model ===
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
model.conf = 0.1
model.iou = 0.05
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# === Webcam ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(39, 0)
cap.set(28, 30)

# === Secret and QR ===
ephemeral_secret = secrets.token_bytes(32)
secret_hash = keccak(ephemeral_secret)
print(f"🪪 Display this in QR form: {secret_hash.hex()}")

def render_challenge_qr(text):
    qr = qrcode.make(text)
    return np.array(qr.convert('RGB'))

qr_img = render_challenge_qr(secret_hash.hex())
qr_img = cv2.resize(qr_img, (100, 100))

def decode_embedded_qr(qr_img):
    decoded = decode(qr_img)
    if decoded:
        return decoded[0].data.decode().strip()
    return ""

# === QR State ===
qr_confirm_count = 0
challenge_verified = False
CHALLENGE_FRAMES_REQUIRED = 10
challenge = None

def submit_challenge_tx():
    global challenge
    print("🛰 Sending challenge to blockchain...")
    challenge = request_challenge(secret_hash)
    verify_shared_secret_hash(secret_hash)
    print("✅ Blockchain challenge complete.")

# === ROI Tracker ===
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

        if self.stability_score < 0.20:
            self.stable_count += 1
            if self.stable_count >= 15:
                if self.last_status != "Stopped":
                    self.stopped_frame_index = frame_idx
                self.last_status = "Stopped"
        else:
            self.stable_count = 0
            self.last_status = "Moving"

        return self.last_status

def match_tracker(center, name, threshold=80):
    best, dist = None, float('inf')
    for tid, t in trackers.items():
        if t.class_id == name and t.last_pos is not None:
            d = np.linalg.norm(np.array(t.last_pos) - np.array(center))
            if d < threshold and d < dist:
                best, dist = tid, d
    return best

# === State Variables ===
trackers = {}
stop_by_class = {}
last_seen_frame = {}
seen_frames_by_class = {}
next_id = 0
frame_index = 0
shutdown_triggered = False
shutdown_start_frame = None
telemetry_events = []
final_stopped_names = []

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame read failed.")
        time.sleep(0.5)
        continue

    frame_index += 1

    if not challenge_verified:
        decoded = decode_embedded_qr(qr_img)
        print(f"[QR Decode] Got: {decoded[:12]}... Expected: {secret_hash.hex()[:12]}...")

        if decoded == secret_hash.hex():
            qr_confirm_count += 1
            print(f"[QR] Confirm {qr_confirm_count}/{CHALLENGE_FRAMES_REQUIRED}")
        else:
            qr_confirm_count = 0

        if qr_confirm_count >= CHALLENGE_FRAMES_REQUIRED:
            print("✅ QR threshold met. Starting match tracking and blockchain thread...")
            threading.Thread(target=submit_challenge_tx, daemon=True).start()
            challenge_verified = True
            qr_confirm_count = 0
            continue

        frame[10:110, frame.shape[1] - 110:frame.shape[1] - 10] = qr_img
        overlay_text = f"CHALLENGE: {secret_hash.hex()[:12]}..."
        cv2.putText(frame, overlay_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("ROI Stability Beyblade Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # === Match tracking logic ===
    current_ids = set()
    results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        name = model.names[int(cls)]
        center = (cx, cy)

        matched = match_tracker(center, name)
        if matched is not None:
            tracker = trackers[matched]
        else:
            tracker = ROIStabilityTracker(name)
            trackers[next_id] = tracker
            matched = next_id
            next_id += 1

        tracker.last_pos = center
        status = tracker.update(frame, center, frame_index)
        current_ids.add(matched)
        last_seen_frame[name] = frame_index
        seen_frames_by_class[name] = seen_frames_by_class.get(name, 0) + 1

        if seen_frames_by_class[name] < MIN_SEEN_FRAMES:
            continue

        if status == "Stopped" and name not in stop_by_class:
            stop_by_class[name] = frame_index
            final_stopped_names.append(name)
            log = f"{name} stopped at frame {frame_index}"
            telemetry_events.append(log)
            print(f"[Frame {frame_index}] ❌ {log}")

    for cname, last_seen in last_seen_frame.items():
        if cname not in stop_by_class and frame_index - last_seen > MAX_MISSING_FRAMES:
            if seen_frames_by_class.get(cname, 0) >= MIN_SEEN_FRAMES:
                stop_by_class[cname] = frame_index
                final_stopped_names.append(cname)
                log = f"{cname} lost at frame {frame_index}"
                telemetry_events.append(log)
                print(f"[Frame {frame_index}] ❌ {log}")

    if not shutdown_triggered and len(stop_by_class) > 0:
        shutdown_triggered = True
        shutdown_start_frame = frame_index

    if shutdown_triggered and frame_index - shutdown_start_frame >= AUTO_SHUTDOWN_FRAMES:
        print("\n🛑 Auto-shutdown triggered.")
        break

    for tid in current_ids:
        t = trackers[tid]
        label = f"{t.class_id}: {t.last_status} (Δ={t.stability_score:.2f})"
        color = (0, 255, 0) if t.last_status == "Moving" else (0, 0, 255)
        if t.last_pos:
            cv2.circle(frame, t.last_pos, 20, color, 2)
            cv2.putText(frame, label, (t.last_pos[0] - 70, t.last_pos[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for name in seen_frames_by_class:
        print(f"[{frame_index}] 👁️ {name} seen {seen_frames_by_class[name]} frames")

    frame[10:110, frame.shape[1] - 110:frame.shape[1] - 10] = qr_img
    overlay_text = f"CHALLENGE: {secret_hash.hex()[:12]}..."
    cv2.putText(frame, overlay_text, (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("ROI Stability Beyblade Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Wait for blockchain challenge to complete if necessary
while challenge is None:
    print("⏳ Waiting for blockchain challenge to complete...")
    time.sleep(0.5)





# === Final stop order ===
print("\n📋 Stop Order:")
for i, (name, frame_idx) in enumerate(stop_by_class.items()):
    print(f"{i+1}. {name} stopped at frame {frame_idx}")

# === Win/Lose Conditions ===
if len(stop_by_class) == 1 and len(seen_frames_by_class) >= 2:
    stopped = next(iter(stop_by_class))
    possible_opponents = [name for name in seen_frames_by_class if name != stopped and seen_frames_by_class[name] >= MIN_SEEN_FRAMES]

    if possible_opponents:
        moving = possible_opponents[0]
        winner = moving
        loser = stopped
        tie = False
        name1, name2 = winner, loser
    else:
        print("⚠️ No valid opponent found.")
        exit()

elif len(stop_by_class) >= 2:
    name1, name2 = list(stop_by_class.keys())[:2]
    frame1 = stop_by_class[name1]
    frame2 = stop_by_class[name2]

    if frame1 == frame2:
        winner = loser = "None"
        tie = True
    elif frame1 < frame2:
        winner, loser = name2, name1
        tie = False
    else:
        winner, loser = name1, name2
        tie = False
else:
    print("⚠️ Not enough Beyblades tracked to determine a valid match.")
    exit()




# === Request challenge from blockchain using the secret hash
challenge = request_challenge(secret_hash)
onchain_hash = contract.functions.getSharedSecretHash(WALLET_ADDRESS).call()

# === Required for challenge to be requested and verified
time.sleep(3)
verify_shared_secret_hash(secret_hash)
with open(MODEL_PATH, "rb") as f:
    model_hash = hashlib.sha256(f.read()).digest()

def to_hex_string32(b):
    return ''.join(f'{x:02x}' for x in b)



# === Structured typed match hash
typed_data = encode(
    ["string", "bytes32", "string", "string", "bool"],
    [MODEL_VERSION, model_hash, winner, loser, tie]
)

match_hash = keccak(typed_data)
message = match_hash + challenge
packed = encode_packed(['bytes32', 'bytes32', 'bytes32'], [secret_hash,  match_hash, challenge])
leaves = telemetry_events  # use raw event strings

# === Submit match to blockchain 
print("📤 Telemetry being submitted:")
for i, leaf in enumerate(telemetry_events):
    print(f"  [{i}] {repr(leaf)}")




print(f"\n🏁 Submitting result: {name1} vs {name2} | Winner: {winner} | Loser: {loser} | Tie: {tie}")
tx_hash = record_match_with_hmac(
    model_version=MODEL_VERSION,
    model_hash=model_hash,
    winner=winner,
    loser=loser,
    tie=tie,
    telemetry_leaves=leaves,
    challenge=challenge,
    hmac_signature = encode_packed(['bytes32', 'bytes32', 'bytes32'], [secret_hash,  match_hash, challenge])

)
print(f"✅ Submitted match to blockchain: {tx_hash}")
