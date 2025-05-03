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
from blockchain_logger import request_challenge, record_match_with_hmac
import qrcode
from pyzbar.pyzbar import decode
import threading

# === Load .env ===
load_dotenv()

# === Settings ===
MIN_SEEN_FRAMES = 100  # Minimum frames before tracking is considered valid
AUTO_SHUTDOWN_FRAMES = 180  # Frames before auto-terminate
MAX_MISSING_FRAMES = 180  # If not seen in this many frames, declare lost
MODEL_VERSION = "v1.0"
MODEL_PATH = "yolov5/runs/train/spinchain-yolo/weights/best.pt"

PRIVATE_KEY = os.getenv("PRIVATE_KEY")
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")

# === Load YOLOv5 model ===
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
model.conf = 0.1  # Confidence threshold
model.iou = 0.05  # IOU threshold
model.to('cuda' if torch.cuda.is_available() else 'cpu') # YOLO defualt to cuda then fall back to cpu if unavailable

# === Webcam ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60) # Frame cap/set
cap.set(39, 0) # Brightness adjustment
cap.set(28, 30) # Auto focus

# === Secret and QR ===
# Changes the secret hash every match
ephemeral_secret = secrets.token_bytes(32)
secret_hash = keccak(ephemeral_secret) # Used for HMAC, challenge, and QR
print(f"🪪 Display this in QR form: {secret_hash.hex()}")

def render_challenge_qr(text):
    # Generate QR code image
    qr = qrcode.make(text)
    return np.array(qr.convert('RGB'))

# Uses the secret hash that always changes to create a unique QR code every match
qr_img = render_challenge_qr(secret_hash.hex())
qr_img = cv2.resize(qr_img, (100, 100)) # QR Size

def decode_embedded_qr(qr_img):
    # Read the QR code
    decoded = decode(qr_img)
    if decoded:
        return decoded[0].data.decode().strip()
    return ""

# === QR State ===
qr_confirm_count = 0
challenge_verified = False
CHALLENGE_FRAMES_REQUIRED = 10 # Frames QR needs to be read to confirm the video
challenge = None    # Stores the blockchain challenge later

def submit_challenge_tx():
    # Submit challenge to contract in the backround to allow the rest of the program to continue without waiting for challege/response
    global challenge
    print("🛰 Sending challenge to blockchain...")
    # Gets challenge from the contract using the secret hash
    challenge = request_challenge(secret_hash)    
    print("✅ Blockchain challenge complete.")
    return challenge

# === ROI Tracker ===
class ROIStabilityTracker:
    def __init__(self, class_id):
        self.class_id = class_id              # Beyblade class name (e.g., "Driger")
        self.last_status = "Moving"           # Last known motion status
        self.last_pos = None                  # Last known (x, y) center position
        self.last_gray = None                 # Last grayscale ROI (for optical flow)
        self.stability_score = float('inf')   # Average optical flow magnitude
        self.score_history = deque(maxlen=5)  # Buffer of recent motion scores
        self.stable_count = 0                 # Frames seen as stable in a row
        self.pos_history = deque(maxlen=12)   # Buffer of recent center positions
        self.stopped_frame_index = None       # Frame when first marked as stopped

    def update(self, frame, center, frame_idx):
        # Track movement stability using optical flow
        self.pos_history.append(center)                                 # Add the current center to the position history 
        smoothed = np.mean(self.pos_history, axis=0).astype(int)        # Smooth out sudden position changes by averaging recent position
        x, y = smoothed

        # Define a Region of Interest (ROI) centered at the smoothed position
        half = 60
        roi = frame[max(y - half, 0):min(y + half, frame.shape[0]),
                    max(x - half, 0):min(x + half, frame.shape[1])]
        
        # If ROI is too small (near border), skip stability check (prevents ID mistakes when beyblades are too close to each other)
        if roi.shape[0] < 32 or roi.shape[1] < 32:
            return "Moving"

        # Convert ROI to grayscale and apply blur to reduce noise
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # If a previous grayscale ROI exists, compute optical flow
        if self.last_gray is not None and self.last_gray.shape == gray.shape:
            # Use Farneback dense optical flow to get pixel-wise motion vectors
            flow = cv2.calcOpticalFlowFarneback(self.last_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])         # Convert to magnitude
            self.score_history.append(np.mean(mag))                      # Save the average motion magnitude
            self.stability_score = np.mean(self.score_history)           # Average over recent scores
        else:
            self.stability_score = float('inf')

        self.last_gray = gray.copy()

        # If already marked as stopped, allow for minor jitter without resetting state
        if self.last_status == "Stopped":
            dx = abs(center[0] - self.last_pos[0])
            dy = abs(center[1] - self.last_pos[1])
            if dx < 15 and dy < 15:
                return self.last_status
            else:
                self.last_status = "Moving"                 # Movement detected again (sometimes they rotate in one spot)

        # If movement magnitude is low enough, count toward stable state
        if self.stability_score < 0.20:
            self.stable_count += 1
            # If stable for 15 frames, update status
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
trackers = {}                  # ID → ROIStabilityTracker instance
stop_by_class = {}             # class name → frame when stopped
last_seen_frame = {}           # class name → last seen frame
seen_frames_by_class = {}      # class name → total seen frame count
next_id = 0                    # next unique tracker ID
frame_index = 0                # current global frame number
shutdown_triggered = False    # True after first stop/loss
shutdown_start_frame = None   # frame when shutdown started
telemetry_events = []         # list of match logs (for blockchain)
final_stopped_names = []      # ordered list of stopped/lost class names

# === Main Loop ===
# Read the next frame from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame read failed.")
        time.sleep(0.5)
        continue

    frame_index += 1        # Track global frame count


    # If challenge is not yet verified, show and decode QR until confirmed (prevents video playback attacks)
    if not challenge_verified:
        decoded = decode_embedded_qr(qr_img)
        print(f"[QR Decode] Got: {decoded[:12]}... Expected: {secret_hash.hex()[:12]}...")

        if decoded == secret_hash.hex():
            qr_confirm_count += 1
            print(f"[QR] Confirm {qr_confirm_count}/{CHALLENGE_FRAMES_REQUIRED}")
        else:
            qr_confirm_count = 0

        # After enough QR confirmations it sends the blockhain challenge request
        if qr_confirm_count >= CHALLENGE_FRAMES_REQUIRED:
            print("✅ QR threshold met. Starting match tracking and blockchain thread...")
            threading.Thread(target=submit_challenge_tx, daemon=True).start()
            challenge_verified = True
            qr_confirm_count = 0
            continue
        
        # Show QR code overlay while waiting
        frame[10:110, frame.shape[1] - 110:frame.shape[1] - 10] = qr_img
        overlay_text = f"CHALLENGE: {secret_hash.hex()[:12]}..."
        cv2.putText(frame, overlay_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("ROI Stability Beyblade Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # === Match tracking logic ===
    current_ids = set()
    # YOLO object detection
    results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Loop through each detected object
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        name = model.names[int(cls)]                  # Get class label
        center = (cx, cy)

        # Attempt to match with existing tracker based on position and name
        matched = match_tracker(center, name)
        if matched is not None:
            tracker = trackers[matched]
        else:
            tracker = ROIStabilityTracker(name)     # If you lose tracking create a new tracker id
            trackers[next_id] = tracker
            matched = next_id
            next_id += 1

        # Update tracker with new position and status
        tracker.last_pos = center
        status = tracker.update(frame, center, frame_index)
        current_ids.add(matched)
        # Update global frame tracking stats
        last_seen_frame[name] = frame_index
        seen_frames_by_class[name] = seen_frames_by_class.get(name, 0) + 1
        # Skip any new detection that hasn’t been seen long enough (prevents false positives)
        if seen_frames_by_class[name] < MIN_SEEN_FRAMES:
            continue
        
        # If beyblade has stopped moving and hasn’t been logged yet, log it
        if status == "Stopped" and name not in stop_by_class:
            stop_by_class[name] = frame_index
            final_stopped_names.append(name)
            log = f"{name} stopped at frame {frame_index}"
            telemetry_events.append(log)
            print(f"[Frame {frame_index}] ❌ {log}")

    #  Handle objects that disappear (beyblade went out of the arena)
    for cname, last_seen in last_seen_frame.items():
        if cname not in stop_by_class and frame_index - last_seen > MAX_MISSING_FRAMES:
            if seen_frames_by_class.get(cname, 0) >= MIN_SEEN_FRAMES:
                stop_by_class[cname] = frame_index
                final_stopped_names.append(cname)
                log = f"{cname} lost at frame {frame_index}"
                telemetry_events.append(log)
                print(f"[Frame {frame_index}] ❌ {log}")

    # Start shutdown countdown once first stop/loss is detected
    if not shutdown_triggered and len(stop_by_class) > 0:
        shutdown_triggered = True
        shutdown_start_frame = frame_index

    # If shutdown is triggered and enough time has passed it ends the match
    if shutdown_triggered and frame_index - shutdown_start_frame >= AUTO_SHUTDOWN_FRAMES:
        print("\n🛑 Auto-shutdown triggered.")
        break

     # Draw tracking info and overlay onto the frame (live debug information)   
    for tid in current_ids:
        t = trackers[tid]
        label = f"{t.class_id}: {t.last_status} (Δ={t.stability_score:.2f})"
        color = (0, 255, 0) if t.last_status == "Moving" else (0, 0, 255)
        if t.last_pos:
            cv2.circle(frame, t.last_pos, 20, color, 2)
            cv2.putText(frame, label, (t.last_pos[0] - 70, t.last_pos[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Print how many frames each object has been seen (also used for debugging)
    for name in seen_frames_by_class:
        print(f"[{frame_index}] 👁️ {name} seen {seen_frames_by_class[name]} frames")

    # Overlay QR in top right and frame info at bottom
    frame[10:110, frame.shape[1] - 110:frame.shape[1] - 10] = qr_img
    overlay_text = f"CHALLENGE: {secret_hash.hex()[:12]}..."
    cv2.putText(frame, overlay_text, (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Display the frame with all overlays
    cv2.imshow("ROI Stability Beyblade Tracker", frame)

    # User quit
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

# Case 1: Only one Beyblade stopped, but multiple were seen during the match
if len(stop_by_class) == 1 and len(seen_frames_by_class) >= 2:
    stopped = next(iter(stop_by_class))
    # Find an opponent that was seen enough and is not the stopped one
    possible_opponents = [name for name in seen_frames_by_class if name != stopped and seen_frames_by_class[name] >= MIN_SEEN_FRAMES]

    # Declare the other Beyblade (still moving) as the winner
    if possible_opponents:
        moving = possible_opponents[0]
        winner = moving
        loser = stopped
        tie = False
        name1, name2 = winner, loser
    else:
        print("⚠️ No valid opponent found.")
        exit()

# Case 2: Two Beyblades stopped — compare their stop frame indices
elif len(stop_by_class) >= 2:
    name1, name2 = list(stop_by_class.keys())[:2]
    frame1 = stop_by_class[name1]
    frame2 = stop_by_class[name2]

    # They stopped at exactly the same time — it's a tie
    if frame1 == frame2:
        winner = loser = "None"
        tie = True
    # name1 stopped first => name2 is the winner
    elif frame1 < frame2:
        winner, loser = name2, name1
        tie = False
    # name2 stopped first => name1 is the winner
    else:
        winner, loser = name1, name2
        tie = False

# Case 3: Not enough beyblades were seen long enough for a valid match
else:
    print("⚠️ Not enough Beyblades tracked to determine a valid match.")
    exit()





# === Required for challenge to be requested and verified
# Verify the challenge response using local ephemeral secret
verify_shared_secret_hash(secret_hash)
# Submitted as part of the match to ensure that results can be verified later against the same model version
with open(MODEL_PATH, "rb") as f:
    model_hash = hashlib.sha256(f.read()).digest()
# Format bytes as a full 64-char hex string
def to_hex_string32(b):
    return ''.join(f'{x:02x}' for x in b)



# === Structured typed match hash
typed_data = encode(
    ["string", "bytes32", "string", "string", "bool"],
    [MODEL_VERSION, model_hash, winner, loser, tie]
)

# Compute final match hash using Keccak256 on encoded data 
match_hash = keccak(typed_data)

# Packed format used as HMAC payload in the final on-chain record
packed = encode_packed(['bytes32', 'bytes32', 'bytes32'], [secret_hash,  match_hash, challenge])

# === Telemetry events are raw strings like "Dranzer stopped at frame 210" ===
# These are submitted directly and hashed on-chain into a Merkle Tree
leaves = telemetry_events  # use raw event strings

# === Submit match to blockchain 
print("📤 Telemetry being submitted:")
for i, leaf in enumerate(telemetry_events):
    print(f"  [{i}] {repr(leaf)}")




print(f"\n🏁 Submitting result: {name1} vs {name2} | Winner: {winner} | Loser: {loser} | Tie: {tie}")
tx_hash = record_match_with_hmac(
    model_version=MODEL_VERSION,        # "v1.0"
    model_hash=model_hash,              # SHA256 digest of YOLO model used
    winner=winner,                      # Beyblade determined to win
    loser=loser,                        # Beyblade determined to lose
    tie=tie,                            # True if tied
    telemetry_leaves=leaves,            # List of telemetry strings
    challenge=challenge,                # Challenge value to bind HMAC
    hmac_signature=packed               # Final packed HMAC input to bind identity

)
print(f"✅ Submitted match to blockchain: {tx_hash}")
