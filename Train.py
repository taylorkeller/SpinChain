import os
import cv2
import shutil
import random
import numpy as np
from pathlib import Path
import yaml
import subprocess

# --------------------------
# SETTINGS
# --------------------------
source_dir = Path('.')  # Directory with original .jpgs
dataset_dir = Path('./beyblade_dataset')
images_dir = dataset_dir / 'images' / 'train'
labels_dir = dataset_dir / 'labels' / 'train'
AUG_PER_IMAGE = 10       # Number of augmentations per image

# --------------------------
# DETECTION FUNCTIONS
# --------------------------
def detect_bounding_box(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = img.shape[:2]

    # Try circle detection first
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=0
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        x, y, r = circles[0]
        x_center = x / w
        y_center = y / h
        width = height = (2 * r) / w
        return x_center, y_center, width, height

    # Fallback to ellipse fitting
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), _ = ellipse
            x_center = x / w
            y_center = y / h
            width = MA / w
            height = ma / h
            return x_center, y_center, width, height

        # Fallback to bounding box
        x, y, box_w, box_h = cv2.boundingRect(cnt)
        x_center = (x + box_w / 2) / w
        y_center = (y + box_h / 2) / h
        width = box_w / w
        height = box_h / h
        return x_center, y_center, width, height

    # Final fallback: full image
    return 0.5, 0.5, 1.0, 1.0

# --------------------------
# AUGMENTATION FUNCTION
# --------------------------
def augment_image(img, bbox):
    h, w = img.shape[:2]
    x_center, y_center, box_w, box_h = bbox

    # Zoom-out
    if random.random() > 0.5:
        scale = random.uniform(0.3, 0.8)
        new_w, new_h = int(w * scale), int(h * scale)
        small = cv2.resize(img, (new_w, new_h))
        canvas = np.zeros_like(img)
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = small
        img = canvas
        x_center = (x_offset + new_w / 2) / w
        y_center = (y_offset + new_h / 2) / h
        box_w = new_w / w
        box_h = new_h / h

    # Rotation
    angle = random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))

    # Flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
        x_center = 1 - x_center

    # Brightness/contrast
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Blur
    if random.random() > 0.7:
        img = cv2.GaussianBlur(img, (5, 5), 0)

    # Noise
    if random.random() > 0.7:
        noise = np.random.randint(0, 20, img.shape, dtype='uint8')
        img = cv2.add(img, noise)

    # Color tint
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] += random.uniform(-10, 10)
    hsv[..., 1] *= random.uniform(0.9, 1.1)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img, (x_center, y_center, box_w, box_h)

# --------------------------
# PREP FOLDERS
# --------------------------
shutil.rmtree(dataset_dir, ignore_errors=True)
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# --------------------------
# FIND CLASSES
# --------------------------
image_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.jpg')]
class_names = sorted(set(os.path.splitext(f)[0].split('_')[0] for f in image_files))
class_to_index = {name: i for i, name in enumerate(class_names)}

print("🧠 Found classes:", class_to_index)

# --------------------------
# AUGMENT AND SAVE
# --------------------------
img_id = 0
for file in image_files:
    name = os.path.splitext(file)[0]
    class_name = name.split('_')[0]
    class_index = class_to_index[class_name]

    img = cv2.imread(str(source_dir / file))
    if img is None:
        print(f"❌ Could not read {file}")
        continue

    bbox = detect_bounding_box(img)

    # Save original
    out_img_path = images_dir / f"{class_name}_{img_id}.jpg"
    label_path = labels_dir / f"{class_name}_{img_id}.txt"
    cv2.imwrite(str(out_img_path), img)
    label_path.write_text(f"{class_index} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    img_id += 1

    # Augmentations
    for i in range(AUG_PER_IMAGE):
        aug, aug_bbox = augment_image(img, bbox)
        out_img_path = images_dir / f"{class_name}_{img_id}.jpg"
        label_path = labels_dir / f"{class_name}_{img_id}.txt"
        cv2.imwrite(str(out_img_path), aug)
        label_path.write_text(f"{class_index} {aug_bbox[0]:.6f} {aug_bbox[1]:.6f} {aug_bbox[2]:.6f} {aug_bbox[3]:.6f}\n")
        img_id += 1

print(f"\n✅ Dataset created: {img_id} images")

# --------------------------
# CREATE data.yaml
# --------------------------
data_yaml = {
    'train': str(images_dir.resolve()),
    'val': str(images_dir.resolve()),  # same as train here
    'nc': len(class_names),
    'names': class_names
}
yaml_path = dataset_dir / 'data.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f)

print(f"📄 data.yaml saved to: {yaml_path}")

# --------------------------
# TRAIN YOLOv5
# --------------------------
print("\n🚀 Starting YOLOv5 training...\n")

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')

run_command(f"python yolov5/train.py --img 640 --batch -1 --epochs 100 --data {yaml_path} --weights yolov5s.pt --name spinchain-yolo")
