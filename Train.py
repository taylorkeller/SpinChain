import os
import cv2
import shutil
import random
import numpy as np
from pathlib import Path
import yaml
import subprocess
from tqdm import tqdm

# --------------------------
# SETTINGS
# --------------------------
source_dir = Path('.')
dataset_dir = Path('./beyblade_dataset')
images_dir = dataset_dir / 'images' / 'train'
labels_dir = dataset_dir / 'labels' / 'train'
AUG_PER_IMAGE = 5
MULTI_IMG_COUNT = 15000
ARENA_PATH = "arena_bg.jpg"
# bounding box of the dead zone
DEAD_ZONE_NORM = (0.43, 0.83, 0.14, 0.08)  # (x_center, y_center, width, height)


# --------------------------
# DETECTION FUNCTIONS
# --------------------------
def detect_bounding_box(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = img.shape[:2]

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=20, minRadius=15, maxRadius=60)
    if circles is not None:
        x, y, r = np.round(circles[0, 0]).astype("int")
        return (x / w, y / h, 2 * r / w, 2 * r / h), 'circle'

    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) >= 5:
            (x, y), (MA, ma), _ = cv2.fitEllipse(cnt)
            return (x / w, y / h, MA / w, ma / h), 'ellipse'

    return None, 'none'

def is_bbox_too_large(x, y, w, h, threshold=0.9):
    return w > threshold or h > threshold

def is_low_contrast_crop(img, threshold=15):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.std() < threshold

def has_enough_edges(img, threshold=30):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.count_nonzero(edges) > threshold

def is_bbox_meaningful(img, bbox):
    x, y, w, h = bbox
    H, W = img.shape[:2]
    x1 = int((x - w / 2) * W)
    y1 = int((y - h / 2) * H)
    x2 = int((x + w / 2) * W)
    y2 = int((y + h / 2) * H)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    crop = img[y1:y2, x1:x2]
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return False
    return not is_low_contrast_crop(crop) and has_enough_edges(crop)

def bbox_intersects_dead_zone(x, y, w, h, dz=DEAD_ZONE_NORM):
    # Convert to corner format
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2

    dz_x, dz_y, dz_w, dz_h = dz
    dz_x1, dz_y1 = dz_x - dz_w / 2, dz_y - dz_h / 2
    dz_x2, dz_y2 = dz_x + dz_w / 2, dz_y + dz_h / 2

    # Check for overlap
    return not (x2 < dz_x1 or x1 > dz_x2 or y2 < dz_y1 or y1 > dz_y2)

# --------------------------
# AUGMENTATION FUNCTION
# --------------------------
def augment_image(img, bbox, max_box_ratio=0.8):
    h, w = img.shape[:2]
    x_center, y_center, box_w, box_h = bbox

    if random.random() > 0.5:
        scale = random.uniform(0.98, 1.00)
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

    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if random.random() > 0.7:
        noise = np.random.randint(0, 20, img.shape, dtype='uint8')
        img = cv2.add(img, noise)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] += random.uniform(-10, 10)
    hsv[..., 1] *= random.uniform(0.9, 1.1)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if box_w > max_box_ratio or box_h > max_box_ratio:
        return None, None

    return img, (x_center, y_center, box_w, box_h)

# --------------------------
# MULTI-OBJECT IMAGE CREATION
# --------------------------
def generate_composite(i, img_paths, class_indices, out_img_dir, out_lbl_dir, arena_path=None):
    random.seed(i)
    canvas_size = (640, 640)

    if arena_path and Path(arena_path).exists():
        arena = cv2.imread(str(arena_path))
        arena = cv2.resize(arena, canvas_size) if arena is not None else None
    else:
        arena = None

    canvas = arena.copy() if arena is not None else np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)

    annotations = []
    used_regions = []
    num_objs = random.randint(1, 3)
    samples = random.sample(list(zip(img_paths, class_indices)), num_objs)

    for img_path, class_idx in samples:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        bbox, method = detect_bounding_box(img)
        if bbox is None or method not in ('circle', 'ellipse'):
            continue
        if is_bbox_too_large(*bbox):
            continue

        aug_img, aug_bbox = augment_image(img, bbox)
        if aug_img is None:
            continue

        h, w = aug_img.shape[:2]
        scale = random.uniform(0.3, 0.6)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w >= canvas_size[0] or new_h >= canvas_size[1]:
            continue

        resized = cv2.resize(aug_img, (new_w, new_h))

        for _ in range(10):
            x_offset = random.randint(0, canvas_size[0] - new_w)
            y_offset = random.randint(0, canvas_size[1] - new_h)

            overlap = False
            for (ox, oy, ow, oh) in used_regions:
                if not (x_offset + new_w < ox or x_offset > ox + ow or
                        y_offset + new_h < oy or y_offset > oy + oh):
                    overlap = True
                    break
            if not overlap:
                used_regions.append((x_offset, y_offset, new_w, new_h))
                break
        else:
            continue

        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        x_c, y_c, bw, bh = aug_bbox
        abs_x = x_offset + x_c * new_w
        abs_y = y_offset + y_c * new_h
        x_center = abs_x / canvas_size[0]
        y_center = abs_y / canvas_size[1]
        width = bw * new_w / canvas_size[0]
        height = bh * new_h / canvas_size[1]

        if bbox_intersects_dead_zone(x_center, y_center, width, height):
            continue
    
        annotations.append((class_idx, x_center, y_center, width, height))

    if annotations:
        out_img = out_img_dir / f"multi_{i}.jpg"
        out_lbl = out_lbl_dir / f"multi_{i}.txt"
        cv2.imwrite(str(out_img), canvas)
        with open(out_lbl, 'w') as f:
            for class_idx, x, y, w, h in annotations:
                f.write(f"{class_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        return True
    return False

# --------------------------
# DATASET GENERATION
# --------------------------
shutil.rmtree(dataset_dir, ignore_errors=True)
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

image_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.jpg')]
class_names = sorted(set(os.path.splitext(f)[0].split('_')[0] for f in image_files))
class_to_index = {name: i for i, name in enumerate(class_names)}
print("🧠 Found classes:", class_to_index)

img_id = 0
augmented_count = 0
multi_composite_count = 0
valid_images = []
valid_classes = []

for file in tqdm(image_files, desc="🔍 Generating single-object images"):
    name = os.path.splitext(file)[0]
    class_name = name.split('_')[0]
    class_index = class_to_index[class_name]
    img_path = source_dir / file
    img = cv2.imread(str(img_path))
    if img is None or img.shape[2] != 3:
        continue

    bbox, method = detect_bounding_box(img)
    if bbox is None or method not in ('circle', 'ellipse'):
        continue
    if is_bbox_too_large(*bbox):
        continue
    bbox = tuple(np.clip(bbox, 0, 1))
    if bbox_intersects_dead_zone(*bbox):
        continue

    out_img_path = images_dir / f"{class_name}_{img_id}.jpg"
    label_path = labels_dir / f"{class_name}_{img_id}.txt"
    cv2.imwrite(str(out_img_path), img)
    label_path.write_text(f"{class_index} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    img_id += 1

    for _ in range(AUG_PER_IMAGE):
        aug, aug_bbox = augment_image(img, bbox)
        if aug is None:
            continue
        out_img_path = images_dir / f"{class_name}_{img_id}.jpg"
        label_path = labels_dir / f"{class_name}_{img_id}.txt"
        cv2.imwrite(str(out_img_path), aug)
        label_path.write_text(f"{class_index} {aug_bbox[0]:.6f} {aug_bbox[1]:.6f} {aug_bbox[2]:.6f} {aug_bbox[3]:.6f}\n")
        img_id += 1
        augmented_count += 1

    valid_images.append(img_path)
    valid_classes.append(class_index)

print("🧩 Generating multi-object composite images...")
for i in tqdm(range(img_id, img_id + MULTI_IMG_COUNT), desc="🧩 Composite"):
    if generate_composite(i, valid_images, valid_classes, images_dir, labels_dir, arena_path=ARENA_PATH):
        multi_composite_count += 1
img_id += MULTI_IMG_COUNT

print(f"\n✅ Dataset created: {img_id} total images")
print(f"📈 Augmented images created: {augmented_count}")
print(f"🎨 Multi-object composite images created: {multi_composite_count}")

# --------------------------
# CREATE data.yaml
# --------------------------
data_yaml = {
    'train': str(images_dir.resolve()),
    'val': str(images_dir.resolve()),
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
def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')

print("\n🚀 Starting YOLOv5 training...\n")
run_command(f"python yolov5/train.py --img 640 --batch 64 --epochs 120 --data {yaml_path} --weights yolov5s.pt --name spinchain-yolo")
print(f"\n✅ Dataset created: {img_id} total images")
print(f"📈 Augmented images created: {augmented_count}")
print(f"🎨 Multi-object composite images created: {multi_composite_count}")