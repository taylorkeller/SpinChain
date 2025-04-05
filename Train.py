import os
import cv2
import shutil
import random
import numpy as np
from pathlib import Path
import yaml

# --------------------------
# SETTINGS
# --------------------------
source_dir = Path('.')  # Directory with original .jpgs
dataset_dir = Path('./beyblade_dataset')
images_dir = dataset_dir / 'images' / 'train'
labels_dir = dataset_dir / 'labels' / 'train'
AUG_PER_IMAGE = 10       # Number of augmentations per image

# --------------------------
# UTILITY FUNCTIONS
# --------------------------
def augment_image(img):
    h, w = img.shape[:2]

    # Random rotation
    angle = random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))

    # Random flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # Brightness/contrast
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Gaussian blur
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

    return img

# --------------------------
# PREP DATASET FOLDERS
# --------------------------
shutil.rmtree(dataset_dir, ignore_errors=True)
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# --------------------------
# FIND CLASS NAMES
# --------------------------
image_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.jpg')]
class_names = sorted(set(os.path.splitext(f)[0].split('_')[0] for f in image_files))
class_to_index = {name: i for i, name in enumerate(class_names)}

print("🧠 Found classes:", class_to_index)

# --------------------------
# AUGMENT AND SAVE IMAGES + LABELS
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

    # Save original image
    out_img_path = images_dir / f"{class_name}_{img_id}.jpg"
    label_path = labels_dir / f"{class_name}_{img_id}.txt"
    cv2.imwrite(str(out_img_path), img)
    label_path.write_text(f"{class_index} 0.5 0.5 1.0 1.0\n")
    img_id += 1

    # Create augmented versions
    for i in range(AUG_PER_IMAGE):
        aug = augment_image(img)
        out_img_path = images_dir / f"{class_name}_{img_id}.jpg"
        label_path = labels_dir / f"{class_name}_{img_id}.txt"
        cv2.imwrite(str(out_img_path), aug)
        label_path.write_text(f"{class_index} 0.5 0.5 1.0 1.0\n")
        img_id += 1

print(f"\n✅ Dataset created: {img_id} images")

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
print("\n🚀 Starting YOLOv5 training...\n")
import subprocess

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')  # print each line as it's produced

# Replace os.system with run_command
run_command(f"python yolov5/train.py --img 640 --batch -1 --epochs 50 --data {yaml_path} --weights yolov5s.pt --name spinchain-yolo")


# --------------------------
# EVALUATE THE MODEL
# --------------------------
print("\n🔍 Running evaluation...\n")
best_weights = Path("runs/train/spinchain-yolo/weights/best.pt")
os.system(f"python yolov5/val.py --img 640 --data {yaml_path} --weights {best_weights} --save-conf --save-json --project eval_output --name spinchain_eval --exist-ok")

print("\n🎯 Done! Evaluation results saved in: eval_output/")
