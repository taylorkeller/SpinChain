import cv2
import numpy as np
from tkinter import Tk, filedialog
import os

# Define dead zone in YOLO normalized format (x_center, y_center, width, height)
DEAD_ZONE_NORM = (0.53, 0.95, 0.045, 0.11)

def draw_dead_zone(img, dz=DEAD_ZONE_NORM, color=(0, 0, 255), thickness=3):
    h, w = img.shape[:2]
    cx, cy, dw, dh = dz
    x1 = int((cx - dw / 2) * w)
    y1 = int((cy - dh / 2) * h)
    x2 = int((cx + dw / 2) * w)
    y2 = int((cy + dh / 2) * h)
    return cv2.rectangle(img.copy(), (x1, y1), (x2, y2), color, thickness)

def main():
    # Open file picker
    Tk().withdraw()
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        print("❌ No file selected.")
        return

    img = cv2.imread(file_path)
    if img is None:
        print("❌ Failed to read image.")
        return

    img_with_zone = draw_dead_zone(img)

    # Display the result
    cv2.imshow("Image with Dead Zone", img_with_zone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optional: save the image
    out_path = os.path.splitext(file_path)[0] + "_deadzone.jpg"
    cv2.imwrite(out_path, img_with_zone)
    print(f"✅ Saved to: {out_path}")

if __name__ == "__main__":
    main()
