import cv2
import numpy as np
from tkinter import Tk, filedialog
SHRINK_BBOX = 0.80  # between 0 and 1

def detect_bounding_box(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = img.shape[:2]

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=20, minRadius=15, maxRadius=60)
    if circles is not None:
        x, y, r = np.round(circles[0, 0]).astype("int")
        return (x / w, y / h, (2 * r / w) * SHRINK_BBOX, (2 * r / h) * SHRINK_BBOX), 'circle'

    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) >= 5:
            (x, y), (MA, ma), _ = cv2.fitEllipse(cnt)
            return (x / w, y / h, (MA / w) * SHRINK_BBOX, (ma / h) * SHRINK_BBOX), 'ellipse'

    return None, 'none'

def draw_yolo_box(img, bbox):
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    x1 = int((x - bw / 2) * w)
    y1 = int((y - bh / 2) * h)
    x2 = int((x + bw / 2) * w)
    y2 = int((y + bh / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return img

def main():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        print("No file selected.")
        return

    img = cv2.imread(file_path)
    if img is None:
        print("Failed to load image.")
        return

    bbox, method = detect_bounding_box(img)
    if bbox is None:
        print("No bounding box detected.")
        return

    img = draw_yolo_box(img, bbox)
    cv2.imshow("YOLO-Style Bounding Box", img)
    print(f"Detected using: {method}")
    print(f"YOLO Format: x_center={bbox[0]:.6f}, y_center={bbox[1]:.6f}, width={bbox[2]:.6f}, height={bbox[3]:.6f}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
