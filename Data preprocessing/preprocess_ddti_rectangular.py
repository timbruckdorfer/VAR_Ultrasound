import cv2
import numpy as np
import os
import glob

def crop_content_by_contrast(img):
    """
    Crop the main content area of a rectangular image based on contrast/edges.
    Args:
        img (np.ndarray): Input image
    Returns:
        np.ndarray: Cropped image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)
    # Dilate edges to close gaps
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped = img[y:y+h, x:x+w]
        return cropped
    else:
        return img  # fallback

def resize_with_padding(img, target_size=(256, 256)):
    h, w = img.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def preprocess_ddti_images(src_dir, dst_dir, target_size=(256, 256)):
    os.makedirs(dst_dir, exist_ok=True)
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(src_dir, ext)))
    print(f"Found {len(image_files)} images in {src_dir}")
    for i, img_path in enumerate(image_files, 1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading {img_path}")
            continue
        cropped = crop_content_by_contrast(img)
        resized = resize_with_padding(cropped, target_size)
        out_name = os.path.basename(img_path)
        out_path = os.path.join(dst_dir, out_name)
        cv2.imwrite(out_path, resized)
        print(f"[{i}/{len(image_files)}] Saved {out_path}")
    print("Done preprocessing DDTI images.")

if __name__ == "__main__":
    src = "/Users/timbruckdorfer/Documents/Sommersemester25/Computational Surgineering/datasets/Thyroid/DDTI dataset/DDTI/1_or_data/image"
    dst = "/Users/timbruckdorfer/Documents/Sommersemester25/Computational Surgineering/datasets/Datasets_preprocessed/Thyroid_preprocessed/DDTI_preprocessed"
    preprocess_ddti_images(src, dst) 