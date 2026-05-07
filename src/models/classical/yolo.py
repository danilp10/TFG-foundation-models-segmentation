import os
import shutil

import cv2
import numpy as np
from ultralytics import YOLO


def mask_to_yolo_polygon(mask_path, img_w, img_h):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None

    points = contour.squeeze()
    if points.ndim == 1:
        return None

    normalized = []
    for x, y in points:
        normalized.extend([x / img_w, y / img_h])
    return normalized


def convert_kvasir_to_yolo(dataset_path):
    for split in ["train", "val", "test"]:
        images_dir = os.path.join(dataset_path, "images", split)
        masks_dir = os.path.join(dataset_path, "masks", split)
        labels_dir = os.path.join(dataset_path, "labels", split)
        os.makedirs(labels_dir, exist_ok=True)

        for img_file in os.listdir(images_dir):
            if not img_file.endswith(".jpg"):
                continue
            name = os.path.splitext(img_file)[0]
            mask_path = os.path.join(masks_dir, name + ".jpg")
            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(os.path.join(images_dir, img_file))
            h, w = img.shape[:2]

            polygon = mask_to_yolo_polygon(mask_path, w, h)
            if polygon is None:
                continue

            label_path = os.path.join(labels_dir, name + ".txt")
            with open(label_path, "w") as f:
                f.write("0 " + " ".join(f"{v:.6f}" for v in polygon) + "\n")


def write_yolo_yaml(dataset_path, yaml_path, num_classes=1, class_names=("polyp",)):
    names_str = "[" + ", ".join(f"'{n}'" for n in class_names) + "]"
    content = (
        f"path: {dataset_path}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {num_classes}\n"
        f"names: {names_str}\n"
    )
    with open(yaml_path, "w") as f:
        f.write(content)


def train_yolo(yaml_path, output_weights, epochs=50, imgsz=512, batch=4, device=0,
               base_weights="yolov8n-seg.pt"):
    model = YOLO(base_weights)
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False,
    )
    best_weights = model.trainer.best
    shutil.copy(best_weights, output_weights)
    return output_weights
