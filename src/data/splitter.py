import os
import pickle
import random
import shutil

import numpy as np
from PIL import Image
from pycocotools.coco import COCO


def split_kvasir(dataset_path, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Divide el dataset Kvasir en train/val/test copiando imágenes y máscaras
    a las subcarpetas correspondientes. La proporción restante (1 - train -
    val) se reserva para test. La semilla fija el orden aleatorio para que
    el split sea reproducible."""
    images_dir = os.path.join(dataset_path, "images")
    masks_dir = os.path.join(dataset_path, "masks")

    images = [f.replace(".jpg", "") for f in os.listdir(images_dir) if f.endswith(".jpg")]
    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split, names in splits.items():
        os.makedirs(os.path.join(dataset_path, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "masks", split), exist_ok=True)
        for name in names:
            shutil.copy(
                os.path.join(images_dir, name + ".jpg"),
                os.path.join(dataset_path, "images", split, name + ".jpg"),
            )
            shutil.copy(
                os.path.join(masks_dir, name + ".jpg"),
                os.path.join(dataset_path, "masks", split, name + ".jpg"),
            )


def split_pascals(dataset_path, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Divide el dataset PASCAL-S en train/val/test. A diferencia de Kvasir,
    las imágenes están en Image/ (.jpg) y las máscaras en GT/ (.png)."""
    images_dir = os.path.join(dataset_path, "Image")
    masks_dir = os.path.join(dataset_path, "GT")

    images = [f.replace(".jpg", "") for f in os.listdir(images_dir) if f.endswith(".jpg")]
    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split, names in splits.items():
        os.makedirs(os.path.join(dataset_path, "Image", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "GT", split), exist_ok=True)
        for name in names:
            shutil.copy(
                os.path.join(images_dir, name + ".jpg"),
                os.path.join(dataset_path, "Image", split, name + ".jpg"),
            )
            shutil.copy(
                os.path.join(masks_dir, name + ".png"),
                os.path.join(dataset_path, "GT", split, name + ".png"),
            )


def split_isic2016(dataset_root, output_root, train_ratio=0.85, seed=42):
    """Reorganiza ISIC 2016 a una estructura images/{split}, masks/{split}.
    El split test se mantiene tal cual (es el oficial); el split train se
    divide internamente en train y val según train_ratio. Las máscaras del
    dataset original tienen sufijo "_Segmentation.png" que aquí se elimina."""
    train_images_dir = os.path.join(dataset_root, "ISBI2016_ISIC_Part1_Training_Data")
    train_masks_dir = os.path.join(dataset_root, "ISBI2016_ISIC_Part1_Training_GroundTruth")
    test_images_dir = os.path.join(dataset_root, "ISBI2016_ISIC_Part1_Test_Data")
    test_masks_dir = os.path.join(dataset_root, "ISBI2016_ISIC_Part1_Test_GroundTruth")

    train_names = [
        f.replace("_Segmentation.png", "")
        for f in os.listdir(train_masks_dir) if f.endswith("_Segmentation.png")
    ]
    test_names = [
        f.replace("_Segmentation.png", "")
        for f in os.listdir(test_masks_dir) if f.endswith("_Segmentation.png")
    ]

    random.seed(seed)
    random.shuffle(train_names)
    n_train = int(len(train_names) * train_ratio)

    splits = {
        "train": [(n, train_images_dir, train_masks_dir) for n in train_names[:n_train]],
        "val": [(n, train_images_dir, train_masks_dir) for n in train_names[n_train:]],
        "test": [(n, test_images_dir, test_masks_dir) for n in test_names],
    }

    for split, entries in splits.items():
        os.makedirs(os.path.join(output_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_root, "masks", split), exist_ok=True)
        for name, img_src_dir, mask_src_dir in entries:
            shutil.copy(
                os.path.join(img_src_dir, name + ".jpg"),
                os.path.join(output_root, "images", split, name + ".jpg"),
            )
            shutil.copy(
                os.path.join(mask_src_dir, name + "_Segmentation.png"),
                os.path.join(output_root, "masks", split, name + ".png"),
            )


def split_refcocog(dataset_path, subset_size=1000, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Divide RefCOCOg generando un subconjunto de subset_size referencias
    desde el split train original. Cada referencia se guarda como par
    {ann_id}.jpg + {ann_id}.png, donde la máscara se rasteriza desde la
    anotación COCO con annToMask."""
    images_dir = os.path.join(dataset_path, "train2014")
    instances_path = os.path.join(dataset_path, "instances.json")
    refs_path = os.path.join(dataset_path, "refs(umd).p")

    with open(refs_path, "rb") as f:
        refs = pickle.load(f)
    coco = COCO(instances_path)

    random.seed(seed)
    train_refs = [r for r in refs if r["split"] == "train"]
    random.shuffle(train_refs)
    subset = train_refs[:subset_size]

    n = len(subset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": subset[:n_train],
        "val": subset[n_train:n_train + n_val],
        "test": subset[n_train + n_val:],
    }

    for split, split_refs in splits.items():
        os.makedirs(os.path.join(dataset_path, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "masks", split), exist_ok=True)

        for ref in split_refs:
            ann = coco.loadAnns(ref["ann_id"])[0]
            img_info = coco.loadImgs(ref["image_id"])[0]

            src = os.path.join(images_dir, img_info["file_name"])
            if not os.path.exists(src):
                continue

            ann_id = ref["ann_id"]
            dst_img = os.path.join(dataset_path, "images", split, f"{ann_id}.jpg")
            dst_mask = os.path.join(dataset_path, "masks", split, f"{ann_id}.png")

            shutil.copy(src, dst_img)

            mask = coco.annToMask(ann) * 255
            Image.fromarray(mask.astype(np.uint8)).save(dst_mask)
