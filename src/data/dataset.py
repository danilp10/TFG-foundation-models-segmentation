import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class KvasirDataset(Dataset):
    """Dataset estándar de Kvasir para modelos de segmentación clásicos como
    UNet. Devuelve pares (imagen, máscara) sin información de prompt."""

    def __init__(self, dataset_path, split, img_size=512):
        """Carga las muestras del split indicado descartando aquellas cuya
        máscara esté vacía (sin píxeles positivos)."""
        self.img_size = img_size
        self.images_dir = os.path.join(dataset_path, "images", split)
        self.masks_dir = os.path.join(dataset_path, "masks", split)
        self.samples = []

        for img_file in os.listdir(self.images_dir):
            if not img_file.endswith(".jpg"):
                continue
            name = os.path.splitext(img_file)[0]
            img_path = os.path.join(self.images_dir, img_file)
            mask_path = os.path.join(self.masks_dir, name + ".jpg")

            if os.path.exists(mask_path):
                gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if gt is not None and (gt > 127).sum() > 0:
                    self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Devuelve (imagen, máscara) ambas redimensionadas a img_size×img_size.
        La imagen se normaliza al rango [0, 1] y la máscara se binariza."""
        img_path, mask_path = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.squeeze(mask)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = torch.tensor((mask > 127).astype(np.float32)).unsqueeze(0)
        return image, mask


class KvasirSegDataset(Dataset):
    """Dataset de Kvasir para fine tuning de modelos SAM. Devuelve la imagen,
    la máscara y un punto central calculado a partir de la caja delimitadora
    del JSON de anotaciones."""

    def __init__(self, dataset_path, split, bbox_json, img_size=1024, mask_size=256):
        """Carga las muestras leyendo el JSON de bounding boxes para usarlas
        como prompt durante el entrenamiento."""
        self.img_size = img_size
        self.mask_size = mask_size
        self.images_dir = os.path.join(dataset_path, "images", split)
        self.masks_dir = os.path.join(dataset_path, "masks", split)
        self.samples = []

        with open(bbox_json) as f:
            bboxes = json.load(f)

        for img_name, info in bboxes.items():
            img_path = os.path.join(self.images_dir, img_name + ".jpg")
            mask_path = os.path.join(self.masks_dir, img_name + ".jpg")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.samples.append((img_path, mask_path, info["bbox"][0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Devuelve (imagen, máscara, punto central, label). La imagen va a
        img_size, la máscara a mask_size, y el punto se escala con la imagen."""
        img_path, mask_path, bbox = self.samples[idx]

        image = cv2.imread(img_path)
        orig_h, orig_w = image.shape[:2]
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.squeeze(mask)
        mask = cv2.resize(mask, (self.mask_size, self.mask_size))
        mask = torch.tensor((mask > 127).astype(np.float32)).unsqueeze(0)

        xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
        cx = (xmin + (xmax - xmin) / 2) * scale_x
        cy = (ymin + (ymax - ymin) / 2) * scale_y
        point = torch.tensor([[cx, cy]]).float()
        label = torch.tensor([1])

        return image, mask, point, label


class PascalSegDataset(Dataset):
    """Dataset de PASCAL-S para fine tuning de modelos SAM. El punto central
    se calcula como el centroide de los píxeles positivos de la máscara."""

    def __init__(self, dataset_path, split, img_size=1024, mask_size=256):
        """Carga las muestras del split desde las carpetas Image/ y GT/."""
        self.img_size = img_size
        self.mask_size = mask_size
        self.images_dir = os.path.join(dataset_path, "Image", split)
        self.masks_dir = os.path.join(dataset_path, "GT", split)
        self.samples = []

        for img_name in os.listdir(self.images_dir):
            if not img_name.endswith(".jpg"):
                continue
            name = img_name.replace(".jpg", "")
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, name + ".png")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Devuelve (imagen, máscara, centroide, label). Si la máscara está
        vacía, usa el centro geométrico de la imagen como fallback."""
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.squeeze(mask)
        mask = cv2.resize(mask, (self.mask_size, self.mask_size))
        mask_bin = (mask > 127).astype(np.float32)

        ys, xs = np.where(mask_bin > 0)
        if len(xs) > 0:
            cx, cy = float(xs.mean()), float(ys.mean())
        else:
            cx, cy = mask_bin.shape[1] / 2, mask_bin.shape[0] / 2

        mask_tensor = torch.tensor(mask_bin).unsqueeze(0)
        point = torch.tensor([[cx, cy]]).float()
        label = torch.tensor([1])

        return image, mask_tensor, point, label


class ISICSegDataset(Dataset):
    """Dataset de ISIC 2016 para fine tuning de modelos SAM. El punto central
    se calcula a partir de la caja delimitadora del contorno de la máscara."""

    def __init__(self, dataset_path, split, img_size=1024, mask_size=256):
        """Carga las muestras del split desde la carpeta de salida del
        proceso de splitting."""
        self.img_size = img_size
        self.mask_size = mask_size
        self.images_dir = os.path.join(dataset_path, "images", split)
        self.masks_dir = os.path.join(dataset_path, "masks", split)
        self.samples = []

        for img_filename in os.listdir(self.images_dir):
            if not img_filename.lower().endswith(".jpg"):
                continue
            name = img_filename.replace(".jpg", "")
            img_path = os.path.join(self.images_dir, img_filename)
            mask_path = os.path.join(self.masks_dir, name + ".png")
            if os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Devuelve (imagen, máscara, punto central, label). El punto es el
        centro de la bounding box que envuelve a los contornos de la lesión."""
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path)
        orig_h, orig_w = image.shape[:2]
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.squeeze(mask)
        mask = cv2.resize(mask, (self.mask_size, self.mask_size))
        mask = torch.tensor((mask > 127).astype(np.float32)).unsqueeze(0)

        gt_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_full = np.squeeze(gt_full)
        gt_bin = (gt_full > 127).astype(np.uint8)
        contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(np.vstack(contours))

        cx = (x + w / 2) * scale_x
        cy = (y + h / 2) * scale_y
        point = torch.tensor([[cx, cy]]).float()
        label = torch.tensor([1])

        return image, mask, point, label


class RefCOCOgSegDataset(Dataset):
    """Dataset de RefCOCOg para fine tuning de modelos SAM. El punto central
    se calcula como centroide de los píxeles positivos escalado al tamaño
    de imagen del modelo."""

    def __init__(self, dataset_path, split, img_size=1024, mask_size=256):
        """Carga las muestras del split descartando aquellas cuya máscara
        esté vacía."""
        self.img_size = img_size
        self.mask_size = mask_size
        self.images_dir = os.path.join(dataset_path, "images", split)
        self.masks_dir = os.path.join(dataset_path, "masks", split)
        self.samples = []

        for img_file in os.listdir(self.images_dir):
            if not img_file.endswith(".jpg"):
                continue
            name = os.path.splitext(img_file)[0]
            img_path = os.path.join(self.images_dir, img_file)
            mask_path = os.path.join(self.masks_dir, name + ".png")
            if not os.path.exists(mask_path):
                continue
            gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt is None or (gt > 127).sum() == 0:
                continue
            self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Devuelve (imagen, máscara, centroide, label). El centroide se
        calcula sobre la máscara original y luego se escala a img_size."""
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        gt_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_full = np.squeeze(gt_full)
        orig_h, orig_w = gt_full.shape
        ys, xs = np.where(gt_full > 127)
        if len(xs) > 0:
            cx = float(xs.mean()) * (self.img_size / orig_w)
            cy = float(ys.mean()) * (self.img_size / orig_h)
        else:
            cx, cy = self.img_size / 2, self.img_size / 2

        mask = cv2.resize(gt_full, (self.mask_size, self.mask_size))
        mask_tensor = torch.tensor((mask > 127).astype(np.float32)).unsqueeze(0)

        point = torch.tensor([[cx, cy]]).float()
        label = torch.tensor([1])
        return image, mask_tensor, point, label
