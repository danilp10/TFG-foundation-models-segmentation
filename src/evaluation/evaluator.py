import os
import time

import cv2
import numpy as np
import torch

from src.evaluation.metrics import boundary_iou, compute_all_metrics, hausdorff_95, resize_for_hausdorff
from src.evaluation.inference import measure_inference_fine_tuning, measure_inference_fine_tuning_refcocog

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_bbox_from_mask(mask_binary):
    """Calcula la caja delimitadora (xmin, ymin, xmax, ymax) de la región
    positiva de una máscara binaria. Devuelve None si no hay región."""
    contours, _ = cv2.findContours(mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    return x, y, x + w, y + h


def _empty_metric_buffers():
    """Crea el diccionario de listas vacías que recogerá las métricas durante
    el bucle de evaluación."""
    return {k: [] for k in [
        "iou", "precision", "recall", "f1", "dice", "specificity", "f2",
        "pixel_accuracy", "boundary_iou", "hausdorff_95", "latency", "vram",
    ]}


def _accumulate(buf, pred_mask, gt_mask, latency, vram, use_resize_for_hausdorff=False):
    """Calcula todas las métricas de una predicción y las añade al buffer."""
    iou, precision, recall, f1, dice, specificity, f2, pixel_accuracy = compute_all_metrics(pred_mask, gt_mask)
    buf["iou"].append(iou)
    buf["precision"].append(precision)
    buf["recall"].append(recall)
    buf["f1"].append(f1)
    buf["dice"].append(dice)
    buf["specificity"].append(specificity)
    buf["f2"].append(f2)
    buf["pixel_accuracy"].append(pixel_accuracy)
    buf["boundary_iou"].append(boundary_iou(pred_mask, gt_mask))
    if use_resize_for_hausdorff:
        pred_mask, gt_mask = resize_for_hausdorff(pred_mask, gt_mask)
    buf["hausdorff_95"].append(hausdorff_95(pred_mask, gt_mask))
    buf["latency"].append(latency)
    buf["vram"].append(vram)


def _summarize(buf, model_name):
    """Calcula las medias del buffer y devuelve el diccionario en el formato
    esperado por el CSV de resultados."""
    return {
        "modelo": [model_name],
        "mean_iou": [np.mean(buf["iou"])],
        "mean_f1": [np.mean(buf["f1"])],
        "mean_recall": [np.mean(buf["recall"])],
        "mean_precision": [np.mean(buf["precision"])],
        "mean_dice": [np.mean(buf["dice"])],
        "mean_specificity": [np.mean(buf["specificity"])],
        "mean_f2": [np.mean(buf["f2"])],
        "mean_pixel_accuracy": [np.mean(buf["pixel_accuracy"])],
        "mean_boundary_iou": [np.mean(buf["boundary_iou"])],
        "mean_hausdorff_95": [np.mean(buf["hausdorff_95"])],
        "mean_latency_ms": [np.mean(buf["latency"])],
        "mean_vram_mb": [np.mean(buf["vram"])],
    }


def evaluate_finetuned(model, model_name, samples_iter, prompt_fn,
                       use_resize_for_hausdorff=False, prompt_type="point"):
    """Evalúa un modelo SAM/SAM2/SAM2.1/SAM3 ya fine-tuneado sobre el
    iterador de samples dado. La función prompt_fn(img_path, gt_mask) decide
    el prompt usado en cada muestra, prompt_type indica si la inferencia se
    hace con punto central ('point') o con caja delimitadora ('bbox')."""
    buf = _empty_metric_buffers()

    for img_path, gt_mask in samples_iter:
        prompt = prompt_fn(img_path, gt_mask)
        if prompt is None:
            continue

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if prompt_type == "bbox":
            masks, scores, latency, vram = measure_inference_fine_tuning_refcocog(
                model, image, np.array(prompt)
            )
        else:
            masks, scores, latency, vram = measure_inference_fine_tuning(
                model, image, np.array(prompt), np.array([1])
            )

        if masks is None or len(masks) == 0:
            continue

        best_idx = np.argmax(scores)
        pred_mask = masks[best_idx].astype(bool)

        gt_mask = np.squeeze(gt_mask)
        H, W = gt_mask.shape
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        _accumulate(buf, pred_mask, gt_mask, latency, vram, use_resize_for_hausdorff)

    return _summarize(buf, model_name)


def evaluate_zero_shot(model, model_name, samples_iter, inference_fn,
                       use_resize_for_hausdorff=False, best_by_iou=False):
    """Evalúa un modelo zero-shot sobre el iterador de samples dado. La
    función inference_fn(model, img_path, prompt) hace la inferencia. Si
    best_by_iou=True, selecciona la mejor máscara comparándolas con la GT;
    en caso contrario usa la confianza devuelta por el modelo."""
    buf = _empty_metric_buffers()

    for img_path, gt_mask, prompt in samples_iter:
        results, latency, vram = inference_fn(model, img_path, prompt)

        if results is None:
            continue
        if results[0].masks is None or len(results[0].masks.data) == 0:
            continue

        masks = results[0].masks.data.cpu().numpy()
        gt_mask = np.squeeze(gt_mask)
        H, W = gt_mask.shape

        if best_by_iou and len(masks) > 1:
            ious = []
            for m in masks:
                m_r = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                inter = np.logical_and(m_r, gt_mask).sum()
                union = np.logical_or(m_r, gt_mask).sum()
                ious.append(inter / union if union > 0 else 0)
            best_idx = int(np.argmax(ious))
        elif hasattr(results[0], "boxes") and results[0].boxes is not None and len(results[0].boxes.conf) > 0:
            scores = results[0].boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(scores))
        elif hasattr(results[0], "probs") and results[0].probs is not None:
            best_idx = int(np.argmax(results[0].probs))
        else:
            best_idx = 0

        pred_mask = masks[best_idx].astype(bool)
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        _accumulate(buf, pred_mask, gt_mask, latency, vram, use_resize_for_hausdorff)

    return _summarize(buf, model_name)


def evaluate_unet(model, test_samples, model_name, img_size=512):
    """Evalúa un modelo UNet sobre la lista de pares (img_path, mask_path)."""
    model.to(device)
    model.eval()
    buf = _empty_metric_buffers()

    for img_path, mask_path in test_samples:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]

        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = np.squeeze(gt_mask)
        gt_mask = (gt_mask > 127).astype(bool)

        image_resized = cv2.resize(image, (img_size, img_size))
        image_tensor = torch.tensor(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)

        vram_before = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0
        start = time.time()
        with torch.no_grad():
            pred = model(image_tensor)
        latency = (time.time() - start) * 1000
        vram = (torch.cuda.memory_allocated() / 1024 ** 2 - vram_before) if torch.cuda.is_available() else 0

        pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        pred_mask = cv2.resize(pred_mask, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        _accumulate(buf, pred_mask, gt_mask, latency, vram)

    return _summarize(buf, model_name)


def evaluate_yolo(model, dataset_path, model_name, split="test"):
    """Evalúa un modelo YOLO sobre el split indicado del dataset."""
    test_images_dir = os.path.join(dataset_path, "images", split)
    test_masks_dir = os.path.join(dataset_path, "masks", split)
    buf = _empty_metric_buffers()

    for img_file in os.listdir(test_images_dir):
        if not img_file.endswith(".jpg"):
            continue
        name = os.path.splitext(img_file)[0]
        img_path = os.path.join(test_images_dir, img_file)
        mask_path = os.path.join(test_masks_dir, name + ".jpg")
        if not os.path.exists(mask_path):
            continue

        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = np.squeeze(gt_mask)
        gt_mask = (gt_mask > 127).astype(bool)
        H, W = gt_mask.shape

        vram_before = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0
        start = time.time()
        results = model(img_path, verbose=False)
        latency = (time.time() - start) * 1000
        vram = (torch.cuda.memory_allocated() / 1024 ** 2 - vram_before) if torch.cuda.is_available() else 0

        if results[0].masks is None or len(results[0].masks.data) == 0:
            continue

        scores = results[0].boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(scores))
        pred_mask = results[0].masks.data[best_idx].cpu().numpy().astype(np.uint8)
        pred_mask = cv2.resize(pred_mask, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        _accumulate(buf, pred_mask, gt_mask, latency, vram)

    return _summarize(buf, model_name)
