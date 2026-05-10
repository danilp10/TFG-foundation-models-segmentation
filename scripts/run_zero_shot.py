import argparse
import json
import os
import pickle
import time
import cv2
import numpy as np
import yaml
from pycocotools.coco import COCO
from ultralytics import SAM
from ultralytics.models.sam import SAM3SemanticPredictor
from src.evaluation.evaluator import evaluate_zero_shot, get_bbox_from_mask
from src.evaluation.inference import (
    measure_inference_central_point,
    measure_inference_refcocog,
    measure_inference_sam3_prompt_zero_shot,
)
from src.evaluation.results_writer import save_results


TEXT_PROMPTS = {
    "kvasir": "polyp",
    "isic2016": "skin lesion",
    "pascals": "object",
    "mapillary": "object",
    "refcocog": "object",
}


def kvasir_iter(dataset_path):
    """Itera sobre todas las muestras de Kvasir leyendo el JSON de bboxes
    para usar el centro de cada caja como prompt. Produce
    (img_path, gt_mask, [[cx, cy]])."""
    with open(os.path.join(dataset_path, "kavsir_bboxes.json")) as f:
        bboxes = json.load(f)
    for img_name, info in bboxes.items():
        img_path = os.path.join(dataset_path, "images", img_name + ".jpg")
        mask_path = os.path.join(dataset_path, "masks", img_name + ".jpg")
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
        gt = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(bool)
        b = info["bbox"][0]
        cx = b["xmin"] + (b["xmax"] - b["xmin"]) / 2
        cy = b["ymin"] + (b["ymax"] - b["ymin"]) / 2
        yield img_path, gt, [[cx, cy]]


def isic_iter(dataset_root):
    """Itera sobre las muestras de ISIC 2016 (train + test) calculando la
    bbox de la lesión a partir de la máscara y devolviendo el centro como
    prompt. Las máscaras tienen sufijo _Segmentation.png en el dataset
    original, y esto se elimina."""
    splits = [
        (os.path.join(dataset_root, "ISBI2016_ISIC_Part1_Training_Data"),
         os.path.join(dataset_root, "ISBI2016_ISIC_Part1_Training_GroundTruth")),
        (os.path.join(dataset_root, "ISBI2016_ISIC_Part1_Test_Data"),
         os.path.join(dataset_root, "ISBI2016_ISIC_Part1_Test_GroundTruth")),
    ]
    for images_dir, masks_dir in splits:
        for mask_filename in sorted(os.listdir(masks_dir)):
            if not mask_filename.lower().endswith(".png"):
                continue
            stem = mask_filename.replace("_Segmentation.png", "")
            img_path = os.path.join(images_dir, stem + ".jpg")
            mask_path = os.path.join(masks_dir, mask_filename)
            if not os.path.exists(img_path):
                continue
            gt = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(bool)
            bbox = get_bbox_from_mask(gt)
            if bbox is None:
                continue
            xmin, ymin, xmax, ymax = bbox
            yield img_path, gt, [[(xmin + xmax) / 2, (ymin + ymax) / 2]]


def pascals_iter(dataset_path):
    """Itera sobre las muestras de PASCAL-S calculando la bbox del objeto a
    partir de los píxeles positivos de la máscara y devolviendo su centro
    como prompt."""
    images_dir = os.path.join(dataset_path, "Image")
    masks_dir = os.path.join(dataset_path, "GT")
    for img_file in os.listdir(images_dir):
        img_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, img_name + ".png")
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
        gt = np.squeeze(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
        gt = (gt > 127).astype(bool)

        coords = np.argwhere(gt)
        if len(coords) == 0:
            continue
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)
        cx = xmin + (xmax - xmin) / 2
        cy = ymin + (ymax - ymin) / 2
        yield img_path, gt, [[cx, cy]]


def mapillary_iter(dataset_path, max_instances_per_image=5, min_pixels=100, seed=42):
    """Itera sobre las muestras de Mapillary Vistas a nivel de instancia, no
    de imagen: para cada imagen, lee el mapa de instancias y produce hasta
    max_instances_per_image muestras (una por objeto), descartando aquellas
    con menos de min_pixels píxeles. La semilla fija qué instancias se
    seleccionan cuando hay más del máximo permitido."""
    images_dir = os.path.join(dataset_path, "images")
    instances_dir = os.path.join(dataset_path, "instances")
    rng = np.random.default_rng(seed)

    for img_file in os.listdir(images_dir):
        img_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(images_dir, img_file)
        inst_path = os.path.join(instances_dir, img_name + ".png")
        if not os.path.exists(img_path) or not os.path.exists(inst_path):
            continue

        inst_img = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED)
        inst_img = np.squeeze(inst_img)
        if inst_img is None:
            continue

        instance_ids = np.unique(inst_img)
        instance_ids = instance_ids[instance_ids != 0]
        if len(instance_ids) > max_instances_per_image:
            instance_ids = rng.choice(instance_ids, size=max_instances_per_image, replace=False)

        for inst_id in instance_ids:
            gt_mask = (inst_img == inst_id)
            coords = np.argwhere(gt_mask)
            if len(coords) < min_pixels:
                continue
            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0)
            cx = xmin + (xmax - xmin) / 2
            cy = ymin + (ymax - ymin) / 2
            yield img_path, gt_mask, [[cx, cy]]


def refcocog_iter(dataset_path, images_subdir="train2014", split="test"):
    """Itera sobre las referencias de RefCOCOg del split indicado, leyendo
    las anotaciones de COCO. A diferencia de los demás datasets, aquí el
    prompt es la bbox completa [xmin, ymin, xmax, ymax] y no un punto."""
    images_dir = os.path.join(dataset_path, images_subdir)
    refs_path = os.path.join(dataset_path, "refs(umd).p")
    instances_path = os.path.join(dataset_path, "instances.json")

    with open(refs_path, "rb") as f:
        refs = pickle.load(f)
    coco = COCO(instances_path)

    for ref in [r for r in refs if r["split"] == split]:
        ann = coco.loadAnns(ref["ann_id"])[0]
        img_info = coco.loadImgs(ref["image_id"])[0]
        img_path = os.path.join(images_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        gt_mask = coco.annToMask(ann).astype(bool)
        if gt_mask.sum() == 0:
            continue

        x, y, w, h = ann["bbox"]
        bbox = [x, y, x + w, y + h]
        yield img_path, gt_mask, bbox


def text_prompt_iter(samples_iter, text_prompt):
    """Envuelve un iterador de samples reemplazando el prompt original (punto
    o bbox) por un prompt textual fijo. Se usa cuando el modelo es
    sam3_prompt y necesita texto en lugar de coordenadas."""
    for img_path, gt, _ in samples_iter:
        yield img_path, gt, text_prompt


def get_iter(dataset_name, paths):
    """Devuelve el iterador apropiado para el dataset indicado."""
    if dataset_name == "kvasir":
        return kvasir_iter(paths["kvasir"])
    if dataset_name == "isic2016":
        return isic_iter(paths["isic2016"])
    if dataset_name == "pascals":
        return pascals_iter(paths["pascals"])
    if dataset_name == "mapillary":
        return mapillary_iter(paths["mapillary"])
    if dataset_name == "refcocog":
        return refcocog_iter(paths["refcocog"])
    raise ValueError(f"Dataset desconocido: {dataset_name}")


def load_model(model_name, paths):
    """Carga el modelo zero-shot indicado. Para sam3_prompt usa
    SAM3SemanticPredictor (que admite prompts textuales); para el resto usa
    el wrapper SAM de Ultralytics."""
    if model_name == "sam3_prompt":
        overrides = dict(conf=0.01, task="segment", mode="predict",
                         model=paths["sam3_weights"], device="cuda")
        return SAM3SemanticPredictor(overrides=overrides)
    return SAM(paths[f"{model_name}_weights"])


def get_inference_fn(model_name, dataset_name):
    """Selecciona la función de inferencia adecuada: prompt textual para
    sam3_prompt, bbox para refcocog, y punto central para los demás."""
    if model_name == "sam3_prompt":
        return measure_inference_sam3_prompt_zero_shot
    if dataset_name == "refcocog":
        return measure_inference_refcocog
    return measure_inference_central_point


DATASETS_BEST_BY_IOU = {"pascals", "mapillary", "refcocog"}
DATASETS_RESIZE_HAUSDORFF = {"isic2016", "mapillary"}


def main():
    """Punto de entrada para evaluar modelos zero-shot. Lee los argumentos
    CLI, carga el modelo y el iterador correspondientes y delega en
    evaluate_zero_shot. Activa best_by_iou y resize_for_hausdorff según los
    conjuntos definidos arriba."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=[
                            "sam_b", "sam_l",
                            "sam2_b", "sam2_l",
                            "sam2_1_b", "sam2_1_l",
                            "sam3", "sam3_prompt",
                        ])
    parser.add_argument("--dataset", required=True,
                        choices=["kvasir", "isic2016", "pascals", "mapillary", "refcocog"])
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    args = parser.parse_args()

    with open(args.paths_config) as f:
        paths = yaml.safe_load(f)

    model = load_model(args.model, paths)

    samples_iter = get_iter(args.dataset, paths)
    if args.model == "sam3_prompt":
        samples_iter = text_prompt_iter(samples_iter, TEXT_PROMPTS[args.dataset])

    inference_fn = get_inference_fn(args.model, args.dataset)

    start = time.time()
    results = evaluate_zero_shot(
        model, args.model, samples_iter, inference_fn,
        use_resize_for_hausdorff=args.dataset in DATASETS_RESIZE_HAUSDORFF,
        best_by_iou=args.dataset in DATASETS_BEST_BY_IOU,
    )
    eval_time = time.time() - start

    results["eval_time_minutes"] = [round(eval_time / 60, 2)]

    output_path = os.path.join(
        paths["results_zero_shot_dir"],
        f"resultados_sam_{args.dataset}.csv",
    )
    save_results(results, output_path)

if __name__ == "__main__":
    main()
