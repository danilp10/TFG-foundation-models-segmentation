import argparse
import json
import os
import pickle
import time

import cv2
import numpy as np
import torch
import yaml
from pycocotools.coco import COCO
from segment_anything import SamPredictor, sam_model_registry
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import SAM

from src.data.dataset import ISICSegDataset, KvasirSegDataset, PascalSegDataset, RefCOCOgSegDataset
from src.data.splitter import split_isic2016, split_kvasir, split_pascals, split_refcocog
from src.evaluation.evaluator import evaluate_finetuned, get_bbox_from_mask
from src.evaluation.results_writer import save_results
from src.models.sam.finetune import train_sam
from src.models.sam2.finetune import train_sam2
from src.models.sam2_1.finetune import train_sam2_1
from src.models.sam3.finetune import train_sam3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


MODEL_REGISTRY = {
    "sam_b": {"family": "sam", "vit": "vit_b"},
    "sam_l": {"family": "sam", "vit": "vit_l"},
    "sam2_b": {"family": "sam2", "config_key": "sam2_b_config"},
    "sam2_l": {"family": "sam2", "config_key": "sam2_l_config"},
    "sam2_1_b": {"family": "sam2_1", "config_key": "sam2_1_b_config"},
    "sam2_1_l": {"family": "sam2_1", "config_key": "sam2_1_l_config"},
    "sam3": {"family": "sam3"},
}


DATASET_REGISTRY = {
    "kvasir": {
        "split_fn": split_kvasir,
        "dataset_cls": KvasirSegDataset,
        "needs_bbox_json": True,
    },
    "pascals": {
        "split_fn": split_pascals,
        "dataset_cls": PascalSegDataset,
        "needs_bbox_json": False,
    },
    "isic2016": {
        "split_fn": split_isic2016,
        "dataset_cls": ISICSegDataset,
        "needs_bbox_json": False,
    },
    "refcocog": {
        "split_fn": split_refcocog,
        "dataset_cls": RefCOCOgSegDataset,
        "needs_bbox_json": False,
    },
}


def build_train_dataset(dataset_name, paths, img_size, mask_size):
    """Construye la clase Dataset apropiada para el dataset y la pasa al
    entrenamiento. Cada dataset tiene su propia subclase porque la fuente
    del prompt es distinta."""
    info = DATASET_REGISTRY[dataset_name]
    cls = info["dataset_cls"]
    path_key = "isic2016_split" if dataset_name == "isic2016" else dataset_name
    dataset_path = paths[path_key]

    if info["needs_bbox_json"]:
        return cls(dataset_path, "train", paths["kvasir_bboxes"], img_size=img_size, mask_size=mask_size)
    return cls(dataset_path, "train", img_size=img_size, mask_size=mask_size)


def kvasir_prompt_fn(bboxes):
    """Devuelve una función que, dado img_path, busca el bbox del JSON y
    calcula el punto central."""
    def fn(img_path):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        info = bboxes.get(img_name)
        if info is None:
            return None
        b = info["bbox"][0]
        return [[b["xmin"] + (b["xmax"] - b["xmin"]) / 2, b["ymin"] + (b["ymax"] - b["ymin"]) / 2]]
    return fn


def centroid_prompt_fn(gt_mask):
    """Devuelve el centroide de los píxeles positivos de la máscara como
    punto único."""
    ys, xs = np.where(gt_mask)
    if len(xs) == 0:
        return None
    return [[float(xs.mean()), float(ys.mean())]]


def isic_prompt_fn(gt_mask):
    """Devuelve el centro de la bbox calculada del contorno de la máscara."""
    bbox = get_bbox_from_mask(gt_mask)
    if bbox is None:
        return None
    xmin, ymin, xmax, ymax = bbox
    return [[xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2]]


def refcocog_bbox_prompt_fn(refs_by_ann_id, coco):
    """Devuelve una función que, dado img_path, busca la anotación de COCO
    correspondiente y devuelve la bbox como [xmin, ymin, xmax, ymax]."""
    def fn(img_path):
        ann_id = int(os.path.splitext(os.path.basename(img_path))[0])
        if ann_id not in refs_by_ann_id:
            return None
        ann = coco.loadAnns(ann_id)[0]
        x, y, w, h = ann["bbox"]
        return [x, y, x + w, y + h]
    return fn


def kvasir_samples_iter(dataset_path, bboxes):
    """Itera sobre las muestras de test de Kvasir."""
    for img_name in bboxes.keys():
        img_path = os.path.join(dataset_path, "images", "test", img_name + ".jpg")
        mask_path = os.path.join(dataset_path, "masks", "test", img_name + ".jpg")
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
        gt = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(bool)
        yield img_path, gt


def pascals_samples_iter(dataset_path):
    """Itera sobre las muestras de test de PASCAL-S."""
    images_dir = os.path.join(dataset_path, "Image", "test")
    masks_dir = os.path.join(dataset_path, "GT", "test")
    for img_file in sorted(os.listdir(images_dir)):
        if not img_file.endswith(".jpg"):
            continue
        name = img_file.replace(".jpg", "")
        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, name + ".png")
        if not os.path.exists(mask_path):
            continue
        gt = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(bool)
        yield img_path, gt


def isic_samples_iter(dataset_path):
    """Itera sobre las muestras de test de ISIC 2016."""
    images_dir = os.path.join(dataset_path, "images", "test")
    masks_dir = os.path.join(dataset_path, "masks", "test")
    for img_file in sorted(os.listdir(images_dir)):
        if not img_file.lower().endswith(".jpg"):
            continue
        name = img_file.replace(".jpg", "")
        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, name + ".png")
        if not os.path.exists(mask_path):
            continue
        gt = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(bool)
        if gt.sum() == 0:
            continue
        yield img_path, gt


def refcocog_samples_iter(dataset_path):
    """Itera sobre las muestras de test de RefCOCOg."""
    images_dir = os.path.join(dataset_path, "images", "test")
    masks_dir = os.path.join(dataset_path, "masks", "test")
    for img_file in sorted(os.listdir(images_dir)):
        if not img_file.endswith(".jpg"):
            continue
        name = os.path.splitext(img_file)[0]
        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, name + ".png")
        if not os.path.exists(mask_path):
            continue
        gt = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(bool)
        yield img_path, gt


def get_samples_iter_and_prompt(dataset_name, paths):
    """Devuelve el iterador de samples y la función de prompt apropiados
    para el dataset. RefCOCOg usa bbox del JSON de COCO y el resto usa punto
    central calculado de la máscara o de un bbox."""
    if dataset_name == "kvasir":
        with open(paths["kvasir_bboxes"]) as f:
            bboxes = json.load(f)
        return kvasir_samples_iter(paths["kvasir"], bboxes), kvasir_prompt_fn(bboxes)
    if dataset_name == "pascals":
        return pascals_samples_iter(paths["pascals"]), centroid_prompt_fn
    if dataset_name == "isic2016":
        return isic_samples_iter(paths["isic2016_split"]), isic_prompt_fn
    if dataset_name == "refcocog":
        with open(os.path.join(paths["refcocog"], "refs(umd).p"), "rb") as f:
            refs = pickle.load(f)
        coco = COCO(os.path.join(paths["refcocog"], "instances.json"))
        refs_by_ann_id = {r["ann_id"]: r for r in refs}
        return refcocog_samples_iter(paths["refcocog"]), refcocog_bbox_prompt_fn(refs_by_ann_id, coco)
    raise ValueError(f"Dataset desconocido: {dataset_name}")


def load_finetuned_predictor(model_name, weights_path, base_weights, config_path=None):
    """Reconstruye el predictor del modelo cargando los pesos finetuneados
    sobre el modelo base."""
    info = MODEL_REGISTRY[model_name]
    family = info["family"]

    if family == "sam":
        sam = sam_model_registry[info["vit"]](checkpoint=weights_path)
        sam.to(device).eval()
        return SamPredictor(sam)
    if family in ("sam2", "sam2_1"):
        sam2 = build_sam2(config_path, base_weights)
        sd = torch.load(weights_path)["model"]
        sam2.load_state_dict(sd)
        sam2.to(device).eval()
        return SAM2ImagePredictor(sam2)
    if family == "sam3":
        wrapper = SAM(base_weights)
        sd = torch.load(weights_path)["model"]
        wrapper.model.load_state_dict(sd)
        wrapper.model.to(device).eval()
        return wrapper
    raise ValueError(f"Familia desconocida: {family}")


def run_train(model_name, train_dataset, weights_path, output_weights, config_path, training_cfg):
    """Despacha al train_* correspondiente según la familia del modelo."""
    info = MODEL_REGISTRY[model_name]
    family = info["family"]

    if family == "sam":
        return train_sam(train_dataset, weights_path, output_weights, vit=info["vit"], **training_cfg)
    if family == "sam2":
        return train_sam2(train_dataset, weights_path, config_path, output_weights, **training_cfg)
    if family == "sam2_1":
        return train_sam2_1(train_dataset, weights_path, config_path, output_weights, **training_cfg)
    if family == "sam3":
        return train_sam3(train_dataset, weights_path, output_weights, **training_cfg)
    raise ValueError(f"Familia desconocida: {family}")


def main():
    """Punto de entrada para entrenar y evaluar modelos SAM/SAM2/SAM2.1/SAM3
    fine-tuneados. Lee los YAML de configuración, hace el split del dataset
    (a menos que se pase --skip-split), entrena la variante elegida del
    modelo y la evalúa sobre el split test. Mide tiempos de entrenamiento
    y evaluación y guarda los resultados en CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["sam_b", "sam_l", "sam2_b", "sam2_l", "sam2_1_b", "sam2_1_l", "sam3"])
    parser.add_argument("--dataset", required=True, choices=["kvasir", "pascals", "isic2016", "refcocog"])
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--skip-split", action="store_true")
    args = parser.parse_args()

    with open(args.paths_config) as f:
        paths = yaml.safe_load(f)
    with open(args.training_config) as f:
        training_all = yaml.safe_load(f)
        training = training_all[MODEL_REGISTRY[args.model]["family"]]

    if not args.skip_split:
        split_info = DATASET_REGISTRY[args.dataset]
        if args.dataset == "isic2016":
            split_info["split_fn"](paths["isic2016"], paths["isic2016_split"])
        else:
            split_info["split_fn"](paths[args.dataset])

    train_dataset = build_train_dataset(
        args.dataset, paths,
        img_size=training["img_size"], mask_size=training["mask_size"],
    )

    base_weights = paths[f"{args.model}_weights"]
    config_path = paths.get(MODEL_REGISTRY[args.model].get("config_key", ""))
    output_weights = os.path.join(paths["finetuned_weights_dir"], f"{args.model}_{args.dataset}.pt")
    family = MODEL_REGISTRY[args.model]["family"]
    output_csv = os.path.join(paths["results_finetuning_dir"], f"resultados_{family}.csv")

    training_cfg = {k: training[k] for k in ("epochs", "batch_size", "lr") if k in training}

    start_train = time.time()
    trained_weights = run_train(args.model, train_dataset, base_weights, output_weights, config_path, training_cfg)
    train_time = time.time() - start_train

    predictor = load_finetuned_predictor(args.model, trained_weights, base_weights, config_path)
    samples_iter, prompt_fn = get_samples_iter_and_prompt(args.dataset, paths)

    use_resize = args.dataset == "isic2016"
    prompt_type = "bbox" if args.dataset == "refcocog" else "point"

    start_eval = time.time()
    results = evaluate_finetuned(
        predictor, f"{args.model}_{args.dataset}",
        samples_iter, prompt_fn,
        use_resize_for_hausdorff=use_resize,
        prompt_type=prompt_type,
    )
    eval_time = time.time() - start_eval

    results["train_time_minutes"] = [round(train_time / 60, 2)]
    results["eval_time_minutes"] = [round(eval_time / 60, 2)]
    save_results(results, output_csv)


if __name__ == "__main__":
    main()
    