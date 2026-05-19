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
from ultralytics import SAM, YOLO
from ultralytics.models.sam import SAM3SemanticPredictor
import segmentation_models_pytorch as smp

from src.evaluation.evaluator import get_bbox_from_mask
from src.evaluation.inference import (
    measure_inference_central_point,
    measure_inference_refcocog,
    measure_inference_sam3_prompt_zero_shot,
    measure_inference_fine_tuning,
    measure_inference_fine_tuning_refcocog,
)
from src.evaluation.results_writer import save_results

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


MODEL_REGISTRY = {
    "sam_b": {"family": "sam", "vit": "vit_b"},
    "sam_l": {"family": "sam", "vit": "vit_l"},
    "sam2_b": {"family": "sam2", "config_key": "sam2_b_config"},
    "sam2_l": {"family": "sam2", "config_key": "sam2_l_config"},
    "sam2_1_b": {"family": "sam2", "config_key": "sam2_1_b_config"},
    "sam2_1_l": {"family": "sam2", "config_key": "sam2_1_l_config"},
    "sam3": {"family": "sam3"},
    "sam3_text": {"family": "sam3"},
}


TEXT_PROMPTS = {
    "kvasir": "polyp",
    "isic2016": "skin lesion",
    "pascals": "object",
    "mapillary": None,
    "refcocog": None,
}


def kvasir_samples(paths):
    """Devuelve la lista de samples de test de Kvasir-SEG en el formato
    (img_path, mask_path, prompt, prompt_type), donde el prompt es el
    centro de la bbox del JSON de anotaciones."""
    with open(paths["kvasir_bboxes"]) as f:
        bboxes = json.load(f)
    samples = []
    for name, info in bboxes.items():
        img = os.path.join(paths["kvasir"], "images", "test", name + ".jpg")
        mask = os.path.join(paths["kvasir"], "masks", "test", name + ".jpg")
        if os.path.exists(img) and os.path.exists(mask):
            b = info["bbox"][0]
            cx = b["xmin"] + (b["xmax"] - b["xmin"]) / 2
            cy = b["ymin"] + (b["ymax"] - b["ymin"]) / 2
            samples.append((img, mask, [[cx, cy]], "point"))
    return samples


def pascals_samples(paths):
    """Devuelve la lista de samples de test de PASCAL-S. El prompt es el
    centroide de los píxeles positivos de la máscara."""
    images_dir = os.path.join(paths["pascals"], "Image", "test")
    masks_dir = os.path.join(paths["pascals"], "GT", "test")
    samples = []
    for img_file in sorted(os.listdir(images_dir)):
        if not img_file.endswith(".jpg"):
            continue
        name = img_file.replace(".jpg", "")
        img = os.path.join(images_dir, img_file)
        mask = os.path.join(masks_dir, name + ".png")
        if not os.path.exists(mask):
            continue
        gt = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        gt = np.squeeze(gt)
        ys, xs = np.where(gt > 127)
        if len(xs) == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        samples.append((img, mask, [[cx, cy]], "point"))
    return samples


def isic_samples(paths):
    """Devuelve la lista de samples de test de ISIC 2016. El prompt es el
    centro de la bbox del contorno principal de la máscara."""
    images_dir = os.path.join(paths["isic2016_split"], "images", "test")
    masks_dir = os.path.join(paths["isic2016_split"], "masks", "test")
    samples = []
    for img_file in sorted(os.listdir(images_dir)):
        if not img_file.lower().endswith(".jpg"):
            continue
        name = img_file.replace(".jpg", "")
        img = os.path.join(images_dir, img_file)
        mask = os.path.join(masks_dir, name + ".png")
        if not os.path.exists(mask):
            continue
        gt = cv2.imread(mask, cv2.IMREAD_GRAYSCALE) > 127
        bbox = get_bbox_from_mask(gt.astype(np.uint8))
        if bbox is None:
            continue
        xmin, ymin, xmax, ymax = bbox
        samples.append((img, mask, [[(xmin + xmax) / 2, (ymin + ymax) / 2]], "point"))
    return samples


def refcocog_samples(paths):
    """Devuelve la lista de samples de test de RefCOCOg en formato geométrico
    (bbox). Cada sample contiene también la frase referencial original, que
    se utilizará cuando el modelo sea sam3_prompt en modo zero-shot."""
    with open(os.path.join(paths["refcocog"], "refs(umd).p"), "rb") as f:
        refs = pickle.load(f)
    coco = COCO(os.path.join(paths["refcocog"], "instances.json"))
    images_dir = os.path.join(paths["refcocog"], "images", "test")
    masks_dir = os.path.join(paths["refcocog"], "masks", "test")
    refs_by_ann = {r["ann_id"]: r for r in refs}
    samples = []
    for img_file in sorted(os.listdir(images_dir)):
        if not img_file.endswith(".jpg"):
            continue
        name = os.path.splitext(img_file)[0]
        ann_id = int(name)
        ann = coco.loadAnns(ann_id)[0]
        x, y, w, h = ann["bbox"]
        img = os.path.join(images_dir, img_file)
        mask = os.path.join(masks_dir, name + ".png")
        if not os.path.exists(mask):
            continue
        text = refs_by_ann[ann_id]["sentences"][0]["raw"]
        samples.append((img, mask, [x, y, x + w, y + h], "bbox", text))
    return samples


def mapillary_samples(paths, max_instances_per_image=5, min_pixels=100, seed=42):
    """Devuelve la lista de samples de test de Mapillary Vistas a nivel de
    instancia, replicando la lógica del iterador zero-shot. Cada sample
    incluye también el nombre de clase obtenido de config.json, que se
    utilizará cuando el modelo sea sam3_prompt."""
    images_dir = os.path.join(paths["mapillary"], "images")
    instances_dir = os.path.join(paths["mapillary"], "instances")
    config_path = os.path.join(paths["mapillary"], "config.json")

    with open(config_path) as f:
        config = json.load(f)
    class_names = {i: label["readable"] for i, label in enumerate(config["labels"])}

    rng = np.random.default_rng(seed)
    samples = []

    for img_file in sorted(os.listdir(images_dir)):
        name = os.path.splitext(img_file)[0]
        img_path = os.path.join(images_dir, img_file)
        inst_path = os.path.join(instances_dir, name + ".png")
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
            class_id = int(inst_id) >> 8
            text = class_names.get(class_id, "object")
            samples.append((img_path, inst_path, [[cx, cy]], "point", text))
    return samples


SAMPLES_FN = {
    "kvasir": kvasir_samples,
    "pascals": pascals_samples,
    "isic2016": isic_samples,
    "refcocog": refcocog_samples,
    "mapillary": mapillary_samples,
}


def load_predictor_finetune(model_name, weights_path, base_weights, config_path):
    """Reconstruye el predictor del modelo cargando los pesos fine-tuneados
    sobre la arquitectura base correspondiente. Despacha al constructor
    adecuado en función de la familia del modelo indicada en el
    MODEL_REGISTRY."""
    info = MODEL_REGISTRY[model_name]
    family = info["family"]
    if family == "sam":
        sam = sam_model_registry[info["vit"]](checkpoint=weights_path)
        sam.to(device).eval()
        return SamPredictor(sam)
    if family == "sam2":
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


def load_predictor_zero_shot(model_name, paths):
    """Carga el modelo zero-shot indicado sin aplicar ningún checkpoint
    fine-tuneado. Para sam3_prompt usa SAM3SemanticPredictor (que admite
    prompts textuales); para el resto usa el wrapper SAM de Ultralytics."""
    if model_name == "sam3_prompt":
        overrides = dict(conf=0.01, task="segment", mode="predict",
                         model=paths["sam3_weights"], device="cuda")
        return SAM3SemanticPredictor(overrides=overrides)
    return SAM(paths[f"{model_name}_weights"])


def measure_finetuned(predictor, model_name, samples):
    """Ejecuta una pasada completa de inferencia sobre samples usando el
    predictor fine-tuneado correspondiente. Solo registra latencia y
    consumo de VRAM, descartando las máscaras producidas para no recalcular
    métricas de calidad."""
    is_sam3 = model_name.startswith("sam3")
    latencies = []
    vrams = []
    for sample in samples:
        img_path, _, prompt, prompt_type = sample[:4]
        if is_sam3:
            if prompt_type == "bbox":
                _, lat, vram = measure_inference_refcocog(predictor, img_path, prompt)
            else:
                _, lat, vram = measure_inference_central_point(predictor, img_path, prompt)
        else:
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            if prompt_type == "bbox":
                _, _, lat, vram = measure_inference_fine_tuning_refcocog(predictor, image, np.array(prompt))
            else:
                _, _, lat, vram = measure_inference_fine_tuning(predictor, image, np.array(prompt), np.array([1]))
        latencies.append(lat)
        vrams.append(vram)
    return latencies, vrams


def measure_zero_shot(model, model_name, dataset_name, samples):
    """Ejecuta una pasada completa de inferencia zero-shot sobre samples.
    Despacha a la función de medición apropiada según el modelo: prompt
    textual para sam3_prompt (extraído del sample en RefCOCOg y Mapillary o
    del diccionario TEXT_PROMPTS en el resto de dominios), bbox para
    RefCOCOg con prompts geométricos, y punto central para el resto."""
    latencies = []
    vrams = []
    is_text_model = model_name == "sam3_prompt"

    for sample in samples:
        img_path, _, prompt, prompt_type = sample[:4]

        if is_text_model:
            if dataset_name in ("refcocog", "mapillary"):
                text_prompt = sample[4]
            else:
                text_prompt = TEXT_PROMPTS[dataset_name]
            _, lat, vram = measure_inference_sam3_prompt_zero_shot(model, img_path, text_prompt)
        elif prompt_type == "bbox":
            _, lat, vram = measure_inference_refcocog(model, img_path, prompt)
        else:
            _, lat, vram = measure_inference_central_point(model, img_path, prompt)

        latencies.append(lat)
        vrams.append(vram)
    return latencies, vrams


def measure_unet(model, samples, img_size=512):
    """Ejecuta una pasada completa de inferencia sobre samples con un modelo
    UNet ya cargado, midiendo únicamente latencia y consumo de VRAM."""
    model.to(device).eval()
    latencies = []
    vrams = []
    for sample in samples:
        img_path = sample[0]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_size, img_size))
        tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model(tensor)
        torch.cuda.synchronize()
        latencies.append((time.time() - start) * 1000)
        vrams.append(torch.cuda.max_memory_allocated() / 1024 ** 2)
    return latencies, vrams


def measure_yolo(model, samples):
    """Ejecuta una pasada completa de inferencia sobre samples con un modelo
    YOLOv8-Seg ya cargado, midiendo únicamente latencia y consumo de VRAM."""
    latencies = []
    vrams = []
    for sample in samples:
        img_path = sample[0]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        _ = model(img_path, verbose=False)
        torch.cuda.synchronize()
        latencies.append((time.time() - start) * 1000)
        vrams.append(torch.cuda.max_memory_allocated() / 1024 ** 2)
    return latencies, vrams


def measure_finetuned_text(inference_obj, samples):
    """Ejecuta una pasada completa sobre samples usando la pipeline
    Grounding DINO + SAM 3 fine-tuneado. Cada sample aporta su frase
    referencial original (quinto elemento) como prompt textual."""
    latencies = []
    vrams = []
    for sample in samples:
        img_path = sample[0]
        text_prompt = sample[4]
        _, lat, vram = inference_obj(None, img_path, text_prompt)
        latencies.append(lat)
        vrams.append(vram)
    return latencies, vrams


def summarize(model_name, dataset_name, all_latencies, all_vrams, warmup):
    """Calcula la media y desviación estándar de latencia y VRAM sobre las
    imágenes de una pasada, descartando las primeras warmup imágenes para
    eliminar el efecto del calentamiento de los kernels CUDA."""
    lat = np.array(all_latencies[warmup:])
    vram = np.array(all_vrams[warmup:])
    return {
        "modelo": [f"{model_name}_{dataset_name}"],
        "mean_latency_ms": [float(np.mean(lat))],
        "std_latency_ms": [float(np.std(lat))],
        "mean_vram_mb": [float(np.mean(vram))],
        "std_vram_mb": [float(np.std(vram))],
        "n_images": [len(lat)],
    }


def main():
    """Punto de entrada para evaluar la eficiencia (latencia y consumo de
    VRAM) de un modelo, sin recalcular métricas de calidad. El parámetro
    --mode controla si la evaluación se realiza sobre el modelo fine-tuneado
    correspondiente o sobre el modelo zero-shot con sus pesos base. En el
    modo finetune, UNet y YOLO también son evaluables sobre Kvasir-SEG. Por
    cada pasada se descartan las primeras --warmup imágenes para reducir la
    varianza inicial de CUDA y se deja una fila en el CSV de resultados con
    la media y desviación estándar de las métricas registradas."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", choices=["finetune", "zero_shot"], default="finetune")
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output-suffix", default="eff")
    args = parser.parse_args()

    with open(args.paths_config) as f:
        paths = yaml.safe_load(f)

    samples = SAMPLES_FN[args.dataset](paths)

    if args.mode == "finetune":
        if args.model == "unet":
            weights = os.path.join(paths["finetuned_weights_dir"], "unet_resnet34_kvasir.pt")
            model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
            model.load_state_dict(torch.load(weights, weights_only=True))
            measure_fn = lambda: measure_unet(model, samples)
        elif args.model == "yolo":
            weights = os.path.join(paths["finetuned_weights_dir"], "yolov8n_seg_kvasir.pt")
            model = YOLO(weights)
            measure_fn = lambda: measure_yolo(model, samples)
        elif args.model == "sam3_text":
            from src.evaluation.inference import GroundingDinoSAM3Inference
            finetuned = os.path.join(paths["finetuned_weights_dir"], "sam3_refcocog.pt")
            base = paths["sam3_weights"]
            wrapper = SAM(base)
            sd = torch.load(finetuned)["model"]
            wrapper.model.load_state_dict(sd)
            for p in wrapper.model.parameters():
                p.data = p.data.to(device)
            for b in wrapper.model.buffers():
                b.data = b.data.to(device)
            wrapper.model.eval()
            inference_obj = GroundingDinoSAM3Inference(
                sam_wrapper=wrapper,
                gd_config_path=paths["grounding_dino_config"],
                gd_weights_path=paths["grounding_dino_weights"],
            )
            measure_fn = lambda: measure_finetuned_text(inference_obj, samples)
        else:
            finetuned = os.path.join(paths["finetuned_weights_dir"], f"{args.model}_{args.dataset}.pt")
            base = paths[f"{args.model}_weights"]
            config = paths.get(MODEL_REGISTRY[args.model].get("config_key", ""))
            predictor = load_predictor_finetune(args.model, finetuned, base, config)
            measure_fn = lambda: measure_finetuned(predictor, args.model, samples)
    else:
        model = load_predictor_zero_shot(args.model, paths)
        measure_fn = lambda: measure_zero_shot(model, args.model, args.dataset, samples)

    print(f"Warm-up pass...")
    measure_fn()

    output_csv = os.path.join(paths["results_finetuning_dir"], f"resultados_{args.output_suffix}.csv")

    for i in range(args.passes):
        print(f"Pass {i + 1}/{args.passes}...")
        latencies, vrams = measure_fn()
        results = summarize(args.model, args.dataset, latencies, vrams, args.warmup)
        results["pass"] = [i + 1]
        results["mode"] = [args.mode]
        save_results(results, output_csv)
        print(f"  latency={results['mean_latency_ms'][0]:.2f} ± {results['std_latency_ms'][0]:.2f} ms")
        print(f"  vram={results['mean_vram_mb'][0]:.2f} ± {results['std_vram_mb'][0]:.2f} MB")


if __name__ == "__main__":
    main()
