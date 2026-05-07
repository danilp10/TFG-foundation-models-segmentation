import argparse
import os
import pickle
import time

import torch
import yaml
from pycocotools.coco import COCO
from ultralytics import SAM

from src.evaluation.evaluator import evaluate_zero_shot
from src.evaluation.inference import GroundingDinoSAM3Inference
from src.evaluation.results_writer import save_results

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_finetuned_sam3(base_weights, finetuned_weights):
    """Carga el modelo SAM3 base desde Ultralytics y luego sobrescribe con
    los pesos fine-tuneados. Mueve parámetros y buffers a GPU manualmente
    porque el wrapper de Ultralytics no los traslada con un .to() normal."""
    wrapper = SAM(base_weights)
    sd = torch.load(finetuned_weights)["model"]
    wrapper.model.load_state_dict(sd)
    for param in wrapper.model.parameters():
        param.data = param.data.to(device)
    for buffer in wrapper.model.buffers():
        buffer.data = buffer.data.to(device)
    wrapper.model.eval()
    return wrapper


def refcocog_text_iter(dataset_path, images_subdir="train2014", split_dir="test"):
    """Itera sobre las muestras de test de RefCOCOg cargando las anotaciones
    de COCO y filtrando solo las referencias cuyo ann_id corresponda con un
    fichero presente en images/test. Para cada muestra produce
    (img_path, gt_mask, texto_descriptivo) usando la primera oración de lenguaje 
    natural asociada a la referencia."""
    refs_path = os.path.join(dataset_path, "refs(umd).p")
    instances_path = os.path.join(dataset_path, "instances.json")
    test_images_dir = os.path.join(dataset_path, "images", split_dir)
    images_dir = os.path.join(dataset_path, images_subdir)

    with open(refs_path, "rb") as f:
        refs = pickle.load(f)
    coco = COCO(instances_path)

    ann_ids_test = {
        int(os.path.splitext(f)[0])
        for f in os.listdir(test_images_dir)
        if f.endswith(".jpg")
    }
    refs_test = [r for r in refs if r["ann_id"] in ann_ids_test]

    for ref in refs_test:
        ann = coco.loadAnns(ref["ann_id"])[0]
        img_info = coco.loadImgs(ref["image_id"])[0]
        img_path = os.path.join(images_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        gt_mask = coco.annToMask(ann).astype(bool)
        if gt_mask.sum() == 0:
            continue

        text_prompt = ref["sentences"][0]["raw"]
        yield img_path, gt_mask, text_prompt


def main():
    """Evalúa SAM3 fine-tuneado sobre RefCOCOg usando prompts textuales en
    lugar de bounding boxes. La pipeline pasa el texto a Grounding DINO para
    localizar el objeto, y luego SAM3 segmenta la región detectada."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--finetuned-weights", default=None,
                        help="Ruta a los pesos finetuneados. Por defecto usa "
                             "{finetuned_weights_dir}/sam3_refcocog.pt")
    args = parser.parse_args()

    with open(args.paths_config) as f:
        paths = yaml.safe_load(f)

    base_weights = paths["sam3_weights"]
    finetuned_weights = args.finetuned_weights or os.path.join(
        paths["finetuned_weights_dir"], "sam3_refcocog.pt"
    )

    model = load_finetuned_sam3(base_weights, finetuned_weights)

    inference = GroundingDinoSAM3Inference(
        sam_wrapper=model,
        gd_config_path=paths["grounding_dino_config"],
        gd_weights_path=paths["grounding_dino_weights"],
    )

    samples_iter = refcocog_text_iter(paths["refcocog"])

    start = time.time()
    results = evaluate_zero_shot(
        model, "sam3_refcocog_text",
        samples_iter,
        inference,
    )
    eval_time = time.time() - start

    results["train_time_minutes"] = [0]
    results["eval_time_minutes"] = [round(eval_time / 60, 2)]

    output_path = os.path.join(paths["results_finetuning_dir"], "resultados_sam.csv")
    save_results(results, output_path)


if __name__ == "__main__":
    main()
