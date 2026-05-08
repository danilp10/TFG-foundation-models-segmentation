import argparse
import os
import time

import segmentation_models_pytorch as smp
import torch
import yaml
from ultralytics import YOLO

from src.data.dataset import KvasirDataset
from src.data.splitter import split_kvasir
from src.evaluation.evaluator import evaluate_unet, evaluate_yolo
from src.evaluation.results_writer import save_results
from src.models.classical.unet import train_unet
from src.models.classical.yolo import convert_kvasir_to_yolo, train_yolo, write_yolo_yaml

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_unet(paths, training):
    """Entrena UNet sobre Kvasir, guarda los pesos resultantes y reconstruye
    el modelo con los pesos guardados para luego evaluarlo sobre el split
    test. Mide tiempos de entrenamiento y evaluación y los añade al CSV de
    resultados."""
    output_weights = os.path.join(paths["finetuned_weights_dir"], "unet_resnet34_kvasir.pt")
    output_csv = os.path.join(paths["results_finetuning_dir"], "resultados_unet.csv")

    train_dataset = KvasirDataset(paths["kvasir"], "train", img_size=training["img_size"])

    start_train = time.time()
    train_unet(
        train_dataset, output_weights,
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        lr=training["lr"],
        encoder_name=training["encoder_name"],
        encoder_weights=training["encoder_weights"],
    )
    train_time = time.time() - start_train

    model = smp.Unet(encoder_name=training["encoder_name"], encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(output_weights, weights_only=True))

    test_dataset = KvasirDataset(paths["kvasir"], "test", img_size=training["img_size"])

    start_eval = time.time()
    results = evaluate_unet(
        model, test_dataset.samples,
        model_name="unet_resnet34_kvasir",
        img_size=training["img_size"],
    )
    eval_time = time.time() - start_eval

    results["train_time_minutes"] = [round(train_time / 60, 2)]
    results["eval_time_minutes"] = [round(eval_time / 60, 2)]
    save_results(results, output_csv)


def run_yolo(paths, training):
    """Convierte las máscaras de Kvasir al formato de polígonos de YOLO,
    genera el .yaml de configuración, entrena YOLOv8n-seg y evalúa el
    modelo resultante. Mide tiempos de entrenamiento y evaluación y los
    añade al CSV de resultados."""
    output_weights = os.path.join(paths["finetuned_weights_dir"], "yolov8n_seg_kvasir.pt")
    output_csv = os.path.join(paths["results_finetuning_dir"], "resultados_yolo.csv")
    yaml_path = os.path.join(paths["kvasir"], "dataset_yolo.yaml")

    convert_kvasir_to_yolo(paths["kvasir"])
    write_yolo_yaml(paths["kvasir"], yaml_path)

    start_train = time.time()
    train_yolo(
        yaml_path, output_weights,
        epochs=training["epochs"],
        imgsz=training["img_size"],
        batch=training["batch_size"],
    )
    train_time = time.time() - start_train

    model = YOLO(output_weights)

    start_eval = time.time()
    results = evaluate_yolo(model, paths["kvasir"], model_name="yolov8n_seg_kvasir")
    eval_time = time.time() - start_eval

    results["train_time_minutes"] = [round(train_time / 60, 2)]
    results["eval_time_minutes"] = [round(eval_time / 60, 2)]
    save_results(results, output_csv)


def main():
    """Punto de entrada para entrenar y evaluar modelos de segmentación
    clásicos (UNet o YOLO) sobre Kvasir. Lee los YAML de configuración,
    hace el split del dataset (a menos que se pase --skip-split) y delega
    en la función run_* correspondiente."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["unet", "yolo"])
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--skip-split", action="store_true")
    args = parser.parse_args()

    with open(args.paths_config) as f:
        paths = yaml.safe_load(f)
    with open(args.training_config) as f:
        training = yaml.safe_load(f)[args.model]

    if not args.skip_split:
        split_kvasir(paths["kvasir"])

    if args.model == "unet":
        run_unet(paths, training)
    else:
        run_yolo(paths, training)


if __name__ == "__main__":
    main()
    