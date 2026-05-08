import argparse
import subprocess
import sys
import time
from datetime import datetime


ZERO_SHOT_MODELS = [
    "sam_b", "sam_l",
    "sam2_b", "sam2_l",
    "sam2_1_b", "sam2_1_l",
    "sam3", "sam3_prompt",
]
ZERO_SHOT_DATASETS = ["kvasir", "isic2016", "pascals", "mapillary", "refcocog"]

FINETUNE_MODELS = [
    "sam_b", "sam_l",
    "sam2_b", "sam2_l",
    "sam2_1_b", "sam2_1_l",
    "sam3",
]
FINETUNE_DATASETS = ["kvasir", "pascals", "isic2016", "refcocog"]

CLASSICAL_MODELS = ["unet", "yolo"]


def run(cmd, log_file):
    """Ejecuta un comando como subproceso y registra el resultado en log_file.
    Devuelve True si el comando termina con código 0, False en caso contrario."""
    start = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] >>> {' '.join(cmd)}")

    log_file.write(f"\n[{timestamp}] >>> {' '.join(cmd)}\n")
    log_file.flush()

    result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else f"ERROR (code {result.returncode})"
    print(f"    {status} - {elapsed / 60:.2f} min")
    log_file.write(f"    {status} - {elapsed / 60:.2f} min\n")
    log_file.flush()

    return result.returncode == 0


def main():
    """Lanza todos los experimentos en serie como subprocesos independientes
    para garantizar que la VRAM se libera por completo entre ejecuciones.
    Permite filtrar qué grupos correr con flags. Si un experimento falla,
    continúa con el siguiente y registra todos los resultados en un log."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-shot", action="store_true", help="Lanzar zero-shot")
    parser.add_argument("--finetune", action="store_true", help="Lanzar fine-tuning")
    parser.add_argument("--classical", action="store_true", help="Lanzar UNet y YOLO")
    parser.add_argument("--text-eval", action="store_true",
                        help="Lanzar evaluación con texto sobre SAM3 fine-tuneado en refcocog")
    parser.add_argument("--all", action="store_true", help="Lanzar todo")
    parser.add_argument("--log", default="run_all.log", help="Fichero de log")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo imprimir los comandos sin ejecutarlos")
    args = parser.parse_args()

    if args.all:
        args.zero_shot = args.finetune = args.classical = args.text_eval = True

    if not any([args.zero_shot, args.finetune, args.classical, args.text_eval]):
        parser.error("Indica al menos un grupo: --zero-shot, --finetune, --classical, --text-eval o --all")

    commands = []

    if args.zero_shot:
        for model in ZERO_SHOT_MODELS:
            for dataset in ZERO_SHOT_DATASETS:
                commands.append([sys.executable, "-m", "scripts.run_zero_shot",
                                 "--model", model, "--dataset", dataset])

    if args.finetune:
        for model in FINETUNE_MODELS:
            for dataset in FINETUNE_DATASETS:
                commands.append([sys.executable, "-m", "scripts.run_finetune",
                                 "--model", model, "--dataset", dataset])

    if args.classical:
        for model in CLASSICAL_MODELS:
            commands.append([sys.executable, "-m", "scripts.train_classical",
                             "--model", model])

    if args.text_eval:
        commands.append([sys.executable, "-m", "scripts.run_finetune_text_eval"])

    print(f"\nTotal de experimentos a lanzar: {len(commands)}")

    if args.dry_run:
        for cmd in commands:
            print(" ".join(cmd))
        return

    overall_start = time.time()
    n_ok = 0
    n_fail = 0

    with open(args.log, "w", encoding="utf-8") as log_file:
        log_file.write(f"=== Inicio: {datetime.now()} ===\n")
        log_file.write(f"Total experimentos: {len(commands)}\n")

        for i, cmd in enumerate(commands, 1):
            print(f"\n[{i}/{len(commands)}]", end=" ")
            if run(cmd, log_file):
                n_ok += 1
            else:
                n_fail += 1

        total_elapsed = time.time() - overall_start
        summary = (
            f"\n=== Resumen ===\n"
            f"OK:    {n_ok}\n"
            f"FAIL:  {n_fail}\n"
            f"Total: {total_elapsed / 60:.2f} min ({total_elapsed / 3600:.2f} h)\n"
        )
        log_file.write(summary)
        print(summary)


if __name__ == "__main__":
    main()
