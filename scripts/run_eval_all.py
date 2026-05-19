import argparse
import subprocess
import sys
import time
from datetime import datetime


SAM_FAMILY = ["sam_b", "sam_l", "sam2_b", "sam2_l", "sam2_1_b", "sam2_1_l", "sam3"]
SAM_FAMILY_ZS = SAM_FAMILY + ["sam3_prompt"]
DATASETS_FT = ["kvasir", "pascals", "isic2016", "refcocog"]
DATASETS_ZS = ["kvasir", "pascals", "isic2016", "refcocog", "mapillary"]
CLASSICAL = [("unet", "kvasir"), ("yolo", "kvasir")]


def build_commands(mode, passes, warmup, output_suffix):
    """Construye la lista de comandos según el modo seleccionado. En modo
    'finetune' evalúa todos los modelos entrenados, incluyendo UNet y YOLO.
    En modo 'zero_shot' evalúa toda la familia SAM (incluido el prompt
    textual) sobre los cinco datasets sin checkpoint específico. En modo
    'all' lanza ambos bloques en serie."""
    commands = []

    if mode in ("finetune", "all"):
        for model in SAM_FAMILY:
            for dataset in DATASETS_FT:
                commands.append([
                    sys.executable, "-m", "scripts.run_eval_only",
                    "--model", model, "--dataset", dataset,
                    "--mode", "finetune",
                    "--passes", str(passes),
                    "--warmup", str(warmup),
                    "--output-suffix", f"{output_suffix}_ft",
                ])
        for model, dataset in CLASSICAL:
            commands.append([
                sys.executable, "-m", "scripts.run_eval_only",
                "--model", model, "--dataset", dataset,
                "--mode", "finetune",
                "--passes", str(passes),
                "--warmup", str(warmup),
                "--output-suffix", f"{output_suffix}_ft",
            ])
        commands.append([
            sys.executable, "-m", "scripts.run_eval_only",
            "--model", "sam3_text", "--dataset", "refcocog",
            "--mode", "finetune",
            "--passes", str(passes),
            "--warmup", str(warmup),
            "--output-suffix", f"{output_suffix}_ft",
        ])

    if mode in ("zero_shot", "all"):
        for model in SAM_FAMILY_ZS:
            for dataset in DATASETS_ZS:
                commands.append([
                    sys.executable, "-m", "scripts.run_eval_only",
                    "--model", model, "--dataset", dataset,
                    "--mode", "zero_shot",
                    "--passes", str(passes),
                    "--warmup", str(warmup),
                    "--output-suffix", f"{output_suffix}_zs",
                ])

    return commands


def run(cmd, log_file):
    """Ejecuta un comando como subproceso aislado y registra su salida en
    log_file. Cada subproceso se ejecuta en su propio espacio de memoria
    para garantizar la liberación completa de VRAM entre experimentos."""
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
    """Punto de entrada del orquestador. Lanza la evaluación de eficiencia
    (latencia y VRAM) para todas las combinaciones modelo×dataset según el
    modo indicado. El modo 'finetune' cubre los modelos ajustados, el modo
    'zero_shot' cubre la familia SAM con pesos base, y el modo 'all' lanza
    ambos en serie."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["finetune", "zero_shot", "all"], default="all")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output-suffix", default="eff")
    parser.add_argument("--log", default="run_eval_all.log")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commands = build_commands(args.mode, args.passes, args.warmup, args.output_suffix)
    print(f"\nTotal de experimentos a lanzar: {len(commands)}")

    if args.dry_run:
        for cmd in commands:
            print(" ".join(cmd))
        return

    overall_start = time.time()
    n_ok = n_fail = 0

    with open(args.log, "w", encoding="utf-8") as log_file:
        log_file.write(f"=== Inicio: {datetime.now()} ===\n")
        log_file.write(f"Modo: {args.mode}\n")
        log_file.write(f"Total experimentos: {len(commands)}\n")
        log_file.write(f"Pasadas por experimento: {args.passes}\n")

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
