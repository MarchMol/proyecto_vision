"""
train.py
--------
Entrena YOLOv8n (nano) para detectar la zona de colocación de logos
en camisetas y tazas/botellas.

¿Por qué YOLOv8n?
  - Es el modelo más pequeño de la familia YOLOv8, ideal para CPU.
  - Transfer learning desde pesos COCO permite buenas métricas incluso
    con datasets pequeños (~150 imágenes).
  - La tarea es detección de una región (bounding box), exactamente lo
    que YOLO resuelve.

Sin GPU — opciones:
  1. LOCAL (CPU)    : Este script corre directo. Estimado ~20-40 min / 50 epochs.
  2. Google Colab   : Gratis, GPU T4. Sube el dataset a Google Drive y monta
                      la unidad. Cambia device='cpu' → device=0.
  3. Kaggle Kernels : Gratis, GPU P100/T4. Similar a Colab.

Uso:
    python src/train.py
    python src/train.py --epochs 50 --batch 8
"""

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"


def build_runtime_yaml() -> Path:
    """
    Genera un data.yaml temporal con el path absoluto correcto para esta máquina.
    El archivo original usa un path hardcoded de otra máquina.
    """
    with open(DATA_YAML) as f:
        config = yaml.safe_load(f)

    config["path"] = str(DATASET_DIR.resolve())

    runtime_yaml = DATASET_DIR / "data_runtime.yaml"
    with open(runtime_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return runtime_yaml


def train(
    epochs: int = 100,
    batch: int = 4,
    imgsz: int = 640,
    patience: int = 20,
    device: str = "cpu",
) -> None:
    """
    Lanza el entrenamiento de YOLOv8n.

    Parámetros clave para CPU:
      - batch=4   : batch pequeño para no saturar RAM.
      - workers=0 : obligatorio en Windows; evita deadlocks con DataLoader.
      - imgsz=640 : resolución estándar YOLO. Bajar a 416 acelera en CPU.
      - patience   : early stopping si val/mAP no mejora en N epochs.
    """
    runtime_yaml = build_runtime_yaml()
    print(f"Dataset config: {runtime_yaml}")
    print(f"Dispositivo   : {device}")
    print(f"Epochs        : {epochs}  |  Batch: {batch}  |  Img size: {imgsz}")
    print("-" * 55)

    # Descarga automática de pesos preentrenados en COCO (37 MB)
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(runtime_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,          # 0 es obligatorio en Windows con CPU
        patience=patience,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,           # learning rate final = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Augmentaciones — útiles con datasets pequeños
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        # Rutas de salida
        project=str(RUNS_DIR / "train"),
        name="logo_placement",
        exist_ok=True,
        save=True,
        save_period=10,     # guarda checkpoint cada 10 epochs
        plots=True,         # genera curvas de entrenamiento
        verbose=True,
    )

    best_weights = RUNS_DIR / "train" / "logo_placement" / "weights" / "best.pt"
    print("\n" + "=" * 55)
    print("Entrenamiento completado.")
    print(f"Mejores pesos : {best_weights}")
    print(f"mAP50         : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"mAP50-95      : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")

    # Limpia el yaml temporal
    runtime_yaml.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena YOLOv8n para detección de zona de logo")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4,
                        help="Tamaño de batch. Usa 4-8 en CPU, 16-32 con GPU.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Tamaño de imagen. Usa 416 para acelerar en CPU.")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping: epochs sin mejora antes de detener.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="'cpu' o '0' para GPU. En Colab usa '0'.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device,
    )
