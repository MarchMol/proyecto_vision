"""
tools/iou_merge.py
------------------
Fusiona las anotaciones de múltiples anotadores y genera el ground truth
del conjunto de prueba en formato YOLO.

Flujo:
  1. Lee todos los JSONs de test_set/annotations/
  2. Agrupa las cajas por imagen
  3. Calcula IoU inter-anotador (acuerdo) para cada imagen
  4. Genera la caja fusionada como promedio de todas las cajas
  5. Exporta test_set/labels/<imagen>.txt en formato YOLO
  6. Genera test_set/agreement_report.json con métricas de acuerdo

Uso:
    python tools/iou_merge.py
    python tools/iou_merge.py --min-iou 0.4
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "test_set" / "annotations"
IMAGES_DIR = PROJECT_ROOT / "test_set" / "images"
LABELS_DIR = PROJECT_ROOT / "test_set" / "labels"

CLASS_IDS = {"bottles": 0, "tshirts": 1}


# ── geometry ──────────────────────────────────────────────────────────────────

def iou(a: list, b: list) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def average_box(boxes: list[list]) -> list[int]:
    return [int(sum(b[i] for b in boxes) / len(boxes)) for i in range(4)]


def box_to_yolo(box: list, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


# ── main ──────────────────────────────────────────────────────────────────────

def run(min_iou: float = 0.3) -> None:
    ann_files = sorted(ANNOTATIONS_DIR.glob("*.json"))
    if not ann_files:
        print(f"No se encontraron archivos JSON en {ANNOTATIONS_DIR}")
        print("Corre primero tools/annotator.py para al menos un anotador.")
        return

    # Index annotations by image name
    by_image: dict[str, list[dict]] = {}
    annotators: list[str] = []

    for ann_file in ann_files:
        with open(ann_file, encoding="utf-8") as f:
            data = json.load(f)
        annotators.append(data["annotator"])
        for ann in data["annotations"]:
            if not ann.get("box"):
                continue
            by_image.setdefault(ann["image"], []).append(
                {
                    "annotator": data["annotator"],
                    "class": ann.get("class", "bottles"),
                    "box": ann["box"],
                }
            )

    print(f"Anotadores encontrados : {', '.join(annotators)}")
    print(f"Imágenes con ≥1 caja   : {len(by_image)}")
    print("-" * 65)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    exported = skipped = single = 0

    for img_name in sorted(by_image):
        anns = by_image[img_name]
        boxes = [a["box"] for a in anns]
        classes = [a["class"] for a in anns]

        # Inter-annotator IoU
        if len(boxes) >= 2:
            ious = [iou(a, b) for a, b in combinations(boxes, 2)]
            mean_iou = sum(ious) / len(ious)
        else:
            mean_iou = None
            single += 1

        # Majority class
        cls_name = max(set(classes), key=classes.count)
        cls_id = CLASS_IDS.get(cls_name, 0)

        # Merged (average) ground-truth box
        merged = average_box(boxes)

        # Decide status
        if mean_iou is not None and mean_iou < min_iou:
            status = f"low_agreement"
            skipped += 1
        else:
            status = "ok"

        iou_str = f"{mean_iou:.3f}" if mean_iou is not None else "solo 1 anotador"
        flag = "⚠" if status != "ok" else " "
        print(
            f"{flag} {img_name:<35} "
            f"n={len(anns)}  acuerdo={iou_str:<8}  cls={cls_name}"
        )

        report.append(
            {
                "image": img_name,
                "annotators": len(anns),
                "mean_iou": round(mean_iou, 4) if mean_iou is not None else None,
                "class": cls_name,
                "merged_box": merged,
                "status": status,
            }
        )

        if status != "ok":
            continue

        # Resolve image path to get dimensions
        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            found = list(IMAGES_DIR.rglob(img_name))
            if not found:
                print(f"  WARN: imagen no encontrada para dimensiones: {img_name}")
                continue
            img_path = found[0]

        img_w, img_h = Image.open(img_path).size
        cx, cy, w, h = box_to_yolo(merged, img_w, img_h)

        label_path = LABELS_DIR / (Path(img_name).stem + ".txt")
        label_path.write_text(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        exported += 1

    # Summary report
    report_path = PROJECT_ROOT / "test_set" / "agreement_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("-" * 65)
    print(f"Etiquetas exportadas    : {exported}")
    print(f"Omitidas (IoU < {min_iou:.1f})  : {skipped}")
    print(f"Solo 1 anotador         : {single}")
    print(f"Reporte guardado en     : {report_path}")

    # Quick agreement summary
    multi = [r for r in report if r["mean_iou"] is not None]
    if multi:
        avg_agreement = sum(r["mean_iou"] for r in multi) / len(multi)
        print(f"IoU promedio global     : {avg_agreement:.3f}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Fusiona anotaciones y genera ground truth YOLO para el conjunto de prueba"
    )
    p.add_argument(
        "--min-iou", type=float, default=0.3,
        help="IoU mínimo inter-anotador para incluir la imagen (default: 0.3)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(min_iou=args.min_iou)