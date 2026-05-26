"""
tools/iou_merge.py
------------------
Lee test_images/labels_raw/, fusiona las anotaciones de múltiples anotadores
con un promedio ponderado por consenso, y escribe el ground truth final en
test_images/labels/ en formato YOLO estándar.

Algoritmo (weighted-consensus merge):
    Para cada imagen con N anotadores:
      1. Calcular IoU par-a-par entre todas las cajas.
      2. consensus_score[i] = mean(iou(i, j) para j ≠ i)
         → cajas que coinciden con las demás pesan más.
      3. Si todos los scores son 0 (desacuerdo total) → pesos uniformes 1/N.
      4. Caja fusionada = promedio ponderado de (cx, cy, w, h).
      5. Clase = mayoría de votos.
    Toda imagen se exporta siempre; --min-iou solo determina el flag en el
    reporte (no omite imágenes).

Formato de entrada  (labels_raw/<stem>.txt):
    <anotador> <class_id> <cx> <cy> <w> <h>   — una línea por anotador

Formato de salida   (labels/<stem>.txt):
    <class_id> <cx> <cy> <w> <h>              — YOLO estándar

Uso:
    python tools/iou_merge.py
    python tools/iou_merge.py --min-iou 0.4
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LABELS_RAW_DIR = PROJECT_ROOT / "test_images" / "labels_raw"
LABELS_OUT_DIR = PROJECT_ROOT / "test_images" / "labels"
REPORT_PATH    = PROJECT_ROOT / "test_images" / "agreement_report.json"

CLASS_NAMES = {0: "bottles", 1: "tshirts"}


# ── geometry (normalized coords throughout) ───────────────────────────────────

def yolo_to_xyxy(box: list[float]) -> list[float]:
    cx, cy, w, h = box
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def iou(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def consensus_merge(entries: list[dict]) -> tuple[int, list[float], float | None, list[float]]:
    """
    entries: [{"name": str, "cls_id": int, "box": [cx,cy,w,h]}, ...]
    Returns (cls_id, merged_box, mean_iou_or_None, weights).
    """
    n = len(entries)
    xyxy = [yolo_to_xyxy(e["box"]) for e in entries]

    if n == 1:
        weights = [1.0]
        mean_iou = None
    else:
        pair_ious = {
            (i, j): iou(xyxy[i], xyxy[j])
            for i, j in combinations(range(n), 2)
        }
        scores = []
        for i in range(n):
            neighbours = [pair_ious[tuple(sorted((i, j)))] for j in range(n) if j != i]
            scores.append(sum(neighbours) / (n - 1))

        total = sum(scores)
        weights = [s / total for s in scores] if total > 0 else [1.0 / n] * n

        all_pairs = list(pair_ious.values())
        mean_iou = sum(all_pairs) / len(all_pairs)

    merged = [
        sum(w * e["box"][i] for w, e in zip(weights, entries))
        for i in range(4)
    ]

    classes = [e["cls_id"] for e in entries]
    cls_id = max(set(classes), key=classes.count)

    return cls_id, merged, mean_iou, weights


# ── I/O ───────────────────────────────────────────────────────────────────────

def read_raw_labels(raw_dir: Path) -> dict[str, list[dict]]:
    """Return {stem: [{"name", "cls_id", "box"}, ...]} for every txt file."""
    by_image: dict[str, list[dict]] = {}
    for txt in sorted(raw_dir.glob("*.txt")):
        entries = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 6:
                continue
            name = parts[0]
            cls_id = int(parts[1])
            box = list(map(float, parts[2:]))
            entries.append({"name": name, "cls_id": cls_id, "box": box})
        if entries:
            by_image[txt.stem] = entries
    return by_image


# ── main ──────────────────────────────────────────────────────────────────────

def run(min_iou: float = 0.3) -> None:
    if not LABELS_RAW_DIR.exists():
        print(f"No se encontró la carpeta {LABELS_RAW_DIR}")
        print("Corre primero tools/annotator.py para al menos un anotador.")
        return

    by_image = read_raw_labels(LABELS_RAW_DIR)
    if not by_image:
        print(f"No se encontraron entradas en {LABELS_RAW_DIR}")
        return

    all_annotators = sorted({e["name"] for entries in by_image.values() for e in entries})
    print(f"Anotadores encontrados : {', '.join(all_annotators)}")
    print(f"Imágenes con ≥1 caja   : {len(by_image)}")
    print("-" * 70)

    LABELS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    low_agreement = 0

    for stem in sorted(by_image):
        entries = by_image[stem]
        cls_id, merged, mean_iou, weights = consensus_merge(entries)
        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))

        flagged = mean_iou is not None and mean_iou < min_iou
        if flagged:
            low_agreement += 1

        iou_str = f"{mean_iou:.3f}" if mean_iou is not None else "solo 1 anotador"
        flag = "⚠" if flagged else " "
        weight_str = "  ".join(
            f"{e['name']}={w:.2f}" for e, w in zip(entries, weights)
        )
        print(
            f"{flag} {stem:<20} n={len(entries)}"
            f"  acuerdo={iou_str:<8}  cls={cls_name:<10}  pesos: {weight_str}"
        )

        cx, cy, w, h = merged
        out = LABELS_OUT_DIR / f"{stem}.txt"
        out.write_text(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")

        report.append({
            "image": stem,
            "annotators": [e["name"] for e in entries],
            "weights": {e["name"]: round(wt, 4) for e, wt in zip(entries, weights)},
            "mean_iou": round(mean_iou, 4) if mean_iou is not None else None,
            "class": cls_name,
            "merged_box": [round(v, 6) for v in merged],
            "flagged_low_agreement": flagged,
        })

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("-" * 70)
    print(f"Etiquetas exportadas          : {len(by_image)}")
    print(f"Flaggeadas (IoU < {min_iou:.1f})       : {low_agreement}")
    print(f"Reporte guardado en           : {REPORT_PATH}")

    multi = [r for r in report if r["mean_iou"] is not None]
    if multi:
        avg = sum(r["mean_iou"] for r in multi) / len(multi)
        print(f"IoU promedio global (N≥2)     : {avg:.3f}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Fusiona anotaciones raw con weighted-consensus merge"
    )
    p.add_argument(
        "--min-iou", type=float, default=0.3,
        help="IoU mínimo para marcar imagen como de bajo acuerdo (default: 0.3). "
             "No omite la imagen — solo la flaggea en el reporte.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(min_iou=args.min_iou)
