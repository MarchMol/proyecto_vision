"""
verify_dataset.py
-----------------
Verifica la integridad del dataset antes de entrenar:
  - Cada imagen tiene su label correspondiente
  - Los labels tienen formato YOLO válido (5 valores por línea)
  - Las coordenadas del bounding box están normalizadas [0, 1]
  - Muestra estadísticas por clase

Uso:
    python src/verify_dataset.py
"""

from pathlib import Path
import sys


DATASET_ROOT = Path(__file__).parent.parent / "dataset"
CLASS_NAMES = {0: "bottles", 1: "tshirts"}
SPLITS = ["train", "val"]


def verify_label_file(label_path: Path) -> list[str]:
    """Devuelve lista de errores encontrados en un archivo de label."""
    errors = []
    lines = label_path.read_text().strip().splitlines()

    if not lines:
        errors.append(f"{label_path.name}: archivo vacío")
        return errors

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(f"{label_path.name} línea {i+1}: se esperaban 5 valores, hay {len(parts)}")
            continue

        try:
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"{label_path.name} línea {i+1}: valores no numéricos")
            continue

        if cls_id not in CLASS_NAMES:
            errors.append(f"{label_path.name} línea {i+1}: clase {cls_id} no definida en CLASS_NAMES")

        for name, val in zip(["cx", "cy", "w", "h"], [cx, cy, w, h]):
            if not (0.0 <= val <= 1.0):
                errors.append(f"{label_path.name} línea {i+1}: {name}={val:.4f} fuera del rango [0,1]")

    return errors


def verify_split(split: str) -> dict:
    """Verifica un split (train/val) y devuelve estadísticas."""
    images_dir = DATASET_ROOT / "images" / split
    labels_dir = DATASET_ROOT / "labels" / split

    if not images_dir.exists():
        print(f"  [WARN] No existe: {images_dir}")
        return {}

    images = set(p.stem for p in images_dir.glob("*.jpg")) | set(p.stem for p in images_dir.glob("*.png"))
    labels = set(p.stem for p in labels_dir.glob("*.txt")) if labels_dir.exists() else set()

    missing_labels = images - labels
    orphan_labels = labels - images

    stats = {"total_images": len(images), "total_labels": len(labels),
             "missing_labels": len(missing_labels), "orphan_labels": len(orphan_labels),
             "class_counts": {v: 0 for v in CLASS_NAMES.values()},
             "errors": []}

    for label_path in labels_dir.glob("*.txt"):
        file_errors = verify_label_file(label_path)
        stats["errors"].extend(file_errors)

        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cls_id = int(parts[0])
                    if cls_id in CLASS_NAMES:
                        stats["class_counts"][CLASS_NAMES[cls_id]] += 1
                except ValueError:
                    pass

    if missing_labels:
        stats["errors"].append(f"Imágenes sin label: {sorted(missing_labels)[:5]}{'...' if len(missing_labels) > 5 else ''}")
    if orphan_labels:
        stats["errors"].append(f"Labels sin imagen: {sorted(orphan_labels)[:5]}{'...' if len(orphan_labels) > 5 else ''}")

    return stats


def main():
    print("=" * 55)
    print("  Verificación del Dataset")
    print("=" * 55)
    print(f"Raíz del dataset: {DATASET_ROOT.resolve()}\n")

    if not DATASET_ROOT.exists():
        print(f"ERROR: No se encontró el directorio {DATASET_ROOT}")
        sys.exit(1)

    all_ok = True

    for split in SPLITS:
        print(f"[{split.upper()}]")
        stats = verify_split(split)

        if not stats:
            continue

        print(f"  Imágenes : {stats['total_images']}")
        print(f"  Labels   : {stats['total_labels']}")
        print(f"  Clases   : {stats['class_counts']}")

        if stats["errors"]:
            all_ok = False
            print(f"  Errores  : {len(stats['errors'])}")
            for err in stats["errors"][:10]:
                print(f"    - {err}")
            if len(stats["errors"]) > 10:
                print(f"    ... y {len(stats['errors']) - 10} más")
        else:
            print("  Estado   : OK")
        print()

    if all_ok:
        print("Dataset verificado correctamente. Listo para entrenar.")
    else:
        print("Se encontraron problemas. Revisa los errores antes de entrenar.")
        sys.exit(1)


if __name__ == "__main__":
    main()
