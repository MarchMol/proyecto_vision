"""
python tools/augmentation.py                     # default
python tools/augmentation.py --factor 6          # 6 variantes por imagen
python tools/augmentation.py --splits train      # solo train
python tools/augmentation.py --no-copy-originals # sin copiar originales
python tools/augmentation.py --bg-colors 8       # 8 colores de fondo distintos

pip install albumentations opencv-python-headless pyyaml tqdm
"""

import argparse
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
import yaml
import albumentations as A
from tqdm import tqdm

# CONFIG POR DEFECTO
DEFAULT_DATASET_DIR = Path("dataset")
DEFAULT_OUTPUT_DIR = Path("dataset_aug")
DEFAULT_FACTOR = 4  # variantes aumentadas por imagen original
DEFAULT_SPLITS = ["train", "val"]
SEED = 42

# Umbral para detectar fondo blanco
WHITE_THRESHOLD = 240

# Simula pequeño ruido en las anotaciones
BBOX_JITTER = 0.005


# CAMBIO DE COLOR DE FONDO

BG_PALETTE = [
    (200, 220, 255),
    (220, 255, 200),
    (255, 220, 200),
    (230, 210, 255),
    (255, 255, 180),
    (200, 245, 245),
    (255, 210, 230),
    (215, 215, 215),
    (240, 230, 210),
    (210, 240, 220),
]


def replace_white_background(image: np.ndarray, bg_color: tuple) -> np.ndarray:
    mask = np.all(image >= WHITE_THRESHOLD, axis=2)  # True donde el pixel es blanco
    out = image.copy()
    out[mask] = bg_color
    return out


# ruido en labels


def jitter_bboxes(bboxes: list, max_jitter: float = BBOX_JITTER) -> list:
    """
    Añade pequeño ruido gaussiano a cada coordenada de los bounding boxes.
    """
    jittered = []
    for cx, cy, w, h in bboxes:
        cx = float(np.clip(cx + np.random.normal(0, max_jitter), 0.0, 1.0))
        cy = float(np.clip(cy + np.random.normal(0, max_jitter), 0.0, 1.0))
        w = float(np.clip(w + np.random.normal(0, max_jitter), 0.01, 1.0))
        h = float(np.clip(h + np.random.normal(0, max_jitter), 0.01, 1.0))
        jittered.append((cx, cy, w, h))
    return jittered


def apply_random_padding(image: np.ndarray, bboxes: list, max_pad: int = 40):
    """
    Añade padding blanco aleatorio por cada lado.
    """
    h, w = image.shape[:2]

    pad_top = random.randint(0, max_pad)
    pad_bottom = random.randint(0, max_pad)
    pad_left = random.randint(0, max_pad)
    pad_right = random.randint(0, max_pad)

    # Aplicar padding con fondo blanco
    padded = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    new_h, new_w = padded.shape[:2]

    # Recalcular bboxes YOLO
    new_bboxes = []
    for cx, cy, bw, bh in bboxes:
        # Convertir a pixeles absolutos en imagen original
        abs_cx = cx * w
        abs_cy = cy * h
        abs_bw = bw * w
        abs_bh = bh * h

        # Desplazar por el padding añadido
        new_cx = (abs_cx + pad_left) / new_w
        new_cy = (abs_cy + pad_top) / new_h
        new_bw = abs_bw / new_w
        new_bh = abs_bh / new_h

        new_bboxes.append(
            (
                float(np.clip(new_cx, 0.0, 1.0)),
                float(np.clip(new_cy, 0.0, 1.0)),
                float(np.clip(new_bw, 0.01, 1.0)),
                float(np.clip(new_bh, 0.01, 1.0)),
            )
        )

    return padded, new_bboxes


# PIPELINE ALBUMENTATIONS


def build_pipeline() -> A.Compose:
    """
    - Rotacion ligera
    - Ruido gaussiano en la imagen
    - Padding blanco aleatorio
    """
    return A.Compose(
        [
            # rotate_limit=10 → maximo ±10
            A.Rotate(
                limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(255, 255, 255),  # relleno blanco al rotar
                p=0.7,
            ),
            # Ruido gaussiano en la imagen
            A.GaussNoise(
                std_range=(0.02, 0.08),
                p=0.6,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.5,
        ),
    )


# UTILIDADES YOLO


def read_yolo_labels(label_path: Path):
    class_ids, bboxes = [], []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                class_ids.append(int(parts[0]))
                bboxes.append(list(map(float, parts[1:])))
    return class_ids, bboxes


def write_yolo_labels(label_path: Path, class_ids, bboxes):
    lines = []
    for cls, (cx, cy, w, h) in zip(class_ids, bboxes):
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines))


# PROCESO PRINCIPAL


def augment_split(
    split: str,
    dataset_dir: Path,
    output_dir: Path,
    factor: int,
    pipeline: A.Compose,
    copy_originals: bool,
    bg_colors: int,
):
    img_in_dir = dataset_dir / "images" / split
    lbl_in_dir = dataset_dir / "labels" / split
    img_out_dir = output_dir / "images" / split
    lbl_out_dir = output_dir / "labels" / split

    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg"}
    img_paths = sorted(
        [p for p in img_in_dir.iterdir() if p.suffix.lower() in img_exts]
    )

    if not img_paths:
        print(f"Sin imagenes en {img_in_dir}. Se omite '{split}'.")
        return

    # Subconjunto de la paleta que se usara
    palette = BG_PALETTE[: max(1, min(bg_colors, len(BG_PALETTE)))]

    print(
        f"\n  Split '{split}': {len(img_paths)} originales -> "
        f"{len(img_paths) * (factor + (1 if copy_originals else 0))} imagenes totales"
    )

    written = skipped = 0

    for img_path in tqdm(img_paths, desc=f"  {split}", unit="img"):
        stem = img_path.stem
        lbl_path = lbl_in_dir / f"{stem}.txt"

        # Copiar og
        if copy_originals:
            shutil.copy2(img_path, img_out_dir / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_out_dir / lbl_path.name)

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"No se pudo leer {img_path.name}. Se omite.")
            skipped += 1
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        class_ids, bboxes = read_yolo_labels(lbl_path)

        for idx in range(factor):
            try:
                # Cambio de color de fondo
                bg_color = random.choice(palette)
                img_bg = replace_white_background(image, bg_color)

                # Pipeline
                result = pipeline(image=img_bg, bboxes=bboxes, class_labels=class_ids)
                aug_img = result["image"]
                aug_bboxes = list(result["bboxes"])
                aug_clsids = result["class_labels"]

                # Padding blanco aleatorio
                if random.random() < 0.5:
                    aug_img, aug_bboxes = apply_random_padding(
                        aug_img, aug_bboxes, max_pad=40
                    )

                # ruido en labels
                if aug_bboxes:
                    aug_bboxes = jitter_bboxes(aug_bboxes)

            except Exception as exc:
                print(f"Error en {img_path.name} (var {idx}): {exc}")
                skipped += 1
                continue

            out_stem = f"{stem}_aug{idx:03d}"
            out_img_path = img_out_dir / f"{out_stem}{img_path.suffix}"
            out_lbl_path = lbl_out_dir / f"{out_stem}.txt"

            cv2.imwrite(str(out_img_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            write_yolo_labels(out_lbl_path, aug_clsids, aug_bboxes)
            written += 1

    print(f"  {written} imagenes generadas | {skipped} errores")


def copy_yaml(dataset_dir: Path, output_dir: Path):
    src = dataset_dir / "data.yaml"
    dst = output_dir / "data.yaml"
    if not src.exists():
        print("data.yaml no encontrado.")
        return
    with open(src) as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(output_dir.resolve())
    with open(dst, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"\n  data.yaml -> {dst}")


# CLI


def parse_args():
    p = argparse.ArgumentParser(
        description="Data augmentation YOLO  |  rotacion - cambio de fondo - ruido - padding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--factor",
        type=int,
        default=DEFAULT_FACTOR,
        help="Variantes aumentadas por imagen original",
    )
    p.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    p.add_argument(
        "--bg-colors",
        type=int,
        default=len(BG_PALETTE),
        help=f"Cuántos colores de la paleta usar (1-{len(BG_PALETTE)})",
    )
    p.add_argument(
        "--no-copy-originals",
        action="store_true",
        help="No copiar imágenes originales a dataset_aug",
    )
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = args.dataset.resolve()
    output_dir = args.output.resolve()
    copy_originals = not args.no_copy_originals

    print("=" * 60)
    print("  YOLO Data Augmentation")
    print("=" * 60)
    print(f"Dataset fuente  : {dataset_dir}")
    print(f"Salida          : {output_dir}")
    print(f"Factor          : {args.factor}x por imagen")
    print(f"Splits          : {args.splits}")
    print(f"Colores de fondo: {args.bg_colors} de {len(BG_PALETTE)} disponibles")
    print(f"Copiar og.    : {copy_originals}")
    print(f"Semilla         : {args.seed}")
    print("=" * 60)
    print()

    pipeline = build_pipeline()

    for split in args.splits:
        augment_split(
            split=split,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            factor=args.factor,
            pipeline=pipeline,
            copy_originals=copy_originals,
            bg_colors=args.bg_colors,
        )

    copy_yaml(dataset_dir, output_dir)

    print("\n Augmentation completa.")
    print(f"Resultado en: {output_dir}")


if __name__ == "__main__":
    main()
