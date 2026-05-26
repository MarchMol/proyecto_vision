"""
normalize_test_images.py  (quick and dirty — delete after use)

Reads test/bottles/ and test/tshirts/, resizes every image so its longest
side is TARGET_PX (preserving aspect ratio), converts to RGB JPG, and saves
in YOLO format:
  test_images/images/images-<id>.jpg
  test_images/labels/images-<id>.txt  (empty — no annotations)
"""

from pathlib import Path
from PIL import Image
import pillow_avif

TARGET_PX = 640
SRC_ROOT = Path("test")
DST_ROOT = Path("test_images")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif", ".tiff", ".tif"}

img_dir = DST_ROOT / "images"
lbl_dir = DST_ROOT / "labels"
img_dir.mkdir(parents=True, exist_ok=True)
lbl_dir.mkdir(parents=True, exist_ok=True)


def resize_to_longest(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest == target:
        return img
    scale = target / longest
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


counter = 1

for cls_dir in ("bottles", "tshirts"):
    src_dir = SRC_ROOT / cls_dir
    if not src_dir.exists():
        print(f"\n{cls_dir}: carpeta no encontrada, se omite ({src_dir})")
        continue
    files = [f for f in src_dir.iterdir() if f.suffix.lower() in EXTS]
    print(f"\n{cls_dir}: {len(files)} imágenes")

    for src in sorted(files):
        try:
            img = Image.open(src).convert("RGB")
        except Exception as e:
            print(f"  SKIP {src.name}: {e}")
            continue

        img = resize_to_longest(img, TARGET_PX)

        stem = f"images-{counter:04d}"
        img.save(img_dir / f"{stem}.jpg", format="JPEG", quality=88)
        (lbl_dir / f"{stem}.txt").touch()

        print(f"  {src.name} -> {stem}.jpg  {img.size}")
        counter += 1

print(f"\nListo. {counter - 1} imágenes guardadas en {DST_ROOT}/")