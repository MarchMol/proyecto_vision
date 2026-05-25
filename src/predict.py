"""
predict.py
----------
Usa el modelo entrenado para predecir dónde colocar el logo
en una imagen de camisa o taza.

Salida por imagen:
  - Bounding box con coordenadas pixel (x1, y1, x2, y2)
  - Clase detectada (bottles / tshirts)
  - Confianza de la predicción
  - Imagen guardada con la zona de logo marcada

Uso:
    # Una imagen
    python src/predict.py --source dataset/images/val/image-0214.jpg

    # Carpeta completa
    python src/predict.py --source dataset/images/val/

    # Umbral de confianza personalizado
    python src/predict.py --source mi_imagen.jpg --conf 0.4

    # Modelo alternativo
    python src/predict.py --source mi_imagen.jpg --weights runs/train/logo_placement/weights/last.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "train" / "logo_placement" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "predict"

CLASS_NAMES = {0: "bottles", 1: "tshirts"}
# Color BGR por clase
CLASS_COLORS = {0: (0, 165, 255), 1: (0, 200, 0)}


def predict_image(
    model: YOLO,
    image_path: Path,
    conf_threshold: float = 0.25,
    save: bool = True,
) -> list[dict]:
    """
    Corre inferencia en una imagen y devuelve las predicciones.

    Retorna lista de dicts con:
        class_id, class_name, confidence, x1, y1, x2, y2, cx, cy, w, h
    """
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        device="cpu",
        verbose=False,
    )

    predictions = []
    img = cv2.imread(str(image_path))
    img_h, img_w = img.shape[:2]

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Centro y dimensiones en pixeles
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            w = x2 - x1
            h = y2 - y1

            pred = {
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
                "confidence": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx, "cy": cy, "w": w, "h": h,
            }
            predictions.append(pred)

            # Dibuja el bounding box
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{pred['class_name']} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - lh - 8), (x1 + lw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Cruz en el centro (punto de colocación del logo)
            cv2.drawMarker(img, (cx, cy), color,
                           markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    if save and predictions:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(out_path), img)

    return predictions


def run(
    source: str,
    weights: str | None = None,
    conf: float = 0.25,
    save: bool = True,
) -> None:
    """Punto de entrada principal. Acepta ruta a imagen o carpeta."""
    weights_path = Path(weights) if weights else DEFAULT_WEIGHTS

    if not weights_path.exists():
        print(f"ERROR: No se encontraron pesos en {weights_path}")
        print("Entrena primero con: python src/train.py")
        return

    model = YOLO(str(weights_path))
    print(f"Modelo cargado: {weights_path.name}")
    print(f"Umbral conf   : {conf}")
    print("-" * 50)

    source_path = Path(source)
    if source_path.is_dir():
        images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
    else:
        images = [source_path]

    total_detections = 0

    for img_path in images:
        preds = predict_image(model, img_path, conf_threshold=conf, save=save)
        total_detections += len(preds)

        if preds:
            for p in preds:
                print(
                    f"{img_path.name} → {p['class_name']} "
                    f"({p['confidence']:.2f}) "
                    f"centro=({p['cx']}, {p['cy']}) "
                    f"bbox=({p['x1']},{p['y1']})-({p['x2']},{p['y2']})"
                )
        else:
            print(f"{img_path.name} → sin detecciones (conf < {conf})")

    print("-" * 50)
    print(f"Imágenes procesadas : {len(images)}")
    print(f"Total detecciones   : {total_detections}")
    if save and total_detections > 0:
        print(f"Resultados guardados: {OUTPUT_DIR.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predicción de zona de logo con YOLOv8")
    parser.add_argument("--source", required=True,
                        help="Ruta a imagen o carpeta de imágenes")
    parser.add_argument("--weights", type=str, default=None,
                        help="Ruta a los pesos .pt (default: best.pt del último entrenamiento)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Umbral mínimo de confianza (0-1)")
    parser.add_argument("--no-save", action="store_true",
                        help="No guardar imágenes con predicciones")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        save=not args.no_save,
    )
