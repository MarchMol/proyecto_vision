"""
demo.py
-------
Demo interactivo del modelo de detección de zona de logo.

Sube una imagen de producto (camisa o botella) y un logo (PNG con fondo
transparente recomendado). El sistema:
  1. Detecta la zona del logo con YOLOv8
  2. Redimensiona el logo para ajustarlo a la zona detectada
  3. Compone y renderiza la imagen resultante

Uso:
    streamlit run src/demo.py
"""

import tempfile
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "train" / "logo_placement" / "weights" / "best.pt"
CLASS_NAMES = {0: "bottles", 1: "tshirts"}

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Logo Placement Demo",
    page_icon="🎨",
    layout="wide",
)

st.title("Logo Placement Demo")
st.markdown(
    "Sube una imagen de **camisa o botella** y un **logo** "
    "(PNG con fondo transparente recomendado). "
    "El modelo detecta automáticamente la zona de impresión y superpone el logo."
)

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Configuración")
conf_thresh = st.sidebar.slider("Umbral de confianza", 0.05, 0.90, 0.25, 0.05)
opacity = st.sidebar.slider("Opacidad del logo", 0.1, 1.0, 0.85, 0.05)
padding = st.sidebar.slider("Padding interior (px)", 0, 40, 8)

weights_path = DEFAULT_WEIGHTS
custom_weights = st.sidebar.file_uploader("Pesos del modelo (.pt)", type=["pt"])
if custom_weights:
    tmp_wt = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    tmp_wt.write(custom_weights.read())
    tmp_wt.flush()
    weights_path = Path(tmp_wt.name)

# ── file uploaders ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    product_file = st.file_uploader(
        "Imagen del producto", type=["jpg", "jpeg", "png"]
    )
    if product_file:
        st.image(product_file, caption="Producto original", use_container_width=True)

with col2:
    logo_file = st.file_uploader(
        "Logo (PNG con alpha recomendado)", type=["png", "jpg", "jpeg"]
    )
    if logo_file:
        st.image(logo_file, caption="Logo", use_container_width=True)


# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


def fit_logo(logo: Image.Image, box_w: int, box_h: int, pad: int) -> Image.Image:
    """Resize logo to fit inside the bounding box with padding, preserving aspect ratio."""
    max_w = max(box_w - 2 * pad, 1)
    max_h = max(box_h - 2 * pad, 1)
    fitted = logo.copy()
    fitted.thumbnail((max_w, max_h), Image.LANCZOS)
    return fitted


def composite_logo(
    product: Image.Image,
    logo: Image.Image,
    x1: int, y1: int, x2: int, y2: int,
    pad: int,
    opacity: float,
) -> Image.Image:
    result = product.copy().convert("RGBA")
    box_w, box_h = x2 - x1, y2 - y1

    logo_rgba = logo.convert("RGBA") if logo.mode != "RGBA" else logo.copy()
    logo_fitted = fit_logo(logo_rgba, box_w, box_h, pad)

    # Apply opacity to alpha channel
    r, g, b, a = logo_fitted.split()
    a = a.point(lambda p: int(p * opacity))
    logo_fitted = Image.merge("RGBA", (r, g, b, a))

    # Center within bounding box
    lw, lh = logo_fitted.size
    px = x1 + pad + max((box_w - 2 * pad - lw) // 2, 0)
    py = y1 + pad + max((box_h - 2 * pad - lh) // 2, 0)

    result.alpha_composite(logo_fitted, dest=(px, py))
    return result.convert("RGB")


def annotate_bbox(
    img: Image.Image, x1: int, y1: int, x2: int, y2: int, label: str
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([x1, y1, x2, y2], outline=(0, 210, 100), width=3)
    draw.rectangle([x1, y1 - 22, x1 + len(label) * 9, y1], fill=(0, 210, 100))
    draw.text((x1 + 4, y1 - 18), label, fill=(10, 10, 10))
    return out


# ── inference ─────────────────────────────────────────────────────────────────

ready = product_file is not None and logo_file is not None
if st.button("Generar composición", type="primary", disabled=not ready):

    if not weights_path.exists():
        st.error(
            f"No se encontraron pesos del modelo en `{weights_path}`.\n\n"
            "Entrena el modelo primero con `python src/train.py`, o sube un archivo "
            "`.pt` en el panel lateral."
        )
        st.stop()

    with st.spinner("Detectando zona de logo..."):
        product_img = Image.open(product_file).convert("RGB")
        logo_img = Image.open(logo_file)

        # Save product to a temp file for YOLO (needs a path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            product_img.save(tmp.name, quality=95)
            tmp_path = tmp.name

        model = load_model(str(weights_path))
        results = model.predict(
            source=tmp_path, conf=conf_thresh, device="cpu", verbose=False
        )
        Path(tmp_path).unlink(missing_ok=True)

        boxes = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                boxes.append((x1, y1, x2, y2, cls_id, conf))

    if not boxes:
        st.warning(
            "No se detectó ninguna zona de logo en la imagen. "
            "Prueba bajando el umbral de confianza en el panel lateral."
        )
        st.stop()

    # Use the highest-confidence detection
    boxes.sort(key=lambda b: b[5], reverse=True)
    x1, y1, x2, y2, cls_id, conf = boxes[0]
    cls_label = CLASS_NAMES.get(cls_id, str(cls_id))
    label_str = f"{cls_label} {conf:.0%}"

    st.success(
        f"Detectado: **{cls_label}** · confianza **{conf:.1%}** · "
        f"zona ({x1},{y1}) → ({x2},{y2})"
    )

    annotated = annotate_bbox(product_img, x1, y1, x2, y2, label_str)
    result_img = composite_logo(product_img, logo_img, x1, y1, x2, y2, padding, opacity)

    out_col1, out_col2 = st.columns(2)
    with out_col1:
        st.subheader("Zona detectada")
        st.image(annotated, use_container_width=True)
    with out_col2:
        st.subheader("Resultado final")
        st.image(result_img, use_container_width=True)

    buf = BytesIO()
    result_img.save(buf, format="JPEG", quality=93)
    st.download_button(
        label="Descargar resultado",
        data=buf.getvalue(),
        file_name="logo_resultado.jpg",
        mime="image/jpeg",
    )