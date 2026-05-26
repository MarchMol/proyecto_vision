"""
demo.py
-------
Demo interactivo del modelo de detección de zona de logo.

Sube una imagen de producto (camisa o botella) y un logo (PNG con fondo
transparente recomendado). El sistema:
  1. Detecta la zona del logo con YOLOv8
  2. Redimensiona el logo para ajustarlo a la zona detectada
  3. Compone y renderiza la imagen resultante

Uso normal:
    streamlit run tools/demo.py

Prueba del demo (sin modelo — usa imagenes y etiquetas de test_images/):
    streamlit run tools/demo.py -- --dry-run
"""

import base64
import sys
import time
from io import BytesIO
from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image, ImageDraw

PROJECT_ROOT    = Path(__file__).parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "train" / "logo_placement" / "weights" / "best.pt"
LABELS_DIR      = PROJECT_ROOT / "test_images" / "labels"
LABELS_RAW_DIR  = PROJECT_ROOT / "test_images" / "labels_raw"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images" / "images"
LOGO_PATH       = Path(__file__).parent / "logomotion.png"

CLASS_NAMES = {0: "bottles", 1: "tshirts"}
DRY_RUN = "--dry-run" in sys.argv

ACCENT        = "#39FF14"   # bright orange
ACCENT_HOVER  = "#32C718"
PREVIEW_MAX_H = 280         # px cap for upload previews

_logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode() if LOGO_PATH.exists() else ""
_logo_img = (
    f'<img src="data:image/png;base64,{_logo_b64}" '
    f'style="height:195px;width:auto;vertical-align:middle;">'
    if _logo_b64 else ""
)


def preview_html(src, caption: str = "") -> str:
    """Render an image as HTML with max-width/max-height so large uploads don't overflow."""
    if isinstance(src, (str, Path)):
        data = Path(src).read_bytes()
        ext  = Path(src).suffix.lower().lstrip(".")
    else:
        data = src.getvalue()
        ext  = Path(getattr(src, "name", "img.jpg")).suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext, "jpeg")
    b64  = base64.b64encode(data).decode()
    cap  = f"<small>{caption}</small>" if caption else ""
    return (
        f'<img src="data:image/{mime};base64,{b64}" '
        f'style="max-width:100%; max-height:{PREVIEW_MAX_H}px; '
        f'object-fit:contain; display:block;">{cap}'
    )


# ── page config + global styles ──────────────────────────────────────────────
st.set_page_config(page_title="LogoMotion", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@800&display=swap');

/* accent color — primary buttons */
button[data-testid="baseButton-primary"] {{
    background-color: {ACCENT} !important;
    border-color:     {ACCENT} !important;
    color: white !important;
}}
button[data-testid="baseButton-primary"]:hover {{
    background-color: {ACCENT_HOVER} !important;
    border-color:     {ACCENT_HOVER} !important;
}}
/* slider — thumb */
[role="slider"] {{
    background-color: {ACCENT} !important;
    border-color:     {ACCENT} !important;
    outline-color:    {ACCENT} !important;
}}
/* slider — focus ring */
[role="slider"]:focus {{
    box-shadow: 0 0 0 4px {ACCENT}44 !important;
}}
/* slider — filled track segment */
[data-testid="stSlider"] [data-baseweb="slider"] div[style*="background"],
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="Track"] div,
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="Inner"] {{
    background-color: {ACCENT} !important;
}}
/* slider — native accent fallback */
[data-testid="stSlider"] * {{
    accent-color: {ACCENT} !important;
}}
/* selectbox highlighted option */
[data-baseweb="menu"] [aria-selected="true"],
[data-baseweb="menu"] li:hover {{
    background-color: {ACCENT}28 !important;
    color: inherit !important;
}}
/* sidebar header */
section[data-testid="stSidebar"] h2 {{
    font-size: 1.35rem !important;
}}
/* links */
a {{ color: {ACCENT} !important; }}
/* custom header */
.lm-header {{
    display: flex;
    align-items: center;
    gap: 20px;
    padding-bottom: 16px;
    border-bottom: 3px solid {ACCENT};
    margin-bottom: 22px;
}}
.lm-title {{
    font-family: 'Poppins', sans-serif;
    font-size: 3.6rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0;
    line-height: 1;
}}
</style>

<div class="lm-header">
    {_logo_img}
    <span class="lm-title">LogoMotion</span>
</div>
""", unsafe_allow_html=True)

if DRY_RUN:
    st.info(
        "**Prueba del demo** — se omite el modelo y se usan imagenes de "
        "`test_images/` con sus etiquetas precalculadas."
    )

# ── session state ─────────────────────────────────────────────────────────────
for _key, _default in [
    ("step", 1),
    ("inf_boxes", None),
    ("product_bytes", None),
    ("logo_bytes", None),
    ("dry_img_str", None),
    ("inf_conf", None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Configuracion")
conf_thresh = st.sidebar.slider("Umbral de confianza", 0.05, 0.90, 0.25, 0.05)
opacity     = st.sidebar.slider("Opacidad del logo", 0.1, 1.0, 0.85, 0.05)
padding     = st.sidebar.slider("Padding interior (px)", 0, 40, 8)

weights_path = DEFAULT_WEIGHTS
if not DRY_RUN:
    custom_weights = st.sidebar.file_uploader("Pesos del modelo (.pt)", type=["pt"])
    if custom_weights:
        tmp_wt = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        tmp_wt.write(custom_weights.read())
        tmp_wt.flush()
        weights_path = Path(tmp_wt.name)

if DRY_RUN:
    test_imgs = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    names = [p.name for p in test_imgs]
    chosen = st.sidebar.selectbox("Imagen de prueba", names) if names else None
    dry_img_path: Path | None = TEST_IMAGES_DIR / chosen if chosen else None
else:
    dry_img_path = None

# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


def fit_logo(logo: Image.Image, box_w: int, box_h: int, pad: int) -> Image.Image:
    max_w = max(box_w - 2 * pad, 1)
    max_h = max(box_h - 2 * pad, 1)
    fitted = logo.copy()
    fitted.thumbnail((max_w, max_h), Image.LANCZOS)
    return fitted


def composite_logo(
    product: Image.Image, logo: Image.Image,
    x1: int, y1: int, x2: int, y2: int,
    pad: int, opacity: float,
) -> Image.Image:
    result = product.copy().convert("RGBA")
    box_w, box_h = x2 - x1, y2 - y1
    logo_rgba = logo.convert("RGBA") if logo.mode != "RGBA" else logo.copy()
    logo_fitted = fit_logo(logo_rgba, box_w, box_h, pad)
    r, g, b, a = logo_fitted.split()
    a = a.point(lambda p: int(p * opacity))
    logo_fitted = Image.merge("RGBA", (r, g, b, a))
    lw, lh = logo_fitted.size
    px = x1 + pad + max((box_w - 2 * pad - lw) // 2, 0)
    py = y1 + pad + max((box_h - 2 * pad - lh) // 2, 0)
    result.alpha_composite(logo_fitted, dest=(px, py))
    return result.convert("RGB")


def annotate_bbox(img: Image.Image, x1: int, y1: int, x2: int, y2: int, label: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([x1, y1, x2, y2], outline=(0, 210, 100), width=3)
    draw.rectangle([x1, y1 - 22, x1 + len(label) * 9, y1], fill=(0, 210, 100))
    draw.text((x1 + 4, y1 - 18), label, fill=(10, 10, 10))
    return out


def read_dry_run_label(img_path: Path) -> tuple[int, float, float, float, float] | None:
    merged = LABELS_DIR / (img_path.stem + ".txt")
    raw    = LABELS_RAW_DIR / (img_path.stem + ".txt")
    if merged.exists():
        parts = merged.read_text(encoding="utf-8").splitlines()[0].split()
        if len(parts) == 5:
            return int(parts[0]), *map(float, parts[1:])
    if raw.exists():
        for line in raw.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 6:
                return int(parts[1]), *map(float, parts[2:])
    return None


# ── tabs ─────────────────────────────────────────────────────────────────────
_tab1, _tab2 = st.tabs(["Generador", "Acerca de"])

with _tab1:
    st.markdown(
        "Sube una imagen de **camisa o botella** y un **logo** "
        "(PNG con fondo transparente recomendado). "
        "El modelo detecta automaticamente la zona de impresion y superpone el logo."
    )

    # ── accordion 1: subida de imagenes ──────────────────────────────────────
with _tab1, st.expander("1. Subida de Imagenes", expanded=(st.session_state.step == 1)):
    col1, col2 = st.columns(2)

    with col1:
        if DRY_RUN:
            if dry_img_path and dry_img_path.exists():
                st.markdown(preview_html(dry_img_path, f"Producto: {dry_img_path.name}"), unsafe_allow_html=True)
            else:
                st.warning("No se encontraron imagenes en test_images/images/")
            product_file = "dry_run"
        else:
            product_file = st.file_uploader("Imagen del producto", type=["jpg", "jpeg", "png"])
            if product_file:
                st.markdown(preview_html(product_file, "Producto original"), unsafe_allow_html=True)

    with col2:
        logo_file = st.file_uploader("Logo (PNG con alpha recomendado)", type=["png", "jpg", "jpeg"])
        if logo_file:
            st.markdown(preview_html(logo_file, "Logo"), unsafe_allow_html=True)

    if DRY_RUN:
        ready = logo_file is not None and dry_img_path is not None
    else:
        ready = product_file is not None and logo_file is not None

    st.write("")
    _, btn_col = st.columns([5, 1])
    with btn_col:
        if st.button("Generar Superposicion ->", disabled=not ready, type="primary"):
            if not DRY_RUN:
                st.session_state.product_bytes = product_file.getvalue()
            else:
                st.session_state.dry_img_str = str(dry_img_path)
            st.session_state.logo_bytes = logo_file.getvalue()
            st.session_state.inf_boxes = None
            st.session_state.step = 2
            st.rerun()

# ── accordion 2: resultado ────────────────────────────────────────────────────
label2 = "2. Resultado" if st.session_state.step == 2 else "2. Resultado  —  pendiente"
with _tab1, st.expander(label2, expanded=(st.session_state.step == 2)):
    if st.session_state.step == 1:
        st.caption("Completa el paso 1 y haz clic en Generar Superposicion para continuar.")
    else:
        # invalidate cache if conf threshold changed since last run
        if st.session_state.inf_conf != conf_thresh:
            st.session_state.inf_boxes = None

        # run inference once; cache boxes in session state
        if st.session_state.inf_boxes is None:
            if DRY_RUN:
                with st.spinner("Simulando deteccion..."):
                    img_path = Path(st.session_state.dry_img_str)
                    product_img_tmp = Image.open(img_path).convert("RGB")
                    img_w, img_h = product_img_tmp.size
                    lbl = read_dry_run_label(img_path)
                    time.sleep(5)
                if lbl is None:
                    st.warning(
                        f"No se encontro etiqueta para `{img_path.name}`. "
                        "Anota la imagen primero con tools/annotator.py."
                    )
                    st.stop()
                cls_id, cx, cy, bw, bh = lbl
                x1 = int((cx - bw / 2) * img_w)
                y1 = int((cy - bh / 2) * img_h)
                x2 = int((cx + bw / 2) * img_w)
                y2 = int((cy + bh / 2) * img_h)
                st.session_state.inf_boxes = [(x1, y1, x2, y2, cls_id, 0.99)]
                st.session_state.inf_conf = conf_thresh
            else:
                if not weights_path.exists():
                    st.error(
                        f"No se encontraron pesos del modelo en `{weights_path}`. "
                        "Entrena el modelo primero con `python src/train.py` o sube un .pt en el panel lateral."
                    )
                    st.stop()
                with st.spinner("Detectando zona de logo..."):
                    product_img_tmp = Image.open(BytesIO(st.session_state.product_bytes)).convert("RGB")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        product_img_tmp.save(tmp.name, quality=95)
                        tmp_path = tmp.name
                    model = load_model(str(weights_path))
                    results = model.predict(source=tmp_path, conf=conf_thresh, device="cpu", verbose=False)
                    Path(tmp_path).unlink(missing_ok=True)
                    boxes_raw = []
                    for result in results:
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                            bcls = int(box.cls.item())
                            bconf = float(box.conf.item())
                            boxes_raw.append((bx1, by1, bx2, by2, bcls, bconf))
                    st.session_state.inf_boxes = boxes_raw
                    st.session_state.inf_conf = conf_thresh

        boxes = st.session_state.inf_boxes

        if not boxes:
            st.warning(
                "No se detecto ninguna zona de logo. "
                "Prueba bajando el umbral de confianza en el panel lateral."
            )
            st.stop()

        # reload images for compositing (runs on every rerun so sliders update live)
        if DRY_RUN:
            product_img = Image.open(Path(st.session_state.dry_img_str)).convert("RGB")
        else:
            product_img = Image.open(BytesIO(st.session_state.product_bytes)).convert("RGB")
        logo_img = Image.open(BytesIO(st.session_state.logo_bytes))

        x1, y1, x2, y2, cls_id, conf = sorted(boxes, key=lambda b: b[5], reverse=True)[0]
        cls_label = CLASS_NAMES.get(cls_id, str(cls_id))
        label_str = f"{cls_label} {conf:.0%}"

        st.success(
            f"Detectado: **{cls_label}** · confianza **{conf:.1%}** · "
            f"zona ({x1},{y1}) -> ({x2},{y2})"
            + (" *(dry-run)*" if DRY_RUN else "")
        )

        annotated  = annotate_bbox(product_img, x1, y1, x2, y2, label_str)
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

        dl_col, reset_col = st.columns([3, 1])
        with dl_col:
            st.download_button(
                label="Descargar resultado",
                data=buf.getvalue(),
                file_name="logo_resultado.jpg",
                mime="image/jpeg",
            )
        with reset_col:
            if st.button("Reiniciar"):
                st.session_state.step = 1
                st.session_state.inf_boxes = None
                st.rerun()

with _tab2:
    st.markdown("## Acerca de Nosotros")
    st.markdown("LogoMotion es un sistema de composición de imagenes para traer tu marca a los productos de tus sueños. "
            "Implementado sistemas de Inteligencia Artificial YOLOv8 y procesamiento de imágenes integrado para identificar cuál"
            "es el mejor lugar para insertar tu marca. Contamos con manejo de una gran variedad de productos de botellas y camisetas."
            )

    st.divider()

    st.markdown("## ¿Cómo Funciona?")
    st.markdown(
        "LogoMotion sigue el siguiente flujo para generar la composición "
        "de Tú Logo sobre el producto indicado:"
    )

    st.image(str(Path(__file__).parent / "flowchart.png"), use_container_width=True)


