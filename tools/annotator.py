"""
tools/annotator.py
------------------
Herramienta tipo CAPTCHA para anotar bounding boxes en imágenes de validación.
Cada miembro del equipo corre el script de forma independiente.
Las anotaciones se fusionan después con tools/iou_merge.py.

Uso:
    python tools/annotator.py --name Francis
    python tools/annotator.py --name Cesar --images test_images/images/

Controles:
    Arrastrar mouse  → dibuja el bounding box
    Enter / clic ✓   → confirmar anotación y pasar a la siguiente
    Space  / clic ↷  → omitir imagen (sin box)
    Flecha ←         → regresar a la imagen anterior
    Escape / clic ✗  → guardar y salir

Formato de etiquetas (test_images/labels/<stem>.txt):
    Una línea por anotador:
        <nombre> <class_id> <cx> <cy> <w> <h>
    coords normalizadas [0-1], igual que YOLO.
    Múltiples líneas = múltiples anotadores; usar IoU para fusionar.
"""

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_IMAGES = PROJECT_ROOT / "test_images" / "images"
LABELS_DIR = PROJECT_ROOT / "test_images" / "labels_raw"

CLASS_NAMES = ["bottles", "tshirts"]
CLASS_IDS = {name: idx for idx, name in enumerate(CLASS_NAMES)}
MAX_W, MAX_H = 900, 620


class Annotator:
    def __init__(self, root: tk.Tk, images: list[Path], name: str):
        self.root = root
        self.images = images
        self.name = name
        self.idx = 0

        self.scale = 1.0
        self.orig_w = self.orig_h = 0
        self.start_x = self.start_y = 0
        self.rect_id = None
        self.box = None  # [x1, y1, x2, y2] in original image coords

        LABELS_DIR.mkdir(parents=True, exist_ok=True)

        self._build_ui()
        self._load_image()

    # ── label file I/O ────────────────────────────────────────────────────────

    def _label_path(self, img_path: Path) -> Path:
        return LABELS_DIR / (img_path.stem + ".txt")

    def _read_my_annotation(self, img_path: Path) -> tuple[str, list] | None:
        """Return (class_name, [x1,y1,x2,y2]) for this annotator, or None."""
        txt = self._label_path(img_path)
        if not txt.exists():
            return None
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 6 and parts[0] == self.name:
                cls_id = int(parts[1])
                cx, cy, w, h = map(float, parts[2:])
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else CLASS_NAMES[0]
                x1 = int((cx - w / 2) * self.orig_w)
                y1 = int((cy - h / 2) * self.orig_h)
                x2 = int((cx + w / 2) * self.orig_w)
                y2 = int((cy + h / 2) * self.orig_h)
                return cls_name, [x1, y1, x2, y2]
        return None

    def _write_my_annotation(self, img_path: Path, box: list | None, cls: str):
        """Upsert this annotator's line in the label file; removes line if box is None."""
        txt = self._label_path(img_path)
        lines = txt.read_text(encoding="utf-8").splitlines() if txt.exists() else []
        lines = [l for l in lines if not (l.split() and l.split()[0] == self.name)]
        if box:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2 / self.orig_w
            cy = (y1 + y2) / 2 / self.orig_h
            w = (x2 - x1) / self.orig_w
            h = (y2 - y1) / self.orig_h
            cls_id = CLASS_IDS.get(cls, 0)
            lines.append(f"{self.name} {cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        txt.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

    def _count_done(self) -> int:
        count = 0
        for img_path in self.images:
            txt = self._label_path(img_path)
            if not txt.exists():
                continue
            for line in txt.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts and parts[0] == self.name and len(parts) == 6:
                    count += 1
                    break
        return count

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.title(f"Logo Annotator — {self.name}")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(False, False)

        self.header = tk.Label(
            self.root, text="", font=("Helvetica", 12, "bold"),
            bg="#1e1e2e", fg="#cdd6f4",
        )
        self.header.pack(pady=(10, 2))

        self.canvas = tk.Canvas(
            self.root, cursor="crosshair", bg="#181825", highlightthickness=0,
        )
        self.canvas.pack(padx=14, pady=4)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        cls_frame = tk.Frame(self.root, bg="#1e1e2e")
        cls_frame.pack(pady=5)
        tk.Label(
            cls_frame, text="Clase:", bg="#1e1e2e", fg="#a6e3a1",
            font=("Helvetica", 11),
        ).pack(side="left", padx=6)
        self.cls_var = tk.StringVar(value=CLASS_NAMES[0])
        for cls in CLASS_NAMES:
            tk.Radiobutton(
                cls_frame, text=cls, variable=self.cls_var, value=cls,
                bg="#1e1e2e", fg="#cdd6f4", selectcolor="#313244",
                activebackground="#1e1e2e", font=("Helvetica", 11),
            ).pack(side="left", padx=10)

        self.status = tk.Label(
            self.root, text="", font=("Helvetica", 10),
            bg="#1e1e2e", fg="#6c7086",
        )
        self.status.pack(pady=2)

        btn_frame = tk.Frame(self.root, bg="#1e1e2e")
        btn_frame.pack(pady=10)
        cfg = dict(font=("Helvetica", 11), padx=14, pady=6, bd=0, cursor="hand2")

        tk.Button(
            btn_frame, text="← Anterior", command=self._prev,
            bg="#313244", fg="#cdd6f4", **cfg,
        ).pack(side="left", padx=5)
        tk.Button(
            btn_frame, text="✓ Confirmar", command=self._confirm,
            bg="#a6e3a1", fg="#1e1e2e", **cfg,
        ).pack(side="left", padx=5)
        tk.Button(
            btn_frame, text="↷ Omitir", command=self._skip,
            bg="#fab387", fg="#1e1e2e", **cfg,
        ).pack(side="left", padx=5)
        tk.Button(
            btn_frame, text="✗ Salir", command=self._quit,
            bg="#f38ba8", fg="#1e1e2e", **cfg,
        ).pack(side="left", padx=5)

        self.root.bind("<Return>", lambda _: self._confirm())
        self.root.bind("<space>", lambda _: self._skip())
        self.root.bind("<Escape>", lambda _: self._quit())
        self.root.bind("<Left>", lambda _: self._prev())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    # ── image loading ─────────────────────────────────────────────────────────

    def _load_image(self):
        if self.idx >= len(self.images):
            messagebox.showinfo(
                "Completado",
                f"Anotaste todas las imágenes.\nEtiquetas en:\n{LABELS_DIR}",
            )
            self.root.destroy()
            return

        img_path = self.images[self.idx]
        pil_img = Image.open(img_path).convert("RGB")
        self.orig_w, self.orig_h = pil_img.size

        scale_w = MAX_W / self.orig_w
        scale_h = MAX_H / self.orig_h
        self.scale = min(scale_w, scale_h, 1.0)

        disp_w = int(self.orig_w * self.scale)
        disp_h = int(self.orig_h * self.scale)
        pil_resized = pil_img.resize((disp_w, disp_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_resized)

        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.rect_id = None

        existing = self._read_my_annotation(img_path)
        if existing:
            cls_name, box = existing
            self.box = box
            self.cls_var.set(cls_name)
            self._draw_box_canvas(self.box, "#a6e3a1")
            self._set_status(f"Ya anotado: {self.box}  — redibuja para cambiar")
        else:
            self.box = None
            self._set_status("Arrastra el mouse para dibujar el bounding box")

        n = len(self.images)
        done = self._count_done()
        self.header.config(
            text=f"{img_path.name}   [{self.idx + 1}/{n}]   ({done} confirmadas)",
        )

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw_box_canvas(self, orig_box: list, color: str):
        x1, y1, x2, y2 = [c * self.scale for c in orig_box]
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            x1, y1, x2, y2, outline=color, width=2,
        )

    def _on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def _on_drag(self, event):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline="#89b4fa", width=2,
        )
        w = abs(event.x - self.start_x)
        h = abs(event.y - self.start_y)
        self._set_status(f"Dibujando: {w}×{h} px (canvas)")

    def _on_release(self, event):
        cx1 = min(self.start_x, event.x)
        cy1 = min(self.start_y, event.y)
        cx2 = max(self.start_x, event.x)
        cy2 = max(self.start_y, event.y)

        if cx2 - cx1 < 5 or cy2 - cy1 < 5:
            self._set_status("Box demasiado pequeño — intenta de nuevo")
            return

        s = self.scale
        self.box = [int(cx1 / s), int(cy1 / s), int(cx2 / s), int(cy2 / s)]

        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            cx1, cy1, cx2, cy2, outline="#a6e3a1", width=2,
        )
        self._set_status(
            f"Box: {self.box}  — presiona Enter para confirmar o redibuja para ajustar"
        )

    # ── actions ───────────────────────────────────────────────────────────────

    def _confirm(self):
        if not self.box:
            messagebox.showwarning("Sin caja", "Dibuja un bounding box primero.")
            return
        self._write_my_annotation(self.images[self.idx], self.box, self.cls_var.get())
        self._advance(1)

    def _skip(self):
        self._write_my_annotation(self.images[self.idx], None, self.cls_var.get())
        self._advance(1)

    def _prev(self):
        if self.idx > 0:
            self._advance(-1)

    def _quit(self):
        self.root.destroy()

    def _advance(self, delta: int):
        self.idx += delta
        self.box = None
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self._load_image()

    def _set_status(self, msg: str):
        self.status.config(text=msg)


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Anotador de bounding boxes para conjunto de prueba"
    )
    p.add_argument(
        "--name", required=True,
        help="Tu nombre (ej. Francis, Cesar, Jose)",
    )
    p.add_argument(
        "--images", type=str, default=str(DEFAULT_IMAGES),
        help="Carpeta con imágenes a anotar (default: test_images/images/)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    images_dir = Path(args.images)

    if not images_dir.exists():
        print(f"ERROR: La carpeta '{images_dir}' no existe.")
        print("Corre normalize_test_images.py primero para generar las imágenes.")
        raise SystemExit(1)

    images = sorted(
        list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.jpeg"))
        + list(images_dir.glob("*.png"))
    )

    if not images:
        print(f"No se encontraron imágenes en {images_dir}")
        raise SystemExit(1)

    print(f"Cargando {len(images)} imágenes para anotación por: {args.name}")
    print(f"Etiquetas se guardarán en: {LABELS_DIR}")

    root = tk.Tk()
    Annotator(root, images, args.name)
    root.mainloop()
