from pathlib import Path
from PIL import Image
from io import BytesIO

try:
    import fitz
    HAVE_PYMUPDF = True
except Exception:
    fitz = None
    HAVE_PYMUPDF = False

def load_first_page_image(path: Path, dpi: int = 150) -> Image.Image:
    ext = path.suffix.lower()

    if ext == ".pdf":
        if not HAVE_PYMUPDF:
            raise RuntimeError("PyMuPDF no disponible.")
        doc = fitz.open(str(path))
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
        doc.close()
        return img

    return Image.open(path).convert("RGB")

def crop_relative(img: Image.Image, roi: dict) -> Image.Image:
    w, h = img.size
    x0 = int(w * roi["x0"])
    y0 = int(h * roi["y0"])
    x1 = int(w * roi["x1"])
    y1 = int(h * roi["y1"])
    return img.crop((x0, y0, x1, y1))


def draw_roi_overlay(img: Image.Image, roi: dict, color=(255, 0, 0), width: int = 5) -> Image.Image:
    """
    Retorna una còpia de la imatge amb la ROI dibuixada.
    Això fa que x0/y0/x1/y1 deixin de ser coordenades Gowex invisibles.
    """
    from PIL import ImageDraw

    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)

    w, h = out.size
    x0 = int(w * roi["x0"])
    y0 = int(h * roi["y0"])
    x1 = int(w * roi["x1"])
    y1 = int(h * roi["y1"])

    # Normalitzar per si l'usuari creua sliders
    xa, xb = sorted([x0, x1])
    ya, yb = sorted([y0, y1])

    # Rectangle principal
    for i in range(width):
        draw.rectangle((xa-i, ya-i, xb+i, yb+i), outline=color)

    # Marques cantonades més visibles
    corner = max(12, int(min(w, h) * 0.015))
    draw.line((xa, ya, xa+corner, ya), fill=color, width=width)
    draw.line((xa, ya, xa, ya+corner), fill=color, width=width)

    draw.line((xb, ya, xb-corner, ya), fill=color, width=width)
    draw.line((xb, ya, xb, ya+corner), fill=color, width=width)

    draw.line((xa, yb, xa+corner, yb), fill=color, width=width)
    draw.line((xa, yb, xa, yb-corner), fill=color, width=width)

    draw.line((xb, yb, xb-corner, yb), fill=color, width=width)
    draw.line((xb, yb, xb, yb-corner), fill=color, width=width)

    return out


def split_fixed_slots(crop: Image.Image, fixed_length: int, pad_px: int = 1) -> list:
    """
    Divideix una ROI en N slots iguals.
    MVP fiable per camps de longitud fixa.
    """
    if fixed_length <= 0:
        raise ValueError("fixed_length ha de ser > 0")

    img = crop.convert("RGB")
    w, h = img.size

    slots = []
    for i in range(fixed_length):
        x0 = int(round(i * w / fixed_length))
        x1 = int(round((i + 1) * w / fixed_length))

        # petit marge interior/exterior controlat
        xa = max(0, x0 - pad_px)
        xb = min(w, x1 + pad_px)

        slot = img.crop((xa, 0, xb, h))
        slots.append(slot)

    return slots


def trim_to_ink_bbox(img: Image.Image, threshold: int = 245, pad_px: int = 4) -> Image.Image:
    """
    Retalla marges blancs al voltant del contingut.
    Pensat per crops de codis impresos/blau/negre sobre fons blanc.

    Si no troba tinta, retorna la imatge original.
    """
    import numpy as np

    rgb = img.convert("RGB")
    gray = rgb.convert("L")
    arr = np.array(gray)

    # tinta = píxel no-blanc
    mask = arr < threshold

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return rgb

    w, h = rgb.size
    x0 = max(0, int(xs.min()) - pad_px)
    x1 = min(w, int(xs.max()) + 1 + pad_px)
    y0 = max(0, int(ys.min()) - pad_px)
    y1 = min(h, int(ys.max()) + 1 + pad_px)

    if x1 <= x0 or y1 <= y0:
        return rgb

    return rgb.crop((x0, y0, x1, y1))


def slot_ink_ratio(img: Image.Image, threshold: int = 245) -> float:
    """
    Percentatge aproximat de píxels amb tinta dins d'un slot.
    Serveix per detectar slots en blanc abans de contaminar el dataset.
    """
    import numpy as np

    gray = img.convert("L")
    arr = np.array(gray)

    if arr.size == 0:
        return 0.0

    mask = arr < threshold
    return float(mask.mean())


def crop_box_pixels(img: Image.Image, box: dict) -> Image.Image:
    """
    Retalla una caixa en coordenades de píxel.
    box = {"x0": int, "y0": int, "x1": int, "y1": int}
    """
    w, h = img.size
    x0 = max(0, min(w, int(box["x0"])))
    y0 = max(0, min(h, int(box["y0"])))
    x1 = max(0, min(w, int(box["x1"])))
    y1 = max(0, min(h, int(box["y1"])))

    xa, xb = sorted([x0, x1])
    ya, yb = sorted([y0, y1])

    if xb <= xa:
        xb = min(w, xa + 1)
    if yb <= ya:
        yb = min(h, ya + 1)

    return img.crop((xa, ya, xb, yb))


def draw_char_boxes_overlay(img: Image.Image, boxes: list, active_idx: int = 0) -> Image.Image:
    """
    Dibuixa caixes de caràcters sobre la ROI.
    - caixa activa en vermell
    - resta en groc
    """
    from PIL import ImageDraw

    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)

    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = int(b["x0"]), int(b["y0"]), int(b["x1"]), int(b["y1"])
        color = (255, 0, 0) if i == active_idx else (255, 210, 0)
        width = 2 if i == active_idx else 1
        draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
        draw.text((x0 + 1, max(0, y0 - 10)), str(i + 1), fill=color)

    return out


def resize_for_zoom(img: Image.Image, zoom: int = 5) -> Image.Image:
    """
    Ampliació visual per labeling manual.
    """
    zoom = max(1, int(zoom))
    w, h = img.size
    return img.resize((w * zoom, h * zoom))
