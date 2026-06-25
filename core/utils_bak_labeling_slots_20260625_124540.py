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
