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
