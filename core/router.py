"""
Mòdul d'enrutament i classificació automàtica de formats de document.
Identifica la plantilla del document (Tipus 1: Codi de barres, Tipus 2: Logo, Tipus 3: Excepció/Punts)
per aplicar la ROI, longitud esperada i regles de segmentació adequades.
"""
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
import cv2


FORMAT_DEFINITIONS = {
    "format_1_barcode": {
        "id": "format_1_barcode",
        "name": "Tipus 1 · Codi de Barres",
        "description": "Document amb codi de barres a la capçalera dreta i número de 7 dígits a sota.",
        "expected_length": 7,
        "default_roi": {
            "x0": 0.812,
            "y0": 0.076,
            "x1": 0.948,
            "y1": 0.098,
        },
        "filter_dots": False,
    },
    "format_2_logo": {
        "id": "format_2_logo",
        "name": "Tipus 2 · Logo Carburos Metálicos",
        "description": "Document amb logotip de Carburos Metálicos a la cantonada superior dreta i 6 dígits.",
        "expected_length": 6,
        "default_roi": {
            "x0": 0.825,
            "y0": 0.088,
            "x1": 0.955,
            "y1": 0.118,
        },
        "filter_dots": False,
    },
    "format_3_exception": {
        "id": "format_3_exception",
        "name": "Tipus 3 · Excepció (amb punts)",
        "description": "Document amb capçalera buida a la dreta i codi amb separador/punt (p. ex. 440.217 -> 440217).",
        "expected_length": 6,
        "default_roi": {
            "x0": 0.835,
            "y0": 0.115,
            "x1": 0.925,
            "y1": 0.133,
        },
        "filter_dots": True,
    },
}


def locate_anchor_and_adjust_roi(
    img: Image.Image,
    format_id: str,
    base_roi: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Ajusta dinàmicament les coordenades de la ROI utilitzant ancoratge visual
    respecte a la posició real d'elements de referència (codi de barres, logo o línia de text).
    Tolera desplaçaments físics del document (pàgines mogudes / escanejos descentrats).
    """
    if img is None or img.size[0] == 0 or img.size[1] == 0:
        return base_roi, {"status": "empty_image"}

    w, h = img.size
    adjusted_roi = {k: float(v) for k, v in base_roi.items()}
    debug_info: Dict[str, Any] = {"format_id": format_id, "anchor_found": False}

    if format_id == "format_1_barcode":
        # Zona superior dreta on viu el codi de barres (x: 70-98%, y: 1.5-9%)
        bx0, by0, bx1, by1 = int(0.70 * w), int(0.015 * h), int(0.98 * w), int(0.09 * h)
        header_crop = img.crop((bx0, by0, bx1, by1)).convert("L")
        h_arr = np.array(header_crop)

        blurred = cv2.GaussianBlur(h_arr, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Fusió horitzontal de barres
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour = None
        max_area = 0

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if cw >= 60 and ch >= 12 and (cw / max(1, ch)) >= 1.5:
                if area > max_area:
                    max_area = area
                    best_contour = (x, y, cw, ch)

        if best_contour is not None:
            cx, cy, cw, ch = best_contour
            abs_x0 = bx0 + cx
            abs_y0 = by0 + cy + ch
            adjusted_roi = {
                "x0": float(max(0.70, round((abs_x0 - 8) / w, 4))),
                "x1": float(min(0.99, round((abs_x0 + cw + 8) / w, 4))),
                "y0": float(max(0.02, round((abs_y0 - 2) / h, 4))),
                "y1": float(min(0.18, round((abs_y0 + int(0.028 * h)) / h, 4))),
            }
            debug_info["anchor_found"] = True
            debug_info["barcode_box"] = {"x0": abs_x0, "y0": abs_y0, "x1": abs_x0 + cw, "y1": abs_y0}
            return adjusted_roi, debug_info

    elif format_id == "format_2_logo":
        # Format 2 (Logo): el número de 6 dígits grans es troba sota el logotip (y: 0.07..0.14)
        sx0, sy0, sx1, sy1 = int(0.75 * w), int(0.05 * h), int(0.98 * w), int(0.16 * h)
        crop = img.crop((sx0, sy0, sx1, sy1)).convert("L")
        arr = np.array(crop)
        _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
        min_digit_h = max(14, int(0.009 * h))
        max_digit_h = max(35, int(0.030 * h))
        comps = []
        for i in range(1, num_labels):
            x, y, cw, ch, area = stats[i]
            if cw > 0.40 * (sx1 - sx0) and ch < 12:
                continue
            # Excloure lletres petites de 'Grupo Air Products' (ch ~ 7-11px) i quedar-se amb dígits/lletres grans
            if min_digit_h <= ch <= max_digit_h and 4 <= cw <= 45 and area >= 15:
                comps.append((y, x, cw, ch, area))

        if not comps:
            return base_roi, debug_info

        bands = []
        for c in sorted(comps, key=lambda item: item[0]):
            added = False
            for b in bands:
                avg_y = sum(item[0] for item in b) / len(b)
                if abs(c[0] - avg_y) <= 10:
                    b.append(c)
                    added = True
                    break
            if not added:
                bands.append([c])

        best_band = None
        best_score = -9999.0

        for b in bands:
            b.sort(key=lambda item: item[1])
            cnt = len(b)
            min_x = min(item[1] for item in b)
            max_x = max(item[1] + item[2] for item in b)
            min_y = min(item[0] for item in b)
            max_y = max(item[0] + item[3] for item in b)
            band_w = max_x - min_x
            avg_h = sum(item[3] for item in b) / max(1, cnt)

            score = 100.0 - abs(cnt - 6) * 15.0
            if band_w < 35 or band_w > 260:
                score -= 50.0
            if avg_h >= max(20, int(0.013 * h)):
                score += 25.0

            abs_y_center = (sy0 + (min_y + max_y) / 2) / h
            if 0.072 <= abs_y_center <= 0.125:
                score += 40.0
            elif abs_y_center < 0.07:
                score -= 60.0

            if score > best_score:
                best_score = score
                best_band = (min_x, min_y, max_x, max_y)

        if best_band is not None and best_score > 0:
            bx0, by0, bx1, by1 = best_band
            adjusted_roi = {
                "x0": float(max(0.65, round((sx0 + bx0 - 8) / w, 4))),
                "x1": float(min(0.99, round((sx0 + bx1 + 8) / w, 4))),
                "y0": float(max(0.04, round((sy0 + by0 - 2) / h, 4))),
                "y1": float(min(0.24, round((sy0 + by1 + 4) / h, 4))),
            }
            debug_info["anchor_found"] = True

    elif format_id == "format_3_exception":
        # Format 3 (Excepció): cerca de la línia 'CONTRATO Nº   440.xxx' al voltant de y: 0.08..0.20
        sx0, sy0, sx1, sy1 = int(0.62 * w), int(0.08 * h), int(0.98 * w), int(0.22 * h)
        crop = img.crop((sx0, sy0, sx1, sy1)).convert("L")
        arr = np.array(crop)
        _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
        comps = []
        for i in range(1, num_labels):
            x, y, cw, ch, area = stats[i]
            if cw > 0.45 * (sx1 - sx0) and ch < 14:
                continue
            # Permetre cw fins a 95px per quan els dígits de 440.xxx es toquen entre si
            if 7 <= ch <= 45 and 3 <= cw <= 95 and area >= 10:
                comps.append((y, x, cw, ch, area))

        if not comps:
            return base_roi, debug_info

        bands = []
        for c in sorted(comps, key=lambda item: item[0]):
            added = False
            for b in bands:
                avg_y = sum(item[0] for item in b) / len(b)
                if abs(c[0] - avg_y) <= 12:
                    b.append(c)
                    added = True
                    break
            if not added:
                bands.append([c])

        # Dividir cada banda horitzontal en clústers separats per espais grans (p.ex. entre 'CONTRATO Nº' i '440.xxx')
        gap_split_px = max(16, int(0.015 * w))
        candidate_clusters = []

        for b in bands:
            b.sort(key=lambda item: item[1])
            clusters = [[b[0]]]
            for item in b[1:]:
                prev_end = max(c_it[1] + c_it[2] for c_it in clusters[-1])
                curr_start = item[1]
                if (curr_start - prev_end) >= gap_split_px:
                    clusters.append([item])
                else:
                    clusters[-1].append(item)

            for cl in clusters:
                min_x = min(it[1] for it in cl)
                max_x = max(it[1] + it[2] for it in cl)
                min_y = min(it[0] for it in cl)
                max_y = max(it[0] + it[3] for it in cl)
                cl_w = max_x - min_x
                abs_x0 = (sx0 + min_x) / w
                abs_y = (sy0 + (min_y + max_y) / 2) / h

                # El bloc numèric 440.xxx viu a la dreta (abs_x0 >= 0.78) i té una amplada d'almenys ~35px
                if 0.09 <= abs_y <= 0.185 and abs_x0 >= 0.775 and 30 <= cl_w <= 150:
                    candidate_clusters.append((abs_y, abs_x0, cl))

        if candidate_clusters:
            # Prioritzar el clúster més proper a la zona d'albarà/contracte (y ~ 0.125..0.155) i situat a la dreta
            candidate_clusters.sort(key=lambda item: abs(item[0] - 0.135))
            _, _, chosen_cl = candidate_clusters[0]

            min_x = min(it[1] for it in chosen_cl)
            max_x = max(it[1] + it[2] for it in chosen_cl)
            min_y = min(it[0] for it in chosen_cl)
            max_y = max(it[0] + it[3] for it in chosen_cl)

            adjusted_roi = {
                "x0": float(max(0.75, round((sx0 + min_x - 6) / w, 4))),
                "x1": float(min(0.99, round((sx0 + max_x + 6) / w, 4))),
                "y0": float(max(0.06, round((sy0 + min_y - 4) / h, 4))),
                "y1": float(min(0.24, round((sy0 + max_y + 4) / h, 4))),
            }
            debug_info["anchor_found"] = True

    return adjusted_roi, debug_info


def classify_document_format(
    img: Image.Image,
    custom_rois: Optional[Dict[str, dict]] = None,
    auto_anchor: bool = False,
) -> Dict[str, Any]:
    """
    Classifica automàticament el format del document analitzant la zona superior dreta (x: 75%-96%, y: 2.5%-8.5%).

    - Format 1 (Barcode): Alta densitat de transicions horitzontals (> 2500) i tinta alta.
    - Format 2 (Logo): Tinta moderada/alta (> 10%) però baixa freqüència de transicions.
    - Format 3 (Excepció): Zona superior dreta pràcticament en blanc (< 10% tinta).

    Si auto_anchor és True, reajusta dinàmicament la ROI localitzant l'àncora visual corresponent.
    Retorna un diccionari amb el format detectat, la ROI aplicable i els paràmetres de processament.
    """
    if img is None or img.size[0] == 0 or img.size[1] == 0:
        fmt = FORMAT_DEFINITIONS["format_2_logo"]
        return {
            "format_id": "format_2_logo",
            "format_info": fmt,
            "roi": fmt["default_roi"],
            "expected_length": fmt["expected_length"],
            "filter_dots": fmt["filter_dots"],
            "confidence": 0.0,
            "debug": {"error": "Imatge buida"},
        }

    w, h = img.size

    # Zona d'ancoratge visual superior dreta
    tr_crop = img.crop((int(0.75 * w), int(0.025 * h), int(0.96 * w), int(0.085 * h))).convert("L")
    tr_arr = np.array(tr_crop)

    # Binarització
    _, tr_bin = cv2.threshold(tr_arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Transicions horitzontals (característica distintiva de codis de barres)
    diffs = np.abs(np.diff(tr_bin > 0, axis=1))
    v_transitions = int(np.sum(diffs))
    ink_ratio = float(np.mean(tr_bin > 0))

    if ink_ratio > 0.15 and v_transitions > 2500:
        format_id = "format_1_barcode"
        confidence = min(1.0, v_transitions / 5000.0)
    elif ink_ratio > 0.10:
        format_id = "format_2_logo"
        confidence = min(1.0, ink_ratio / 0.25)
    else:
        format_id = "format_3_exception"
        confidence = max(0.5, 1.0 - (ink_ratio / 0.10))

    fmt = FORMAT_DEFINITIONS[format_id]
    roi = fmt["default_roi"]
    if custom_rois and format_id in custom_rois:
        roi = custom_rois[format_id]

    anchor_debug = {}
    if auto_anchor:
        roi, anchor_debug = locate_anchor_and_adjust_roi(img, format_id, roi)

    return {
        "format_id": format_id,
        "format_info": fmt,
        "roi": roi,
        "expected_length": fmt["expected_length"],
        "filter_dots": fmt["filter_dots"],
        "confidence": float(confidence),
        "debug": {
            "ink_ratio": round(ink_ratio, 4),
            "v_transitions": v_transitions,
            "anchor_crop_size": tr_crop.size,
            "anchor": anchor_debug,
        },
    }


def is_project_router_enabled(config: Optional[Dict[str, Any]]) -> bool:
    """
    Determina si un projecte utilitza l'enrutador multi-format de Carburos (Tipus 1/2/3 de 6-7 dígits)
    o bé la ROI i longitud fixa pròpies del projecte (p. ex. 10 dígits).
    """
    if not config:
        return True
    if "use_router" in config and config["use_router"] is not None:
        return bool(config["use_router"])
    fixed_len = int(config.get("fixed_length", 0) or 0)
    return fixed_len in (6, 7)


def get_document_roi_and_format(
    img: Image.Image,
    config: Optional[Dict[str, Any]] = None,
    use_router: Optional[bool] = None,
    auto_anchor: bool = False,
) -> Dict[str, Any]:
    """
    Resol la ROI, la longitud esperada i les opcions de format d'un document segons la configuració del projecte.
    - Si el Router multi-format està actiu (Carburos 6/7 dígits), classifica automàticament en Tipus 1, 2 o 3.
    - Si està desactivat (p. ex. projectes de 10 dígits o amb ROI única), retorna la ROI i `fixed_length` del projecte.
    """
    cfg = config or {}
    router_enabled = is_project_router_enabled(cfg) if use_router is None else bool(use_router)

    if router_enabled:
        res = classify_document_format(
            img,
            custom_rois=cfg.get("custom_rois"),
            auto_anchor=auto_anchor,
        )
        res["router_active"] = True
        return res

    fixed_len = int(cfg.get("fixed_length", 7) or 7)
    base_roi = cfg.get("roi") or {
        "x0": 0.50,
        "y0": 0.00,
        "x1": 0.98,
        "y1": 0.25,
    }
    filter_dots = bool(cfg.get("filter_dots", False))
    fmt_info = {
        "id": "project_fixed",
        "name": f"Format fix del projecte ({fixed_len} caràcters)",
        "description": "ROI única i longitud fixa definides a la configuració del projecte.",
        "expected_length": fixed_len,
        "default_roi": base_roi,
        "filter_dots": filter_dots,
    }
    return {
        "format_id": "project_fixed",
        "format_info": fmt_info,
        "roi": base_roi,
        "expected_length": fixed_len,
        "filter_dots": filter_dots,
        "confidence": 1.0,
        "router_active": False,
        "debug": {"mode": "project_fixed"},
    }
