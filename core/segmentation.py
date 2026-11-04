"""
Mòdul de segmentació dinàmica de caràcters per histograma X i anàlisi morfològica / components connectats.
Permet extreure caràcters individuals d'una ROI, gestionar caràcters enganxats (valley splitting)
i filtrar automàticament soroll i separadors com punts (p. ex. format tipus 3: 440.217 -> 440217).
Inclou també preprocessament de deskewing (rotació) i segmentació robusta per cv2.connectedComponentsWithStats.
"""
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import numpy as np
import cv2


def deskew_image(img: Image.Image, max_angle: float = 6.0) -> Tuple[Image.Image, float]:
    """
    Detecta i corregeix l'angle d'inclinació (skew) d'una imatge mitjançant anàlisi morfològica i Hough/minAreaRect.
    Només aplica rotació si l'angle detectat està dins de [-max_angle, max_angle].
    """
    if img is None or img.size[0] == 0 or img.size[1] == 0:
        return img, 0.0

    img_np = np.array(img.convert("L"))
    h, w = img_np.shape
    blurred = cv2.GaussianBlur(img_np, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 30:
        return img, 0.0

    angle = 0.0
    found_hough = False

    # Unir caràcters horitzontalment per detectar línies de text reals i evitar traços diagonals curts
    kernel_w = max(5, min(30, int(0.025 * w)))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1)))
    edges = cv2.Canny(closed, 50, 150, apertureSize=3)
    min_line_len = max(25, int(0.10 * w))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=min_line_len, maxLineGap=10)

    if lines is not None and len(lines) > 0:
        angles = []
        for line in lines:
            flat = np.array(line).reshape(-1)
            if len(flat) >= 4:
                x1, y1, x2, y2 = float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])
                if x2 != x1:
                    deg = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                    if abs(deg) <= max_angle:
                        angles.append(deg)
        if len(angles) >= 2:
            angle = float(np.median(angles))
            found_hough = True

    if not found_hough:
        # Fallback amb minAreaRect només quan HoughLinesP no ha trobat línies (p.ex. crops petits)
        rect = cv2.minAreaRect(coords)
        rect_angle = rect[-1]
        if rect_angle < -45:
            rect_angle = -(90 + rect_angle)
        elif rect_angle > 45:
            rect_angle = 90 - rect_angle

        if abs(rect_angle) <= max_angle and abs(rect_angle) >= 0.3:
            angle = float(rect_angle)

    if abs(angle) < 0.25 or abs(angle) > max_angle:
        return img, 0.0

    # Rotació en sentit horari/antihorari compensant l'angle
    fill = (255, 255, 255) if img.mode == "RGB" else 255
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=fill)
    return rotated, float(angle)


def split_wide_segments_by_valleys(
    ink_mask: np.ndarray,
    segments: List[Tuple[int, int]],
    expected_count: Optional[int] = None,
    min_valley_width_px: int = 1,
    valley_min_depth_ratio: float = 0.35,
    min_char_width: int = 3,
) -> List[Tuple[int, int]]:
    """
    Subdivideix segments d'histograma amples cercant valls de mínima densitat de tinta.
    """
    if not segments:
        return []

    if expected_count is not None and len(segments) >= expected_count:
        return segments

    final_segments = []

    for idx, (x1, x2) in enumerate(segments):
        seg_w = x2 - x1 + 1
        if seg_w <= max(1, min_char_width * 2):
            final_segments.append((x1, x2))
            continue

        seg_slice = ink_mask[:, x1:x2 + 1]
        x_proj = np.sum(seg_slice, axis=0)

        seg_max = float(np.max(x_proj)) if x_proj.size else 0.0
        if seg_max <= 0:
            final_segments.append((x1, x2))
            continue

        # Nombre de talls necessaris: per passar d'1 segment a N parts calen N-1 talls
        target_splits = 1
        if expected_count is not None:
            current_total = len(final_segments) + (len(segments) - idx)
            cuts_needed = max(0, int(expected_count) - current_total)
            if cuts_needed <= 0:
                final_segments.append((x1, x2))
                continue
            target_splits = cuts_needed

        valley_cols = x_proj <= (seg_max * valley_min_depth_ratio)
        # Ignorar vores extremes per no tallar marges buits
        margin = max(min_char_width, 2)
        if seg_w > 2 * margin:
            valley_cols[:margin] = False
            valley_cols[seg_w - margin:] = False

        valleys = []
        in_run = False
        run_start = 0
        for i, v in enumerate(valley_cols):
            if v and not in_run:
                in_run = True
                run_start = i
            elif (not v) and in_run:
                in_run = False
                if (i - run_start) >= min_valley_width_px:
                    valleys.append((run_start, i - 1))
        if in_run:
            end_i = len(valley_cols) - 1
            if (end_i - run_start + 1) >= min_valley_width_px:
                valleys.append((run_start, end_i))

        valley_scores = []
        for (v1, v2) in valleys:
            mid = (v1 + v2) // 2
            valley_scores.append((float(x_proj[mid]), v1, v2))
        valley_scores.sort(key=lambda t: t[0])

        min_sep = max(min_char_width, int(seg_w / (target_splits + 1) * 0.55))
        chosen_cuts = []
        for (_, v1, v2) in valley_scores:
            cp = (v1 + v2) // 2
            if margin <= cp <= (seg_w - margin - 1):
                if all(abs(cp - existing) >= min_sep for existing in chosen_cuts):
                    chosen_cuts.append(cp)
                    if len(chosen_cuts) >= target_splits:
                        break

        # Si sabem exactament quantes parts s'esperen (p.ex. 2 o 3 dígits enganxats) i falten talls,
        # busquem el mínim local de projecció X al voltant de cada punt de divisió teòric
        if expected_count is not None and len(chosen_cuts) < target_splits and len(segments) == 1:
            n_parts = target_splits + 1
            fallback_cuts = []
            for k in range(1, n_parts):
                ideal = int(round(seg_w * k / n_parts))
                w_rad = max(2, int(seg_w / (n_parts * 3)))
                lo = max(margin, ideal - w_rad)
                hi = min(seg_w - margin - 1, ideal + w_rad)
                if lo <= hi:
                    best_local = lo + int(np.argmin(x_proj[lo:hi + 1]))
                    if float(x_proj[best_local]) <= seg_max * 0.85:
                        fallback_cuts.append(best_local)
            if len(fallback_cuts) == target_splits:
                chosen_cuts = fallback_cuts

        if not chosen_cuts:
            final_segments.append((x1, x2))
            continue

        cut_positions = sorted(chosen_cuts)

        left = x1
        for cp in cut_positions:
            abs_cut = x1 + cp
            s1 = (left, abs_cut - 1)
            if (s1[1] - s1[0] + 1) >= min_char_width:
                final_segments.append(s1)
            left = abs_cut + 1

        last_seg = (left, x2)
        if (last_seg[1] - last_seg[0] + 1) >= min_char_width:
            final_segments.append(last_seg)
        else:
            if final_segments:
                px = final_segments[-1]
                final_segments[-1] = (px[0], x2)

    return final_segments


def auto_split_characters_by_x_histogram(
    roi_img: Image.Image,
    threshold_method: str = "otsu",
    min_char_width: int = 3,
    min_ink_per_col: int = 1,
    gap_merge_px: int = 2,
    pad_x: int = 2,
    pad_y: int = 2,
    invert_if_needed: bool = True,
    apply_morph_open: bool = True,
    apply_morph_close: bool = False,
    enable_valley_split: bool = False,
    min_valley_width_px: int = 1,
    valley_min_depth_ratio: float = 0.35,
    expected_count: Optional[int] = None,
    filter_dots: bool = False,
) -> Tuple[List[Image.Image], List[Tuple[int, int, int, int]], Dict[str, Any]]:
    """
    Segmenta automàticament una ROI en caràcters mitjançant histograma X.

    Args:
        roi_img: Imatge PIL de la ROI.
        threshold_method: "otsu" o "adaptive".
        min_char_width: Amplada mínima d'un segment per considerar-lo caràcter.
        min_ink_per_col: Mínim de píxels actius en una columna X.
        gap_merge_px: Fusiona segments separats per gaps iguals o menors a aquest valor.
        pad_x, pad_y: Padding en píxels.
        invert_if_needed: Garanteix tinta blanca sobre fons negre.
        apply_morph_open, apply_morph_close: Filtres morfològics.
        enable_valley_split: Divideix segments amples per valls de tinta.
        filter_dots: Si és True, descarta components amb alçada o àrea petita (punts decimals).

    Returns:
        crops: Llista d'imatges PIL de cada caràcter.
        boxes: Llista de caixes (x0, y0, x1, y1) en coordenades de la ROI.
        debug: Diccionari amb dades intermèdies per inspecció.
    """
    debug: Dict[str, Any] = {}
    if roi_img is None or roi_img.size[0] == 0 or roi_img.size[1] == 0:
        return [], [], {"error": "ROI buida."}

    img_np = np.array(roi_img.copy())

    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
    else:
        gray = img_np.copy()
    debug["gray"] = Image.fromarray(gray)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    if threshold_method == "adaptive":
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    if invert_if_needed:
        if cv2.countNonZero(binary) > (binary.shape[0] * binary.shape[1]) // 2:
            binary = cv2.bitwise_not(binary)
    debug["binary_pre_morph"] = Image.fromarray(binary)

    if apply_morph_open:
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if apply_morph_close:
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    debug["binary"] = Image.fromarray(binary)

    ink = binary > 0
    x_projection = np.sum(ink, axis=0)
    debug["x_projection"] = x_projection

    active_cols = x_projection >= min_ink_per_col
    raw_segments: List[Tuple[int, int]] = []
    is_in_segment = False
    start_x = 0
    for x, is_active in enumerate(active_cols):
        if is_active and not is_in_segment:
            is_in_segment = True
            start_x = x
        elif not is_active and is_in_segment:
            is_in_segment = False
            raw_segments.append((start_x, x - 1))
    if is_in_segment:
        raw_segments.append((start_x, len(active_cols) - 1))
    debug["raw_segments"] = raw_segments

    if not raw_segments:
        return [], [], debug

    merged_segments = [raw_segments[0]]
    for i in range(1, len(raw_segments)):
        prev_end = merged_segments[-1][1]
        curr_start = raw_segments[i][0]
        if (curr_start - prev_end - 1) <= gap_merge_px:
            merged_segments[-1] = (merged_segments[-1][0], raw_segments[i][1])
        else:
            merged_segments.append(raw_segments[i])
    debug["merged_segments"] = merged_segments

    # Pre-filtratge de separadors / punts abans del valley split perquè un punt no compti com a dígit
    h, w = gray.shape
    if filter_dots and len(merged_segments) > 1:
        seg_heights = []
        seg_infos = []
        for sx1, sx2 in merged_segments:
            s_slice = ink[:, sx1:sx2 + 1]
            rows = np.where(np.sum(s_slice, axis=1) > 0)[0]
            if len(rows) > 0:
                sy1, sy2 = int(rows[0]), int(rows[-1])
                sh = sy2 - sy1 + 1
                s_area = int(np.sum(s_slice))
                seg_infos.append((sx1, sx2, sy1, sy2, sh, s_area))
                if sh >= 8:
                    seg_heights.append(sh)
        ref_sh = float(np.median(seg_heights)) if seg_heights else 20.0
        pre_filtered = []
        for (sx1, sx2, sy1, sy2, sh, s_area) in seg_infos:
            is_dot = (sh < 0.48 * ref_sh) and (sy1 > h * 0.30)
            if is_dot:
                continue
            pre_filtered.append((sx1, sx2))
        if pre_filtered:
            merged_segments = pre_filtered

    # Valley split si cal separar caràcters units
    if enable_valley_split:
        valley_segments = split_wide_segments_by_valleys(
            ink_mask=ink,
            segments=merged_segments,
            expected_count=expected_count,
            min_valley_width_px=int(min_valley_width_px),
            valley_min_depth_ratio=float(valley_min_depth_ratio),
            min_char_width=int(min_char_width),
        )
        debug["split_segments"] = valley_segments
        active_segments = valley_segments
    else:
        active_segments = merged_segments

    crops: List[Image.Image] = []
    boxes: List[Tuple[int, int, int, int]] = []
    char_candidates = []

    for x1, x2 in active_segments:
        if (x2 - x1 + 1) >= min_char_width:
            segment_slice = ink[:, x1:x2+1]
            y_projection = np.sum(segment_slice, axis=1)
            active_rows = np.where(y_projection > 0)[0]
            if len(active_rows) > 0:
                y1, y2 = int(active_rows[0]), int(active_rows[-1])
                char_h = y2 - y1 + 1
                char_w = x2 - x1 + 1
                area = int(np.sum(segment_slice))
                char_candidates.append({
                    "x1": x1,
                    "x2": x2,
                    "y1": y1,
                    "y2": y2,
                    "w": char_w,
                    "h": char_h,
                    "area": area,
                })

    if not char_candidates:
        return [], [], debug

    # Filtratge de separadors / punts (punts amb alçada petita a la meitat inferior)
    if filter_dots or (expected_count and len(char_candidates) > expected_count):
        heights = [c["h"] for c in char_candidates if c["h"] >= 10]
        ref_h = float(np.median(heights)) if heights else 20.0

        filtered_candidates = []
        for c in char_candidates:
            is_dot = (c["h"] < (0.48 * ref_h)) and (c["y1"] > (h * 0.30))
            is_speckle = c["area"] < 6 or c["h"] < 4
            if filter_dots and is_dot:
                continue
            if is_speckle and len(char_candidates) > (expected_count or 6):
                continue
            filtered_candidates.append(c)
        char_candidates = filtered_candidates

    for c in char_candidates:
        box_x1 = max(0, c["x1"] - pad_x)
        box_y1 = max(0, c["y1"] - pad_y)
        box_x2 = min(w, c["x2"] + pad_x + 1)
        box_y2 = min(h, c["y2"] + pad_y + 1)
        boxes.append((box_x1, box_y1, box_x2, box_y2))
        crops.append(roi_img.crop((box_x1, box_y1, box_x2, box_y2)))

    debug["boxes"] = boxes
    return crops, boxes, debug


def extract_characters_by_connected_components(
    roi_img: Image.Image,
    threshold_method: str = "otsu",
    expected_count: Optional[int] = None,
    filter_dots: bool = False,
    min_char_width: int = 3,
    min_char_height: int = 6,
    min_area: int = 8,
    pad_x: int = 2,
    pad_y: int = 2,
    enable_valley_split: bool = True,
    apply_morph_open: bool = True,
    apply_morph_close: bool = False,
    invert_if_needed: bool = True,
) -> Tuple[List[Image.Image], List[Tuple[int, int, int, int]], Dict[str, Any]]:
    """
    Segmenta automàticament els caràcters d'una ROI utilitzant cv2.connectedComponentsWithStats.
    Filtra soroll, línies paràsites, separadors/punts i extreu els N caràcters més coherents geomètricament.
    """
    debug: Dict[str, Any] = {}
    if roi_img is None or roi_img.size[0] == 0 or roi_img.size[1] == 0:
        return [], [], {"error": "ROI buida."}

    img_np = np.array(roi_img.copy())
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
    else:
        gray = img_np.copy()
    debug["gray"] = Image.fromarray(gray)

    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    if threshold_method == "adaptive":
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    if invert_if_needed:
        if cv2.countNonZero(binary) > (binary.shape[0] * binary.shape[1]) // 2:
            binary = cv2.bitwise_not(binary)

    if apply_morph_open:
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if apply_morph_close:
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    debug["binary"] = Image.fromarray(binary)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    candidates = []
    for label in range(1, num_labels):
        bx = int(stats[label, cv2.CC_STAT_LEFT])
        by = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        # Filtrat de soroll mínim
        if area < min_area or bh < min_char_height or bw < min_char_width:
            continue

        # Filtrat de línies de taula o marges complets
        if bw > 0.85 * w and bh < 8:
            continue
        if bh > 0.85 * h and bw < 5:
            continue

        candidates.append({
            "x1": bx,
            "y1": by,
            "x2": bx + bw - 1,
            "y2": by + bh - 1,
            "w": bw,
            "h": bh,
            "area": area,
            "label": label,
        })

    if not candidates:
        return [], [], debug

    # Filtratge de punts (punts decimals inferiors) i fragments paràsits d'alçada molt petita
    raw_heights = [c["h"] for c in candidates if c["h"] >= 8]
    if raw_heights:
        max_h = max(raw_heights)
        tall_heights = [ch for ch in raw_heights if ch >= 0.65 * max_h]
        ref_h = float(np.median(tall_heights)) if tall_heights else float(np.median(raw_heights))
        tall_cys = [(c["y1"] + c["y2"]) / 2.0 for c in candidates if c["h"] >= 0.65 * max_h]
        ref_cy = float(np.median(tall_cys)) if tall_cys else (h / 2.0)
    else:
        ref_h = 20.0
        ref_cy = h / 2.0

    filtered = []
    for c in candidates:
        c_cy = (c["y1"] + c["y2"]) / 2.0
        is_dot = (c["h"] < (0.48 * ref_h)) and (c["y1"] > (h * 0.30))
        if filter_dots and is_dot:
            continue
        # Descartar text paràsit molt més petit o desalineat verticalment (p.ex. 'Grupo Air Products', 'º', 'rador')
        if ref_h >= 12.0 and c["h"] < (0.55 * ref_h):
            if abs(c_cy - ref_cy) > (0.25 * ref_h) or c["h"] < (0.36 * ref_h):
                continue
        filtered.append(c)

    if filtered:
        candidates = filtered

    if not candidates:
        return [], [], debug

    # Divisió de caràcters units/enganxats si enable_valley_split
    if enable_valley_split:
        widths = [c["w"] for c in candidates if c["w"] >= 4]
        median_w = float(np.median(widths)) if widths else 15.0
        est_single_w = max(4.0, 0.62 * ref_h)
        if expected_count is not None and len(candidates) < expected_count:
            ref_w = min(median_w, est_single_w)
        else:
            ref_w = median_w

        split_candidates = []
        for c in candidates:
            # Si el component és notablement més ample que un dígit estàndard (>= 1.65x amplada d'un dígit)
            if c["w"] >= 1.65 * ref_w:
                n_sub = max(2, int(round(c["w"] / max(1.0, ref_w))))
                comp_slice = (binary[c["y1"]:c["y2"]+1, c["x1"]:c["x2"]+1] > 0).astype(np.uint8)
                valleys = split_wide_segments_by_valleys(
                    ink_mask=comp_slice,
                    segments=[(0, c["w"] - 1)],
                    expected_count=n_sub,
                    min_valley_width_px=1,
                    valley_min_depth_ratio=0.55,
                    min_char_width=min_char_width,
                )
                if len(valleys) > 1:
                    for vx1, vx2 in valleys:
                        sub_w = vx2 - vx1 + 1
                        sub_slice = comp_slice[:, vx1:vx2+1]
                        sub_rows = np.where(np.sum(sub_slice, axis=1) > 0)[0]
                        if len(sub_rows) > 0:
                            sub_y1 = c["y1"] + int(sub_rows[0])
                            sub_y2 = c["y1"] + int(sub_rows[-1])
                            split_candidates.append({
                                "x1": c["x1"] + vx1,
                                "y1": sub_y1,
                                "x2": c["x1"] + vx2,
                                "y2": sub_y2,
                                "w": sub_w,
                                "h": sub_y2 - sub_y1 + 1,
                                "area": int(np.sum(sub_slice)),
                            })
                    continue
            split_candidates.append(c)
        candidates = split_candidates

    # Ordenar d'esquerra a dreta
    candidates.sort(key=lambda c: c["x1"])

    # Si tenim més candidats que expected_count (p. ex. text paràsit "rador" a l'esquerra o taques)
    if expected_count is not None and len(candidates) > expected_count:
        k = expected_count
        best_window = candidates[:k]
        best_score = float("inf")

        for start_idx in range(len(candidates) - k + 1):
            window = candidates[start_idx:start_idx + k]
            wh_list = [c["h"] for c in window]
            cy_list = [(c["y1"] + c["y2"]) / 2.0 for c in window]
            ar_list = [c["w"] / max(1.0, float(c["h"])) for c in window]

            h_std = float(np.std(wh_list))
            cy_std = float(np.std(cy_list))
            # Penalitzar si l'aspect ratio està fora del rang típic d'un dígit (0.2 a 0.9)
            ar_penalty = sum(max(0.0, ar - 1.0) * 15.0 for ar in ar_list)

            # Prioritzar caràcters de bona alçada (evitar taques petites)
            mean_h = float(np.mean(wh_list))
            size_penalty = max(0.0, (ref_h - mean_h)) * 2.0

            score = (h_std * 2.0) + (cy_std * 2.5) + ar_penalty + size_penalty
            if score < best_score:
                best_score = score
                best_window = window

        candidates = best_window

    boxes = []
    crops = []
    for c in candidates:
        box_x1 = max(0, c["x1"] - pad_x)
        box_y1 = max(0, c["y1"] - pad_y)
        box_x2 = min(w, c["x2"] + pad_x + 1)
        box_y2 = min(h, c["y2"] + pad_y + 1)
        boxes.append((box_x1, box_y1, box_x2, box_y2))
        crops.append(roi_img.crop((box_x1, box_y1, box_x2, box_y2)))

    debug["candidates"] = candidates
    debug["boxes"] = boxes
    return crops, boxes, debug


def segment_roi_characters(
    roi_img: Image.Image,
    method: str = "connected_components",
    threshold_method: str = "otsu",
    min_char_width: int = 3,
    min_ink_per_col: int = 1,
    gap_merge_px: int = 2,
    pad_x: int = 2,
    pad_y: int = 2,
    enable_valley_split: bool = True,
    expected_count: Optional[int] = None,
    filter_dots: bool = False,
    apply_morph_open: bool = True,
    apply_morph_close: bool = False,
) -> Tuple[List[Image.Image], List[Tuple[int, int, int, int]], Dict[str, Any]]:
    """
    Interfície unificada per a la segmentació de caràcters dins d'una ROI.
    Permet triar entre 'connected_components' (mètode robust) o 'histogram' (mètode clàssic).
    """
    if method == "connected_components":
        return extract_characters_by_connected_components(
            roi_img=roi_img,
            threshold_method=threshold_method,
            expected_count=expected_count,
            filter_dots=filter_dots,
            min_char_width=min_char_width,
            pad_x=pad_x,
            pad_y=pad_y,
            enable_valley_split=enable_valley_split,
            apply_morph_open=apply_morph_open,
            apply_morph_close=apply_morph_close,
        )

    return auto_split_characters_by_x_histogram(
        roi_img=roi_img,
        threshold_method=threshold_method,
        min_char_width=min_char_width,
        min_ink_per_col=min_ink_per_col,
        gap_merge_px=gap_merge_px,
        pad_x=pad_x,
        pad_y=pad_y,
        enable_valley_split=enable_valley_split,
        expected_count=expected_count,
        filter_dots=filter_dots,
        apply_morph_open=apply_morph_open,
        apply_morph_close=apply_morph_close,
    )
