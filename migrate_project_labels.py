import csv
import hashlib
import os
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent / "data" / "projects" / "cb_v2_old_20260629_135151"
LABELS_CSV = PROJECT_DIR / "labels.csv"
BACKUP_CSV = PROJECT_DIR / "labels.csv.bak"
FIELDS = [
    "timestamp", "project_id", "source_doc", "truth", "pos", "label", "image_path", "mode", "ink_ratio",
    "box_x0", "box_y0", "box_x1", "box_y1", "roi_x0", "roi_y0", "roi_x1", "roi_y1",
    "auto_trim", "trim_threshold", "trim_pad", "zoom",
    "hist_threshold_method", "hist_min_char_width", "hist_min_ink_per_col", "hist_gap_merge_px",
    "hist_pad_x", "hist_pad_y", "hist_apply_morph_open", "hist_apply_morph_close",
]
LEGACY_AUTO_FIELDS = [
    "timestamp", "project_id", "source_doc", "truth", "pos", "label", "image_path", "mode", "ink_ratio",
    "box_x0", "box_y0", "box_x1", "box_y2", "roi_x0", "roi_y0", "roi_x1", "roi_y1",
    "hist_threshold_method", "hist_min_char_width", "hist_min_ink_per_col", "hist_gap_merge_px",
    "hist_pad_x", "hist_pad_y", "hist_apply_morph_open", "hist_apply_morph_close",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_row(header: list[str], values: list[str], row_number: int) -> dict[str, str]:
    if header == FIELDS:
        if len(values) != len(FIELDS):
            raise ValueError(f"Fila {row_number}: s'esperaven {len(FIELDS)} camps, n'hi ha {len(values)}.")
        row = dict(zip(FIELDS, values))
    elif len(values) == len(header):
        row = dict(zip(header, values))
    elif len(values) == len(LEGACY_AUTO_FIELDS):
        row = dict(zip(LEGACY_AUTO_FIELDS, values))
    else:
        raise ValueError(f"Fila {row_number}: {len(values)} camps; capçalera amb {len(header)}.")

    mode = row.get("mode", "")
    if mode not in {"canvas_manual", "auto_histogram"}:
        raise ValueError(f"Fila {row_number}: mode desconegut o absent ({mode!r}).")
    if mode == "auto_histogram" and "box_y2" in row:
        row["box_y1"] = row.pop("box_y2")
    return {field: row.get(field, "") for field in FIELDS}


def main() -> None:
    if not LABELS_CSV.exists() or not BACKUP_CSV.exists():
        raise FileNotFoundError("Calen labels.csv i la còpia de seguretat labels.csv.bak.")
    if sha256(LABELS_CSV) != sha256(BACKUP_CSV):
        raise RuntimeError("La còpia de seguretat no coincideix amb l'original; no s'ha modificat cap fitxer.")

    with LABELS_CSV.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        header = next(reader, None)
        if not header:
            raise ValueError("labels.csv és buit o no té cap capçalera.")
        rows = [normalize_row(header, values, number) for number, values in enumerate(reader, start=2) if values]

    counts = {"canvas_manual": 0, "auto_histogram": 0}
    valid_boxes = {position: 0 for position in range(1, 7)}
    for row in rows:
        counts[row["mode"]] += 1
        try:
            position = int(row["pos"])
            x0, y0, x1, y1 = (float(row[key]) for key in ("box_x0", "box_y0", "box_x1", "box_y1"))
            if position in valid_boxes and x1 > x0 and y1 > y0:
                valid_boxes[position] += 1
        except (TypeError, ValueError):
            continue

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=PROJECT_DIR, delete=False
        ) as destination:
            temp_path = Path(destination.name)
            writer = csv.DictWriter(destination, fieldnames=FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)

        with temp_path.open("r", newline="", encoding="utf-8") as check:
            reader = csv.reader(check)
            output_header = next(reader)
            output_lengths = {len(record) for record in reader}
        if output_header != FIELDS or output_lengths - {len(FIELDS)}:
            raise RuntimeError("La verificació del CSV normalitzat no ha passat.")

        os.replace(temp_path, LABELS_CSV)
        temp_path = None
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()

    print(f"Files migrades: {len(rows)}; manuals={counts['canvas_manual']}; automàtiques={counts['auto_histogram']}")
    print("Caixes vàlides per posició: " + ", ".join(f"{position}={count}" for position, count in valid_boxes.items()))
    print(f"Còpia de seguretat preservada: {BACKUP_CSV.name} ({sha256(BACKUP_CSV)})")


if __name__ == "__main__":
    main()