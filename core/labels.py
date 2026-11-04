import csv
from pathlib import Path

import numpy as np
import pandas as pd


LABELS_CSV_FIELDS = [
    "timestamp", "project_id", "source_doc", "truth", "pos", "label", "image_path", "mode", "ink_ratio",
    "box_x0", "box_y0", "box_x1", "box_y1", "roi_x0", "roi_y0", "roi_x1", "roi_y1",
    "auto_trim", "trim_threshold", "trim_pad", "zoom",
    "hist_threshold_method", "hist_min_char_width", "hist_min_ink_per_col", "hist_gap_merge_px",
    "hist_pad_x", "hist_pad_y", "hist_apply_morph_open", "hist_apply_morph_close",
]


def append_label_rows(labels_csv: Path, rows: list[dict]) -> None:
    """Append labels using the shared schema for all labeling modes."""
    write_header = not labels_csv.exists() or labels_csv.stat().st_size == 0
    if not write_header:
        with labels_csv.open("r", newline="", encoding="utf-8") as file:
            existing_fields = next(csv.reader(file), [])
        if existing_fields != LABELS_CSV_FIELDS:
            raise RuntimeError(
                "L'esquema de labels.csv no coincideix amb el format actual. "
                "Normalitza o migra el fitxer abans de continuar."
            )

    with labels_csv.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LABELS_CSV_FIELDS, extrasaction="raise")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def build_box_template_from_labels(pdir: Path, fixed_length: int) -> list:
    """Build one runtime box per fixed position from labels.csv."""
    labels_csv = pdir / "labels.csv"
    if not labels_csv.exists():
        raise RuntimeError("No existeix labels.csv. Primer etiqueta caràcters a 04 · Dataset / labeling.")

    try:
        with labels_csv.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if not header:
                raise RuntimeError("labels.csv és buit o no té cap capçalera.")
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise RuntimeError(
                        f"Fila {row_number} inconsistent a labels.csv: té {len(row)} camps; "
                        f"la capçalera en defineix {len(header)}. Normalitza el CSV abans del batch."
                    )
        df = pd.read_csv(labels_csv)
    except (csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise RuntimeError(
            f"No es pot interpretar labels.csv ({error}). Comprova que totes les files tinguin "
            "la mateixa capçalera i nombre de columnes; recupera'l des de labels.csv.bak si cal."
        ) from error

    if "pos" not in df.columns and "slot_idx" in df.columns:
        df["pos"] = df["slot_idx"]

    required = ["pos", "box_x0", "box_y0", "box_x1", "box_y1"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"labels.csv no té columnes de caixa necessàries: {missing}")

    position_values = pd.to_numeric(df["pos"], errors="coerce")
    invalid_positions = ~np.isfinite(position_values.to_numpy(dtype=float))
    invalid_positions |= position_values.mod(1).fillna(1).ne(0).to_numpy()
    invalid_positions |= position_values.lt(1).fillna(True).to_numpy()
    invalid_positions |= position_values.gt(int(fixed_length)).fillna(True).to_numpy()
    if invalid_positions.any():
        rows = (np.flatnonzero(invalid_positions) + 2).tolist()
        raise RuntimeError(f"Posicions invàlides a labels.csv, files {rows}; s'esperen valors 1–{fixed_length}.")

    coordinate_columns = ["box_x0", "box_y0", "box_x1", "box_y1"]
    coordinates = df[coordinate_columns].apply(pd.to_numeric, errors="coerce")
    coordinate_values = coordinates.to_numpy(dtype=float)
    valid_geometry = np.isfinite(coordinate_values).all(axis=1)
    valid_geometry &= coordinates["box_x1"].gt(coordinates["box_x0"]).to_numpy()
    valid_geometry &= coordinates["box_y1"].gt(coordinates["box_y0"]).to_numpy()
    if not valid_geometry.all():
        rows = (np.flatnonzero(~valid_geometry) + 2).tolist()
        raise RuntimeError(
            f"Caixes invàlides o incompletes a labels.csv, files {rows}; "
            "calen coordenades numèriques amb x1 > x0 i y1 > y0."
        )

    df["_position"] = position_values.astype(int)
    for column in coordinate_columns:
        df[column] = coordinates[column]

    template = []
    for position in range(1, int(fixed_length) + 1):
        subset = df[df["_position"] == position]
        if subset.empty:
            raise RuntimeError(f"No hi ha cap caixa etiquetada per la posició {position}.")

        template.append({
            "pos": position,
            "x0": int(round(float(subset["box_x0"].mean()))),
            "y0": int(round(float(subset["box_y0"].mean()))),
            "x1": int(round(float(subset["box_x1"].mean()))),
            "y1": int(round(float(subset["box_y1"].mean()))),
            "n": int(len(subset)),
        })

    return template