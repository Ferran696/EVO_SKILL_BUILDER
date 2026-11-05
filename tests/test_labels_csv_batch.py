import csv
import tempfile
import unittest
from pathlib import Path

from core.labels import (
    LABELS_CSV_FIELDS,
    LEGACY_CANVAS_FIELDS,
    append_label_rows,
    build_box_template_from_labels,
    migrate_legacy_labels_csv,
)
from streamlit.testing.v1 import AppTest


class LabelsCsvBatchTests(unittest.TestCase):
    def make_row(self, mode: str, position: int) -> dict:
        row = {
            "timestamp": "20261029_120000",
            "project_id": "test-project",
            "source_doc": "sample.pdf",
            "truth": "123456",
            "pos": position,
            "label": str(position % 10),
            "image_path": f"samples/{position}.png",
            "mode": mode,
            "ink_ratio": 0.1,
            "box_x0": position * 10,
            "box_y0": 2,
            "box_x1": position * 10 + 8,
            "box_y1": 18,
            "roi_x0": 0.1,
            "roi_y0": 0.2,
            "roi_x1": 0.9,
            "roi_y1": 0.8,
        }
        if mode == "canvas_manual":
            row.update({"auto_trim": True, "trim_threshold": 245, "trim_pad": 4, "zoom": 8})
        else:
            row.update({"hist_threshold_method": "otsu", "hist_pad_x": 2, "hist_pad_y": 3})
        return row

    def test_manual_and_auto_rows_share_schema_and_build_six_boxes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            labels_csv = Path(temp_dir) / "labels.csv"
            for position in range(1, 4):
                append_label_rows(labels_csv, [self.make_row("canvas_manual", position)])
            for position in range(4, 7):
                append_label_rows(labels_csv, [self.make_row("auto_histogram", position)])

            with labels_csv.open("r", newline="", encoding="utf-8") as file:
                records = list(csv.reader(file))
            self.assertEqual(records[0], LABELS_CSV_FIELDS)
            self.assertTrue(all(len(record) == len(LABELS_CSV_FIELDS) for record in records))
            auto_row = dict(zip(LABELS_CSV_FIELDS, records[4]))
            self.assertEqual(auto_row["box_y1"], "18")
            self.assertEqual(auto_row["hist_threshold_method"], "otsu")
            self.assertEqual(auto_row["auto_trim"], "")

            template = build_box_template_from_labels(Path(temp_dir), fixed_length=6)
            self.assertEqual([box["pos"] for box in template], [1, 2, 3, 4, 5, 6])
            self.assertEqual(template[3]["y1"], 18)

    def test_migrated_project_labels_build_six_boxes(self):
        project_dir = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "projects"
            / "cb_v2_old_20260629_135151"
        )
        template = build_box_template_from_labels(project_dir, fixed_length=6)
        self.assertEqual([box["pos"] for box in template], [1, 2, 3, 4, 5, 6])
        self.assertTrue(all(box["x1"] > box["x0"] and box["y1"] > box["y0"] for box in template))
        with (project_dir / "labels.csv").open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        manual_rows = [row for row in rows if row["mode"] == "canvas_manual"]
        auto_rows = [row for row in rows if row["mode"] == "auto_histogram"]
        self.assertEqual((len(manual_rows), len(auto_rows)), (43, 18))
        self.assertTrue(all(row["hist_threshold_method"] for row in auto_rows))
        self.assertTrue(all(row["box_y1"] and not row.get("box_y2") for row in auto_rows))
        self.assertTrue((project_dir / "labels.csv.bak").exists())

    def test_batch_page_reaches_document_upload_after_template(self):
        project_name = "cb_v2_old_20260629_135151"
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        self.assertFalse(app.exception)

        app.sidebar.radio[0].set_value("07 · Run batch").run()
        self.assertFalse(app.exception)
        app.selectbox[0].set_value(project_name).run()

        self.assertFalse(app.exception)
        self.assertTrue(any("Plantilla de caixes runtime" in item.value for item in app.markdown))
        self.assertTrue(any("Upload documents nous" in item.value for item in app.markdown))
        self.assertTrue(any("Model carregat" in item.value for item in app.success))

    def test_malformed_csv_reports_actionable_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            labels_csv = Path(temp_dir) / "labels.csv"
            with labels_csv.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(LABELS_CSV_FIELDS)
                writer.writerow(["extra"] * (len(LABELS_CSV_FIELDS) + 1))

            with self.assertRaisesRegex(RuntimeError, "Fila 2 inconsistent a labels.csv"):
                build_box_template_from_labels(Path(temp_dir), fixed_length=6)

    def test_writer_rejects_incompatible_existing_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            labels_csv = Path(temp_dir) / "labels.csv"
            labels_csv.write_text("old,header\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Normalitza o migra"):
                append_label_rows(labels_csv, [self.make_row("canvas_manual", 1)])

    def test_writer_auto_migrates_legacy_21_column_canvas_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            labels_csv = Path(temp_dir) / "labels.csv"
            legacy_row = self.make_row("canvas_manual", 1)
            with labels_csv.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=LEGACY_CANVAS_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerow(legacy_row)

            append_label_rows(labels_csv, [self.make_row("canvas_manual", 2)])

            with labels_csv.open("r", newline="", encoding="utf-8") as file:
                records = list(csv.reader(file))
            self.assertEqual(records[0], LABELS_CSV_FIELDS)
            self.assertEqual(len(records), 3)
            self.assertTrue((Path(temp_dir) / "labels.csv.bak").exists())

    def test_invalid_box_is_reported_with_csv_row(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            labels_csv = Path(temp_dir) / "labels.csv"
            row = self.make_row("canvas_manual", 1)
            row["box_x1"] = row["box_x0"]
            append_label_rows(labels_csv, [row])

            with self.assertRaisesRegex(RuntimeError, r"files \[2\]"):
                build_box_template_from_labels(Path(temp_dir), fixed_length=1)


if __name__ == "__main__":
    unittest.main()