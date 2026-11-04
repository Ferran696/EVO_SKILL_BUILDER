import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from PIL import Image
import numpy as np

from core.router import classify_document_format, locate_anchor_and_adjust_roi, FORMAT_DEFINITIONS
from core.segmentation import (
    auto_split_characters_by_x_histogram,
    extract_characters_by_connected_components,
    segment_roi_characters,
    deskew_image,
)
from core.storage import (
    sanitize_filename_token,
    build_renamed_filenames,
    build_batch_export_zip,
    export_batch_to_folder,
)
from core.utils import load_first_page_image, crop_relative
from app import predict_document_code_from_path, load_trained_model_for_project, load_project_config


class RobustFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_dir = Path("data/projects/contractos_cb_mix_20261103_133240")
        if not cls.project_dir.exists():
            cls.project_dir = Path("data/projects/cb_v2_old_20260629_135151")
        cls.has_project = cls.project_dir.exists()

    def test_deskew_image(self):
        # Crear imatge de prova lleugerament rotada
        img = Image.new("L", (200, 80), 255)
        # Afegir línia negra
        arr = np.array(img)
        arr[35:45, 20:180] = 0
        test_img = Image.fromarray(arr).rotate(2.5, resample=Image.BICUBIC, fillcolor=255)

        rotated, angle = deskew_image(test_img, max_angle=5.0)
        self.assertIsInstance(rotated, Image.Image)
        # L'angle detectat ha d'estar a prop de 2.5 graus
        self.assertTrue(abs(angle) <= 5.0)

    def test_connected_components_segmentation_clean(self):
        # Crear ROI sintètica amb 6 dígits
        roi = Image.new("L", (180, 40), 255)
        arr = np.array(roi)
        # 6 blocs de dígits
        for i in range(6):
            x_start = 10 + i * 28
            arr[10:32, x_start:x_start+16] = 0
        test_roi = Image.fromarray(arr)

        crops, boxes, debug = extract_characters_by_connected_components(
            roi_img=test_roi,
            expected_count=6,
            filter_dots=False,
        )
        self.assertEqual(len(crops), 6)
        self.assertEqual(len(boxes), 6)

    def test_connected_components_filter_dots(self):
        # Crear ROI amb 6 dígits i un punt decimal entremig
        roi = Image.new("L", (220, 40), 255)
        arr = np.array(roi)
        for i in range(3):
            x_start = 10 + i * 28
            arr[10:32, x_start:x_start+16] = 0
        # Punt decimal petit
        arr[26:30, 98:102] = 0
        for i in range(3):
            x_start = 110 + i * 28
            arr[10:32, x_start:x_start+16] = 0
        test_roi = Image.fromarray(arr)

        crops, boxes, debug = extract_characters_by_connected_components(
            roi_img=test_roi,
            expected_count=6,
            filter_dots=True,
        )
        self.assertEqual(len(crops), 6)
        self.assertEqual(len(boxes), 6)

    def test_locate_anchor_and_adjust_roi_fallback(self):
        # Si la imatge és blanca, retorna base_roi sense errors
        blank = Image.new("RGB", (600, 800), (255, 255, 255))
        base_roi = {"x0": 0.8, "y0": 0.1, "x1": 0.9, "y1": 0.2}
        adj_roi, debug = locate_anchor_and_adjust_roi(blank, "format_1_barcode", base_roi)
        self.assertEqual(adj_roi, base_roi)
        self.assertFalse(debug["anchor_found"])

    def test_unified_segment_roi_characters(self):
        roi = Image.new("L", (180, 40), 255)
        arr = np.array(roi)
        for i in range(6):
            x_start = 10 + i * 28
            arr[10:32, x_start:x_start+16] = 0
        test_roi = Image.fromarray(arr)

        crops_cc, boxes_cc, _ = segment_roi_characters(
            roi_img=test_roi,
            method="connected_components",
            expected_count=6,
        )
        self.assertEqual(len(crops_cc), 6)

        crops_hist, boxes_hist, _ = segment_roi_characters(
            roi_img=test_roi,
            method="histogram",
            expected_count=6,
        )
        self.assertEqual(len(crops_hist), 6)

    def test_connected_components_ignores_top_parasite_fragments(self):
        # Simula 20.pdf: 6 dígits principals (h=26) i fragments petits a la vora superior (y=0..5) intercalats en X
        roi = Image.new("L", (180, 44), 255)
        arr = np.array(roi)
        for i in range(6):
            x_start = 10 + i * 26
            arr[12:38, x_start:x_start+15] = 0
        # Fragment paràsit superior ('Grupo Air Products') intercalat entre el 2n i 3r dígit
        arr[0:6, 42:54] = 0
        arr[0:5, 95:105] = 0
        test_roi = Image.fromarray(arr)

        crops, boxes, _ = extract_characters_by_connected_components(
            roi_img=test_roi,
            expected_count=6,
            filter_dots=False,
        )
        self.assertEqual(len(crops), 6)
        for (_, y0, _, y1) in boxes:
            self.assertGreaterEqual(y1 - y0, 20)

    def test_connected_components_splits_multiple_touching_blobs(self):
        # Simula 16.pdf / 17.pdf: un bloc de 3 dígits enganxats + punt + bloc de 2 dígits enganxats + 1 dígit solt
        roi = Image.new("L", (120, 32), 255)
        arr = np.array(roi)
        # Bloc 1: 3 dígits enganxats amb valls primes
        arr[8:23, 8:17] = 0
        arr[14:16, 17:19] = 0  # pont prim
        arr[8:23, 19:28] = 0
        arr[14:16, 28:30] = 0  # pont prim
        arr[8:23, 30:39] = 0
        # Punt decimal
        arr[20:23, 43:46] = 0
        # Bloc 2: 2 dígits enganxats amb vall prima
        arr[8:23, 50:59] = 0
        arr[14:16, 59:61] = 0  # pont prim
        arr[8:23, 61:70] = 0
        # Bloc 3: 1 dígit solt
        arr[8:23, 76:84] = 0

        test_roi = Image.fromarray(arr)
        crops, boxes, _ = extract_characters_by_connected_components(
            roi_img=test_roi,
            expected_count=6,
            filter_dots=True,
        )
        self.assertEqual(len(crops), 6)

    def test_real_batch_documents_16_to_20(self):
        pdir = Path("data/projects/contractos_cb_mix_20261103_133240")
        upload_dir = pdir / "uploads"
        if not (pdir.exists() and (pdir / "models" / "model_latest.pt").exists()):
            self.skipTest("Model o projecte contractos_cb_mix no disponible.")

        config = load_project_config(pdir)
        model, alphabet, device, _ = load_trained_model_for_project(pdir)

        expected = {
            "16.pdf": "440221",
            "17.pdf": "440220",
            "18.pdf": "0529820",
            "19.pdf": "517932",
            "20.pdf": "517919",
        }
        for fname, exp_code in expected.items():
            fpath = upload_dir / fname
            if not fpath.exists():
                continue
            for use_d in (False, True):
                res = predict_document_code_from_path(
                    fpath,
                    config,
                    model,
                    alphabet,
                    device,
                    use_router=True,
                    segmentation_method="connected_components",
                    use_anchor=True,
                    use_deskew=use_d,
                )
                self.assertEqual(res["code"], exp_code, f"Error a {fname} amb deskew={use_d}")

    def test_build_renamed_filenames_patterns_and_collisions(self):
        rows = [
            {"exportar": True, "fitxer": "18.pdf", "codi": "0529820", "status": "OK"},
            {"exportar": True, "fitxer": "19.pdf", "codi": "0529820", "status": "OK"},
            {"exportar": False, "fitxer": "20.pdf", "codi": "517:919*", "status": "REVIEW"},
        ]

        out_default = build_renamed_filenames(rows, pattern="{codi}")
        self.assertEqual(out_default[0]["nou_nom"], "0529820.pdf")
        self.assertEqual(out_default[1]["nou_nom"], "0529820_2.pdf")
        self.assertEqual(out_default[2]["nou_nom"], "517_919_.pdf")

        out_prefix = build_renamed_filenames(rows, pattern="{prefix}{codi}", prefix="ALB_")
        self.assertEqual(out_prefix[0]["nou_nom"], "ALB_0529820.pdf")
        self.assertEqual(out_prefix[1]["nou_nom"], "ALB_0529820_2.pdf")

        out_orig = build_renamed_filenames(rows, pattern="{codi}_{original}")
        self.assertEqual(out_orig[0]["nou_nom"], "0529820_18.pdf")
        self.assertEqual(out_orig[1]["nou_nom"], "0529820_19.pdf")

    def test_batch_export_zip_and_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_dir = tmp_path / "src"
            dst_dir = tmp_path / "dst"
            src_dir.mkdir(parents=True, exist_ok=True)

            (src_dir / "18.pdf").write_bytes(b"%PDF-1.4 mock 18")
            (src_dir / "19.pdf").write_bytes(b"%PDF-1.4 mock 19")
            (src_dir / "20.pdf").write_bytes(b"%PDF-1.4 mock 20")

            rows = build_renamed_filenames([
                {"exportar": True, "fitxer": "18.pdf", "codi": "0529820", "format": "Tipus 1", "min_conf": 0.99, "avg_conf": 0.99, "status": "OK"},
                {"exportar": True, "fitxer": "19.pdf", "codi": "517932", "format": "Tipus 2", "min_conf": 0.98, "avg_conf": 0.99, "status": "OK"},
                {"exportar": False, "fitxer": "20.pdf", "codi": "517919", "format": "Tipus 2", "min_conf": 0.65, "avg_conf": 0.80, "status": "REVIEW"},
            ], pattern="{codi}")

            # 1) Test ZIP export
            zip_bytes = build_batch_export_zip(src_dir, rows)
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
                names = set(zf.namelist())
                self.assertIn("0529820.pdf", names)
                self.assertIn("517932.pdf", names)
                self.assertNotIn("517919.pdf", names)
                self.assertIn("index_exportacio.csv", names)
                self.assertEqual(zf.read("0529820.pdf"), b"%PDF-1.4 mock 18")

            # 2) Test Folder export
            summary = export_batch_to_folder(src_dir, dst_dir, rows)
            self.assertEqual(summary["copied_count"], 2)
            self.assertTrue((dst_dir / "0529820.pdf").exists())
            self.assertTrue((dst_dir / "517932.pdf").exists())
            self.assertFalse((dst_dir / "517919.pdf").exists())
            self.assertTrue((dst_dir / "index_exportacio.csv").exists())


if __name__ == "__main__":
    unittest.main()
