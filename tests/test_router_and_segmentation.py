import unittest
from pathlib import Path
from PIL import Image
import torch

from core.router import (
    classify_document_format,
    FORMAT_DEFINITIONS,
    is_project_router_enabled,
    get_document_roi_and_format,
)
from core.segmentation import auto_split_characters_by_x_histogram
from core.utils import load_first_page_image, crop_relative
from app import predict_document_code_from_path, load_trained_model_for_project, load_project_config


class RouterAndSegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_dir = Path("data/projects/cb_v2_old_20260629_135151")
        cls.uploads_dir = cls.project_dir / "uploads"
        cls.has_uploads = cls.uploads_dir.exists()

    def test_classify_document_formats(self):
        if not self.has_uploads:
            self.skipTest("No hi ha directori uploads per a les proves.")

        # 0.pdf: Codi de barres (Tipus 1)
        doc0 = load_first_page_image(self.uploads_dir / "0.pdf", dpi=150)
        res0 = classify_document_format(doc0)
        self.assertEqual(res0["format_id"], "format_1_barcode")
        self.assertEqual(res0["expected_length"], 7)
        self.assertFalse(res0["filter_dots"])

        # 11.pdf: Logo Carburos Metálicos (Tipus 2)
        doc11 = load_first_page_image(self.uploads_dir / "11.pdf", dpi=150)
        res11 = classify_document_format(doc11)
        self.assertEqual(res11["format_id"], "format_2_logo")
        self.assertEqual(res11["expected_length"], 6)
        self.assertFalse(res11["filter_dots"])

        # 15.pdf: Excepció amb capçalera buida / punts (Tipus 3)
        doc15 = load_first_page_image(self.uploads_dir / "15.pdf", dpi=150)
        res15 = classify_document_format(doc15)
        self.assertEqual(res15["format_id"], "format_3_exception")
        self.assertEqual(res15["expected_length"], 6)
        self.assertTrue(res15["filter_dots"])

    def test_filter_dots_in_format_3_segmentation(self):
        if not self.has_uploads:
            self.skipTest("No hi ha directori uploads.")

        doc15 = load_first_page_image(self.uploads_dir / "15.pdf", dpi=150)
        fmt15 = classify_document_format(doc15)
        roi_crop = crop_relative(doc15, fmt15["roi"])

        crops, boxes, debug = auto_split_characters_by_x_histogram(
            roi_img=roi_crop,
            threshold_method="otsu",
            min_char_width=3,
            enable_valley_split=True,
            expected_count=6,
            filter_dots=True,
        )

        # Ha d'haver filtrat el punt decimal del mig i extreure exactament els 6 dígits
        self.assertEqual(len(crops), 6)
        self.assertEqual(len(boxes), 6)

    def test_predict_document_code_with_router(self):
        if not self.has_uploads:
            self.skipTest("No hi ha uploads.")

        config = load_project_config(self.project_dir)
        model, alphabet, device, _ = load_trained_model_for_project(self.project_dir)

        # Provar predicció sobre 15.pdf (format tipus 3: 440.217 -> codi de 6 dígits net)
        pred15 = predict_document_code_from_path(
            path=self.uploads_dir / "15.pdf",
            config=config,
            model=model,
            alphabet=alphabet,
            device=device,
            use_router=True,
        )

        self.assertEqual(pred15["format_id"], "format_3_exception")
        self.assertEqual(len(pred15["code"]), 6)
        self.assertNotIn(".", pred15["code"])
        self.assertTrue(pred15["code"].isdigit())
        self.assertEqual(pred15["code"], "440217")

    def test_format_2_roi_and_segmentation(self):
        if not self.has_uploads:
            self.skipTest("No hi ha directori uploads.")

        doc11 = load_first_page_image(self.uploads_dir / "11.pdf", dpi=150)
        fmt11 = classify_document_format(doc11)
        self.assertEqual(fmt11["format_id"], "format_2_logo")
        self.assertEqual(fmt11["roi"], FORMAT_DEFINITIONS["format_2_logo"]["default_roi"])

        roi_crop = crop_relative(doc11, fmt11["roi"])
        crops, boxes, debug = auto_split_characters_by_x_histogram(
            roi_img=roi_crop,
            threshold_method="otsu",
            min_char_width=3,
            enable_valley_split=True,
            expected_count=6,
            filter_dots=False,
        )
        self.assertEqual(len(crops), 6)
        self.assertEqual(len(boxes), 6)

    def test_multi_format_truth_validation(self):
        # Comprovar que el format 1 (7 dígits) i els formats 2/3 (6 dígits) es validen adequadament
        if not self.has_uploads:
            self.skipTest("No hi ha uploads.")

        doc0 = load_first_page_image(self.uploads_dir / "0.pdf", dpi=150)
        res0 = classify_document_format(doc0)
        self.assertEqual(res0["expected_length"], 7)

        truth_7 = "0528158"
        self.assertEqual(len(truth_7), res0["expected_length"])

        doc11 = load_first_page_image(self.uploads_dir / "11.pdf", dpi=150)
        res11 = classify_document_format(doc11)
        self.assertEqual(res11["expected_length"], 6)

        truth_6 = "515159"
        self.assertEqual(len(truth_6), res11["expected_length"])

    def test_ten_digit_project_uses_fixed_length_and_project_roi(self):
        blank_img = Image.new("RGB", (600, 800), (255, 255, 255))
        custom_roi = {"x0": 0.635, "y0": 0.17, "x1": 0.735, "y1": 0.19}
        config_10 = {
            "project_id": "albarans_10",
            "fixed_length": 10,
            "roi": custom_roi,
        }
        self.assertFalse(is_project_router_enabled(config_10))
        res = get_document_roi_and_format(blank_img, config=config_10)
        self.assertEqual(res["format_id"], "project_fixed")
        self.assertEqual(res["expected_length"], 10)
        self.assertEqual(res["roi"], custom_roi)
        self.assertFalse(res["filter_dots"])
        self.assertFalse(res["router_active"])


if __name__ == "__main__":
    unittest.main()
