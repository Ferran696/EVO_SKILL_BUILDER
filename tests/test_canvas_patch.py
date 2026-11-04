import unittest
from PIL import Image
import streamlit.elements.image as st_image


class TestCanvasCompatibility(unittest.TestCase):
    def test_st_image_image_to_url_exists_and_callable(self):
        # Assegurar que st_image.image_to_url existeix i accepta la signatura de streamlit-drawable-canvas
        import app
        self.assertTrue(hasattr(st_image, "image_to_url"))

        img = Image.new("RGB", (50, 50), color="white")
        # Crida amb la signatura utilitzada per streamlit-drawable-canvas
        try:
            url = st_image.image_to_url(img, 50, True, "RGB", "PNG", "test-canvas-bg")
            self.assertIsInstance(url, str)
        except Exception as e:
            self.fail(f"image_to_url ha fallat amb l'error: {e}")


if __name__ == "__main__":
    unittest.main()
