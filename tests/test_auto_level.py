import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from auto_level import (
    estimate_skew_angle,
    normalize_line_angle,
    process_directory,
    rotate_image,
)


class AutoLevelTests(unittest.TestCase):
    def test_normalize_line_angle_handles_reverse_line_direction(self):
        self.assertAlmostEqual(normalize_line_angle(179), -1)
        self.assertAlmostEqual(normalize_line_angle(-181), -1)
        self.assertAlmostEqual(normalize_line_angle(90), -90)

    def test_estimate_skew_angle_finds_slightly_tilted_line(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.line(image, (30, 130), (290, 153), (255, 255, 255), 5)

        angle, line_count = estimate_skew_angle(image)

        self.assertGreater(line_count, 0)
        self.assertAlmostEqual(angle, 5.0, delta=1.5)

    def test_rotate_image_expands_canvas(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        rotated = rotate_image(image, 15, crop=False)

        self.assertGreater(rotated.shape[0], image.shape[0])
        self.assertGreater(rotated.shape[1], image.shape[1])

    def test_rotate_image_crops_generated_empty_corners_by_default(self):
        image = np.full((100, 200, 3), 255, dtype=np.uint8)

        rotated = rotate_image(image, 15)

        self.assertEqual(int(rotated.min()), 255)
        self.assertLess(rotated.shape[1], 200)

    def test_process_directory_writes_each_supported_image(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            image = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.line(image, (30, 130), (290, 153), (255, 255, 255), 5)
            self.assertTrue(cv2.imwrite(str(input_dir / "tilted.png"), image))

            result = process_directory(input_dir, output_dir)

            self.assertEqual(len(result.processed), 1)
            self.assertFalse(result.failed)
            self.assertTrue((output_dir / "tilted.png").exists())


if __name__ == "__main__":
    unittest.main()
