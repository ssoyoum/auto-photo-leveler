"""Automatically detect and correct the horizontal tilt of photos.

The module can be used as a command-line tool or imported by another Python
program.  It uses Canny edges and probabilistic Hough lines to estimate the
dominant near-horizontal angle in each image.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


LOGGER = logging.getLogger("auto-photo-leveler")
SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class LevelResult:
    """Metadata produced after leveling one image."""

    input_path: Path
    output_path: Path
    detected_angle: float
    line_count: int


@dataclass(frozen=True)
class BatchResult:
    """Summary of a directory processing run."""

    processed: tuple[LevelResult, ...]
    failed: tuple[Path, ...]


def normalize_line_angle(angle_degrees: float) -> float:
    """Map a line angle to the interval [-90, 90).

    Hough line endpoints may be returned in either direction.  Normalizing
    both directions to the same interval makes a 179-degree line equivalent
    to a -1-degree line.
    """

    return (angle_degrees + 90.0) % 180.0 - 90.0


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR, BGRA, or grayscale input to an 8-bit grayscale image."""

    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError("Expected a 2D grayscale or 3D color image")
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _weighted_median(values: Iterable[float], weights: Iterable[float]) -> float:
    """Return a robust weighted median for line angles."""

    pairs = sorted(zip(values, weights), key=lambda pair: pair[0])
    if not pairs:
        return 0.0

    total_weight = sum(weight for _, weight in pairs)
    halfway = total_weight / 2.0
    accumulated = 0.0
    for value, weight in pairs:
        accumulated += weight
        if accumulated >= halfway:
            return float(value)
    return float(pairs[-1][0])


def estimate_skew_angle(
    image: np.ndarray,
    *,
    max_abs_angle: float = 45.0,
    canny_low: int = 50,
    canny_high: int = 150,
) -> tuple[float, int]:
    """Estimate the dominant near-horizontal line angle in an image.

    Returns ``(angle_degrees, line_count)``.  A positive angle means that a
    horizontal feature rises toward the left / falls toward the right in the
    usual image coordinate system.  OpenCV's positive rotation corrects that
    image tilt.

    Vertical lines are ignored because they are not useful for estimating a
    photo's horizon.  A weighted median reduces the influence of short text
    strokes and isolated edge fragments.
    """

    gray = _to_grayscale(image)
    height, width = gray.shape[:2]
    if min(height, width) < 2:
        return 0.0, 0

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high, apertureSize=3)

    min_line_length = max(20, int(min(height, width) * 0.15))
    threshold = max(20, int(min(height, width) * 0.10))
    max_line_gap = max(10, int(min(height, width) * 0.03))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    angles: list[float] = []
    lengths: list[float] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            angle = normalize_line_angle(math.degrees(math.atan2(dy, dx)))
            if abs(angle) <= max_abs_angle:
                angles.append(angle)
                lengths.append(length)

    if not angles:
        return 0.0, 0
    return _weighted_median(angles, lengths), len(angles)


def rotate_image(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate an image while expanding the canvas to avoid cropping."""

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = int(math.ceil((height * sine) + (width * cosine)))
    new_height = int(math.ceil((height * cosine) + (width * sine)))
    matrix[0, 2] += (new_width / 2.0) - center[0]
    matrix[1, 2] += (new_height / 2.0) - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def process_image(
    input_path: Path,
    output_dir: Path,
    *,
    max_abs_angle: float = 45.0,
) -> LevelResult:
    """Read, level, and save one image."""

    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    detected_angle, line_count = estimate_skew_angle(
        image, max_abs_angle=max_abs_angle
    )
    leveled = rotate_image(image, detected_angle)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name
    if not cv2.imwrite(str(output_path), leveled):
        raise OSError(f"Could not write image: {output_path}")

    return LevelResult(
        input_path=input_path,
        output_path=output_path,
        detected_angle=detected_angle,
        line_count=line_count,
    )


def find_images(input_dir: Path, *, recursive: bool = False) -> list[Path]:
    """Find supported image files in deterministic filename order."""

    if not input_dir.exists():
        return []
    candidates = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        (path for path in candidates if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


def process_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    recursive: bool = False,
    max_abs_angle: float = 45.0,
) -> BatchResult:
    """Process every supported image, continuing after individual failures."""

    processed: list[LevelResult] = []
    failed: list[Path] = []
    image_paths = find_images(input_dir, recursive=recursive)

    for input_path in image_paths:
        try:
            result = process_image(
                input_path, output_dir, max_abs_angle=max_abs_angle
            )
        except (OSError, ValueError, cv2.error) as error:
            LOGGER.error("Failed: %s (%s)", input_path.name, error)
            failed.append(input_path)
            continue

        processed.append(result)
        LOGGER.info(
            "Leveled %s -> %s (angle=%+.2f°, lines=%d)",
            input_path.name,
            result.output_path,
            result.detected_angle,
            result.line_count,
        )

    return BatchResult(tuple(processed), tuple(failed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and correct horizontal tilt in every image in a folder."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="Folder containing source images (default: input)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Folder for leveled images (default: output)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for images in subfolders of the input folder",
    )
    parser.add_argument(
        "--max-angle",
        type=float,
        default=45.0,
        help="Ignore detected lines steeper than this angle (default: 45)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-image processing logs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.max_angle <= 90.0:
        LOGGER.error("--max-angle must be greater than 0 and at most 90")
        return 2
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    result = process_directory(
        args.input_dir,
        args.output_dir,
        recursive=args.recursive,
        max_abs_angle=args.max_angle,
    )

    if not result.processed and not result.failed:
        LOGGER.warning("No supported images found in %s", args.input_dir)
    else:
        LOGGER.warning(
            "Completed: %d processed, %d failed",
            len(result.processed),
            len(result.failed),
        )
    return 1 if result.failed and not result.processed else 0


if __name__ == "__main__":
    raise SystemExit(main())
