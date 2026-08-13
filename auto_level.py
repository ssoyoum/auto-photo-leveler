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


def rotate_image(
    image: np.ndarray,
    angle_degrees: float,
    *,
    border_mode: int = cv2.BORDER_CONSTANT,
    crop: bool = True,
) -> np.ndarray:
    """Rotate an image and automatically remove generated empty corners.

    Replicated edge pixels can make the outside of a rotated photo look
    stretched.  The default constant border is cropped away before returning
    the result, so no artificial pixels remain in the saved image.
    """

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = int(math.ceil((height * sine) + (width * cosine)))
    new_height = int(math.ceil((height * cosine) + (width * sine)))
    matrix[0, 2] += (new_width / 2.0) - center[0]
    matrix[1, 2] += (new_height / 2.0) - center[1]

    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=0,
    )

    if not crop:
        return rotated

    valid_mask = np.full((height, width), 255, dtype=np.uint8)
    valid_mask = cv2.warpAffine(
        valid_mask,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid_mask = cv2.erode(valid_mask, np.ones((3, 3), dtype=np.uint8))

    # Work on a small mask for high-resolution photos; output pixels remain
    # full resolution.
    mask_height, mask_width = valid_mask.shape
    mask_scale = min(1.0, 1200 / max(mask_height, mask_width))
    if mask_scale < 1.0:
        small_mask = cv2.resize(
            valid_mask,
            (
                max(1, int(round(mask_width * mask_scale))),
                max(1, int(round(mask_height * mask_scale))),
            ),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        small_mask = valid_mask

    small_rectangle = _largest_valid_rectangle(small_mask)
    if small_rectangle is None:
        return rotated

    small_x, small_y, small_width, small_height = small_rectangle
    if mask_scale < 1.0:
        x = int(math.ceil(small_x / mask_scale)) + 1
        y = int(math.ceil(small_y / mask_scale)) + 1
        right = int(math.floor((small_x + small_width) / mask_scale)) - 1
        bottom = int(math.floor((small_y + small_height) / mask_scale)) - 1
        rectangle = (x, y, max(1, right - x), max(1, bottom - y))
    else:
        rectangle = small_rectangle

    x, y, crop_width, crop_height = rectangle
    return rotated[y : y + crop_height, x : x + crop_width].copy()


def _largest_valid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the largest all-valid axis-aligned rectangle in a binary mask."""

    valid = mask > 0
    height, width = valid.shape
    histogram = [0] * width
    best_area = 0
    best_rectangle: tuple[int, int, int, int] | None = None

    for row in range(height):
        for column in range(width):
            histogram[column] = histogram[column] + 1 if valid[row, column] else 0

        stack: list[int] = []
        for column in range(width + 1):
            current_height = histogram[column] if column < width else 0
            while stack and histogram[stack[-1]] > current_height:
                top = stack.pop()
                rectangle_height = histogram[top]
                left = stack[-1] + 1 if stack else 0
                rectangle_width = column - left
                area = rectangle_width * rectangle_height
                if area > best_area:
                    best_area = area
                    best_rectangle = (
                        left,
                        row - rectangle_height + 1,
                        rectangle_width,
                        rectangle_height,
                    )
            stack.append(column)

    return best_rectangle


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


def _display_image(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Convert an image for display and fit it inside a review panel."""

    if image.ndim == 2:
        display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        display = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        display = image.copy()

    height, width = display.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        display = cv2.resize(
            display,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return display


def make_review_canvas(
    original: np.ndarray,
    preview: np.ndarray,
    angle: float,
    *,
    panel_width: int = 640,
    panel_height: int = 520,
) -> np.ndarray:
    """Create a side-by-side before/after image for the review window."""

    panel_height -= 42
    left = _display_image(original, panel_width, panel_height)
    right = _display_image(preview, panel_width, panel_height)

    def panel(image: np.ndarray, title: str) -> np.ndarray:
        canvas = np.zeros((panel_height + 42, panel_width, 3), dtype=np.uint8)
        y = 42 + max(0, (panel_height - image.shape[0]) // 2)
        x = max(0, (panel_width - image.shape[1]) // 2)
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
        cv2.putText(
            canvas, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            (255, 255, 255), 2, cv2.LINE_AA
        )
        return canvas

    result = cv2.hconcat([panel(left, "BEFORE"), panel(right, f"AFTER  {angle:+.2f} deg")])
    cv2.putText(
        result,
        "A/D or arrows: adjust | S: save after | O: save original | R: reset | N: skip | Q: quit",
        (16, result.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
        (220, 220, 220), 1, cv2.LINE_AA
    )
    return result


def review_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    recursive: bool = False,
    max_abs_angle: float = 45.0,
    angle_step: float = 0.1,
) -> tuple[int, int]:
    """Review each image and save the user's original/corrected choice."""

    saved = 0
    skipped = 0
    window_name = "Auto Photo Leveler - Review"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        for input_path in find_images(input_dir, recursive=recursive):
            image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                LOGGER.error("Failed to read %s", input_path.name)
                skipped += 1
                continue

            auto_angle, line_count = estimate_skew_angle(image, max_abs_angle=max_abs_angle)
            current_angle = auto_angle
            LOGGER.info("Reviewing %s (auto=%+.2f deg, lines=%d)", input_path.name, auto_angle, line_count)

            while True:
                preview = rotate_image(image, current_angle)
                cv2.imshow(window_name, make_review_canvas(image, preview, current_angle))
                key = cv2.waitKey(0) & 0xFF

                if key in (ord("a"), 81):
                    current_angle -= angle_step
                elif key in (ord("d"), 83):
                    current_angle += angle_step
                elif key == ord("r"):
                    current_angle = auto_angle
                elif key in (ord("s"), ord("o")):
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / input_path.name
                    selected = preview if key == ord("s") else image
                    if not cv2.imwrite(str(output_path), selected):
                        raise OSError(f"Could not write image: {output_path}")
                    saved += 1
                    break
                elif key in (ord("n"), 27):
                    skipped += 1
                    break
                elif key == ord("q"):
                    return saved, skipped
    finally:
        cv2.destroyAllWindows()

    return saved, skipped


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
    parser.add_argument(
        "--review",
        action="store_true",
        help="Open a before/after window and manually approve each image",
    )
    parser.add_argument(
        "--angle-step",
        type=float,
        default=0.1,
        help="Manual adjustment step in degrees for --review (default: 0.1)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.max_angle <= 90.0:
        LOGGER.error("--max-angle must be greater than 0 and at most 90")
        return 2
    if args.angle_step <= 0.0:
        LOGGER.error("--angle-step must be greater than 0")
        return 2
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    if args.review:
        saved, skipped = review_directory(
            args.input_dir,
            args.output_dir,
            recursive=args.recursive,
            max_abs_angle=args.max_angle,
            angle_step=args.angle_step,
        )
        LOGGER.warning("Review completed: %d saved, %d skipped", saved, skipped)
        return 0

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
