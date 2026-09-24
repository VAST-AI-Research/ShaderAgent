"""Logging, image helpers, and small file I/O."""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Iterable, Optional, Tuple

from PIL import Image

try:
    from rich.logging import RichHandler
    HAS_RICH = True
except ImportError:
    RichHandler = None
    HAS_RICH = False

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# One file handler per worker thread, each filtered to its own thread id so that
# batch mode (dataset_workers > 1) keeps per-run logs.log files from mixing.
_thread_file_handlers: "dict[int, logging.FileHandler]" = {}
_thread_file_lock = threading.Lock()


def get_logger(name: str = "shader_pipeline") -> logging.Logger:
    """Return a named logger with a console handler (file logging: :func:`set_log_dir`)."""
    logger = logging.getLogger(name)

    level_name = os.getenv("SHADER_PIPELINE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    if HAS_RICH and RichHandler:
        if not any(isinstance(h, RichHandler) for h in logger.handlers):
            logger.addHandler(RichHandler(rich_tracebacks=True, markup=True))
    else:
        # Fallback for environments without rich, e.g. the bundled Blender Python.
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            # stdout, because the Blender subprocess output is parsed from there.
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
            logger.addHandler(stream_handler)

    return logger


def set_log_dir(save_dir: str) -> None:
    """Attach a per-thread file handler writing to *save_dir*/logs.log.

    Each handler only accepts records emitted from the calling thread, so concurrent
    batch runs don't interleave into each other's log files. A second call from the
    same thread replaces its previous handler.
    """
    owner_id = threading.get_ident()
    root = logging.getLogger()

    os.makedirs(save_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(save_dir, "logs.log"))
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    handler.addFilter(lambda record, tid=owner_id: record.thread == tid)

    with _thread_file_lock:
        prev = _thread_file_handlers.pop(owner_id, None)
        if prev is not None:
            root.removeHandler(prev)
            prev.close()
        _thread_file_handlers[owner_id] = handler
    root.addHandler(handler)


DEFAULT_GRAY = (128, 128, 128)

# Long-edge cap for reference images on VLM-bound paths. Larger inputs blow past API
# gateway body-size limits (e.g. nginx 413) once base64-encoded, and most VLMs crop
# to a similar resolution internally anyway.
DEFAULT_MAX_LONG_EDGE = 1024


def cap_long_edge(img: Image.Image, max_long_edge: int = DEFAULT_MAX_LONG_EDGE) -> Image.Image:
    """Downscale ``img`` so neither side exceeds ``max_long_edge``; no-op if it already
    fits or ``max_long_edge`` is falsy."""
    if not max_long_edge:
        return img
    w, h = img.size
    longest = max(w, h)
    if longest <= max_long_edge:
        return img
    scale = max_long_edge / longest
    return img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)


def to_rgb_on_gray(
    img: Image.Image, bg: Tuple[int, int, int] = DEFAULT_GRAY
) -> Image.Image:
    """Return an RGB image, compositing any alpha channel against ``bg``.

    Reference images often arrive as RGBA with a transparent cutout background.
    ``img.convert("RGB")`` drops the alpha and exposes whatever RGB values sat under
    the transparent pixels (color bars, noise, encoder garbage), which then leak into
    VLM inputs. Compositing onto solid mid-gray instead makes transparent pixels gray
    and blends partial alpha correctly. RGB images are returned unchanged.
    """

    if img.mode in ("RGBA", "LA"):
        img = img.convert("RGBA")
    elif img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    else:
        return img.convert("RGB") if img.mode != "RGB" else img

    background = Image.new("RGBA", img.size, bg + (255,))
    return Image.alpha_composite(background, img).convert("RGB")


def load_image(
    path: Optional[str],
    max_long_edge: Optional[int] = DEFAULT_MAX_LONG_EDGE,
) -> Image.Image:
    """Load an image as a detached RGB copy: alpha composited on mid-gray (see
    :func:`to_rgb_on_gray`), long edge capped at ``max_long_edge`` pixels to stay
    under typical API gateway body-size limits. ``None`` or 0 disables capping.
    """

    if not path:
        raise ValueError("path must be a non-empty string")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image path {path} not found.")

    with Image.open(path) as img:
        sanitized = to_rgb_on_gray(img)
        return cap_long_edge(sanitized, max_long_edge).copy()


def load_images(
    paths: Iterable[str],
    max_long_edge: Optional[int] = DEFAULT_MAX_LONG_EDGE,
) -> list[Image.Image]:
    """Load multiple images; see :func:`load_image`."""

    return [load_image(path, max_long_edge=max_long_edge) for path in paths]



def save_to_file(content, file_path: str, format_type: str = "w") -> None:
    with open(file_path, format_type) as f:
        f.write(content)
        f.flush()
