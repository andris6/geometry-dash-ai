import logging
import os
import pickle
import sys
import time
from collections import deque
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from constants import LOG_FILE, LOG_LEVEL


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logging(name: str = "gd_neat") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(level)
    logger.addHandler(sh)
    # file handler
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)
    return logger


log = setup_logging()


# ─── Timing ──────────────────────────────────────────────────────────────────

class Stopwatch:

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def reset(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        """Return seconds since last reset() (or construction)."""
        return time.perf_counter() - self._start


class FPSCounter:

    def __init__(self, window: int = 60) -> None:
        self._times: deque = deque(maxlen=window)

    def tick(self) -> None:
        self._times.append(time.perf_counter())

    def get(self) -> float:
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / dt if dt > 0 else 0.0


# ─── Pickle I/O ──────────────────────────────────────────────────────────────

def save_object(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log.debug("Saved object to %s", path)


def load_object(path: str) -> Optional[Any]:
    if not os.path.isfile(path):
        log.warning("load_object: file not found: %s", path)
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    log.debug("Loaded object from %s", path)
    return obj


# ─── Geometry helpers ────────────────────────────────────────────────────────

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize(value: float, lo: float, hi: float) -> float:
    span = hi - lo
    if span == 0:
        return 0.0
    return clamp((value - lo) / span, 0.0, 1.0)


def rect_center(rect: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = rect
    return x + w / 2.0, y + h / 2.0


def rects_overlap(r1: Tuple[int, int, int, int],
                  r2: Tuple[int, int, int, int]) -> bool:
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return not (
        x1 + w1 <= x2 or
        x2 + w2 <= x1 or
        y1 + h1 <= y2 or
        y2 + h2 <= y1
    )


# ─── Image helpers ───────────────────────────────────────────────────────────

def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def to_bgr(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        return frame
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def draw_text(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    color=(0, 255, 0),
    scale: float = 0.45,
    thickness: int = 1,
) -> None:
    cv2.putText(
        img, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness,
        cv2.LINE_AA,
    )


def draw_rect(
    img: np.ndarray,
    rect: Tuple[int, int, int, int],
    color=(0, 255, 0),
    thickness: int = 1,
) -> None:
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)


# ─── Stats accumulator ───────────────────────────────────────────────────────

class GenerationStats:

    CSV_PATH = "generation_stats.csv"
    _HEADER  = "generation,best_fitness,avg_fitness,worst_fitness,num_species,elapsed_s\n"

    def __init__(self) -> None:
        self._rows = []
        # Write header if file is new
        if not os.path.isfile(self.CSV_PATH):
            with open(self.CSV_PATH, "w") as f:
                f.write(self._HEADER)

    def record(
        self,
        generation: int,
        best: float,
        avg: float,
        worst: float,
        num_species: int,
        elapsed_s: float,
    ) -> None:
        row = f"{generation},{best:.4f},{avg:.4f},{worst:.4f},{num_species},{elapsed_s:.2f}\n"
        self._rows.append(row)
        with open(self.CSV_PATH, "a") as f:
            f.write(row)
        log.info(
            "Gen %d | best=%.2f avg=%.2f worst=%.2f species=%d t=%.1fs",
            generation, best, avg, worst, num_species, elapsed_s,
        )


# ─── Graceful shutdown helper ─────────────────────────────────────────────────

_shutdown_requested = False


def request_shutdown() -> None:
    global _shutdown_requested
    _shutdown_requested = True
    log.info("Shutdown requested.")


def shutdown_requested() -> bool:
    return _shutdown_requested


# ─── Directory bootstrap ──────────────────────────────────────────────────────

def ensure_dirs(*dirs: str) -> None:
    """Create directories if they do not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        log.debug("Ensured directory: %s", d)

