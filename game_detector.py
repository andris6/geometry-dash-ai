import time
from typing import Optional, Tuple

import cv2
import mss
import numpy as np

from constants import (
    CAPTURE_HEIGHT,
    CAPTURE_OFFSET_X,
    CAPTURE_OFFSET_Y,
    CAPTURE_WIDTH,
    GAME_WINDOW_TITLE,
)
from utils import log, resize_frame

# Try win32gui
try:
    import win32gui
    _WIN32GUI_AVAILABLE = True
except ImportError:
    _WIN32GUI_AVAILABLE = False

# Fallback: pygetwindow
try:
    import pygetwindow as gw
    _PYGETWINDOW_AVAILABLE = True
except ImportError:
    _PYGETWINDOW_AVAILABLE = False


# ─── Window Finder ───────────────────────────────────────────────────────────

class WindowInfo:
    """Describes the location of the game window on the screen."""

    def __init__(self, left: int, top: int, width: int, height: int) -> None:
        self.left   = left
        self.top    = top
        self.width  = width
        self.height = height

    def __repr__(self) -> str:
        return (f"WindowInfo(left={self.left}, top={self.top}, "
                f"width={self.width}, height={self.height})")


def find_window(title: str = GAME_WINDOW_TITLE) -> Optional[WindowInfo]:
    if _WIN32GUI_AVAILABLE:
        return _find_win32(title)
    if _PYGETWINDOW_AVAILABLE:
        return _find_pygetwindow(title)
    log.warning(
        "Neither win32gui nor pygetwindow is available. "
        "Falling back to full-screen capture. Install pygetwindow for better results."
    )
    return None


def _find_win32(title: str) -> Optional[WindowInfo]:
    hwnd = None

    def _cb(h, _):
        nonlocal hwnd
        if title.lower() in win32gui.GetWindowText(h).lower():
            hwnd = h

    win32gui.EnumWindows(_cb, None)
    if hwnd is None:
        log.debug("win32gui: window '%s' not found.", title)
        return None
    try:
        rect = win32gui.GetWindowRect(hwnd)   # (left, top, right, bottom)
        left, top, right, bottom = rect
        return WindowInfo(left, top, right - left, bottom - top)
    except Exception as exc:
        log.warning("win32gui.GetWindowRect failed: %s", exc)
        return None


def _find_pygetwindow(title: str) -> Optional[WindowInfo]:
    wins = gw.getWindowsWithTitle(title)
    if not wins:
        log.debug("pygetwindow: window '%s' not found.", title)
        return None
    w = wins[0]
    return WindowInfo(w.left, w.top, w.width, w.height)


# ─── Screen Capturer ─────────────────────────────────────────────────────────

class GameDetector:

    def __init__(
        self,
        title: str = GAME_WINDOW_TITLE,
        capture_width: int = CAPTURE_WIDTH,
        capture_height: int = CAPTURE_HEIGHT,
        offset_x: int = CAPTURE_OFFSET_X,
        offset_y: int = CAPTURE_OFFSET_Y,
    ) -> None:
        self._title          = title
        self._capture_width  = capture_width
        self._capture_height = capture_height
        self._offset_x       = offset_x
        self._offset_y       = offset_y

        self._window: Optional[WindowInfo] = None
        self._region: Optional[dict]       = None   # MSS monitor dict
        self._sct: Optional[mss.MSSBase]   = None   # reusable MSS context

    # ── Public API ──────────────────────────────────────────────────────────

    def find_and_attach(self) -> bool:
        self._window = find_window(self._title)
        if self._window is None:
            log.warning(
                "Geometry Dash window not found. "
                "Make sure the game is running and visible."
            )
            self._region = self._full_screen_region()
        else:
            log.info("Found window: %s", self._window)
            self._region = self._compute_region(self._window)

        # Open a persistent MSS context (avoids per-frame context overhead)
        self._sct = mss.mss()
        return self._window is not None

    def grab(self) -> Optional[np.ndarray]:
        if self._sct is None:
            log.error("grab() called before find_and_attach().")
            return None
        try:
            raw = self._sct.grab(self._region)
            # raw is BGRA; convert to BGR and resize
            frame = np.array(raw, dtype=np.uint8)[..., :3]   # drop alpha
            frame = resize_frame(frame, self._capture_width, self._capture_height)
            return frame
        except Exception as exc:
            log.error("Screen capture failed: %s", exc)
            return None

    def refresh_window(self) -> bool:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
        return self.find_and_attach()

    @property
    def region(self) -> Optional[dict]:
        """The MSS monitor dict currently in use."""
        return self._region

    @property
    def window_info(self) -> Optional[WindowInfo]:
        return self._window

    # ── Internals ───────────────────────────────────────────────────────────

    def _compute_region(self, wi: WindowInfo) -> dict:
        left   = wi.left   + self._offset_x
        top    = wi.top    + self._offset_y
        width  = wi.width  - self._offset_x
        height = wi.height - self._offset_y
        # Clamp to positive
        width  = max(width,  1)
        height = max(height, 1)
        return {"left": left, "top": top, "width": width, "height": height}

    @staticmethod
    def _full_screen_region() -> dict:
        """Fallback: capture the entire primary monitor."""
        with mss.mss() as sct:
            mon = sct.monitors[1]   # monitors[0] is "all monitors" aggregate
        return {"left": mon["left"], "top": mon["top"],
                "width": mon["width"], "height": mon["height"]}

    def __del__(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass


# ─── Convenience helper ───────────────────────────────────────────────────────

def wait_for_window(title: str = GAME_WINDOW_TITLE,
                    timeout: float = 30.0,
                    poll_interval: float = 1.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if find_window(title) is not None:
            log.info("Window '%s' detected.", title)
            return True
        time.sleep(poll_interval)
    log.warning("Timed out waiting for window '%s'.", title)    
    return False

