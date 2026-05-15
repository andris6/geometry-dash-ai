import cv2
import numpy as np

from constants import (
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    DEBUG_BOX_COLOR,
    DEBUG_FONT_COLOR,
    DEBUG_FONT_SCALE,
    DEBUG_PORTAL_COLOR,
    DEBUG_WARN_COLOR,
    MODE_NAMES,
    OVERLAY_ALPHA,
    PLAYER_X_FRACTION,
    SPEED_NAMES,
    VISION_HEIGHT,
    VISION_WIDTH,
)
from game_state import GameState
from utils import draw_rect, draw_text, to_bgr


class DebugOverlay:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        frame_bgr: np.ndarray,
        state: GameState,
        *,
        generation:  int   = 0,
        genome_idx:  int   = 0,
        nn_output:   float = 0.0,
        action:      str   = "none",
        fitness:     float = 0.0,
        fps:         float = 0.0,
    ) -> np.ndarray:
        """
        Return an annotated copy of *frame_bgr* (does not modify in-place).
        Returns the original frame untouched if overlay is disabled.
        """
        if not self.enabled or frame_bgr is None:
            return frame_bgr

        # Work on a copy so we do not pollute the original
        img = frame_bgr.copy()
        img = to_bgr(img)

        # Scale factors from vision space → capture space
        sx = CAPTURE_WIDTH  / VISION_WIDTH
        sy = CAPTURE_HEIGHT / VISION_HEIGHT

        # ── Obstacle boxes ───────────────────────────────────────────────────
        for obs in state.obstacles:
            rx = int(obs.x      * VISION_WIDTH  * sx)
            ry = int(obs.y      * VISION_HEIGHT * sy)
            rw = int(obs.width  * VISION_WIDTH  * sx)
            rh = int(obs.height * VISION_HEIGHT * sy)
            draw_rect(img, (rx, ry, rw, rh), DEBUG_BOX_COLOR, 1)

        # ── Portal boxes ─────────────────────────────────────────────────────
        for portal in state.portals:
            px_c = int(portal.x      * VISION_WIDTH  * sx)
            py_c = int(portal.y      * VISION_HEIGHT * sy)
            pw   = int(portal.width  * VISION_WIDTH  * sx)
            ph   = int(portal.height * VISION_HEIGHT * sy)
            rx   = max(0, px_c - pw // 2)
            ry   = max(0, py_c - ph // 2)
            draw_rect(img, (rx, ry, pw, ph), DEBUG_PORTAL_COLOR, 1)
            draw_text(img, portal.portal_type[:8], (rx, max(0, ry - 4)),
                      DEBUG_PORTAL_COLOR, scale=0.35)

        # ── Player marker ─────────────────────────────────────────────────────
        px = int(PLAYER_X_FRACTION * CAPTURE_WIDTH)
        py = int(state.player.y_position * CAPTURE_HEIGHT)
        cv2.drawMarker(img, (px, py), (0, 255, 0),
                       cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)

        # ── HUD text block ────────────────────────────────────────────────────
        lines = [
            f"Gen {generation:3d}  Genome {genome_idx:3d}",
            f"Mode: {MODE_NAMES.get(state.mode, '?')}",
            f"Speed: {SPEED_NAMES.get(state.speed, '?')}",
            f"Grav: {'INVERTED' if state.gravity_inverted else 'normal'}",
            f"Mini: {state.is_mini}  Dual: {state.is_dual}  Mirror: {state.is_mirrored}",
            f"Y: {state.player.y_position:.3f}",
            f"Progress: {state.progress * 100:.1f}%",
            f"Alive: {state.alive_time:.1f}s",
            f"Portals: {state.portals_passed}",
            f"NN out: {nn_output:+.3f}  → {action}",
            f"Fitness: {fitness:.2f}",
            f"FPS: {fps:.1f}",
        ]

        # Semi-transparent dark banner on the left
        banner_w = 220
        banner_h = len(lines) * 14 + 10
        overlay  = img[:banner_h, :banner_w].copy()
        cv2.rectangle(overlay, (0, 0), (banner_w, banner_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, OVERLAY_ALPHA,
                        img[:banner_h, :banner_w], 1 - OVERLAY_ALPHA,
                        0, img[:banner_h, :banner_w])

        for i, line in enumerate(lines):
            color = DEBUG_WARN_COLOR if "INVERTED" in line else DEBUG_FONT_COLOR
            draw_text(img, line, (5, 12 + i * 14), color, DEBUG_FONT_SCALE)

        # ── Progress bar ──────────────────────────────────────────────────────
        bar_y  = CAPTURE_HEIGHT - 6
        bar_x1 = 0
        bar_x2 = int(state.progress * CAPTURE_WIDTH)
        cv2.line(img, (bar_x1, bar_y), (bar_x2, bar_y), (0, 200, 255), 3)

        return img

    def toggle(self) -> None:
        self.enabled = not self.enabled
        from utils import log
        log.info("Debug overlay: %s", "ON" if self.enabled else "OFF")

