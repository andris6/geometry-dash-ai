import cv2
import numpy as np

from constants import (
    MODE_BALL,
    MODE_CUBE,
    MODE_JETPACK,
    MODE_NAMES,
    MODE_PLATFORMER,
    MODE_ROBOT,
    MODE_SHIP,
    MODE_SPIDER,
    MODE_SWING,
    MODE_UFO,
    MODE_WAVE,
    PLAYER_X_FRACTION,
    SPEED_PORTAL_TO_INDEX,
    VISION_HEIGHT,
    VISION_WIDTH,
)
from game_state import GameState
from portal_detector import (
    FORM_PORTALS,
    SPEED_PORTALS,
    SPEED_PORTAL_TO_INDEX,
)
from utils import log, normalize, resize_frame, to_grayscale


_FORM_PORTAL_TO_MODE = {
    "cube":   MODE_CUBE,
    "ship":   MODE_SHIP,
    "ball":   MODE_BALL,
    "ufo":    MODE_UFO,
    "wave":   MODE_WAVE,
    "robot":  MODE_ROBOT,
    "spider": MODE_SPIDER,
    "swing":  MODE_SWING,
    "jetpack":MODE_JETPACK,
}

# Distance threshold (normalised) at which we treat a portal as "entered"
_PORTAL_ENTER_DIST = 0.08


class ModeManager:
    def __init__(self) -> None:
        self._prev_portals_by_type: dict = {}

    def update(self, frame_bgr: np.ndarray, state: GameState) -> None:
        self._apply_portal_transitions(state)
        if frame_bgr is not None:
            self._icon_shape_heuristic(frame_bgr, state)

    # ── Portal transition logic ───────────────────────────────────────────────

    def _apply_portal_transitions(self, state: GameState) -> None:
        entered_types = set()

        for portal in state.portals:
            if portal.distance > _PORTAL_ENTER_DIST:
                break   # sorted by distance; no need to look further

            ptype = portal.portal_type

            # Avoid applying the same portal type twice in one batch
            if ptype in entered_types:
                continue
            entered_types.add(ptype)

            # ── Form portals ─────────────────────────────────────────────────
            if ptype in _FORM_PORTAL_TO_MODE:
                new_mode = _FORM_PORTAL_TO_MODE[ptype]
                if state.mode != new_mode:
                    log.debug(
                        "Mode transition: %s → %s",
                        MODE_NAMES.get(state.mode, "?"),
                        MODE_NAMES.get(new_mode,   "?"),
                    )
                    state.mode = new_mode

            # ── Gravity portals ───────────────────────────────────────────────
            elif ptype == "gravity_up":
                state.gravity_inverted = False
            elif ptype == "gravity_dn":
                state.gravity_inverted = True

            # ── Size portals ──────────────────────────────────────────────────
            elif ptype == "mini":
                state.is_mini = True
            elif ptype == "normal_sz":
                state.is_mini = False

            # ── Speed portals ─────────────────────────────────────────────────
            elif ptype in SPEED_PORTALS:
                state.speed = SPEED_PORTAL_TO_INDEX[ptype]

            # ── Mirror portal ─────────────────────────────────────────────────
            elif ptype == "mirror":
                state.is_mirrored = not state.is_mirrored

            # ── Dual portal ───────────────────────────────────────────────────
            elif ptype == "dual":
                state.is_dual = not state.is_dual

    # ── Icon-shape heuristic ──────────────────────────────────────────────────

    def _icon_shape_heuristic(
        self, frame_bgr: np.ndarray, state: GameState
    ) -> None:
        small = resize_frame(frame_bgr, VISION_WIDTH, VISION_HEIGHT)
        gray  = to_grayscale(small)

        px = int(PLAYER_X_FRACTION * VISION_WIDTH)
        py = int(state.player.y_position * VISION_HEIGHT)

        # 20×20 region around player centre (clamped to image)
        r = 10
        x0, y0 = max(0, px - r), max(0, py - r)
        x1, y1 = min(VISION_WIDTH, px + r), min(VISION_HEIGHT, py + r)
        crop = gray[y0:y1, x0:x1]

        if crop.size == 0:
            return

        _, bin_icon = cv2.threshold(crop, 80, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(
            bin_icon, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            return

        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 20:
            return

        _x, _y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            return

        aspect   = w / h
        hull     = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        inferred = _infer_mode_from_shape(aspect, solidity)
        if inferred is not None and inferred != state.mode:
            # Only override if the portal list doesn't contradict this
            portal_modes = {
                _FORM_PORTAL_TO_MODE[p.portal_type]
                for p in state.portals
                if p.portal_type in _FORM_PORTAL_TO_MODE
            }
            if not portal_modes:
                # No form portals visible; trust the shape heuristic
                log.debug(
                    "Icon heuristic: mode %s → %s (aspect=%.2f, solidity=%.2f)",
                    MODE_NAMES.get(state.mode, "?"),
                    MODE_NAMES.get(inferred,   "?"),
                    aspect, solidity,
                )
                state.mode = inferred


def _infer_mode_from_shape(aspect: float, solidity: float) -> int | None:
    if 0.8 <= aspect <= 1.2 and solidity > 0.80:
        return MODE_CUBE
    if 0.8 <= aspect <= 1.2 and 0.70 <= solidity <= 0.85:
        return MODE_BALL   # circle ≈ cube aspect but slightly lower solidity
    if aspect < 0.6 and solidity > 0.55:
        return MODE_SHIP   # tall narrow shape
    if solidity < 0.45:
        return MODE_WAVE   # jagged / diagonal shape
    if aspect > 1.4 and solidity > 0.55:
        return MODE_UFO    # wide, medium solidity
    return None            # ambiguous

