from typing import List

import cv2
import numpy as np

from constants import (
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    PLAYER_X_FRACTION,
    PORTAL_COLOR_RANGES,
    PORTAL_MIN_AREA,
    VISION_HEIGHT,
    VISION_WIDTH,
)
from game_state import GameState, PortalInfo
from utils import log, normalize, resize_frame


# Portal types that change the active gamemode
FORM_PORTALS = {"cube", "ship", "ball", "ufo", "wave", "robot", "spider", "swing"}

# Portal types that change speed
SPEED_PORTALS = {"speed_slow", "speed_norm", "speed_fast", "speed_vf", "speed_ex"}

SPEED_PORTAL_TO_INDEX = {
    "speed_slow": 0,
    "speed_norm": 1,
    "speed_fast": 2,
    "speed_vf":   3,
    "speed_ex":   4,
}


class PortalDetector:

    def __init__(self) -> None:
        # Pre-compile HSV ranges into numpy arrays for speed
        self._ranges = {
            ptype: (
                np.array(lo, dtype=np.uint8),
                np.array(hi, dtype=np.uint8),
            )
            for ptype, (lo, hi) in PORTAL_COLOR_RANGES.items()
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray, state: GameState) -> None:
        if frame_bgr is None:
            state.portals = []
            return

        # Work at vision resolution for speed
        small = resize_frame(frame_bgr, VISION_WIDTH, VISION_HEIGHT)
        hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        player_px  = int(PLAYER_X_FRACTION * VISION_WIDTH)
        portals: List[PortalInfo] = []

        for ptype, (lo, hi) in self._ranges.items():
            mask = cv2.inRange(hsv, lo, hi)
            # Small morphological close to join nearby blobs
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < PORTAL_MIN_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                # Only include portals ahead of or very close to the player
                if cx < player_px - 5:
                    continue
                distance = normalize(cx - player_px, 0, VISION_WIDTH)
                portals.append(PortalInfo(
                    portal_type = ptype,
                    x           = normalize(cx,   0, VISION_WIDTH),
                    y           = normalize(y + h // 2, 0, VISION_HEIGHT),
                    width       = normalize(w,    0, VISION_WIDTH),
                    height      = normalize(h,    0, VISION_HEIGHT),
                    distance    = distance,
                ))

        # Sort by distance so index 0 is the nearest portal
        portals.sort(key=lambda p: p.distance)

        # Count portals that were just crossed (distance crossed 0 since last tick)
        prev_nearest = state.portals[0].distance if state.portals else 1.0
        new_nearest  = portals[0].distance       if portals      else 1.0
        if prev_nearest < 0.05 and new_nearest > 0.05:
            # The nearest portal flipped from "right in front" to "behind"
            state.portals_passed += 1

        state.portals = portals

    # ── Utility ──────────────────────────────────────────────────────────────

    @staticmethod
    def get_form_portals(state: GameState) -> List[PortalInfo]:
        return [p for p in state.portals if p.portal_type in FORM_PORTALS]

    @staticmethod
    def get_speed_portals(state: GameState) -> List[PortalInfo]:
        return [p for p in state.portals if p.portal_type in SPEED_PORTALS]

    @staticmethod
    def is_gravity_portal_ahead(state: GameState) -> bool:
        return any(
            p.portal_type in ("gravity_up", "gravity_dn") and p.distance < 0.5
            for p in state.portals
        )

