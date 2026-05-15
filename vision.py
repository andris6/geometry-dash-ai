from typing import List, Tuple

import cv2
import numpy as np

from constants import (
    BINARY_THRESHOLD,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    FEATURE_MISC,
    FEATURE_OBSTACLE_DISTS,
    FEATURE_OBSTACLE_HTS,
    FEATURE_VECTOR_SIZE,
    LOOKAHEAD_COLUMNS,
    LOOKAHEAD_SPACING,
    OBSTACLE_MIN_AREA,
    PLAYER_X_FRACTION,
    PLAYER_Y_FRACTION,
    SPEED_MULTIPLIERS,
    VISION_HEIGHT,
    VISION_WIDTH,
)
from game_state import GameState, ObstacleInfo
from utils import log, normalize, resize_frame, to_grayscale


# ─── VisionSystem ────────────────────────────────────────────────────────────

class VisionSystem:
    def process(self, frame_bgr: np.ndarray, state: GameState) -> None:
        if frame_bgr is None:
            return

        state.frame_bgr = frame_bgr

        # ── 1. Resize to vision resolution ──────────────────────────────────
        small = resize_frame(frame_bgr, VISION_WIDTH, VISION_HEIGHT)

        # ── 2. Grayscale ─────────────────────────────────────────────────────
        gray = to_grayscale(small)

        # ── 3. Binary threshold ───────────────────────────────────────────────
        # We use THRESH_BINARY_INV so dark obstacles on a light or
        # medium background appear as white blobs.
        _, binary = cv2.threshold(
            gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV
        )

        # ── 4. Contour detection ─────────────────────────────────────────────
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ── 5. Build obstacle list ────────────────────────────────────────────
        player_px = int(PLAYER_X_FRACTION * VISION_WIDTH)
        obstacles = _contours_to_obstacles(contours, player_px)
        state.obstacles = obstacles

        # ── 6. Track player Y ─────────────────────────────────────────────────
        state.player.y_position = _estimate_player_y(gray, state)

        # ── 7. Estimate progress ──────────────────────────────────────────────
        state.progress = _estimate_progress(small)

        # ── 8. Build feature vector ───────────────────────────────────────────
        state.feature_vector = _build_feature_vector(state)

    def get_binary_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        small  = resize_frame(frame_bgr, VISION_WIDTH, VISION_HEIGHT)
        gray   = to_grayscale(small)
        _, bin_mask = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        return bin_mask


# ─── Internal helpers ────────────────────────────────────────────────────────

def _contours_to_obstacles(
    contours: list,
    player_px: int,
) -> List[ObstacleInfo]:

    obstacles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < OBSTACLE_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        right_edge = x + w
        if right_edge <= player_px:
            continue  # already behind the player
        distance = normalize(x - player_px, 0, VISION_WIDTH)
        obstacles.append(ObstacleInfo(
            x        = normalize(x,  0, VISION_WIDTH),
            y        = normalize(y,  0, VISION_HEIGHT),
            width    = normalize(w,  0, VISION_WIDTH),
            height   = normalize(h,  0, VISION_HEIGHT),
            distance = max(0.0, distance),
        ))
    # Sort by distance so index 0 is always the closest obstacle
    obstacles.sort(key=lambda o: o.distance)
    return obstacles


def _build_lookahead_slices(
    obstacles: List[ObstacleInfo],
) -> Tuple[np.ndarray, np.ndarray]:

    distances = np.ones(LOOKAHEAD_COLUMNS, dtype=np.float32)
    heights   = np.zeros(LOOKAHEAD_COLUMNS, dtype=np.float32)

    slice_width = LOOKAHEAD_SPACING / VISION_WIDTH  # normalised

    for i in range(LOOKAHEAD_COLUMNS):
        lo = i       * slice_width
        hi = (i + 1) * slice_width
        # Gather obstacles whose left edge falls in this slice
        in_slice = [o for o in obstacles if lo <= o.x < hi]
        if in_slice:
            nearest = min(in_slice, key=lambda o: o.distance)
            distances[i] = nearest.distance
            heights[i]   = nearest.height

    return distances, heights


def _estimate_player_y(gray: np.ndarray, state: GameState) -> float:
    px_col = int(PLAYER_X_FRACTION * VISION_WIDTH)
    search_half = 10   # pixels either side of player column to inspect
    col_lo = max(0, px_col - search_half)
    col_hi = min(VISION_WIDTH, px_col + search_half)
    strip = gray[:, col_lo:col_hi]

    # The player icon is typically brighter than the dark background
    col_mean = strip.mean(axis=1)   # shape (VISION_HEIGHT,)
    if col_mean.max() < 10:
        return PLAYER_Y_FRACTION   # nothing found

    best_row = int(np.argmax(col_mean))
    return normalize(best_row, 0, VISION_HEIGHT)


def _estimate_progress(small_bgr: np.ndarray) -> float:
    bottom_row = small_bgr[-3:, :, :]   # last 3 rows
    brightness = bottom_row.mean(axis=2).mean(axis=0)   # (VISION_WIDTH,)
    bright_px  = int(np.sum(brightness > 180))
    return normalize(bright_px, 0, VISION_WIDTH)


def _build_feature_vector(state: GameState) -> np.ndarray:
    vec = np.zeros(FEATURE_VECTOR_SIZE, dtype=np.float32)
    n   = FEATURE_OBSTACLE_DISTS   # = LOOKAHEAD_COLUMNS

    dists, heights = _build_lookahead_slices(state.obstacles)
    vec[:n]       = dists
    vec[n:n + n]  = heights

    base = n + n   # start of misc section
    vec[base + 0] = normalize(state.mode,   0, 9)
    vec[base + 1] = 1.0 if state.gravity_inverted else 0.0
    vec[base + 2] = normalize(state.speed,  0, 4)
    vec[base + 3] = state.player.y_position
    vec[base + 4] = 1.0 if state.is_mini    else 0.0
    vec[base + 5] = 1.0 if state.is_dual    else 0.0
    vec[base + 6] = 1.0 if state.is_mirrored else 0.0
    vec[base + 7] = state.progress

    return vec

