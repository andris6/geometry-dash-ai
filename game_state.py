import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from constants import (
    MODE_CUBE,
    SPEED_NORMAL,
    FEATURE_VECTOR_SIZE,
)

import numpy as np


@dataclass
class ObstacleInfo:
    x: float          # left edge, 0-1 normalised in VISION_WIDTH
    y: float          # top edge,  0-1 normalised in VISION_HEIGHT
    width: float      # 0-1 normalised
    height: float     # 0-1 normalised
    distance: float   # horizontal distance from player, 0-1 normalised


@dataclass
class PortalInfo:
    portal_type: str   # e.g. "ship", "gravity_up", "speed_fast", …
    x: float           # centre x, 0-1 normalised
    y: float           # centre y, 0-1 normalised
    width: float
    height: float
    distance: float    # horizontal distance from player, 0-1 normalised


@dataclass
class PlayerState:
    y_position: float = 0.5        # 0 = top, 1 = bottom (normalised)
    velocity_y: float = 0.0        # signed, normalised
    is_alive: bool    = True


@dataclass
class GameState:
    """
    mode            : active gamemode index (see constants.MODE_*)
    gravity_inverted: True when gravity is flipped
    speed           : active speed index (see constants.SPEED_*)
    is_mini         : True when mini-mode is active
    is_dual         : True when dual-mode is active
    is_mirrored     : True when mirror portal is active
    is_platformer   : True when in platformer section
    player          : primary player state
    player2         : secondary player (dual mode only)
    obstacles       : list of nearby obstacles from vision
    portals         : list of detected portals in the lookahead window
    progress        : level completion fraction [0, 1]
    alive_time      : seconds the current genome has been alive
    death_detected  : True when a death event was detected this tick
    portals_passed  : count of portals the player has passed
    input_history   : circular log of recent (timestamp, action) tuples
    feature_vector  : pre-computed neural-network input (length FEATURE_VECTOR_SIZE)
    frame_bgr       : most recent raw capture (for debug overlay / vision)
    """

    # ── Mode / physics ──────────────────────────────────────────────────────
    mode:             int   = MODE_CUBE
    gravity_inverted: bool  = False
    speed:            int   = SPEED_NORMAL
    is_mini:          bool  = False
    is_dual:          bool  = False
    is_mirrored:      bool  = False
    is_platformer:    bool  = False

    # ── Player entities ──────────────────────────────────────────────────────
    player:  PlayerState = field(default_factory=PlayerState)
    player2: PlayerState = field(default_factory=PlayerState)

    # ── Scene objects ────────────────────────────────────────────────────────
    obstacles: List[ObstacleInfo] = field(default_factory=list)
    portals:   List[PortalInfo]   = field(default_factory=list)

    # ── Progress & fitness bookkeeping ───────────────────────────────────────
    progress:        float = 0.0
    alive_time:      float = 0.0
    death_detected:  bool  = False
    portals_passed:  int   = 0

    # ── Input history (for spam penalty) ────────────────────────────────────
    input_history: List[Tuple[float, str]] = field(default_factory=list)

    # ── Pre-computed NN input ─────────────────────────────────────────────────
    feature_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(FEATURE_VECTOR_SIZE, dtype=np.float32)
    )

    # ── Raw frame (kept for vision / overlay) ────────────────────────────────
    frame_bgr: Optional[np.ndarray] = field(default=None, repr=False)

    # ── Internal timing ──────────────────────────────────────────────────────
    _start_time: float = field(default_factory=time.perf_counter, init=False,
                               repr=False)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.mode             = MODE_CUBE
        self.gravity_inverted = False
        self.speed            = SPEED_NORMAL
        self.is_mini          = False
        self.is_dual          = False
        self.is_mirrored      = False
        self.is_platformer    = False

        self.player   = PlayerState()
        self.player2  = PlayerState()

        self.obstacles    = []
        self.portals      = []
        self.progress     = 0.0
        self.alive_time   = 0.0
        self.death_detected = False
        self.portals_passed = 0

        self.input_history  = []
        self.feature_vector = np.zeros(FEATURE_VECTOR_SIZE, dtype=np.float32)
        self.frame_bgr      = None
        self._start_time    = time.perf_counter()

    def update_alive_time(self) -> None:
        self.alive_time = time.perf_counter() - self._start_time

    def record_input(self, action: str) -> None:
        self.input_history.append((time.perf_counter(), action))
        # Keep only last 200 events to bound memory
        if len(self.input_history) > 200:
            self.input_history = self.input_history[-200:]

    def recent_input_rate(self, window_sec: float = 1.0) -> float:
        now = time.perf_counter()
        cutoff = now - window_sec
        recent = [t for t, _ in self.input_history if t >= cutoff]
        return len(recent)

    def nearest_obstacle_distance(self) -> float:
        if not self.obstacles:
            return 1.0
        return min(o.distance for o in self.obstacles)

    def nearest_portal(self) -> Optional[PortalInfo]:
        if not self.portals:
            return None
        return min(self.portals, key=lambda p: p.distance)

