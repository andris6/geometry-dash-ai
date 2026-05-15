import time
from typing import Optional

import pyautogui

from constants import (
    ALT_KEY,
    JUMP_KEY,
    LEFT_KEY,
    MIN_TAP_MS,
    MODE_BALL,
    MODE_CUBE,
    MODE_JETPACK,
    MODE_PLATFORMER,
    MODE_ROBOT,
    MODE_SHIP,
    MODE_SPIDER,
    MODE_SWING,
    MODE_UFO,
    MODE_WAVE,
    ROBOT_MAX_HOLD,
)
from game_state import GameState
from utils import log

# Disable pyautogui's own pause between calls (we control timing ourselves)
pyautogui.PAUSE = 0.0


class Controller:
    """
    Translates a single float NN output into gamemode-correct key events.

    Usage::

        ctrl = Controller()
        ctrl.act(nn_output, state)    # every tick
        ctrl.release_all()            # on genome end
    """

    def __init__(self) -> None:
        self._holding:       bool  = False   # True when JUMP_KEY is held down
        self._hold_start:    float = 0.0     # time.perf_counter() when hold began
        self._last_action:   str   = "none"

    # ── Public API ────────────────────────────────────────────────────────────

    def act(self, nn_output: float, state: GameState) -> str:
        """
        Decide and execute an action based on *nn_output* and *state.mode*.

        Returns a short string label for debug overlay / logging.
        """
        mode   = state.mode
        intent = _output_to_intent(nn_output)

        # Mirror mode: flip left/right conceptually (we only flip key logic
        # for platformer; auto-scrolling modes are not affected)
        if state.is_mirrored and mode == MODE_PLATFORMER:
            if intent == "left":  intent = "right"
            elif intent == "right": intent = "left"

        action = self._dispatch(mode, intent, state)
        state.record_input(action)
        self._last_action = action
        return action

    def release_all(self) -> None:
        """Release any held keys.  Call at end of genome evaluation."""
        if self._holding:
            try:
                pyautogui.keyUp(JUMP_KEY)
            except Exception:
                pass
            self._holding = False
        self._last_action = "release_all"

    @property
    def last_action(self) -> str:
        return self._last_action

    # ── Dispatch per mode ────────────────────────────────────────────────────

    def _dispatch(self, mode: int, intent: str, state: GameState) -> str:
        if mode == MODE_CUBE:
            return self._cube(intent)
        elif mode == MODE_SHIP:
            return self._ship(intent)
        elif mode == MODE_BALL:
            return self._ball(intent)
        elif mode == MODE_UFO:
            return self._ufo(intent)
        elif mode == MODE_WAVE:
            return self._wave(intent)
        elif mode == MODE_ROBOT:
            return self._robot(intent, state)
        elif mode == MODE_SPIDER:
            return self._spider(intent)
        elif mode == MODE_SWING:
            return self._swing(intent)
        elif mode == MODE_JETPACK:
            return self._jetpack(intent)
        elif mode == MODE_PLATFORMER:
            return self._platformer(intent)
        else:
            return self._cube(intent)   # safe fallback

    # ── Mode-specific action logic ────────────────────────────────────────────

    def _cube(self, intent: str) -> str:
        """Tap to jump; holding does nothing extra in vanilla cube."""
        if intent == "press":
            self._tap()
            return "jump"
        elif intent == "hold" and not self._holding:
            self._tap()
            return "jump"
        return "none"

    def _ship(self, intent: str) -> str:
        """Hold = fly up, release = fall."""
        if intent in ("press", "hold"):
            if not self._holding:
                self._key_down()
            return "fly_up"
        else:  # release / none
            if self._holding:
                self._key_up()
            return "fly_down"

    def _ball(self, intent: str) -> str:
        """Tap to invert gravity."""
        if intent == "press":
            self._tap()
            return "gravity_flip"
        return "none"

    def _ufo(self, intent: str) -> str:
        """Each tap adds upward impulse."""
        if intent == "press":
            self._tap()
            return "ufo_impulse"
        return "none"

    def _wave(self, intent: str) -> str:
        """Hold = diagonal up, release = diagonal down."""
        if intent in ("press", "hold"):
            if not self._holding:
                self._key_down()
            return "wave_up"
        else:
            if self._holding:
                self._key_up()
            return "wave_down"

    def _robot(self, intent: str, state: GameState) -> str:
        """
        Variable hold: the NN output magnitude controls jump height.
        We hold for a scaled duration proportional to |nn_output|, capped
        at ROBOT_MAX_HOLD.
        """
        if intent in ("press", "hold") and not self._holding:
            # Begin hold
            self._key_down()
            return "robot_jump_start"
        elif intent not in ("press", "hold") and self._holding:
            hold_dur = time.perf_counter() - self._hold_start
            if hold_dur >= ROBOT_MAX_HOLD:
                self._key_up()
                return "robot_jump_release"
        # Check timeout
        if self._holding:
            if time.perf_counter() - self._hold_start >= ROBOT_MAX_HOLD:
                self._key_up()
                return "robot_jump_max"
        return "none"

    def _spider(self, intent: str) -> str:
        """Instant gravity swap on press."""
        if intent == "press":
            self._tap()
            return "spider_swap"
        return "none"

    def _swing(self, intent: str) -> str:
        """Toggle gravity on each press (while airborne GD handles this)."""
        if intent == "press":
            self._tap()
            return "swing_toggle"
        return "none"

    def _jetpack(self, intent: str) -> str:
        """Hold = thrust up."""
        if intent in ("press", "hold"):
            if not self._holding:
                self._key_down()
            return "jet_thrust"
        else:
            if self._holding:
                self._key_up()
            return "jet_off"

    def _platformer(self, intent: str) -> str:
        """
        Platformer mode: jump on press.
        (Full left/right movement can be added by extending NN outputs.)
        """
        if intent == "press":
            self._tap()
            return "plat_jump"
        return "none"

    # ── Low-level key helpers ─────────────────────────────────────────────────

    def _tap(self) -> None:
        """Very short key press (single frame tap)."""
        try:
            pyautogui.keyDown(JUMP_KEY)
            time.sleep(MIN_TAP_MS / 1000.0)
            pyautogui.keyUp(JUMP_KEY)
        except Exception as exc:
            log.warning("Tap failed: %s", exc)
        self._holding = False

    def _key_down(self) -> None:
        if not self._holding:
            try:
                pyautogui.keyDown(JUMP_KEY)
            except Exception as exc:
                log.warning("keyDown failed: %s", exc)
            self._holding    = True
            self._hold_start = time.perf_counter()

    def _key_up(self) -> None:
        if self._holding:
            try:
                pyautogui.keyUp(JUMP_KEY)
            except Exception as exc:
                log.warning("keyUp failed: %s", exc)
            self._holding = False


# ── Intent helper ─────────────────────────────────────────────────────────────

def _output_to_intent(nn_output: float) -> str:
    """
    Map a scalar NN output (tanh, range -1..1) to a discrete intent.

        > 0.5  → "press"  (trigger action)
       -0.5..0.5 → "hold" if already triggered, else "none"
        < -0.5 → "release"

    We use hysteresis-free mapping here; the per-mode handlers implement
    their own stateful logic.
    """
    if nn_output > 0.5:
        return "press"
    elif nn_output < -0.5:
        return "release"
    else:
        return "hold"

