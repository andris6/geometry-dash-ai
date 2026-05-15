import argparse
import sys
import time

import cv2
import neat
import pyautogui

from constants import (
    BEST_GENOME_PATH,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    FRAME_INTERVAL,
    NEAT_CONFIG_PATH,
    MAX_EVAL_TIME,
)
from controller import Controller
from debug_overlay import DebugOverlay
from game_detector import GameDetector
from game_state import GameState
from mode_manager import ModeManager
from portal_detector import PortalDetector
from train import (
    RESTART_DELAY,
    RESTART_KEY,
    _detect_death,
    _reset_death_tracker,
    compute_fitness,
)
from utils import FPSCounter, log, load_object, request_shutdown, shutdown_requested
from vision import VisionSystem


def replay(genome_path: str, debug: bool, auto_restart: bool) -> None:

    # ── Load genome + config ───────────────────────────────────────────────
    genome = load_object(genome_path)
    if genome is None:
        log.error("Could not load genome from %s", genome_path)
        sys.exit(1)
    log.info("Loaded genome (fitness=%.2f) from %s", genome.fitness, genome_path)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        NEAT_CONFIG_PATH,
    )
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # ── Subsystems ────────────────────────────────────────────────────────
    detector   = GameDetector()
    found      = detector.find_and_attach()
    if not found:
        log.warning("GD window not found; replaying against whatever is on screen.")

    vision_sys = VisionSystem()
    portal_d   = PortalDetector()
    mode_mgr   = ModeManager()
    ctrl       = Controller()
    ovl        = DebugOverlay(enabled=debug)
    fps_ctr    = FPSCounter(60)

    run = 0
    while not shutdown_requested():
        run += 1
        log.info("Replay run #%d", run)

        state = GameState()
        state.reset()
        _reset_death_tracker()

        # Restart the level
        ctrl.release_all()
        time.sleep(0.05)
        pyautogui.press(RESTART_KEY)
        time.sleep(RESTART_DELAY)

        prev_t = time.perf_counter()

        while not shutdown_requested():
            now = time.perf_counter()
            dt  = now - prev_t
            if dt < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - dt)
            prev_t = time.perf_counter()
            fps_ctr.tick()

            frame = detector.grab()
            vision_sys.process(frame, state)
            portal_d.detect(frame, state)
            mode_mgr.update(frame, state)
            state.update_alive_time()

            if _detect_death(frame, state):
                log.info("Death detected at %.1f s, progress=%.1f%%",
                         state.alive_time, state.progress * 100)
                break

            if state.alive_time > MAX_EVAL_TIME:
                log.info("Time limit reached.")
                break

            inputs = state.feature_vector.tolist()
            output = net.activate(inputs)
            nn_out = output[0]

            try:
                action = ctrl.act(nn_out, state)
            except pyautogui.FailSafeException:
                log.warning("PyAutoGUI fail-safe triggered.")
                break

            fitness = compute_fitness(state)

            if ovl.enabled and frame is not None:
                ann = ovl.render(
                    frame, state,
                    generation=run,
                    genome_idx=0,
                    nn_output=nn_out,
                    action=action,
                    fitness=fitness,
                    fps=fps_ctr.get(),
                )
                cv2.imshow("GD Replay", ann)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):   # q or ESC
                    request_shutdown()
                elif key == ord("d"):
                    ovl.toggle()

        ctrl.release_all()

        if not auto_restart or shutdown_requested():
            break

    cv2.destroyAllWindows()
    log.info("Replay finished after %d run(s).", run)


def main():
    parser = argparse.ArgumentParser(description="Replay the best NEAT genome.")
    parser.add_argument("--debug",      action="store_true",
                        help="Enable debug overlay.")
    parser.add_argument("--genome",     default=BEST_GENOME_PATH,
                        help="Path to genome pickle file.")
    parser.add_argument("--no-restart", action="store_true",
                        help="Stop after first death instead of looping.")
    args = parser.parse_args()

    replay(
        genome_path  = args.genome,
        debug        = args.debug,
        auto_restart = not args.no_restart,
    )


if __name__ == "__main__":
    main()

