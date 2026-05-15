import os
import signal
import sys
import time

import cv2
import neat
import pyautogui

from constants import (
    BEST_GENOME_PATH,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    CHECKPOINT_DIR,
    CHECKPOINT_INTERVAL,
    CHECKPOINT_PREFIX,
    DEATH_WHITE_THRESH,
    FEATURE_VECTOR_SIZE,
    FITNESS_ALIVE_PER_SEC,
    FITNESS_PORTAL_BONUS,
    FITNESS_PROGRESS_SCALE,
    FITNESS_SPAM_PENALTY,
    FRAME_INTERVAL,
    MAX_EVAL_TIME,
    NEAT_CONFIG_PATH,
    TARGET_FPS,
)
from controller import Controller
from debug_overlay import DebugOverlay
from game_detector import GameDetector
from game_state import GameState
from mode_manager import ModeManager
from portal_detector import PortalDetector
from utils import (
    FPSCounter,
    GenerationStats,
    Stopwatch,
    ensure_dirs,
    load_object,
    log,
    request_shutdown,
    save_object,
    shutdown_requested,
)
from vision import VisionSystem

# Restart key used when the player dies in GD (default: 'r' or backspace)
RESTART_KEY = "backspace"
# Delay after pressing restart before we start evaluating (seconds)
RESTART_DELAY = 1.5
# How long progress must be frozen before we declare death (seconds)
FREEZE_TIMEOUT = 2.5

# ─── Signal handler ──────────────────────────────────────────────────────────

def _sigint_handler(sig, frame):
    log.info("SIGINT received; requesting graceful shutdown.")
    request_shutdown()

signal.signal(signal.SIGINT, _sigint_handler)


# ─── Subsystem singletons ─────────────────────────────────────────────────────
detector  = GameDetector()
vision    = VisionSystem()
portal_d  = PortalDetector()
mode_mgr  = ModeManager()
controller= Controller()
overlay   = DebugOverlay(enabled=False)
fps_ctr   = FPSCounter(window=60)
gen_stats = GenerationStats()


# ─── Death detection ──────────────────────────────────────────────────────────

def _detect_death(frame_bgr, state: GameState) -> bool:
    if frame_bgr is None:
        return False

    # Method 1 – white flash
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    white_frac = float((gray > 230).sum()) / gray.size
    if white_frac > DEATH_WHITE_THRESH:
        log.debug("Death: white flash (%.2f)", white_frac)
        return True

    # Method 2 – progress freeze (only after 3 s alive to ignore loading)
    if state.alive_time > 3.0:
        if not hasattr(_detect_death, "_last_progress"):
            _detect_death._last_progress = -1.0
            _detect_death._freeze_start  = time.perf_counter()

        if abs(state.progress - _detect_death._last_progress) > 0.001:
            _detect_death._last_progress = state.progress
            _detect_death._freeze_start  = time.perf_counter()
        else:
            frozen_secs = time.perf_counter() - _detect_death._freeze_start
            if frozen_secs > FREEZE_TIMEOUT:
                log.debug("Death: progress frozen for %.1f s", frozen_secs)
                _detect_death._last_progress = -1.0
                return True

    return False


def _reset_death_tracker():
    if hasattr(_detect_death, "_last_progress"):
        del _detect_death._last_progress


# ─── Level restart ────────────────────────────────────────────────────────────

def _restart_level():
    controller.release_all()
    time.sleep(0.1)
    pyautogui.press(RESTART_KEY)
    time.sleep(RESTART_DELAY)
    _reset_death_tracker()


# ─── Fitness calculation ──────────────────────────────────────────────────────

def compute_fitness(state: GameState) -> float:
    spam   = state.recent_input_rate(window_sec=1.0)
    spam_p = max(0.0, spam - 5) * FITNESS_SPAM_PENALTY   # allow up to 5 presses/s free

    fitness = (
        state.alive_time       * FITNESS_ALIVE_PER_SEC
        + state.progress       * FITNESS_PROGRESS_SCALE
        + state.portals_passed * FITNESS_PORTAL_BONUS
        - spam_p
    )
    return max(0.0, fitness)


# ─── Single-genome evaluation ─────────────────────────────────────────────────

def eval_genome(genome, config, generation: int = 0, genome_idx: int = 0) -> float:
    net   = neat.nn.FeedForwardNetwork.create(genome, config)
    state = GameState()
    state.reset()

    _restart_level()

    timer     = Stopwatch()
    prev_time = time.perf_counter()

    fitness = 0.0

    while True:
        # ── Timing ────────────────────────────────────────────────────────────
        now  = time.perf_counter()
        dt   = now - prev_time
        if dt < FRAME_INTERVAL:
            time.sleep(FRAME_INTERVAL - dt)
        prev_time = time.perf_counter()
        fps_ctr.tick()

        # ── Capture ───────────────────────────────────────────────────────────
        frame = detector.grab()

        # ── Vision pipeline ───────────────────────────────────────────────────
        vision.process(frame, state)
        portal_d.detect(frame, state)
        mode_mgr.update(frame, state)
        state.update_alive_time()

        # ── Death check ────────────────────────────────────────────────────────
        if _detect_death(frame, state):
            state.death_detected = True
            break

        if state.alive_time > MAX_EVAL_TIME:
            log.info("Genome %d reached time limit (%.0f s)", genome_idx, MAX_EVAL_TIME)
            break

        if shutdown_requested():
            break

        # ── NN forward pass ────────────────────────────────────────────────────
        inputs     = state.feature_vector.tolist()
        output     = net.activate(inputs)
        nn_out     = output[0]

        # ── Controller ────────────────────────────────────────────────────────
        try:
            action = controller.act(nn_out, state)
        except pyautogui.FailSafeException:
            log.warning("PyAutoGUI fail-safe triggered; ending genome eval.")
            break

        # ── Fitness accumulation ──────────────────────────────────────────────
        fitness = compute_fitness(state)

        # ── Debug overlay ──────────────────────────────────────────────────────
        if overlay.enabled and frame is not None:
            ann = overlay.render(
                frame, state,
                generation  = generation,
                genome_idx  = genome_idx,
                nn_output   = nn_out,
                action      = action,
                fitness     = fitness,
                fps         = fps_ctr.get(),
            )
            cv2.imshow("GD NEAT Debug", ann)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("d"):
                overlay.toggle()
            elif key == ord("q"):
                request_shutdown()

    controller.release_all()
    return fitness


# ─── Population evaluator ─────────────────────────────────────────────────────

class GenomeEvaluator:

    def __init__(self):
        self.generation  = 0
        self._gen_timer  = Stopwatch()

    def __call__(self, genomes, config):
        self._gen_timer.reset()
        fitnesses = []

        for idx, (genome_id, genome) in enumerate(genomes):
            if shutdown_requested():
                genome.fitness = 0.0
                continue

            log.info(
                "Gen %d | genome %d/%d (id=%d)",
                self.generation, idx + 1, len(genomes), genome_id,
            )

            f = eval_genome(genome, config,
                             generation=self.generation,
                             genome_idx=idx)
            genome.fitness = f
            fitnesses.append(f)
            log.info("  fitness=%.2f  progress=%.1f%%", f, 0.0)

        if fitnesses:
            best    = max(fitnesses)
            worst   = min(fitnesses)
            avg     = sum(fitnesses) / len(fitnesses)
            elapsed = self._gen_timer.elapsed()
            gen_stats.record(
                self.generation, best, avg, worst,
                num_species=0,   # filled in by train()
                elapsed_s=elapsed,
            )

        self.generation += 1


# ─── Main training entry point ────────────────────────────────────────────────

def train():
    ensure_dirs(CHECKPOINT_DIR)

    # ── Attach window ──────────────────────────────────────────────────────────
    found = detector.find_and_attach()
    if not found:
        log.warning(
            "Game window not found.  Make sure Geometry Dash is running."
        )

    # ── Load NEAT config ───────────────────────────────────────────────────────
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        NEAT_CONFIG_PATH,
    )

    # ── Load or create population ──────────────────────────────────────────────
    checkpoint_file = _latest_checkpoint()
    if checkpoint_file:
        log.info("Restoring population from checkpoint: %s", checkpoint_file)
        pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
    else:
        log.info("Starting fresh NEAT population.")
        pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(
        neat.Checkpointer(
            generation_interval=CHECKPOINT_INTERVAL,
            filename_prefix=os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX),
        )
    )

    evaluator = GenomeEvaluator()

    try:
        winner = pop.run(evaluator, n=10_000)
    except KeyboardInterrupt:
        log.info("Training interrupted by user.")
        winner = stats.best_genome()
    finally:
        controller.release_all()
        cv2.destroyAllWindows()

    if winner is not None:
        save_object(winner, BEST_GENOME_PATH)
        log.info("Best genome saved to %s  (fitness=%.2f)",
                 BEST_GENOME_PATH, winner.fitness)


def _latest_checkpoint() -> str | None:
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    files = [
        os.path.join(CHECKPOINT_DIR, f)
        for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith(CHECKPOINT_PREFIX)
    ]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


if __name__ == "__main__":
    train()

