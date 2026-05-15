# ─── Window / Capture ───────────────────────────────────────────────────────
GAME_WINDOW_TITLE   = "Geometry Dash"
CAPTURE_WIDTH       = 800
CAPTURE_HEIGHT      = 600
CAPTURE_OFFSET_X    = 0     # tweak if window has borders
CAPTURE_OFFSET_Y    = 0
TARGET_FPS          = 30
FRAME_INTERVAL      = 1.0 / TARGET_FPS   # seconds between captures

# ─── Vision pipeline ────────────────────────────────────────────────────────
VISION_WIDTH        = 200   # network input grid width
VISION_HEIGHT       = 100   # network input grid height
BINARY_THRESHOLD    = 80    # grayscale threshold for obstacle detection
OBSTACLE_MIN_AREA   = 50    # minimum contour area (pixels²) to count

# Lookahead columns scanned for obstacles (from player_x forward)
LOOKAHEAD_COLUMNS   = 5     # number of vertical scan slices
LOOKAHEAD_SPACING   = 8     # pixels between each slice (in VISION_WIDTH space)

# Player approximate position in the vision frame (0-1 normalized)
PLAYER_X_FRACTION   = 0.25
PLAYER_Y_FRACTION   = 0.50

# ─── Feature vector ─────────────────────────────────────────────────────────
# Neural-network input features (keep this in sync with vision.py extract_features)
NUM_LOOKAHEAD_SLICES    = LOOKAHEAD_COLUMNS
FEATURE_OBSTACLE_DISTS  = NUM_LOOKAHEAD_SLICES   # nearest obstacle distance per column
FEATURE_OBSTACLE_HTS    = NUM_LOOKAHEAD_SLICES   # height of nearest obstacle per column
FEATURE_MISC            = 8                       # mode, gravity, speed, y_pos, mini, dual, mirror, progress
FEATURE_VECTOR_SIZE     = FEATURE_OBSTACLE_DISTS + FEATURE_OBSTACLE_HTS + FEATURE_MISC

# ─── Game modes ─────────────────────────────────────────────────────────────
MODE_CUBE       = 0
MODE_SHIP       = 1
MODE_BALL       = 2
MODE_UFO        = 3
MODE_WAVE       = 4
MODE_ROBOT      = 5
MODE_SPIDER     = 6
MODE_SWING      = 7
MODE_JETPACK    = 8
MODE_PLATFORMER = 9

MODE_NAMES = {
    MODE_CUBE:       "Cube",
    MODE_SHIP:       "Ship",
    MODE_BALL:       "Ball",
    MODE_UFO:        "UFO",
    MODE_WAVE:       "Wave",
    MODE_ROBOT:      "Robot",
    MODE_SPIDER:     "Spider",
    MODE_SWING:      "Swing",
    MODE_JETPACK:    "Jetpack",
    MODE_PLATFORMER: "Platformer",
}

# ─── Speed multipliers ───────────────────────────────────────────────────────
SPEED_SLOW        = 0
SPEED_NORMAL      = 1
SPEED_FAST        = 2
SPEED_VERY_FAST   = 3
SPEED_EXTREME     = 4

SPEED_MULTIPLIERS = {
    SPEED_SLOW:      0.5,
    SPEED_NORMAL:    1.0,
    SPEED_FAST:      1.5,
    SPEED_VERY_FAST: 2.0,
    SPEED_EXTREME:   3.0,
}

SPEED_NAMES = {
    SPEED_SLOW:      "Slow",
    SPEED_NORMAL:    "Normal",
    SPEED_FAST:      "Fast",
    SPEED_VERY_FAST: "Very Fast",
    SPEED_EXTREME:   "Extreme",
}

# ─── Portal color hints (BGR) ────────────────────────────────────────────────
# These are approximate hue ranges used for HSV-based portal detection.
# Tuned for default GD skin / bright portal sprites.
PORTAL_COLOR_RANGES = {
    "cube":       ([20,  100, 100], [35,  255, 255]),   # orange-ish
    "ship":       ([100, 100, 100], [130, 255, 255]),   # blue
    "ball":       ([140, 100, 100], [160, 255, 255]),   # purple
    "ufo":        ([75,  100, 100], [95,  255, 255]),   # cyan-green
    "wave":       ([55,  100, 100], [75,  255, 255]),   # green
    "robot":      ([0,   100, 100], [15,  255, 255]),   # red-orange
    "spider":     ([150, 80,  80],  [175, 255, 255]),   # magenta
    "swing":      ([160, 80,  80],  [180, 255, 255]),   # pink
    "gravity_up": ([85,  60,  60],  [105, 255, 255]),   # light blue
    "gravity_dn": ([25,  60,  60],  [45,  255, 255]),   # yellow
    "mini":       ([50,  60,  60],  [70,  255, 255]),   # lime
    "normal_sz":  ([110, 60,  60],  [130, 255, 255]),   # cornflower blue
    "mirror":     ([10,  60,  60],  [25,  255, 255]),   # amber
    "dual":       ([165, 60,  60],  [180, 255, 255]),   # hot pink
    "speed_slow": ([0,   0,   180], [180, 30,  255]),   # near-white
    "speed_norm": ([60,  60,  60],  [80,  255, 255]),   # yellow-green
    "speed_fast": ([30,  100, 100], [50,  255, 255]),   # orange-yellow
    "speed_vf":   ([5,   150, 150], [20,  255, 255]),   # deep orange
    "speed_ex":   ([0,   200, 200], [10,  255, 255]),   # red
}

# Portal minimum bounding-box area to avoid noise
PORTAL_MIN_AREA = 400

# ─── Death / restart detection ───────────────────────────────────────────────
# Region (x, y, w, h) in the FULL capture where we look for the death flash.
DEATH_REGION       = (0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT)
DEATH_WHITE_THRESH = 0.55   # fraction of pixels that must be near-white

# Pixel color for "progress bar" sampling to detect freezes
PROGRESS_BAR_Y_FRAC = 0.97   # bottom of screen
PROGRESS_SAMPLE_X   = [0.1, 0.3, 0.5, 0.7, 0.9]

# ─── Input ───────────────────────────────────────────────────────────────────
JUMP_KEY   = "space"
ALT_KEY    = "up"       # alternative / secondary press
LEFT_KEY   = "left"     # platformer
RIGHT_KEY  = "right"    # platformer

# Max hold duration (seconds) for robot variable jump
ROBOT_MAX_HOLD = 0.5

# Minimum tap duration (ms)
MIN_TAP_MS = 30

# ─── NEAT ────────────────────────────────────────────────────────────────────
NEAT_CONFIG_PATH    = "config-feedforward.txt"
CHECKPOINT_DIR      = "checkpoints"
CHECKPOINT_PREFIX   = "neat-checkpoint-"
CHECKPOINT_INTERVAL = 5        # save every N generations
BEST_GENOME_PATH    = "best_genome.pkl"

# Fitness weights
FITNESS_ALIVE_PER_SEC    = 1.0
FITNESS_PROGRESS_SCALE   = 500.0   # bonus per 0..1 progress fraction
FITNESS_PORTAL_BONUS     = 5.0     # reward for each portal passed
FITNESS_SPAM_PENALTY     = 0.05    # subtracted per excessive input
FITNESS_DEATH_PENALTY    = 0.0     # added at death (can be negative to punish early death)

# Max time per genome evaluation (seconds)
MAX_EVAL_TIME = 60.0

# ─── Debug overlay ───────────────────────────────────────────────────────────
DEBUG_FONT_SCALE = 0.45
DEBUG_FONT_COLOR = (0, 255, 0)      # green
DEBUG_WARN_COLOR = (0, 100, 255)    # orange
DEBUG_BOX_COLOR  = (255, 0, 0)      # red for obstacles
DEBUG_PORTAL_COLOR = (0, 255, 255)  # cyan for portals
OVERLAY_ALPHA    = 0.6

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_FILE    = "gd_neat.log"
LOG_LEVEL   = "INFO"    # DEBUG | INFO | WARNING | ERROR

