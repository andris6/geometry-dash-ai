import argparse
import sys

from utils import ensure_dirs, log, setup_logging
from constants import CHECKPOINT_DIR


def _print_info():
    import platform
    log.info("Python   : %s", sys.version.split()[0])
    log.info("Platform : %s", platform.platform())
    try:
        import cv2
        log.info("OpenCV   : %s", cv2.__version__)
    except ImportError:
        log.warning("OpenCV not installed.")
    try:
        import neat
        log.info("neat-python: available")
    except ImportError:
        log.warning("neat-python not installed.")
    try:
        import mss
        log.info("mss      : available")
    except ImportError:
        log.warning("mss not installed.")
    try:
        import pyautogui
        log.info("pyautogui: %s", pyautogui.__version__)
    except ImportError:
        log.warning("pyautogui not installed.")
    try:
        import numpy as np
        log.info("numpy    : %s", np.__version__)
    except ImportError:
        log.warning("numpy not installed.")


def main():
    parser = argparse.ArgumentParser(
        prog="gd_neat",
        description="Geometry Dash NEAT AI – train and replay",
    )
    sub = parser.add_subparsers(dest="command")

    # train subcommand
    sub.add_parser("train", help="Start or resume NEAT training.")

    # replay subcommand
    rep = sub.add_parser("replay", help="Replay the best saved genome.")
    rep.add_argument("--debug",      action="store_true")
    rep.add_argument("--genome",     default=None)
    rep.add_argument("--no-restart", action="store_true")

    # info subcommand
    sub.add_parser("info", help="Print system info and exit.")

    args = parser.parse_args()

    # Bootstrap
    setup_logging()
    ensure_dirs(CHECKPOINT_DIR)

    if args.command == "train" or args.command is None:
        log.info("Starting training mode.")
        from train import train
        train()

    elif args.command == "replay":
        from constants import BEST_GENOME_PATH
        genome_path = args.genome or BEST_GENOME_PATH
        log.info("Starting replay mode (genome=%s).", genome_path)
        from replay_best import replay
        replay(
            genome_path  = genome_path,
            debug        = args.debug,
            auto_restart = not args.no_restart,
        )

    elif args.command == "info":
        _print_info()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

