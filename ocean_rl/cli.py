"""Public command dispatcher; system runners retain their own arguments."""
import argparse


def main(argv=None):
    """Dispatch a selected system, preserving the original racetrack CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--system", choices=("racetrack",), default="racetrack")
    args, remaining = parser.parse_known_args(argv)
    if args.system == "racetrack":
        from ocean_rl.experiments.racetrack import main as run
        return run(remaining)
