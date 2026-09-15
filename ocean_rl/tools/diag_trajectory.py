"""Print velocity and acceleration along the learned greedy route (no noise).

Usage:
    py -3.12 -m ocean_rl.tools.diag_trajectory
    py -3.12 -m ocean_rl.tools.diag_trajectory --map maps/tiled/n_shape.tmj --checkpoint results/01_n_shape_seed0_policy.npz
"""
import argparse
import sys
from pathlib import Path
import numpy as np

from ocean_rl.environments.racetrack import RacetrackEnv
from ocean_rl.paths import MAPS_DIR, EXPERIMENT_ROOT, RESULTS_DIR


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map",           type=Path, default=EXPERIMENT_ROOT / "maps/tiled/n_shape.tmj")
    p.add_argument("--checkpoint",    type=Path, default=RESULTS_DIR / "01_n_shape_seed0_policy.npz")
    p.add_argument("--max-velocity",  type=int,  default=None, help="Exclusive upper speed bound")
    p.add_argument("--movement-mode", choices=["forward-only", "bidirectional"], default=None)
    p.add_argument("--max-steps",     type=int,  default=300)
    p.add_argument("--seed",          type=int,  default=0)
    return p


def load_policy(checkpoint: Path, env: RacetrackEnv):
    """Reconstruct greedy Q-policy from a saved .npz checkpoint."""
    data   = np.load(checkpoint)
    states = [tuple(s) for s in data["states"]]
    Q_mat  = data["Q"]
    Q      = {s: Q_mat[i] for i, s in enumerate(states)}

    def act(state):
        valid  = env.get_valid_actions(state)
        values = Q.get(tuple(int(x) for x in state))
        if values is None or np.all(values == 0):
            return int(valid[0])
        return int(valid[np.argmax(values[valid])])

    return act, len(states)


def main(argv=None):
    args = build_parser().parse_args(argv)

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}\n"
                 "Run the experiment first to generate the policy file.")

    kwargs = {}
    if args.max_velocity is not None:
        kwargs["max_velocity"] = args.max_velocity
    if args.movement_mode is not None:
        kwargs["movement_mode"] = args.movement_mode

    env = RacetrackEnv.from_file(args.map, fail_prob=0.0, **kwargs)
    act, n_states = load_policy(args.checkpoint, env)

    print(f"\nMap            : {args.map.name}")
    print(f"Checkpoint     : {args.checkpoint.name}  ({n_states} states in Q-table)")
    print(f"Max velocity   : {env.max_velocity}  (speeds {env.velocity_min} .. {env.velocity_max})")
    print(f"Movement mode  : {env.movement_mode}")
    print(f"fail_prob      : 0.0  (deterministic trace)\n")

    obs, _ = env.reset(seed=args.seed)
    state  = tuple(int(x) for x in obs)

    col_w = 5
    header = (f"{'Step':>{col_w}}  {'Row':>{col_w}}  {'Col':>{col_w}}  "
              f"{'vx':>{col_w}}  {'vy':>{col_w}}  "
              f"{'ax_req':>{col_w}}  {'ay_req':>{col_w}}  "
              f"{'ax_eff':>{col_w}}  {'ay_eff':>{col_w}}  "
              f"{'speed':>{col_w}}  {'note'}")
    print(header)
    print("-" * len(header))

    for step in range(1, args.max_steps + 1):
        row, col, vx, vy = state
        action = act(state)
        ax_req, ay_req = env.actions[action]

        obs, reward, terminated, truncated, info = env.step(action)
        new_state = tuple(int(x) for x in obs)

        if info.get("crashed"):
            ax_eff, ay_eff = 0, 0
            note = "CRASH -> reset"
        else:
            _, _, nvx, nvy = new_state
            ax_eff = nvx - vx
            ay_eff = nvy - vy
            note = "FINISH" if terminated else ""

        speed = np.sqrt(vx**2 + vy**2)
        print(f"{step:>{col_w}}  {row:>{col_w}}  {col:>{col_w}}  "
              f"{vx:>{col_w}}  {vy:>{col_w}}  "
              f"{ax_req:>{col_w}}  {ay_req:>{col_w}}  "
              f"{ax_eff:>{col_w}}  {ay_eff:>{col_w}}  "
              f"{speed:>{col_w}.2f}  {note}")

        state = new_state
        if terminated or truncated:
            break

    print()
    if terminated:
        row, col, vx, vy = state
        print(f"[SUCCESS] Reached finish at (row={row}, col={col}) in {step} steps.")
    else:
        print(f"[FAILED] Did not finish within {args.max_steps} steps.")


if __name__ == "__main__":
    main()
