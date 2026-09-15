"""Run registered RL algorithms on imported racetrack maps."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ocean_rl.environments.racetrack import RacetrackEnv
from ocean_rl.errors import EpisodeTruncatedError
from ocean_rl.validation import positive_integer
from ocean_rl.visualization.racetrack import plot_results, plot_combined_learning_curves
from ocean_rl.evaluation.racetrack import evaluate_racetrack
from ocean_rl.registry import create_agent, algorithm_names, resolve_options
from ocean_rl.paths import EXPERIMENT_ROOT, MAPS_DIR, RESULTS_DIR
from ocean_rl.objectives import objective_names


HERE = EXPERIMENT_ROOT


def run_map_experiment(env_name, env, num_episodes=50000, *, seed=0, verbose=True, algorithm="off-policy-mc", agent_options=None):
    agent = create_agent(algorithm, env, seed=seed, **(agent_options or {}))
    episodes, returns = agent.train(num_episodes, max(1, num_episodes // 10), verbose=verbose)
    evaluation = evaluate_racetrack(env, agent.act, seed=seed)
    return episodes, returns, evaluation


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=["racetrack"], default="racetrack", help="RL system (currently racetrack)")
    parser.add_argument("--algorithm", choices=algorithm_names(), default="off-policy-mc")
    parser.add_argument("--objective", choices=objective_names(), default=None,
                        help="Reward objective; overrides map settings (default minimum-time)")
    parser.add_argument("--agent-options", type=Path, help="JSON object of algorithm-specific parameters")
    parser.add_argument("--maps", nargs="+", type=Path, default=[MAPS_DIR / name for name in ("track_1.json", "house.json", "maze.json")])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--movement-mode", choices=["forward-only", "bidirectional"], default=None)
    parser.add_argument("--max-velocity", type=int, default=None, help="Exclusive upper speed bound")
    parser.add_argument("--max-steps", type=int, default=None, help="Truncate episodes at this limit; MC skips their updates")
    parser.add_argument("--fail-prob", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--init-q", type=float, default=None)
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument("--eval-max-steps", type=int, default=1000, help="Separate limit for evaluation rollouts")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--plot", action="store_true", help="Save optional Matplotlib figures")
    parser.add_argument("--show", action="store_true", help="Show figures as well as saving them")
    parser.add_argument("--log", metavar="N", type=int, default=0,
                        help="Print per-episode diagnostics every N episodes (0 = disabled).")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Write episode log lines to this file instead of (or as well as) stdout.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        positive_integer(args.episodes, "episodes")
        positive_integer(args.eval_repeats, "eval_repeats")
        positive_integer(args.eval_max_steps, "eval_max_steps")
        if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
            raise ValueError("seeds must be distinct nonnegative integers")
        overrides = {name: value for name, value in {
            "objective": args.objective, "movement_mode": args.movement_mode, "max_velocity": args.max_velocity,
            "max_episode_steps": args.max_steps, "fail_prob": args.fail_prob}.items() if value is not None}
        agent_options = {} if args.agent_options is None else json.loads(args.agent_options.read_text(encoding='utf-8-sig'))
        if not isinstance(agent_options, dict) or set(agent_options) & {'env', 'seed'}:
            raise ValueError('agent-options must be an object without env/seed; use --seeds')
        agent_options.update({key: value for key, value in {'gamma': args.gamma, 'epsilon': args.epsilon, 'init_q': args.init_q}.items() if value is not None})
        agent_options = resolve_options(args.algorithm, agent_options)
        # Validate every map and agent setting before the first potentially long run.
        environments = [RacetrackEnv.from_file(path, **overrides) for path in args.maps]
        for env in environments:
            create_agent(args.algorithm, env, **agent_options)
    except (ValueError, TypeError, OSError, ImportError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined, failed = {}, False
    for index, (source, prototype) in enumerate(zip(args.maps, environments)):
        for seed in args.seeds:
            env = prototype.clone(seed=seed)
            agent = create_agent(args.algorithm, env, seed=seed, **agent_options)
            name = f"{index + 1:02d}_{source.stem}_seed{seed}"
            if args.algorithm != 'off-policy-mc':
                name = f"{args.algorithm}_{name}"
            if env.objective != 'minimum-time':
                name = f"{env.objective}_{name}"
            print(f"\n{name}: {env.height} x {env.width}, {env.movement_mode}")
            error = None
            episodes, returns = [], []
            log_interval = args.log if args.log > 0 else 0
            log_file = open(args.log_file, 'a', encoding='utf-8') if args.log_file else None
            try:
                episodes, returns = agent.train(
                    args.episodes,
                    max(1, args.episodes // 10),
                    log_interval=log_interval,
                    log_file=log_file,
                    run_label=name,
                )
            except EpisodeTruncatedError as exc:
                error, failed = str(exc), True
                print(error)
            finally:
                if log_file is not None:
                    log_file.close()
            evaluation = evaluate_racetrack(env, agent.act, seed=seed, repeats=args.eval_repeats, max_steps=args.eval_max_steps)
            finished = [item for item in evaluation if item["finished"]]
            report = {
                "source": str(source.resolve()), "map_sha256": hashlib.sha256(env.grid.tobytes()).hexdigest(),
                "grid": env.grid.tolist(), "seed": seed,
                "environment": {"objective": env.objective, "movement_mode": env.movement_mode, "max_velocity": env.max_velocity,
                                "fail_prob": env.fail_prob, "max_episode_steps": env.max_episode_steps},
                "algorithm": args.algorithm, "agent": agent_options,
                "evaluation_settings": {"max_steps": args.eval_max_steps, "repeats": args.eval_repeats, "fail_prob": 0.0},
                "status": agent.status, "error": error, "training": agent.training_history,
                "evaluation": evaluation,
                "metrics": {"success_rate": len(finished) / len(evaluation),
                            "mean_distance_to_finish": float(np.mean([x['distance'] for x in finished])) if finished else None,
                            "mean_steps_to_finish": float(np.mean([x["steps"] for x in finished])) if finished else None,
                            "mean_crashes": float(np.mean([x["crashes"] for x in evaluation])),
                            "truncation_rate": float(np.mean([x["truncated"] for x in evaluation]))},
            }
            destination = args.output_dir / f"{name}.json"
            destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            checkpoint = agent.save_checkpoint(args.output_dir / f"{name}_policy")
            report['checkpoint'] = str(checkpoint)
            destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(f"Success rate {report['metrics']['success_rate']:.1%}; saved {destination}")
            combined[name] = episodes, returns, evaluation
            if args.plot or args.show:
                plot_results(episodes, returns, evaluation, env, args.output_dir / f"{name}.png", show=args.show, algorithm_name=args.algorithm)
    if (args.plot or args.show) and combined:
        plot_combined_learning_curves(combined, args.output_dir / "comparison.png", show=args.show)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

