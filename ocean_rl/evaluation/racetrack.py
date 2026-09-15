"""Racetrack evaluation shared by any policy callable."""
from ocean_rl.validation import positive_integer

def evaluate_racetrack(env, policy, *, seed=0, max_steps=1000, repeats=1):
    """Noise-free rollouts in a separate environment; crash resets stay random."""
    repeats = positive_integer(repeats, "repeats")
    limit = positive_integer(max_steps, "max_steps")
    evaluation = env.clone(fail_prob=0.0, max_episode_steps=limit, seed=seed)
    results = []
    for start in evaluation.start_cells:
        for repeat in range(repeats):
            state, _ = evaluation.reset(options={"start_cell": tuple(start)})
            path, crash_steps, total_return = [tuple(map(int, state[:2]))], [], 0.0
            total_distance = 0.0
            for step in range(1, limit + 1):
                action = int(policy(state))
                state, reward, terminated, truncated, info = evaluation.step(action)
                path.append(tuple(map(int, state[:2])))
                total_return += reward
                total_distance += info['distance']
                if info.get("crashed", False):
                    crash_steps.append(step)
                if terminated or truncated:
                    break
            results.append({"start": tuple(map(int, start)), "repeat": repeat, "steps": step,
                            "path": path, "crash_steps": crash_steps, "crashes": len(crash_steps),
                            "return": total_return, "distance": total_distance,
                            "objective": evaluation.objective,
                            "finished": bool(terminated), "truncated": bool(truncated)})
    return results
