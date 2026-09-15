"""Sparse off-policy Monte Carlo control with recorded behavior probabilities.

State keys are raw environment observations, including signed velocities. Each
map needs its own agent. No geometric information enters the update equations.
"""
from pathlib import Path
import json
import math
import numpy as np
from ocean_rl.validation import positive_integer
from ocean_rl.errors import EpisodeTruncatedError
from ocean_rl.contracts import DiscreteEnvironment


class OffPolicyMCAgent:
    """Weighted importance sampling (Sutton & Barto, Chapter 5).

    Q and log_C are dictionaries keyed by (row, col, vx, vy), each value an
    action vector. Log cumulative weights avoid overflow on long episodes.
    Greedy target ties use the first valid action; behavior distributes its
    exploitation probability over all tied maximizers, plus epsilon exploration.
    """
    def __init__(self, env: DiscreteEnvironment, gamma=1.0, epsilon=0.10, init_q=-1000.0, *, seed=None):
        if getattr(getattr(env, 'reward_objective', None), 'requires_undiscounted', False) and gamma != 1:
            raise ValueError('This objective requires gamma=1 to minimize undiscounted total cost')
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        if not np.isfinite(epsilon) or not 0 < epsilon <= 1:
            raise ValueError("epsilon must be finite and in (0, 1] for coverage")
        if not np.isfinite(init_q):
            raise ValueError("init_q must be finite")
        self.env, self.gamma, self.epsilon = env, float(gamma), float(epsilon)
        self.init_q = float(init_q)
        self.num_actions = positive_integer(env.action_count, "action_count")
        self.rng = np.random.default_rng(seed)
        self.Q, self.log_C, self.pi = {}, {}, {}
        self.training_history = []
        self.status = {"completed": False, "episodes": 0, "truncated_episodes": 0}

    @staticmethod
    def getMetadata():
        return json.loads(Path(__file__).with_name("off_policy.meta.json").read_text(encoding="utf-8-sig"))

    def _values(self, state):
        key = tuple(int(x) for x in state)
        if key not in self.Q:
            self.Q[key] = np.full(self.num_actions, self.init_q)
            self.log_C[key] = np.full(self.num_actions, -np.inf)
        return self.Q[key]

    def greedy_action(self, state):
        valid = self.env.get_valid_actions(state)
        return int(valid[int(np.argmax(self._values(state)[valid]))])

    def behavior_distribution(self, state):
        valid = np.asarray(self.env.get_valid_actions(state), dtype=int)
        values = self._values(state)[valid]
        best = valid[values == values.max()]
        probabilities = np.zeros(self.num_actions)
        probabilities[valid] = self.epsilon / len(valid)
        probabilities[best] += (1.0 - self.epsilon) / len(best)
        return probabilities

    def select_behavior_action(self, state):
        return int(self.rng.choice(self.num_actions, p=self.behavior_distribution(state)))

    def get_behavior_probability(self, state, action):
        """Current probability only; train records this BEFORE policy updates."""
        return float(self.behavior_distribution(state)[action])

    def _update_episode(self, transitions):
        # Each transition contains (state, action, reward, generating probability).
        G, log_W = 0.0, 0.0
        for state, action, reward, probability in reversed(transitions):
            G = reward + self.gamma * G
            values = self._values(state)
            key = tuple(state)
            cumulative = float(np.logaddexp(self.log_C[key][action], log_W))
            self.log_C[key][action] = cumulative
            # Q <- Q + (W / C) * (G - Q), evaluated using log weights.
            values[action] += math.exp(log_W - cumulative) * (G - values[action])
            best = self.greedy_action(state)
            self.pi[key] = best
            if action != best:
                break
            log_W -= math.log(probability)

    def train(self, num_episodes=150000, check_interval=10000, *, verbose=True,
              log_interval=0, log_file=None, run_label=None):
        num_episodes = positive_integer(num_episodes, "num_episodes")
        check_interval = positive_integer(check_interval, "check_interval")
        history_episodes, history_returns, block = [], [], []
        self.status = {"completed": False, "episodes": 0, "truncated_episodes": 0}
        _warned_truncation = False  # emit the truncation warning at most once
        for episode in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            transitions, total_return, crashes = [], 0.0, 0
            while True:
                probabilities = self.behavior_distribution(state)
                action = int(self.rng.choice(self.num_actions, p=probabilities))
                next_state, reward, terminated, truncated, info = self.env.step(action)
                transitions.append((tuple(int(x) for x in state), action, float(reward), float(probabilities[action])))
                total_return += reward
                crashes += int(info.get("crashed", False))
                state = next_state
                if terminated or truncated:
                    break
            record = {"episode": episode, "return": float(total_return), "steps": len(transitions),
                      "crashes": crashes, "finished": bool(terminated), "truncated": bool(truncated)}
            self.training_history.append(record)
            # --- Optional per-episode diagnostic logging -----------------------------------
            if log_interval > 0 and episode % log_interval == 0:
                trunc_so_far = self.status["truncated_episodes"] + int(truncated and not terminated)
                trunc_rate = trunc_so_far / episode
                label = f"[{run_label}] " if run_label else ""
                line = (
                    f"{label}ep={episode:>7d}  steps={len(transitions):>5d}  "
                    f"return={total_return:>8.2f}  "
                    f"{'FINISHED' if terminated else ('TRUNC   ' if truncated else 'CRASH   ')}  "
                    f"crashes={crashes:>3d}  trunc_rate={trunc_rate:.3f}"
                )
                print(line)
                if log_file is not None:
                    print(line, file=log_file, flush=True)
            # -------------------------------------------------------------------------------
            if truncated and not terminated:
                # MC cannot compute an unbiased return from an incomplete episode;
                # skip the Q-value update and continue rather than aborting training.
                self.status["truncated_episodes"] += 1
                if verbose and not _warned_truncation:
                    print(
                        f"Warning: episode {episode} reached max_episode_steps="
                        f"{self.env.max_episode_steps}. No MC update for this episode. "
                        "Subsequent truncations will be silently skipped. "
                        "Consider increasing --max-steps or switching to a TD algorithm."
                    )
                    _warned_truncation = True
                continue  # no Q-value update for this incomplete episode
            self._update_episode(transitions)
            self.status["episodes"] += 1
            block.append(total_return)
            if episode % check_interval == 0 or episode == num_episodes:
                history_episodes.append(episode)
                history_returns.append(float(np.mean(block)) if block else float("nan"))
                if verbose:
                    completed = self.status["episodes"]
                    truncated_count = self.status["truncated_episodes"]
                    print(f"Episode {episode}/{num_episodes}: mean return "
                          f"{np.mean(block) if block else float('nan'):.2f} "
                          f"({completed} updates, {truncated_count} truncated)")
                block.clear()
        self.status["completed"] = True
        # Fail explicitly when no Q-value update was ever made — the policy is untrained.
        if self.status["truncated_episodes"] == num_episodes:
            self.status["completed"] = False
            raise EpisodeTruncatedError(
                f"All {num_episodes} episodes reached max_episode_steps="
                f"{self.env.max_episode_steps}. Zero Q-value updates were made; "
                "the policy is completely untrained. Increase --max-steps or use a "
                "bootstrapped algorithm (e.g., Q-learning / SARSA)."
            )
        return history_episodes, history_returns

    def act(self, state):
        """Deterministic target action without allocating or changing Q."""
        valid = self.env.get_valid_actions(state)
        values = self.Q.get(tuple(state))
        return int(valid[0] if values is None else valid[np.argmax(values[valid])])

    def save_checkpoint(self, prefix):
        """Save inspectable state/Q/log-weight arrays; not a resumable RNG snapshot."""
        destination = Path(str(prefix) + '.npz')
        states = sorted(self.Q)
        state_width = len(states[0]) if states else 0
        np.savez_compressed(destination,
                            states=np.asarray(states, dtype=np.int32).reshape(len(states), state_width),
                            Q=np.asarray([self.Q[s] for s in states]).reshape(-1, self.num_actions),
                            log_C=np.asarray([self.log_C[s] for s in states]).reshape(-1, self.num_actions))
        return destination

    def evaluate_learned_policy(self, *, seed=0, max_steps=1000, repeats=1):
        """Compatibility entry point; evaluation lives outside the algorithm."""
        from ocean_rl.evaluation.racetrack import evaluate_racetrack
        return evaluate_racetrack(self.env, self.act, seed=seed, max_steps=max_steps, repeats=repeats)
