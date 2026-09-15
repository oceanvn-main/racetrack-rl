"""Small interfaces for sample-based, discrete-state RL components.

Algorithms own their update rules and episode/step schedules. Experiments own
map selection, evaluation, reporting and plotting. New continuous or visual
systems should define suitable contracts, rather than force data into this one.
"""
from typing import Protocol, Any, Callable
from pathlib import Path


class DiscreteEnvironment(Protocol):
    action_count: int
    max_episode_steps: int

    def reset(self, seed=None, options=None) -> tuple[Any, dict]: ...
    def step(self, action: int) -> tuple[Any, float, bool, bool, dict]: ...
    def get_valid_actions(self, state) -> list[int]: ...


class Agent(Protocol):
    training_history: list[dict]
    status: dict

    def train(self, num_episodes: int, check_interval: int, *, verbose: bool = True) -> tuple[list, list]: ...
    def act(self, state) -> int:
        """Deterministic evaluation action; must not mutate learner state."""
        ...
    def save_checkpoint(self, prefix: Path) -> Path:
        """Serialize algorithm-owned state and return its actual output path."""
        ...


AgentFactory = Callable[..., Agent]
