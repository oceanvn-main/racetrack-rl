"""Stateless reward strategies, independent of learners and map formats.

Distance is measured along the physical segment to the first collision/finish
boundary. Reset teleportation is never included. Minimum distance is an
undiscounted, successful-episode objective; it does not reward nontermination.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Transition:
    distance: float
    terminated: bool
    crashed: bool


class Objective(Protocol):
    allow_stationary: bool

    def reward(self, transition: Transition) -> float: ...


class MinimumTime:
    allow_stationary = True
    requires_undiscounted = False

    def reward(self, transition):
        return 0.0 if transition.terminated else -1.0


class GeometricDistance:
    allow_stationary = False
    requires_undiscounted = True

    def reward(self, transition):
        return -transition.distance


_FACTORIES = {}


def register_objective(name, factory):
    """Register a no-argument factory for a stateless reward strategy."""
    if not isinstance(name, str) or not name or name in _FACTORIES:
        raise ValueError('Objective name must be nonempty and unique')
    strategy = factory()
    if not callable(getattr(strategy, 'reward', None)) or not isinstance(
            getattr(strategy, 'allow_stationary', None), bool):
        raise TypeError('Objective requires reward(transition) and boolean allow_stationary')
    _FACTORIES[name] = factory


def objective_names():
    return tuple(sorted(_FACTORIES))


def create_objective(name):
    if name not in _FACTORIES:
        raise ValueError(f'Unknown objective {name!r}; choose from {objective_names()}')
    return _FACTORIES[name]()


register_objective('minimum-time', MinimumTime)
register_objective('geometric-distance', GeometricDistance)
