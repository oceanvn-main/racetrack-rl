"""Algorithm registration and construction, independent of environments/rendering."""
from dataclasses import dataclass
from inspect import signature
from copy import deepcopy
import re
from ocean_rl.contracts import AgentFactory


@dataclass(frozen=True)
class AlgorithmSpec:
    factory: AgentFactory
    defaults: dict


_REGISTRY = {}


def register_algorithm(name, factory, *, defaults=None):
    """Register a factory(env, *, seed=None, **options); duplicate names fail."""
    if not isinstance(name, str) or re.fullmatch(r'[a-z][a-z0-9_-]*', name) is None:
        raise ValueError('Algorithm name must start with a lowercase letter and use lowercase letters, digits, underscores or hyphens')
    if name in _REGISTRY:
        raise ValueError(f'Algorithm already registered: {name}')
    if not callable(factory):
        raise TypeError('Algorithm factory must be callable')
    if defaults is not None and not isinstance(defaults, dict):
        raise TypeError('Algorithm defaults must be a dictionary')
    if set(defaults or {}) & {'env', 'seed'}:
        raise ValueError('env and seed are supplied by the experiment')
    _REGISTRY[name] = AlgorithmSpec(factory, deepcopy(defaults or {}))


def algorithm_names():
    return tuple(sorted(_REGISTRY))


def resolve_options(name, options=None):
    if name not in _REGISTRY:
        raise ValueError(f'Unknown algorithm {name!r}; available: {algorithm_names()}')
    if options is not None and not isinstance(options, dict):
        raise TypeError('Algorithm options must be a dictionary')
    resolved = deepcopy(_REGISTRY[name].defaults)
    resolved.update(options or {})
    if set(resolved) & {'env', 'seed'}:
        raise ValueError('env and seed are separate factory arguments')
    return resolved


def create_agent(name, env, *, seed=None, **options):
    resolved = resolve_options(name, options)
    factory = _REGISTRY[name].factory
    try:
        signature(factory).bind(env, seed=seed, **resolved)
    except TypeError as exc:
        raise ValueError(f'Invalid options for algorithm {name}: {exc}') from exc
    agent = factory(env, seed=seed, **resolved)
    for method in ('train', 'act', 'save_checkpoint'):
        if not callable(getattr(agent, method, None)):
            raise TypeError(f'Algorithm {name} must implement {method}')
    if not hasattr(agent, 'training_history') or not hasattr(agent, 'status'):
        raise TypeError(f'Algorithm {name} must expose training_history and status')
    return agent


# Built-in registration. New algorithms register here or in a custom launcher
# before calling the existing experiment main(). No change to map/environment code.
from ocean_rl.algorithms.monte_carlo.off_policy import OffPolicyMCAgent
register_algorithm('off-policy-mc', OffPolicyMCAgent,
                   defaults={'gamma': 1.0, 'epsilon': .1, 'init_q': -1000.0})
