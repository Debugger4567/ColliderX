"""Archived legacy event generator module.

This module is intentionally kept as a graveyard file to prevent accidental
routing through the deprecated pre-`collision.simulate_events` pipeline.

All public APIs fail fast with a clear migration message.
"""

from typing import Any

_MIGRATION_MESSAGE = (
    "physics.event_generator is archived and disabled. "
    "Use physics.collision.simulate_events instead."
)


def _archived(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(_MIGRATION_MESSAGE)


def simulate_event(*args: Any, **kwargs: Any):
    _archived(*args, **kwargs)


def simulate_batch(*args: Any, **kwargs: Any):
    _archived(*args, **kwargs)


def estimate_w_max(*args: Any, **kwargs: Any):
    _archived(*args, **kwargs)


def generate_decay_kinematics(*args: Any, **kwargs: Any):
    _archived(*args, **kwargs)


def get_available_parents(*args: Any, **kwargs: Any):
    _archived(*args, **kwargs)
