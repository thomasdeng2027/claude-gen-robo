"""Task registry for the keypoint pipeline.

Maps task names → DEFAULTS dicts (defined in tasks/<task>.py).
Use get_task_defaults(name) to fetch a dict of KeypointTaskSpec fields,
then override any field with explicit CLI args or from_saved_poses().
"""

from __future__ import annotations
from typing import Any

from tasks import bottle_lift      as _bottle_lift      # noqa: F401
from tasks import flip_bottle      as _flip_bottle      # noqa: F401
from tasks import drill_lift       as _drill_lift       # noqa: F401
from tasks import bowl_lift        as _bowl_lift        # noqa: F401
from tasks import cube_rotate      as _cube_rotate      # noqa: F401
from tasks import pour_water       as _pour_water       # noqa: F401
from tasks import cube_semicircle  as _cube_semicircle  # noqa: F401
from tasks import bottle_pour      as _bottle_pour      # noqa: F401

_REGISTRY: dict[str, dict] = {
    "bottle_lift":      _bottle_lift.DEFAULTS,
    "bottle_pour":      _bottle_pour.DEFAULTS,
    "flip_bottle":      _flip_bottle.DEFAULTS,
    "drill_lift":       _drill_lift.DEFAULTS,
    "bowl_lift":        _bowl_lift.DEFAULTS,
    "cube_rotate":      _cube_rotate.DEFAULTS,
    "pour_water":       _pour_water.DEFAULTS,
    "cube_semicircle":  _cube_semicircle.DEFAULTS,
}

TASK_NAMES = list(_REGISTRY.keys())


def get_task_defaults(task_name: str) -> dict[str, Any]:
    """Return a shallow copy of the DEFAULTS dict for *task_name*."""
    if task_name not in _REGISTRY:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {TASK_NAMES}"
        )
    return dict(_REGISTRY[task_name])
