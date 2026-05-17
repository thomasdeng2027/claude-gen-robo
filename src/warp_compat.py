"""
warp_compat.py — swap Isaac Sim's bundled warp for pip warp 1.13 after
SimulationApp has finished starting up.

PROBLEM
-------
Isaac Sim 5.1 bundles warp 1.8.2 whose C-extensions and OGN rendering nodes
(OgnDecal, omni.warp, syntheticdata) must be initialised using the bundled
warp API.  Pre-loading pip warp 1.13 before SimulationApp breaks those nodes
(missing warp.utils.warn, warp.sim, etc.) and causes the RTX render pipeline
to produce black frames.

curobo v2 requires warp 1.13+ features (wp.func(module=…), struct types) that
the bundled 1.8.2 does not have.

FIX
---
Let SimulationApp start normally (bundled warp 1.8.2 handles all rendering
init), then after the app is ready evict the bundled warp from sys.modules and
promote the conda pip warp 1.13 to the front of sys.path so that subsequent
``import warp`` calls (from cuRobo, user code) resolve to pip warp 1.13.

This is safe because Isaac Sim's C++ plugins hold direct references to the
compiled warp module; re-pointing the Python name does not affect them.

USAGE
-----
Import this module AFTER SimulationApp() returns, BEFORE any cuRobo imports:

    simulation_app = SimulationApp({"headless": True})
    import warp_compat          # swap bundled warp → pip warp 1.13
    import torch                # cuRobo and torch imports follow
    from curobo...

Safe to import multiple times (idempotent after first run).
"""
from __future__ import annotations
import sys, site


def _fix() -> None:
    # Already on pip warp (not extscache)?  Nothing to do.
    try:
        import importlib.util
        spec = importlib.util.find_spec("warp")
        if spec and spec.origin and "extscache" not in spec.origin:
            import warp as _wp
            print(f"[warp-compat] pip warp {_wp.__version__} already active", flush=True)
            return
    except Exception:
        pass

    # Find the conda env site-packages (contains pip warp 1.13).
    candidates = [p for p in site.getsitepackages() if "site-packages" in p]
    if not candidates:
        print("[warp-compat] WARNING: no site-packages found; skipping warp swap", flush=True)
        return
    conda_site = candidates[0]

    # Evict bundled warp from the module cache.
    for key in list(sys.modules.keys()):
        if key == "warp" or key.startswith("warp."):
            del sys.modules[key]

    # Promote conda site-packages so pip warp wins the import race.
    if conda_site in sys.path:
        sys.path.remove(conda_site)
    sys.path.insert(0, conda_site)

    import warp as wp  # noqa: F401
    print(f"[warp-compat] warp {wp.__version__} from {wp.__file__}", flush=True)


_fix()
