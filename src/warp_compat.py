"""
warp_compat.py — swap Isaac Sim's bundled warp for pip warp 1.13 after
SimulationApp has finished starting up.

Isaac Sim installs a custom sys.meta_path finder during startup that intercepts
`import warp` and always returns bundled warp 1.8.2, even after we evict it from
sys.modules.  The only reliable fix is to:

  1. Remove Isaac Sim's warp meta_path finders from sys.meta_path.
  2. Load pip warp 1.13 directly from its absolute file path (bypassing the
     import machinery entirely) via importlib.util.spec_from_file_location.
  3. Inject it as sys.modules["warp"] so all subsequent `import warp` calls
     return the already-cached pip warp.

curobo v2 requires warp 1.13+ features (wp.func(module=…), struct types) that
bundled 1.8.2 does not have.  Bundled warp 1.8.2 also generates invalid sm_120
(Blackwell/RTX 5090) PTX → CUDA error 715.

USAGE
-----
    simulation_app = SimulationApp({"headless": True})
    import warp_compat          # swap bundled warp → pip warp 1.13
    # ... omni imports ...
    warp_compat.ensure()        # defensive re-check before cuRobo imports
    import torch
    from curobo._src...
"""
from __future__ import annotations
import sys, site, os


_PIP_WARP_INIT_PATH: str | None = None  # absolute path to pip warp __init__.py


def _is_bundled(mod) -> bool:
    """Return True if *mod* is Isaac Sim's bundled warp (from extscache)."""
    if mod is None:
        return False
    f = getattr(mod, "__file__", "") or ""
    return "extscache" in f


def _find_pip_warp_init() -> str | None:
    """Return the absolute path to pip warp 1.13's __init__.py."""
    global _PIP_WARP_INIT_PATH
    if _PIP_WARP_INIT_PATH and os.path.exists(_PIP_WARP_INIT_PATH):
        return _PIP_WARP_INIT_PATH

    candidates = []
    for p in site.getsitepackages():
        if "extscache" not in p and "site-packages" in p:
            init = os.path.join(p, "warp", "__init__.py")
            if os.path.exists(init):
                candidates.append(init)

    for init in candidates:
        # Prefer the one that's NOT in extscache (extra safety check)
        if "extscache" not in init:
            _PIP_WARP_INIT_PATH = init
            return init
    return None


def _load_pip_warp_direct(init_path: str):
    """
    Load pip warp directly from *init_path*, bypassing sys.meta_path finders.

    Isaac Sim installs a meta_path finder that intercepts `import warp` and
    always returns bundled warp.  We bypass it by using spec_from_file_location
    with an explicit file path, then executing the module ourselves.
    """
    import importlib.util

    # Evict all warp.* from sys.modules first.
    for key in list(sys.modules.keys()):
        if key == "warp" or key.startswith("warp."):
            del sys.modules[key]

    # Remove Isaac Sim's warp meta_path finders.
    # These are entries whose __module__ or string representation references
    # 'warp' or 'isaacsim' or 'omniverse'.
    to_remove = []
    for finder in sys.meta_path:
        finder_str = str(type(finder)) + str(getattr(finder, "__module__", ""))
        finder_file = getattr(finder, "__file__", "") or ""
        if (
            "warp" in finder_str.lower()
            or "extscache" in finder_str.lower()
            or "extscache" in finder_file
        ):
            to_remove.append(finder)
    for finder in to_remove:
        print(f"[warp-compat] removing meta_path finder: {finder}", flush=True)
        sys.meta_path.remove(finder)

    # Add pip warp's parent dir to front of sys.path so warp submodule imports work.
    conda_site = os.path.dirname(os.path.dirname(init_path))  # .../site-packages
    if conda_site in sys.path:
        sys.path.remove(conda_site)
    sys.path.insert(0, conda_site)

    # Remove extscache warp paths from sys.path.
    sys.path[:] = [
        p for p in sys.path
        if not ("extscache" in p and "warp" in p.lower())
    ]

    # Force-load pip warp's config.py directly BEFORE loading __init__.py so
    # that when __init__.py does `from . import config`, Python finds it
    # already in sys.modules["warp.config"] and skips the meta_path finders.
    # Isaac Sim's OmniFinder intercepts the "warp.config" import and returns
    # the bundled 1.8.2 config (missing optimization_level, track_memory, …);
    # pre-populating sys.modules["warp.config"] with the pip config bypasses it.
    warp_dir = os.path.dirname(init_path)
    config_path = os.path.join(warp_dir, "config.py")
    if os.path.exists(config_path):
        config_spec = importlib.util.spec_from_file_location("warp.config", config_path)
        config_mod = importlib.util.module_from_spec(config_spec)
        sys.modules["warp.config"] = config_mod
        config_spec.loader.exec_module(config_mod)
        print(f"[warp-compat] pre-loaded pip warp config (version={getattr(config_mod, 'version', '?')}) from {config_path}", flush=True)

    # Load pip warp directly from file path.
    spec = importlib.util.spec_from_file_location("warp", init_path,
        submodule_search_locations=[warp_dir])
    pip_warp = importlib.util.module_from_spec(spec)
    sys.modules["warp"] = pip_warp

    # Attach the pre-loaded config as an attribute on the module object NOW,
    # before exec_module runs __init__.py.  warp._src.utils is imported at
    # __init__.py line 235 (before "from . import config" at line 465), and
    # utils.py's @wp.func decorator reaches context.py which does
    # `warp.config.max_unroll` — so warp.config must be set on the module
    # object before exec_module starts.
    if "warp.config" in sys.modules:
        pip_warp.config = sys.modules["warp.config"]

    spec.loader.exec_module(pip_warp)

    # Re-attach the pre-loaded config in case exec_module overwrote it.
    if "warp.config" in sys.modules and sys.modules["warp.config"] is not getattr(pip_warp, "config", None):
        pip_warp.config = sys.modules["warp.config"]

    return pip_warp


def _fix() -> None:
    """Load pip warp 1.13 into sys.modules, bypassing Isaac Sim's meta_path hook."""
    cur = sys.modules.get("warp")
    cur_file = getattr(cur, "__file__", "") or ""

    # Already on pip warp? Done.
    if cur is not None and "extscache" not in cur_file:
        print(f"[warp-compat] pip warp {cur.__version__} already active ({cur_file})", flush=True)
        return

    print(f"[warp-compat] bundled warp detected ({cur_file or 'none'}) — applying swap", flush=True)

    # Debug: show meta_path finders
    print(f"[warp-compat] sys.meta_path ({len(sys.meta_path)} entries):", flush=True)
    for i, f in enumerate(sys.meta_path):
        print(f"  [{i}] {type(f).__name__} from {getattr(f, '__module__', '?')}", flush=True)

    init_path = _find_pip_warp_init()
    if not init_path:
        print("[warp-compat] WARNING: pip warp __init__.py not found — skipping swap", flush=True)
        return
    print(f"[warp-compat] pip warp found at: {init_path}", flush=True)

    wp = _load_pip_warp_direct(init_path)
    print(f"[warp-compat] loaded pip warp {wp.__version__} from {wp.__file__}", flush=True)


# ── public API ────────────────────────────────────────────────────────────────

def ensure() -> None:
    """
    Re-apply the warp swap if bundled warp crept back in.
    Call right before cuRobo imports.
    """
    cur = sys.modules.get("warp")
    cur_file = getattr(cur, "__file__", "") or ""
    if cur is None or "extscache" in cur_file:
        print(f"[warp-compat] ensure(): bundled/missing warp ({cur_file or 'None'}) — re-applying", flush=True)
        _fix()
    else:
        print(f"[warp-compat] ensure(): pip warp {cur.__version__} active ({cur_file})", flush=True)


# ── auto-run on first import ──────────────────────────────────────────────────
_fix()
