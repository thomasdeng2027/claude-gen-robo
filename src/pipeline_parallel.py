"""
pipeline_parallel.py — Multi-task parallel pipeline.

Runs multiple KeypointTaskSpec tasks in round-robin, N_ENVS parallel envs each.

Key design points:
  [1] Deduplication: parallel-Claude path shares _execute_and_record() with
      sequential path — no more 40-line copy-paste.
  [2] Rolling history: self.messages grows each iteration; trimmed to keep
      initial prompt + last MAX_HISTORY_TURNS assistant/user pairs.
  [3] No global TASK_MEMORY: each task uses its own spec.task_description.
  [4] best_reach_score updated on every path (sequential + parallel).
  [5] Round-robin stability: picks task with fewest iterations, not modulo.
  [6] Adaptive camera: PARALLEL_BOILERPLATE already handles any grid size.
  [7] parse_results: uses parse_results_parallel (no dead branch).
  [8] Best-code persistence: best_code.py written whenever success rate improves.

Usage:
  python pipeline_parallel.py --tasks bottle_lift cube_rotate --n-envs 8
  python pipeline_parallel.py --tasks all --n-envs 16 --parallel-claude
  python pipeline_parallel.py --tasks bottle_lift --n-envs 4 --max-iter 20
"""

from __future__ import annotations

import argparse
import json as _json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field as _dc_field, replace as _dc_replace
from pathlib import Path
from typing import Optional

import anthropic

# ── Make repo importable ─────────────────────────────────────────────────────
_PKG_DIR   = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
# Add repo root (for metasim/roboverse_pack) and src/ itself (for local modules)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

# ── Task / prompt infrastructure ────────────────────────────────────────────
from task_spec import KeypointTaskSpec      # noqa: E402
from task_configs import (
    get_task_defaults, TASK_NAMES,
)
from trajectory import load_trajectory
from logger import PipelineLogger
from prompts import (
    PARALLEL_SYSTEM_PROMPT,
    build_initial_prompt_parallel,
    build_feedback_prompt_parallel,
    _filled_parallel_boilerplate,
    PARALLEL_BOILERPLATE_SMOKE_SUFFIX,
)

# ── Runtime config ───────────────────────────────────────────────────────────
# KPT_RUNS_DIR controls where ALL pipeline outputs land (per-run logs + cross-iter
# stats).  Override with:  export KPT_RUNS_DIR=/some/writable/path
# Default: <pkg_dir>/runs/keypoint_pipeline  (same directory where all other
# pipeline runs are stored, i.e. keypoint_pipeline_par/runs/keypoint_pipeline/).
_KPT_RUNS_DIR = Path(
    os.environ.get("KPT_RUNS_DIR", str(_PKG_DIR / "runs" / "keypoint_pipeline"))
)

MODEL              = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_ITERATIONS     = int(os.environ.get("KPT_MAX_ITER",          "12"))
SUBPROCESS_TIMEOUT = int(os.environ.get("KPT_SUBPROC_TIMEOUT", "1800"))
ISAAC_PYTHON       = os.environ.get("ISAAC_PYTHON",
                                    "/scr/jingyuny/miniconda3/envs/claude-data-gen/bin/python")
DEFAULT_N_ENVS     = 1024
# How many prior (assistant, user) pairs to keep in the rolling history.
# The initial user prompt is always kept in position 0.
MAX_HISTORY_TURNS  = 3


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess runner
# ─────────────────────────────────────────────────────────────────────────────

def run_script(script_path: str, frames_dir: str) -> tuple[str, str, int]:
    env = os.environ.copy()
    env["ISAAC_FRAMES_DIR"] = frames_dir
    _local_home = "/tmp/isaac_home"
    os.makedirs(_local_home, exist_ok=True)
    env["HOME"]             = _local_home
    env["OMNI_USER_HOME"]   = _local_home
    env["XDG_DATA_HOME"]    = f"{_local_home}/.local/share"
    env["XDG_CACHE_HOME"]   = f"{_local_home}/.cache"
    env["OMNI_CACHE_PATH"]  = f"{_local_home}/cache"
    env["OMNI_STRUCTUREDLOG_ENABLED"] = "0"
    env["CUROBO_KERNEL_BACKEND"]      = "pybind"
    env["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
    if "DISPLAY" not in env:
        env["DISPLAY"] = ":0"
    _curobo_root = "/juno/u/jingyuny/curobo"
    existing_pp = env.get("PYTHONPATH", "")
    extra_paths = f"{_REPO_ROOT}{os.pathsep}{_curobo_root}"
    env["PYTHONPATH"] = f"{extra_paths}{os.pathsep}{existing_pp}" if existing_pp else extra_paths

    out_buf: list[str] = []
    err_buf: list[str] = []

    def _drain(stream, buf, prefix):
        for raw in stream:
            line = raw.rstrip("\n")
            buf.append(line)
            print(f"  [{prefix}] {line}", flush=True)

    try:
        proc = subprocess.Popen(
            [ISAAC_PYTHON, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        t_o = threading.Thread(target=_drain, args=(proc.stdout, out_buf, "out"), daemon=True)
        t_e = threading.Thread(target=_drain, args=(proc.stderr, err_buf, "err"), daemon=True)
        t_o.start(); t_e.start()

        deadline = time.monotonic() + SUBPROCESS_TIMEOUT
        while proc.poll() is None:
            if time.monotonic() > deadline:
                proc.kill()
                t_o.join(timeout=5); t_e.join(timeout=5)
                return (
                    "\n".join(out_buf),
                    "\n".join(err_buf) + f"\nTimeoutExpired: exceeded {SUBPROCESS_TIMEOUT}s",
                    -1,
                )
            time.sleep(1)
        t_o.join(timeout=10); t_e.join(timeout=10)
        return "\n".join(out_buf), "\n".join(err_buf), proc.returncode
    except Exception as e:
        return "", f"Subprocess launch error: {e}", -1


# ─────────────────────────────────────────────────────────────────────────────
# Code extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Result parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_results_parallel(
    stdout: str, spec: KeypointTaskSpec
) -> tuple[bool, Optional[float], list[dict]]:
    """Parallel: return (success, min_kp_dist_across_envs, frame_states)."""
    frame_states: list[dict] = []
    min_kp: Optional[float] = None
    batch_metrics: Optional[dict] = None
    saw_success = False
    saw_failure = False

    for line in stdout.splitlines():
        m = re.match(r"FINAL_KP_MAX_DIST_MIN:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
        if m:
            try:
                min_kp = float(m.group(1))
            except ValueError:
                pass
        m = re.match(r"BATCH_METRICS:\s*(\{.*\})", line)
        if m:
            try:
                batch_metrics = _json.loads(m.group(1))
            except _json.JSONDecodeError:
                pass
        m = re.match(r"FRAME_STATE:\s*(\{.*\})", line)
        if m:
            try:
                frame_states.append(_json.loads(m.group(1)))
            except _json.JSONDecodeError:
                pass
        if line.strip() == "SUCCESS":
            saw_success = True
        if line.strip() == "FAILURE":
            saw_failure = True

    if batch_metrics:
        success_count = batch_metrics.get("success_count", 0)
        min_kp = batch_metrics.get("min_kp_dist", min_kp)
        return success_count > 0, min_kp, frame_states
    if min_kp is not None:
        return (min_kp < spec.success_tolerance), min_kp, frame_states
    if saw_success and not saw_failure:
        return True, None, frame_states
    return False, None, frame_states


# ─────────────────────────────────────────────────────────────────────────────
# Video encoding
# ─────────────────────────────────────────────────────────────────────────────

def _find_ffmpeg() -> Optional[str]:
    # Prefer the ffmpeg co-located with the current Python interpreter (conda env)
    # over any system ffmpeg — the conda-forge ffmpeg includes libx264/libx265.
    for c in [
        Path(sys.executable).parent / "ffmpeg",
        Path(sys.executable).parent.parent / "bin" / "ffmpeg",
    ]:
        if c.exists():
            return str(c)
    p = shutil.which("ffmpeg")
    if p:
        return p
    for c in [Path("/usr/bin/ffmpeg"), Path("/usr/local/bin/ffmpeg")]:
        if c.exists():
            return str(c)
    return None


def encode_video(frames_dir: str, output_mp4: str, fps: int) -> bool:
    frames_path = Path(frames_dir)
    frames = sorted(frames_path.glob("frame_*.png"))
    if not frames:
        print(f"  [video] no frames in {frames_dir} — skipping")
        return False
    print(f"  [video] encoding {len(frames)} frames → {output_mp4}")

    ff = _find_ffmpeg()
    if ff:
        # Use concat demuxer: write a list file so frame numbers don't need to be
        # consecutive (frames saved every _RENDER_EVERY steps → gaps in numbering).
        list_file = str(frames_path / "_frame_list.txt")
        with open(list_file, "w") as _lf:
            for f in frames:
                _lf.write(f"file '{f.resolve()}'\nduration {1.0/fps}\n")
        # Try libx264 first; fall back to mpeg4 if the codec isn't compiled in.
        for codec, extra in [
            ("libx264", ["-pix_fmt", "yuv420p", "-crf", "18"]),
            ("mpeg4",   ["-q:v", "5"]),
        ]:
            cmd = [ff, "-y", "-f", "concat", "-safe", "0",
                   "-i", list_file, "-c:v", codec] + extra + [output_mp4]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"  [video] saved (ffmpeg/{codec}): {output_mp4}")
                return True
            print(f"  [video] ffmpeg/{codec} failed: {r.stderr.splitlines()[-1] if r.stderr else 'no output'}")

    try:
        import imageio
        # Use the FFMPEG plugin explicitly to avoid imageio picking TiffWriter.
        writer = imageio.get_writer(output_mp4, format="FFMPEG", fps=fps,
                                    codec="mpeg4", quality=5)
        import imageio.v2 as imageio_v2
        for f in frames:
            writer.append_data(imageio_v2.imread(str(f)))
        writer.close()
        print(f"  [video] saved (imageio): {output_mp4}")
        return True
    except Exception as e:
        print(f"  [video] imageio failed: {e}")

    try:
        import cv2
        sample = cv2.imread(str(frames[0]))
        h, wd = sample.shape[:2]
        # Try mp4v (MPEG-4 Part 2) — universally available in OpenCV builds.
        # Fall back to XVID if mp4v also fails.
        for fourcc_str in ("mp4v", "XVID"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out_path = output_mp4 if fourcc_str == "mp4v" else output_mp4.replace(".mp4", ".avi")
            vw = cv2.VideoWriter(out_path, fourcc, fps, (wd, h))
            if not vw.isOpened():
                print(f"  [video] cv2/{fourcc_str}: VideoWriter failed to open")
                continue
            for f in frames:
                img = cv2.imread(str(f))
                if img is not None:
                    vw.write(img)
            vw.release()
            if Path(out_path).stat().st_size > 1024:
                print(f"  [video] saved (cv2/{fourcc_str}): {out_path}")
                return True
            print(f"  [video] cv2/{fourcc_str}: output file too small, skipping")
    except Exception as e:
        print(f"  [video] cv2 failed: {e}")

    print("  [video] could not encode — install ffmpeg or imageio[ffmpeg]")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Cross-run statistics persistence
# ─────────────────────────────────────────────────────────────────────────────

def _load_json_safe(path: Path) -> dict:
    """Load a JSON file safely; return empty dict on any error."""
    try:
        if path.exists():
            return _json.loads(path.read_text())
    except Exception:
        pass
    return {}


def _update_aggregate_stats(stats_dir: Path, run_summary: dict) -> None:
    """Append this run's summary to aggregate_stats.json and recompute distributions."""
    import statistics as _st

    stats_dir.mkdir(parents=True, exist_ok=True)
    agg_path = stats_dir / "aggregate_stats.json"
    agg = _load_json_safe(agg_path)

    # Append a compact record for this run
    runs: list[dict] = agg.get("runs", [])
    runs.append({
        "iteration":            run_summary.get("iteration"),
        "success":              run_summary.get("success"),
        "success_rate":         run_summary.get("success_rate"),
        "reach_score":          run_summary.get("reach_score"),
        "mean_kp_dist":         run_summary.get("mean_kp_dist"),
        "min_kp_dist":          run_summary.get("min_kp_dist"),
        "final_kp":             run_summary.get("final_kp"),
        "failure_mode":         run_summary.get("failure_mode"),
        "grasp_stable_any":     run_summary.get("grasp_stable_any"),
        "reach_failure_rate":   run_summary.get("reach_failure_rate"),
        "drop_rate":            run_summary.get("drop_rate"),
        "any_env_lifted":       run_summary.get("any_env_lifted"),
        "dense_score_mean":     run_summary.get("dense_score_mean"),
        "reach_max_cf":         run_summary.get("reach_max_cf"),
        "max_high_vel_contact": run_summary.get("max_high_vel_contact"),
        "cp_err_pre":           run_summary.get("cp_err_pre"),
        "cp_err_post":          run_summary.get("cp_err_post"),
        "transport_cp_err":     run_summary.get("transport_cp_err"),
        "approach_dir":         run_summary.get("approach_dir"),
        "ik_fail_rate":         run_summary.get("ik_fail_rate"),
    })

    def _dist(vals: list) -> dict:
        if not vals:
            return {}
        return {
            "min":  min(vals),
            "mean": _st.mean(vals),
            "std":  _st.stdev(vals) if len(vals) > 1 else 0.0,
            "max":  max(vals),
        }

    success_rates       = [r["success_rate"]       for r in runs if r.get("success_rate")       is not None]
    reach_scores        = [r["reach_score"]        for r in runs if r.get("reach_score")        is not None]
    min_kp_dists        = [r["min_kp_dist"]        for r in runs if r.get("min_kp_dist")        is not None]
    mean_kp_dists       = [r["mean_kp_dist"]       for r in runs if r.get("mean_kp_dist")       is not None]
    grasp_flags         = [r["grasp_stable_any"]   for r in runs if r.get("grasp_stable_any")   is not None]
    reach_failure_rates = [r["reach_failure_rate"] for r in runs if r.get("reach_failure_rate") is not None]
    drop_rates          = [r["drop_rate"]          for r in runs if r.get("drop_rate")          is not None]
    lift_flags          = [r["any_env_lifted"]     for r in runs if r.get("any_env_lifted")     is not None]
    dense_scores        = [r["dense_score_mean"]   for r in runs if r.get("dense_score_mean")   is not None]
    reach_max_cfs       = [r["reach_max_cf"]       for r in runs if r.get("reach_max_cf")       is not None]
    hvc_vals_agg        = [r["max_high_vel_contact"] for r in runs if r.get("max_high_vel_contact") is not None]
    cp_err_pre_vals     = [r["cp_err_pre"]       for r in runs if r.get("cp_err_pre")       is not None]
    cp_err_post_vals    = [r["cp_err_post"]      for r in runs if r.get("cp_err_post")      is not None]
    transport_cp_vals   = [r["transport_cp_err"] for r in runs if r.get("transport_cp_err") is not None]
    ik_fail_rates       = [r["ik_fail_rate"]     for r in runs if r.get("ik_fail_rate")     is not None]

    agg = {
        "total_runs":               len(runs),
        "success_count":            sum(1 for r in runs if r.get("success")),
        "overall_success_rate":     sum(1 for r in runs if r.get("success")) / max(1, len(runs)),
        "success_rate_stats":       _dist(success_rates),
        "reach_score_stats":        _dist(reach_scores),
        "kp_dist_stats": {
            "min":  _dist(min_kp_dists),
            "mean": _dist(mean_kp_dists),
        },
        "reach_failure_rate_stats": _dist(reach_failure_rates),
        "drop_rate_stats":          _dist(drop_rates),
        "grasp_stable_rate": sum(1 for g in grasp_flags if g) / max(1, len(grasp_flags))
                             if grasp_flags else None,
        "any_env_lifted_rate":  sum(1 for l in lift_flags if l) / max(1, len(lift_flags))
                                if lift_flags else None,
        "dense_score_stats":    _dist(dense_scores),
        "reach_max_cf_stats":   _dist(reach_max_cfs),
        "high_vel_contact_stats": _dist(hvc_vals_agg),
        "cp_err_pre_stats":     _dist(cp_err_pre_vals),
        "cp_err_post_stats":    _dist(cp_err_post_vals),
        "transport_cp_err_stats": _dist(transport_cp_vals),
        "ik_fail_rate_stats":     _dist(ik_fail_rates),
        "runs": runs,
    }
    agg_path.write_text(_json.dumps(agg, indent=2))


def _grasp_stage_score(summary: dict) -> int:
    """Stage quality score for best-iter selection.

    4 = trajectory_coupled: arm actively carrying object along the trajectory
        (lift > 2 cm AND EE within 15 cm of object during transport) — the gold
        signal that the approach, grasp, AND transport all worked.
    3 = any env lifted the object (> 2 cm) — grasp worked even if transport didn't
    2 = stable grasp detected — hand closed properly even if no lift
    1 = default (approach only or no meaningful contact)

    This ordering ensures the selector always prefers iterations that actually
    grasped and lifted over ones that merely got close, preventing regression to
    approach-only strategies when a lift was achieved.
    """
    if summary.get("trajectory_coupled_any"):
        return 4
    if summary.get("any_env_lifted"):
        return 3
    if summary.get("grasp_stable_any"):
        return 2
    return 1


def _update_best_iter_stats(stats_dir: Path, run_summary: dict, code: str) -> bool:
    """Overwrite best_iter_stats.json if this iteration's result is better.

    Priority order:
      1. Higher success_rate
      2. Higher grasp stage (lifted > stable grasp > approach > pushing exploit)
      3. Lower min_kp_dist (tiebreaker within same stage)

    Returns True if the best was updated.
    """
    stats_dir.mkdir(parents=True, exist_ok=True)
    best_path = stats_dir / "best_iter_stats.json"
    best = _load_json_safe(best_path)

    cur_rate  = float(run_summary.get("success_rate") or 0.0)
    cur_kp    = float(run_summary.get("min_kp_dist")  or 999.0)
    cur_stage = _grasp_stage_score(run_summary)
    prev_rate  = float(best.get("success_rate") or 0.0)
    prev_kp    = float(best.get("min_kp_dist")  or 999.0)
    prev_stage = _grasp_stage_score(best)

    rate_better  = cur_rate > prev_rate + 1e-6
    rate_tied    = abs(cur_rate - prev_rate) < 1e-6
    stage_better = cur_stage > prev_stage
    stage_tied   = cur_stage == prev_stage
    kp_better    = cur_kp < prev_kp - 1e-4

    # Primary: success_rate.  Secondary: grasp stage.  Tertiary: min_kp_dist.
    is_better = rate_better or (rate_tied and (stage_better or (stage_tied and kp_better)))
    if not is_better:
        return False

    best = {
        "iteration":          run_summary.get("iteration"),
        "success":            run_summary.get("success"),
        "success_rate":       cur_rate,
        "n_envs":             run_summary.get("n_envs"),
        "reach_score":        run_summary.get("reach_score"),
        "reach_arm_joints":   run_summary.get("reach_arm_joints"),
        "min_kp_dist":        cur_kp,
        "mean_kp_dist":       run_summary.get("mean_kp_dist"),
        "final_kp":           run_summary.get("final_kp"),
        "reach_failure_rate": run_summary.get("reach_failure_rate"),
        "drop_rate":          run_summary.get("drop_rate"),
        "failure_mode":       run_summary.get("failure_mode"),
        "grasp_stable_any":         run_summary.get("grasp_stable_any"),
        "any_env_lifted":           run_summary.get("any_env_lifted"),
        "trajectory_coupled_any":   run_summary.get("trajectory_coupled_any"),
        # Last 2000 chars of code that produced the best result (for diagnostics)
        "code_snippet":             code[-2000:] if code else "",
    }
    best_path.write_text(_json.dumps(best, indent=2))
    return True


def _env_composite_score(env_data: dict) -> float:
    """Composite quality score for a single env's performance.

    Weights: 7×coupled + 5×contact + 4×grasp + 3×lift + 1×kp_progress
    Higher is better. Trajectory following (coupled) and grasp dominate;
    kp_dist is a tiebreaker only.
    """
    # Contact score: EE proximity to object during/after grasp
    fee = env_data.get("final_ee_to_obj")
    if fee is None or float(fee) < 0:
        contact_score = 0.0
    elif float(fee) < 0.05:
        contact_score = 1.0
    elif float(fee) < 0.15:
        contact_score = (0.15 - float(fee)) / 0.10
    else:
        contact_score = 0.0

    # Lift score: how high the object was raised (clamped at 12 cm — full task lift)
    max_lift = float(env_data.get("max_lift") or 0.0)
    lift_score = min(max(max_lift, 0.0) / 0.12, 1.0)

    # Coupled score: trajectory_coupled (object moving WITH hand along trajectory)
    # This is the gold-standard signal: contact + lift + trajectory following all working.
    coupled_score = 1.0 if env_data.get("trajectory_coupled") else 0.0

    # Grasp stable score: hand closed on object without knocking it over
    grasp_score = 1.0 if env_data.get("grasp_stable") else 0.0

    # KP progress score (tiebreaker — lower kp_dist is better)
    kp = float(env_data.get("final_kp_dist") or 0.5)
    kp_score = 1.0 - min(kp / 0.5, 1.0)

    # Weights: trajectory_coupled >> contact ~ grasp >> lift >> kp_dist
    # Coupled is highest: it implies contact, lift, AND trajectory following.
    # Grasp is ranked above lift alone: closing on object without toppling is harder than a bump-lift.
    return 7.0 * coupled_score + 5.0 * contact_score + 4.0 * grasp_score + 3.0 * lift_score + kp_score


def _update_best_env_stats(stats_dir: Path, env_data: dict) -> bool:
    """Overwrite best_env_stats.json if this env's composite score is the all-time best.

    env_data fields:
      iteration, env_id, final_kp_dist, score,
      reach_min_ee_to_obj, approach_type, obj_knocked_at_reach,
      grasp_stable, trajectory_coupled, final_ee_to_obj, max_lift,
      arm_joints (optional)

    Returns True if the file was updated.
    """
    stats_dir.mkdir(parents=True, exist_ok=True)
    best_path = stats_dir / "best_env_stats.json"
    best = _load_json_safe(best_path)

    # Always recompute from the canonical composite formula — never trust a pre-set
    # "score" field from the generated code, which may use kp-only or other metrics.
    cur_score  = _env_composite_score(env_data)
    prev_score = _env_composite_score(best) if best else -1.0
    if cur_score <= prev_score + 1e-4:
        return False

    best_path.write_text(_json.dumps(env_data, indent=2))
    return True


def _read_cross_iter_stats(stats_dir: Path) -> tuple[dict, dict, dict]:
    """Return (aggregate_stats, best_iter_stats, best_env_stats) dicts."""
    return (
        _load_json_safe(stats_dir / "aggregate_stats.json"),
        _load_json_safe(stats_dir / "best_iter_stats.json"),
        _load_json_safe(stats_dir / "best_env_stats.json"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test_parallel(
    spec: KeypointTaskSpec, run_dir: Path, trajectory: list[dict], n_envs: int
) -> bool:
    script = (
        _filled_parallel_boilerplate(
            spec, repo_root=str(_REPO_ROOT), trajectory=trajectory,
            n_envs=n_envs, env_spacing=spec.env_spacing,
        )
        + PARALLEL_BOILERPLATE_SMOKE_SUFFIX
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                     prefix="kpt_par_smoke_", dir=tempfile.gettempdir()) as f:
        f.write(script)
        script_path = f.name

    smoke_frames = "/tmp/kpt_par_smoke_frames"
    if Path(smoke_frames).exists():
        shutil.rmtree(smoke_frames)
    Path(smoke_frames).mkdir(parents=True, exist_ok=True)

    print(f"  Parallel smoke test ({n_envs} envs): {script_path}")
    stdout, stderr, rc = run_script(script_path, smoke_frames)
    ok = "SMOKE_OK" in stdout
    if ok:
        print("  Parallel smoke test PASSED.")
        encode_video(smoke_frames, str(run_dir / "smoke_test.mp4"), spec.video_fps)
    else:
        print("\n" + "!" * 60)
        print("  Parallel smoke test FAILED.")
        print("!" * 60)
        print("\n--- stdout (last 4000 chars) ---")
        print(stdout[-4000:] if len(stdout) > 4000 else stdout)
        print("\n--- stderr (last 2000 chars) ---")
        print(stderr[-2000:] if len(stderr) > 2000 else stderr)
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Run record and helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunRecord:
    iteration: int
    success: bool
    return_code: int
    generated_code_path: str
    stdout_path: str
    stderr_path: str
    video_path: str
    final_kp: Optional[float]
    iter_min_kp: float
    reach_score: Optional[float]
    reach_arm_joints: Optional[str]
    success_rate: Optional[float]
    success_count: Optional[int]
    failure_mode: str
    summary: dict = _dc_field(default_factory=dict)


def _tag_value(stdout: str, tag: str) -> Optional[str]:
    for line in stdout.splitlines():
        if line.startswith(tag + ":") or line.startswith(tag + " "):
            return line
    return None


def _float_tag(stdout: str, tag: str) -> Optional[float]:
    line = _tag_value(stdout, tag)
    if not line:
        return None
    try:
        return float(line.split(":", 1)[1].strip().split()[0])
    except Exception:
        return None


def _build_run_summary(
    iteration: int,
    stdout: str,
    stderr: str,
    frame_states: list[dict],
    success: bool,
    final_kp: Optional[float],
    rc: int,
) -> dict:
    batch_metrics: Optional[dict] = None
    bm_line = _tag_value(stdout, "BATCH_METRICS")
    if bm_line:
        try:
            batch_metrics = _json.loads(bm_line.split(":", 1)[1].strip())
        except Exception:
            pass

    transport_kp = [
        fs.get("kp_min") or fs.get("kp_max_dist")
        for fs in frame_states
        if fs.get("phase") in ("transport", "Transport")
        and isinstance(fs.get("kp_min") or fs.get("kp_max_dist"), (int, float))
    ]
    iter_min_kp = min(transport_kp) if transport_kp else 999.0
    if final_kp is not None and final_kp < iter_min_kp:
        iter_min_kp = final_kp

    # Only flag a real Python crash — Isaac Sim floods stderr with "Error"/"Warning" lines
    # that are not Python exceptions. Only trigger on an actual traceback or non-zero exit.
    _stderr_lines = stderr.splitlines()
    has_error = rc not in (0, -1) or any(
        "Traceback (most recent call last)" in l or "SyntaxError" in l
        for l in _stderr_lines
    )
    reach_score          = _float_tag(stdout, "REACH_SCORE")

    # Detect whether arm reached the object
    transport_fs = [fs for fs in frame_states if fs.get("phase") in ("transport", "Transport")]
    _arm_never_reached = (reach_score is not None and reach_score > 0.20)
    # Primary: transport FRAME_STATE lift_per_env (present when transport phase runs normally)
    any_env_lifted = bool(transport_fs and any(
        any(float(l) > 0.02 for l in fs.get("lift_per_env", []))
        for fs in transport_fs if fs.get("lift_per_env")
    ))
    # Fallback 1: scan ALL frame_states for lift_per_env > 0.02 (catches grasp-phase lifts
    # when TRANSPORT_SKIP runs and no transport FRAME_STATE is printed)
    if not any_env_lifted:
        any_env_lifted = any(
            any(float(l) > 0.02 for l in fs.get("lift_per_env", []))
            for fs in frame_states if fs.get("lift_per_env")
        )
    # Fallback 2: BEST_GRASP_STATE tag (always printed, max_lift recorded per-env)
    if not any_env_lifted:
        _bg_line = _tag_value(stdout, "BEST_GRASP_STATE")
        if _bg_line:
            try:
                _bg = _json.loads(_bg_line.split(":", 1)[1].strip())
                if float(_bg.get("max_lift", 0.0)) > 0.02:
                    any_env_lifted = True
            except Exception:
                pass
    # Fallback 3: END_MEMORY any_env_lifted flag (written by the generated script itself)
    if not any_env_lifted:
        _em_line = _tag_value(stdout, "END_MEMORY")
        if _em_line:
            try:
                _em = _json.loads(_em_line.split(":", 1)[1].strip())
                if _em.get("any_env_lifted"):
                    any_env_lifted = True
            except Exception:
                pass
    # Fallback 4: PER_ENV_STATS max_lift in any phase
    # Format: "PER_ENV_STATS {json}" (space) or "PER_ENV_STATS: {json}" (colon)
    if not any_env_lifted:
        for _line in stdout.splitlines():
            if _line.startswith("PER_ENV_STATS"):
                try:
                    _brace4 = _line.find("{")
                    _pe = _json.loads(_line[_brace4:]) if _brace4 >= 0 else {}
                    if float(_pe.get("max_lift", 0.0)) > 0.02:
                        any_env_lifted = True
                        break
                except Exception:
                    pass
    # Parse trajectory_coupled_any: arm carried object along recorded trajectory.
    # FRAME_STATE uses "coupled_envs" (integer count) — check > 0.
    # Also accept trajectory_coupled_any / trajectory_coupled bool fields.
    trajectory_coupled_any = any(
        fs.get("trajectory_coupled_any")
        or fs.get("trajectory_coupled")
        or (fs.get("coupled_envs") or 0) > 0
        for fs in transport_fs
    )
    if not trajectory_coupled_any:
        # BEST_ENV_SUMMARY has trajectory_coupled_any — try it first (colon separator, parses cleanly)
        _bes_line = _tag_value(stdout, "BEST_ENV_SUMMARY")
        if _bes_line:
            try:
                _bes = _json.loads(_bes_line.split(":", 1)[1].strip())
                trajectory_coupled_any = bool(_bes.get("trajectory_coupled_any"))
            except Exception:
                pass
    if not trajectory_coupled_any:
        # END_MEMORY: may use space separator ("END_MEMORY {...}") — extract from first '{'
        _em_tc_line = _tag_value(stdout, "END_MEMORY")
        if _em_tc_line:
            try:
                _brace = _em_tc_line.find("{")
                if _brace >= 0:
                    _em_tc = _json.loads(_em_tc_line[_brace:])
                    trajectory_coupled_any = bool(_em_tc.get("trajectory_coupled_any"))
            except Exception:
                pass
    if not trajectory_coupled_any:
        # Scan PER_ENV_STATS for any env with trajectory_coupled: true
        # Format: "PER_ENV_STATS {json}" (space separator) — extract from first '{'
        for _line in stdout.splitlines():
            if _line.startswith("PER_ENV_STATS"):
                try:
                    _brace = _line.find("{")
                    if _brace >= 0:
                        _pe = _json.loads(_line[_brace:])
                        if _pe.get("trajectory_coupled"):
                            trajectory_coupled_any = True
                            break
                except Exception:
                    pass

    reach_arm_joints     = (_tag_value(stdout, "REACH_ARM_JOINTS")
                            or _tag_value(stdout, "GRASP_HOLD_ARM_JOINTS"))
    success_rate         = batch_metrics.get("success_rate")         if batch_metrics else None
    success_count        = batch_metrics.get("success_count")        if batch_metrics else None
    reach_failure_rate   = batch_metrics.get("reach_failure_rate")   if batch_metrics else None
    drop_rate            = batch_metrics.get("drop_rate")            if batch_metrics else None
    dense_score_mean     = batch_metrics.get("dense_score_mean")     if batch_metrics else None
    score_kp_progress    = batch_metrics.get("score_kp_progress")    if batch_metrics else None
    score_lift           = batch_metrics.get("score_lift")            if batch_metrics else None
    score_grasp          = batch_metrics.get("score_grasp")           if batch_metrics else None
    score_tilt_pen       = batch_metrics.get("score_tilt_pen")        if batch_metrics else None
    score_hvc_pen        = batch_metrics.get("score_hvc_pen")         if batch_metrics else None

    # Contact-pair geometry scores (Phase 3 pre/post closure + transport)
    _cp_lines = [l for l in stdout.splitlines() if l.startswith("CONTACT_PAIR_ERROR:")]
    cp_err_pre  = None  # first occurrence = pre_close (step 0)
    cp_err_post = None  # last occurrence  = post_close (final step)
    if _cp_lines:
        try:
            cp_err_pre  = float(_cp_lines[0].split(":")[1].strip().split()[0])
        except Exception:
            pass
        try:
            cp_err_post = float(_cp_lines[-1].split(":")[1].strip().split()[0])
        except Exception:
            pass
    transport_cp_err = _float_tag(stdout, "TRANSPORT_CONTACT_PAIR_ERROR")

    # Contact / collision stats
    reach_max_cf = _float_tag(stdout, "REACH_MAX_CF")   # peak contact force during approach (N)

    # Approach direction — printed by generated code as "APPROACH_DIR_CORRECTED: [x, y, z]"
    approach_dir = None
    _adir_line = _tag_value(stdout, "APPROACH_DIR_CORRECTED")
    if _adir_line:
        try:
            import ast as _ast
            approach_dir = _ast.literal_eval(_adir_line.split(":", 1)[1].strip())
        except Exception:
            pass

    # IK failure rate: fraction of IK_FAIL_FINAL lines across all calls.
    # Each "IK_FAIL_FINAL" line represents one call where cuRobo couldn't converge.
    # Count how many total IK calls happened vs how many had FAIL_FINAL.
    _ik_fail_lines  = [l for l in stdout.splitlines() if "IK_FAIL_FINAL" in l]
    _ik_total_lines = [l for l in stdout.splitlines()
                       if any(t in l for t in ("IK_FAIL_COUNT", "IK_SERVO_FAIL",
                                               "IK_FK_ERR_M", "IK_FAIL_FINAL",
                                               "IK_RETRY_RECOVERED"))]
    # Rough estimate: each servo call or cold call is one IK dispatch.
    # Count lines containing "IK_FAIL_FINAL" as failed dispatches.
    # Count lines containing "IK_FK_ERR_M" or "IK_RETRY_RECOVERED" as successful dispatches.
    _ik_ok_lines = [l for l in stdout.splitlines()
                    if "IK_FK_ERR_M" in l or "IK_RETRY_RECOVERED" in l]
    _ik_total = len(_ik_fail_lines) + len(_ik_ok_lines)
    ik_fail_rate = (len(_ik_fail_lines) / _ik_total) if _ik_total > 0 else None
    # high_vel_contact: max across all transport FRAME_STATEs
    hvc_vals = [
        fs.get("high_vel_contact_env0")
        for fs in frame_states
        if isinstance(fs.get("high_vel_contact_env0"), (int, float))
    ]
    max_high_vel_contact = max(hvc_vals) if hvc_vals else None
    mean_high_vel_contact = (sum(hvc_vals) / len(hvc_vals)) if hvc_vals else None

    if success:
        failure_mode = "success"
    elif has_error:
        failure_mode = "crash"
    elif "Timeout" in stderr:
        failure_mode = "timeout"
    elif _arm_never_reached:
        failure_mode = "reach_failure"
    elif reach_score is not None and reach_score > 0.15:
        failure_mode = "reach_failed"
    elif final_kp is not None:
        failure_mode = "kp_miss"
    else:
        failure_mode = "unknown"

    return {
        "iteration":        iteration,
        "success":          success,
        "return_code":      rc,
        "has_error":        has_error,
        "failure_mode":     failure_mode,
        "final_kp":         final_kp,
        "iter_min_kp":      iter_min_kp if iter_min_kp < 999.0 else None,
        "reach_score":      reach_score,
        "reach_arm_joints": reach_arm_joints,
        "success_rate":       success_rate,
        "success_count":      success_count,
        "min_kp_dist":        batch_metrics.get("min_kp_dist")  if batch_metrics else None,
        "mean_kp_dist":       batch_metrics.get("mean_kp_dist") if batch_metrics else None,
        "reach_failure_rate": reach_failure_rate,
        "drop_rate":          drop_rate,
        "any_env_lifted":           any_env_lifted,
        "trajectory_coupled_any":   trajectory_coupled_any,
        "dense_score_mean":         dense_score_mean,
        "score_kp_progress":    score_kp_progress,
        "score_lift":           score_lift,
        "score_grasp":          score_grasp,
        "score_tilt_pen":       score_tilt_pen,
        "score_hvc_pen":        score_hvc_pen,
        "reach_max_cf":         reach_max_cf,
        "max_high_vel_contact": max_high_vel_contact,
        "mean_high_vel_contact": mean_high_vel_contact,
        "cp_err_pre":           cp_err_pre,
        "cp_err_post":          cp_err_post,
        "transport_cp_err":     transport_cp_err,
        "approach_dir":         approach_dir,
        "ik_fail_rate":         ik_fail_rate,
    }


def _format_best_runs(
    all_runs: dict[int, RunRecord],
    best_by_metric: dict[str, int],
) -> str:
    if not best_by_metric or not all_runs:
        return ""
    lines = ["=== BEST RUN MEMORY (reference when regressing) ==="]
    for metric in ("reach", "min_kp", "success_rate", "overall"):
        iter_n = best_by_metric.get(metric)
        if iter_n is None:
            continue
        run = all_runs.get(iter_n)
        if run is None:
            continue
        s = run.summary
        parts: list[str] = []
        if s.get("reach_score") is not None:
            parts.append(f"reach={s['reach_score']:.4f}m")
        if s.get("iter_min_kp") is not None:
            parts.append(f"min_kp={s['iter_min_kp']:.4f}m")
        if s.get("final_kp") is not None:
            parts.append(f"final_kp={s['final_kp']:.4f}m")
        if s.get("success_rate") is not None:
            parts.append(f"success_rate={s['success_rate']:.1%}")
        detail = "  |  ".join(parts) if parts else "—"
        lines.append(f"  best_{metric}: iter {iter_n}  [{detail}]")
        lines.append(f"    code: {run.generated_code_path}")
        if metric == "reach" and s.get("reach_arm_joints"):
            lines.append(f"    arm_joints: {s['reach_arm_joints']}")
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Per-task state
# ─────────────────────────────────────────────────────────────────────────────

class TaskState:
    """Encapsulates everything for one task: Claude history, best tracking, etc."""

    def __init__(
        self,
        spec: KeypointTaskSpec,
        trajectory: list[dict],
        logger: PipelineLogger,
    ):
        self.spec       = spec
        self.trajectory = trajectory
        self.logger     = logger
        self.n_envs     = spec.n_envs
        self.client     = anthropic.Anthropic()

        # Cross-iteration stats directory (scoped to this pipeline run)
        self.stats_dir  = logger.run_dir / "cross_iter_stats" / spec.task_name
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [{spec.task_name}] cross-iter stats dir: {self.stats_dir}")

        # Pre-compute and cache the filled boilerplate for stitching motion-only responses
        self.boilerplate_text: str = _filled_parallel_boilerplate(
            spec, repo_root=str(_REPO_ROOT), trajectory=trajectory,
            n_envs=spec.n_envs, env_spacing=spec.env_spacing,
        )

        self.iteration  = 0
        self.done       = False
        self.success    = False

        # ── Rolling conversation history ──────────────────────────────────
        # self.messages grows: [user_0, asst_0, user_1, asst_1, ..., user_N]
        # On each call_claude(), we trim to [user_0] + last MAX_HISTORY_TURNS pairs.
        self.messages: list[dict] = []

        # ── Best-tracking ─────────────────────────────────────────────────
        self.best_min_kp:              float = 999.0
        self.best_reach_score:         float = 999.0
        self.best_reach_arm_joints:    Optional[str] = None
        self.best_success_rate:        float = 0.0
        self.best_final_ee_obj_dist:   float = 999.0  # lower = robot held the object at end
        self.all_runs:    dict[int, RunRecord] = {}
        self.best_by_metric: dict[str, int]   = {}
        self.history: list[dict] = []

        # ── Carry-forward for prompt builder ─────────────────────────────
        self.last_assistant: str = ""
        self.generated_code: str = ""
        self.stdout:         str = ""
        self.stderr:         str = ""
        self.frame_states:   list[dict] = []

        # ── Stagewise optimization state ──────────────────────────────
        # Stages: "approach" → "grasp" → "transport"
        # Drives which part of the controller Claude is asked to focus on.
        self.current_stage:        str            = "approach"
        self.best_approach_joints: Optional[str]  = None   # joints from BEST_ENV_SUMMARY
        self.best_per_env_stats:   list[dict]     = []     # per-env stats from best-reach iter
        self.best_env_summary:     Optional[dict] = None   # BEST_ENV_SUMMARY from best iter
        self.last_end_memory:      Optional[dict] = None   # END_MEMORY from previous run
        # Best full joint state (arm + hand) at grasp closure — seeded to Phase 3 next iter.
        # Updated whenever BEST_GRASP_STATE reports a higher max_lift than the stored value.
        self.best_grasp_state:     Optional[dict] = None   # BEST_GRASP_STATE from best iter

        # ── Stuck / exploration tracking ──────────────────────────────
        # Incremented whenever the obj_tilt knocked-over condition repeats
        # consecutively.  Reset to 0 when a better result is seen.
        self.stuck_counter:        int            = 0
        self.last_tilt:            Optional[float] = None
        self.last_min_kp:          Optional[float] = None

    # ── Claude API ────────────────────────────────────────────────────────

    def call_claude(self) -> str:
        """Build messages and call Claude.  Rolling history kept."""
        # Always read the latest cross-run stats before calling Claude
        agg_stats, best_stats, best_env_stats = _read_cross_iter_stats(self.stats_dir)

        # Restore best_approach_joints from persisted stats if not already set in memory.
        # Handles the case where the pipeline process restarted (TaskState lost but files remain).
        if self.best_approach_joints is None:
            _stored_joints = None
            # Priority 1: best_iter_stats reach_arm_joints (from REACH_ARM_JOINTS tag in stdout)
            if best_stats and best_stats.get("reach_arm_joints"):
                _stored_joints = best_stats["reach_arm_joints"]
            # Priority 2: best_env_stats arm_joints (best_approach_arm_joints from BEST_ENV_SUMMARY)
            if not _stored_joints and best_env_stats and best_env_stats.get("arm_joints"):
                _joints_list = best_env_stats["arm_joints"]
                if isinstance(_joints_list, list):
                    _stored_joints = str(_joints_list)
            if _stored_joints:
                self.best_approach_joints = str(_stored_joints)
                print(f"  [{self.spec.task_name}] best_approach_joints restored from persisted stats")

        if self.iteration == 1:
            user_text = build_initial_prompt_parallel(
                self.spec,
                repo_root=str(_REPO_ROOT),
                trajectory=self.trajectory,
                n_envs=self.n_envs,
                aggregate_stats=agg_stats or None,
                best_iter_stats=best_stats or None,
                best_env_stats=best_env_stats or None,
            )
            self.messages = [{"role": "user", "content": user_text}]
        else:
            best_runs_summary = _format_best_runs(self.all_runs, self.best_by_metric)
            user_text = build_feedback_prompt_parallel(
                self.spec,
                self.generated_code,
                self.stdout,
                self.stderr,
                self.iteration - 1,
                self.frame_states,
                self.best_min_kp,
                self.history,
                repo_root=str(_REPO_ROOT),
                trajectory=self.trajectory,
                n_envs=self.n_envs,
                best_reach_score=self.best_reach_score,
                best_reach_arm_joints=self.best_reach_arm_joints,
                best_runs_summary=best_runs_summary,
                current_stage=self.current_stage,
                best_approach_joints=self.best_approach_joints,
                best_per_env_stats=self.best_per_env_stats,
                best_env_summary=self.best_env_summary,
                last_end_memory=self.last_end_memory,
                best_final_ee_obj_dist=self.best_final_ee_obj_dist,
                aggregate_stats=agg_stats or None,
                best_iter_stats=best_stats or None,
                best_env_stats=best_env_stats or None,
                stuck_counter=self.stuck_counter,
                best_grasp_state=self.best_grasp_state,
            )
            # Strip boilerplate from the assistant message before storing in history.
            # The feedback prompt asks Claude to return motion-only; iter-1 responses
            # include the full boilerplate (~1 000 lines) which would be re-sent every
            # call.  _stitch_motion_only() re-prepends it at execution time.
            _stored_asst = self.last_assistant
            _bp_end = "# === END PARALLEL BOILERPLATE ==="
            if _bp_end in _stored_asst:
                _stored_asst = _stored_asst.split(_bp_end, 1)[1].strip()
            self.messages.append({"role": "assistant", "content": _stored_asst})
            self.messages.append({"role": "user",      "content": user_text})
            # Trim: keep messages[0] (initial prompt) + last MAX_HISTORY_TURNS pairs
            max_len = 1 + 2 * MAX_HISTORY_TURNS
            if len(self.messages) > max_len:
                self.messages = [self.messages[0]] + self.messages[-2 * MAX_HISTORY_TURNS:]

        def _stream_text(msgs: list[dict], max_tok: int = 32000) -> tuple[str, str]:
            """Stream a request and return (full_text, stop_reason).

            Uses prompt caching on the system prompt and the initial user message
            (messages[0]) — both are static across all iterations of a task and
            account for the bulk of repeated input tokens.
            """
            # Cache the system prompt (never changes)
            cached_system = [{"type": "text", "text": PARALLEL_SYSTEM_PROMPT,
                               "cache_control": {"type": "ephemeral"}}]

            # Cache messages[0] (initial prompt — always the same for this task)
            cached_msgs: list[dict] = []
            for i, msg in enumerate(msgs):
                if i == 0 and msg["role"] == "user" and isinstance(msg["content"], str):
                    cached_msgs.append({
                        "role": "user",
                        "content": [{"type": "text", "text": msg["content"],
                                     "cache_control": {"type": "ephemeral"}}],
                    })
                else:
                    cached_msgs.append(msg)

            with self.client.messages.stream(
                model=MODEL,
                max_tokens=max_tok,
                system=cached_system,
                messages=cached_msgs,
            ) as stream:
                text = stream.get_final_text()
                stop_reason = stream.get_final_message().stop_reason
            return text, stop_reason

        _stdout_keep_lines = 60  # shrinks each overflow attempt

        def _truncate_stdout_in_msg(msg: dict, keep: int) -> dict:
            """Return a copy of a user message with STDOUT TAIL section truncated."""
            if msg["role"] != "user":
                return msg
            content = msg["content"]
            marker = "=== STDOUT TAIL ==="
            next_sec = "=== STDERR ==="
            s = content.find(marker)
            e = content.find(next_sec, s) if s >= 0 else -1
            if s >= 0 and e > s:
                lines = content[s:e].splitlines()
                if len(lines) > keep + 2:
                    trunc = "\n".join(lines[:keep + 1]) + f"\n[... {len(lines)-keep-1} lines omitted to fit context ...]\n"
                    content = content[:s] + trunc + content[e:]
            return {"role": msg["role"], "content": content}

        def _trim_history_for_context():
            nonlocal _stdout_keep_lines
            # First passes: truncate STDOUT TAIL in all but the most recent user message
            if _stdout_keep_lines > 5:
                _stdout_keep_lines = max(5, _stdout_keep_lines // 3)
                trimmed = []
                for i, msg in enumerate(self.messages):
                    # Always keep the last user message intact; truncate older ones
                    if msg["role"] == "user" and i < len(self.messages) - 1:
                        trimmed.append(_truncate_stdout_in_msg(msg, _stdout_keep_lines))
                    else:
                        trimmed.append(msg)
                self.messages = trimmed
                print(f"  [context-trim {self.spec.task_name}] truncated STDOUT TAIL to {_stdout_keep_lines} lines in older messages")
            else:
                # Last resort: drop the oldest non-initial pair
                if len(self.messages) > 3:
                    self.messages = [self.messages[0]] + self.messages[3:]
                    print(f"  [context-trim {self.spec.task_name}] dropped oldest history pair ({len(self.messages)} msgs remain)")

        max_tokens = 32000
        for attempt in range(5):
            try:
                text, stop_reason = _stream_text(self.messages, max_tokens)
                # If truncated, ask Claude to continue until the code block is closed
                for _cont in range(3):
                    if stop_reason != "max_tokens":
                        break
                    print(f"  [{self.spec.task_name}] response truncated, continuing...")
                    cont_messages = self.messages + [
                        {"role": "assistant", "content": text},
                        {"role": "user",      "content": "Your response was cut off. Please continue exactly where you left off, completing the Python code block."},
                    ]
                    cont_text, stop_reason = _stream_text(cont_messages, max_tokens)
                    text += cont_text
                return text
            except anthropic.RateLimitError as e:
                wait = 60 * (attempt + 1)
                print(f"  [rate-limit {self.spec.task_name}] waiting {wait}s ({e})")
                time.sleep(wait)
            except anthropic.BadRequestError as e:
                if "context limit" in str(e) or "input length" in str(e):
                    # Context overflow: trim history and reduce output budget, retry immediately
                    _trim_history_for_context()
                    max_tokens = max(4000, max_tokens - 8000)
                    print(f"  [context-overflow {self.spec.task_name}] reduced max_tokens to {max_tokens}")
                else:
                    raise
            except Exception as e:
                # Catch transient network errors (dropped connections, incomplete reads, etc.)
                if attempt < 4:
                    wait = 10 * (attempt + 1)
                    print(f"  [network-error {self.spec.task_name}] attempt {attempt+1}/5, retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"API retries exhausted for {self.spec.task_name}")

    # ── Shared execution + result recording ───────────────────────────────

    def _stitch_motion_only(self, code: str) -> str:
        """If Claude returned only the motion section (no boilerplate), prepend it."""
        if "# === VERIFIED PARALLEL BOILERPLATE" not in code:
            return (
                self.boilerplate_text
                + "\n# === END PARALLEL BOILERPLATE ===\n\n"
                + code.strip()
            )
        return code

    def _execute_and_record(self, code: str) -> dict:
        """
        Syntax-check, write, execute, encode video, parse results, update
        bests, log.  Returns a result dict.
        Shared by both sequential and parallel-Claude execution paths.
        """
        code = self._stitch_motion_only(code)
        # Syntax check
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            print(f"  [{self.spec.task_name}] syntax error: {e}")
            self.stdout, self.stderr = "", str(e)
            iter_dir = self.logger.run_dir / f"iteration_{self.iteration:02d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            (iter_dir / "generated.py").write_text(code)
            (iter_dir / "stdout.txt").write_text("")
            (iter_dir / "stderr.txt").write_text(str(e))
            self.logger.log_iteration(
                iteration=self.iteration, generated_code=code,
                stdout="", stderr=str(e), success=False, note="syntax_error",
            )
            return {"success": False, "success_rate": 0.0, "final_kp": None,
                    "min_kp": 999.0, "frame_states": []}

        # Write temp script
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            prefix=f"kpt_par_{self.spec.task_name}_i{self.iteration:02d}_",
            dir=tempfile.gettempdir(),
        ) as fh:
            fh.write(code)
            script_path = fh.name

        frames_dir = f"/tmp/kpt_par_frames_{self.spec.task_name}_iter{self.iteration:02d}"
        if Path(frames_dir).exists():
            shutil.rmtree(frames_dir)
        Path(frames_dir).mkdir(parents=True, exist_ok=True)

        print(f"  [{self.spec.task_name}] iter={self.iteration} executing: {script_path}")
        t0 = time.monotonic()
        self.stdout, self.stderr, rc = run_script(script_path, frames_dir)
        elapsed = time.monotonic() - t0
        print(f"  [{self.spec.task_name}] rc={rc}  elapsed={elapsed:.0f}s")

        # Encode video
        iter_dir = self.logger.run_dir / f"iteration_{self.iteration:02d}" / self.spec.task_name
        iter_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(iter_dir / "render.mp4")
        encode_video(frames_dir, video_path, self.spec.video_fps)

        # Save artifacts
        code_path   = str(iter_dir / "generated.py")
        stdout_path = str(iter_dir / "stdout.txt")
        stderr_path = str(iter_dir / "stderr.txt")
        try:
            (iter_dir / "generated.py").write_text(code)
            (iter_dir / "stdout.txt").write_text(self.stdout)
            (iter_dir / "stderr.txt").write_text(self.stderr)
        except Exception as _e:
            print(f"  [{self.spec.task_name}] artifact write error: {_e}")

        # Parse results
        success, final_kp, frame_states = parse_results_parallel(self.stdout, self.spec)
        self.frame_states = frame_states

        # ── Update best reach ──────────────────────────────────────────
        for line in self.stdout.splitlines():
            if line.startswith("REACH_SCORE:"):
                try:
                    rs = float(line.split(":")[1].strip())
                    if rs < self.best_reach_score:
                        self.best_reach_score = rs
                        for l2 in self.stdout.splitlines():
                            if l2.startswith("REACH_ARM_JOINTS:"):
                                self.best_reach_arm_joints = l2
                                break
                        print(f"  [{self.spec.task_name}] new best reach: {self.best_reach_score:.4f}m")
                except ValueError:
                    pass
                break

        # ── Update best keypoint distance ─────────────────────────────
        transport_kp = [
            fs.get("kp_min") or fs.get("kp_max_dist")
            for fs in frame_states
            if fs.get("phase") in ("transport", "Transport")
            and isinstance(fs.get("kp_min") or fs.get("kp_max_dist"), (int, float))
        ]
        iter_min = min(transport_kp) if transport_kp else 999.0
        if final_kp is not None and final_kp < iter_min:
            iter_min = final_kp
        if iter_min < self.best_min_kp:
            self.best_min_kp = iter_min
            print(f"  [{self.spec.task_name}] new best kp_dist: {self.best_min_kp:.4f}m")

        # ── Parse success_rate and final_ee_obj_dist from BATCH_METRICS ──
        success_rate = 0.0
        for line in self.stdout.splitlines():
            m = re.match(r"BATCH_METRICS:\s*(\{.*\})", line)
            if m:
                try:
                    bm = _json.loads(m.group(1))
                    success_rate = float(bm.get("success_rate", 0.0))
                    ee_dist_min = bm.get("final_ee_obj_dist_min")
                    if ee_dist_min is not None and float(ee_dist_min) >= 0:
                        ee_dist_min = float(ee_dist_min)
                        if ee_dist_min < self.best_final_ee_obj_dist:
                            self.best_final_ee_obj_dist = ee_dist_min
                            print(f"  [{self.spec.task_name}] new best final_ee_obj_dist: {self.best_final_ee_obj_dist:.4f}m")
                except Exception:
                    pass
                break

        # ── Parse structured per-env logs ─────────────────────────────
        per_env_stats:    list[dict]    = []
        best_env_summary: Optional[dict] = None
        end_memory:       Optional[dict] = None
        best_grasp_state_this_iter: Optional[dict] = None
        for line in self.stdout.splitlines():
            # Accept both "PER_ENV_STATS {json}" (space) and "PER_ENV_STATS: {json}" (colon)
            m = re.match(r"PER_ENV_STATS:?\s*(\{.*\})", line)
            if m:
                try: per_env_stats.append(_json.loads(m.group(1)))
                except Exception: pass
                continue
            m = re.match(r"BEST_ENV_SUMMARY:\s*(\{.*\})", line)
            if m:
                try: best_env_summary = _json.loads(m.group(1))
                except Exception: pass
                continue
            m = re.match(r"END_MEMORY:\s*(\{.*\})", line)
            if m:
                try: end_memory = _json.loads(m.group(1))
                except Exception: pass
                continue
            m = re.match(r"BEST_GRASP_STATE:\s*(\{.*\})", line)
            if m:
                try: best_grasp_state_this_iter = _json.loads(m.group(1))
                except Exception: pass
        self.last_end_memory = end_memory

        # Update best_grasp_state whenever this iter's max_lift exceeds stored value.
        # Even a tiny lift (> 0) is more informative than a zero-lift grasp.
        if best_grasp_state_this_iter is not None:
            new_lift = float(best_grasp_state_this_iter.get("max_lift", 0.0))
            old_lift = float(self.best_grasp_state.get("max_lift", 0.0)) if self.best_grasp_state else -1.0
            if new_lift > old_lift:
                self.best_grasp_state = best_grasp_state_this_iter
                print(f"  [{self.spec.task_name}] best_grasp_state updated "
                      f"(env={best_grasp_state_this_iter.get('env_id')}  "
                      f"max_lift={new_lift:.4f}m  "
                      f"cp_err={best_grasp_state_this_iter.get('cp_err', -1.0):.4f})")

        # Update best approach joints — PRIORITY ORDER:
        # 1. BEST_GRASP_STATE reach_arm_joints when lift ≥ 1mm (proves this config can grasp)
        # 2. BEST_ENV_SUMMARY best_approach_arm_joints when reach improved significantly
        #    BUT only if BEST_GRASP_STATE doesn't have a better config already
        #
        # CRITICAL: BEST_ENV_SUMMARY selects by min EE distance, which can be achieved
        # by ANY IK config family. BEST_GRASP_STATE selects the env that actually lifted
        # the object — its reach_arm_joints are the joints that enabled the grasp.
        grasp_reach_arm = (best_grasp_state_this_iter.get("reach_arm_joints")
                           if best_grasp_state_this_iter else None)
        grasp_lift      = float(best_grasp_state_this_iter.get("max_lift", 0.0)) \
                          if best_grasp_state_this_iter else 0.0
        if grasp_reach_arm and grasp_lift >= 0.001:
            # This iteration's grasp config actually lifted something — prefer it always
            self.best_approach_joints = str(grasp_reach_arm)
            print(f"  [{self.spec.task_name}] best_approach_joints ← GRASP reach_arm "
                  f"(lift={grasp_lift:.4f}m, env={best_grasp_state_this_iter.get('env_id')})")

        # Update best approach joints if this run's reach improved AND grasp didn't set them
        if best_env_summary:
            bes_reach = float(best_env_summary.get("best_reach_dist", 999))
            if bes_reach < self.best_reach_score:
                # Only update arm joints from reach if BEST_GRASP_STATE didn't just set them
                if not (grasp_reach_arm and grasp_lift >= 0.001):
                    joints = best_env_summary.get("best_approach_arm_joints")
                    if joints:
                        self.best_approach_joints = str(joints)
                        print(f"  [{self.spec.task_name}] best_approach_joints ← ENV_SUMMARY "
                              f"(reach={bes_reach:.4f}m, env={best_env_summary.get('best_reach_env')})")
                self.best_per_env_stats = per_env_stats
                self.best_env_summary   = best_env_summary

        # ── Track best individual env across all iterations ───────────
        # Score each env by composite quality (grasp > lift > coupled > kp)
        # Use "final" phase PER_ENV_STATS when available; fall back to kp_batch.
        try:
            kp_batch: Optional[list] = None
            for line in self.stdout.splitlines():
                m = re.match(r"FINAL_KP_MAX_DIST_BATCH:\s*(\[.*\])", line)
                if m:
                    kp_batch = _json.loads(m.group(1))
                    break

            # Build per-env dicts from "final" phase PER_ENV_STATS
            final_by_env: dict[int, dict] = {}
            for s in per_env_stats:
                if s.get("phase") == "final":
                    eid = int(s.get("env_id", -1))
                    if eid >= 0:
                        final_by_env[eid] = s

            # Determine which env to score: prefer final-phase scoring over kp_batch
            if final_by_env:
                # Score every env that has final-phase stats
                best_env_idx = max(
                    final_by_env.keys(),
                    key=lambda eid: _env_composite_score({
                        "final_kp_dist":       final_by_env[eid].get("final_goal_kp",
                                               (kp_batch[eid] if kp_batch and eid < len(kp_batch) else 0.5)),
                        "max_lift":            final_by_env[eid].get("max_lift", 0.0),
                        "trajectory_coupled":  final_by_env[eid].get("trajectory_coupled", False),
                        "grasp_stable":        final_by_env[eid].get("grasp_stable", False),
                        "final_ee_to_obj":     final_by_env[eid].get("final_ee_to_obj", 0.5),
                    })
                )
            elif kp_batch:
                # Fallback: no final stats available, use lowest kp distance
                best_env_idx = int(min(range(len(kp_batch)), key=lambda i: kp_batch[i]))
            else:
                best_env_idx = None

            if best_env_idx is not None:
                best_env_kp = float(kp_batch[best_env_idx]) if (kp_batch and best_env_idx < len(kp_batch)) else 999.0

                # Collect this env's per-phase stats
                env_by_phase = {
                    s["phase"]: s
                    for s in per_env_stats
                    if s.get("env_id") == best_env_idx
                }
                reach_s = env_by_phase.get("reach", {})
                grasp_s = env_by_phase.get("grasp", {})
                final_s = env_by_phase.get("final", {})

                # Lift: prefer "final" phase report; fallback to frame_states
                env_lift = float(final_s["max_lift"]) if final_s.get("max_lift") is not None else None
                if env_lift is None:
                    for fs in frame_states:
                        lift_list = fs.get("lift_per_env")
                        if lift_list and best_env_idx < len(lift_list):
                            v = float(lift_list[best_env_idx])
                            env_lift = v if env_lift is None else max(env_lift, v)

                env_data = {
                    "iteration":            self.iteration,
                    "env_id":               best_env_idx,
                    "final_kp_dist":        best_env_kp,
                    "reach_min_ee_to_obj":  reach_s.get("min_ee_to_obj"),
                    "approach_type":        reach_s.get("approach_type"),
                    "obj_knocked_at_reach": reach_s.get("obj_knocked"),
                    "grasp_stable":         final_s.get("grasp_stable", grasp_s.get("grasp_stable")),
                    "trajectory_coupled":   final_s.get("trajectory_coupled", False),
                    "final_ee_to_obj":      final_s.get("final_ee_to_obj"),
                    "max_lift":             env_lift,
                    "arm_joints":           (best_env_summary.get("best_approach_arm_joints")
                                            if best_env_summary else None),
                }
                env_data["score"] = _env_composite_score(env_data)
                updated_env = _update_best_env_stats(self.stats_dir, env_data)
                if updated_env:
                    print(f"  [{self.spec.task_name}] best_env_stats.json updated "
                          f"(env={best_env_idx}  score={env_data['score']:.3f}  "
                          f"kp={best_env_kp:.4f}m  lift={env_lift or 0:.4f}m  "
                          f"grasp={env_data['grasp_stable']}  coupled={env_data['trajectory_coupled']})")
        except Exception as _env_err:
            print(f"  [{self.spec.task_name}] best_env tracking error (non-fatal): {_env_err}")

        # ── Build RunRecord ───────────────────────────────────────────
        run_summary = _build_run_summary(
            self.iteration, self.stdout, self.stderr,
            frame_states, success, final_kp, rc,
        )
        # Augment run_summary with fields needed for cross-run stats
        run_summary["n_envs"]           = self.n_envs
        run_summary["grasp_stable_any"] = any(s.get("grasp_stable", False)
                                               for s in per_env_stats)

        # Stage detection: use the BEST individual env's reach distance (not aggregate).
        # Aggregate reach_score can be high even when one env clearly reached the object.
        _best_reach = float(best_env_summary.get("best_reach_dist", 999)) \
            if best_env_summary else 999.0
        # Also consider stored all-time best reach
        _stored_best_reach = float(self.best_env_summary.get("best_reach_dist", 999)) \
            if self.best_env_summary else 999.0
        _best_reach = min(_best_reach, _stored_best_reach)
        _reach_ok = _best_reach < 0.10

        # Grasp detection: any env with grasp_stable in "final" phase stats
        _any_stable = any(
            s.get("grasp_stable", False)
            for s in per_env_stats
            if s.get("phase") in ("grasp", "final")
        )
        if not _reach_ok:
            self.current_stage = "approach"
        elif not _any_stable:
            self.current_stage = "grasp"
        else:
            self.current_stage = "transport"
        print(f"  [{self.spec.task_name}] stage={self.current_stage}  "
              f"best_reach={_best_reach:.4f}m  reach_ok={_reach_ok}  grasp_ok={_any_stable}")

        # ── Stuck / exploration counter ───────────────────────────────
        # Increment when the bottle is knocked over AND kp_dist hasn't improved.
        cur_tilt = None
        if frame_states:
            first_transport = next(
                (fs for fs in frame_states if fs.get("phase") == "transport"), None
            )
            if first_transport:
                cur_tilt = first_transport.get("obj_tilt")

        cur_min_kp = run_summary.get("min_kp_dist")
        knocked_over_now = cur_tilt is not None and float(cur_tilt) > 45.0
        no_kp_improvement = (
            self.last_min_kp is None
            or cur_min_kp is None
            or float(cur_min_kp) >= float(self.last_min_kp) - 0.005
        )
        # Do NOT count a run as "stuck" if the arm actually lifted the object —
        # a successful grasp+lift means the tilt reading is likely a formula bug
        # (e.g. the absolute arccos formula always reports ~178° for this bottle mesh),
        # not a real knock-over.  Requiring any_env_lifted=False ensures stuck_counter
        # only fires when the arm genuinely made no contact with the object.
        any_lifted = bool(run_summary.get("any_env_lifted", False))
        if knocked_over_now and no_kp_improvement and not any_lifted:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_tilt   = cur_tilt
        self.last_min_kp = cur_min_kp
        if self.stuck_counter > 0:
            print(f"  [{self.spec.task_name}] stuck_counter={self.stuck_counter} "
                  f"(knocked_over={knocked_over_now}  no_kp_improvement={no_kp_improvement}"
                  f"  any_lifted={any_lifted})")

        # ── Best-code persistence ─────────────────────────────────────
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            best_code_path = self.logger.run_dir / self.spec.task_name / "best_code.py"
            best_code_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                best_code_path.write_text(code)
                print(f"  [{self.spec.task_name}] best_code.py updated "
                      f"(rate={success_rate:.1%})")
            except Exception as _e:
                print(f"  [{self.spec.task_name}] best_code.py write error: {_e}")
        run_rec = RunRecord(
            iteration=self.iteration,
            success=success,
            return_code=rc,
            generated_code_path=code_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            video_path=video_path,
            final_kp=final_kp,
            iter_min_kp=iter_min,
            reach_score=run_summary.get("reach_score"),
            reach_arm_joints=run_summary.get("reach_arm_joints"),
            success_rate=run_summary.get("success_rate"),
            success_count=run_summary.get("success_count"),
            failure_mode=run_summary["failure_mode"],
            summary=run_summary,
        )
        self.all_runs[self.iteration] = run_rec

        # ── Persist cross-iteration statistics ────────────────────────
        try:
            _update_aggregate_stats(self.stats_dir, run_summary)
            updated_best = _update_best_iter_stats(self.stats_dir, run_summary, code)
            if updated_best:
                print(f"  [{self.spec.task_name}] best_iter_stats.json updated "
                      f"(rate={success_rate:.1%}  min_kp={iter_min:.4f}m)")
        except Exception as _stats_err:
            print(f"  [{self.spec.task_name}] stats persistence error (non-fatal): {_stats_err}")

        # Update best_by_metric
        prev_reach = self.all_runs.get(self.best_by_metric.get("reach", -1))
        if run_rec.reach_score is not None and (
            prev_reach is None or prev_reach.reach_score is None
            or run_rec.reach_score < prev_reach.reach_score
        ):
            self.best_by_metric["reach"] = self.iteration

        prev_kp = self.all_runs.get(self.best_by_metric.get("min_kp", -1))
        if iter_min < 999.0 and (prev_kp is None or iter_min < prev_kp.iter_min_kp):
            self.best_by_metric["min_kp"]  = self.iteration
            self.best_by_metric["overall"] = self.iteration

        if run_rec.success_rate is not None:
            prev_sr = self.all_runs.get(self.best_by_metric.get("success_rate", -1))
            if (prev_sr is None or prev_sr.success_rate is None
                    or run_rec.success_rate > prev_sr.success_rate):
                self.best_by_metric["success_rate"] = self.iteration

        # ── Log ───────────────────────────────────────────────────────
        self.history.append({
            "iteration": self.iteration,
            "final_kp":  final_kp,
            "min_kp":    iter_min,
            "success":   success,
        })
        self.history = self.history[-4:]

        self.logger.log_iteration(
            iteration=self.iteration, generated_code=code,
            stdout=self.stdout, stderr=self.stderr, success=success,
            final_kp=final_kp, iter_min_kp=iter_min, best_min_kp=self.best_min_kp,
            n_envs=self.n_envs,
        )

        print(f"  [{self.spec.task_name}] success={success}  "
              f"success_rate={success_rate:.1%}  "
              f"iter_min_kp={iter_min:.4f}m  final_kp={final_kp}")

        return {
            "success":      success,
            "success_rate": success_rate,
            "final_kp":     final_kp,
            "min_kp":       iter_min,
            "frame_states": frame_states,
        }

    # ── One full iteration (call Claude → execute → record) ───────────────

    def run_one_iteration(self) -> dict:
        self.iteration += 1
        print(f"\n{'='*60}")
        print(f"TASK={self.spec.task_name}  ITER={self.iteration}/{MAX_ITERATIONS}"
              f"  N_ENVS={self.n_envs}")
        print(f"{'='*60}")

        print(f"  [{self.spec.task_name}] calling Claude...")
        assistant_text = self.call_claude()
        self.last_assistant = assistant_text
        code = extract_code(assistant_text)
        self.generated_code = code
        print(f"  [{self.spec.task_name}] received {len(code):,} chars")

        result = self._execute_and_record(code)

        if result["success"]:
            self.done    = True
            self.success = True
            print(f"  *** TASK {self.spec.task_name} COMPLETE ***")

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Multi-task runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tasks(
    specs: list[KeypointTaskSpec],
    max_iterations: int = MAX_ITERATIONS,
    parallel_claude: bool = False,
    skip_smoke_test: bool = True,
) -> dict[str, bool]:
    """
    Run all tasks round-robin until each either succeeds or hits max_iterations.

    parallel_claude=True: Claude API calls for all active tasks in each round
    are made concurrently (overlaps network I/O), then sims run sequentially.
    """
    logger = PipelineLogger(base_dir=str(_KPT_RUNS_DIR))

    # Load trajectory for each spec
    trajectories: dict[str, list[dict]] = {}
    for spec in specs:
        traj = load_trajectory(
            spec.trajectory_path,
            pause_threshold=spec.traj_pause_threshold,
        )
        trajectories[spec.task_name] = traj
        print(f"  [{spec.task_name}] trajectory: {len(traj)} frames")

    # Smoke test each task before starting
    if skip_smoke_test:
        print("\n--- Smoke tests SKIPPED (--skip-smoke-test) ---")
    else:
        for spec in specs:
            traj = trajectories[spec.task_name]
            print(f"\n--- Smoke test: {spec.task_name} ({spec.n_envs} envs) ---")
            ok = run_smoke_test_parallel(spec, logger.run_dir, traj, spec.n_envs)
            if not ok:
                print(f"  Smoke test FAILED for {spec.task_name} — aborting.")
                return {s.task_name: False for s in specs}

    states: dict[str, TaskState] = {
        spec.task_name: TaskState(spec, trajectories[spec.task_name], logger)
        for spec in specs
    }

    print(f"\nStarting pipeline: {len(specs)} tasks, "
          f"max {max_iterations} iters/task, "
          f"parallel_claude={parallel_claude}")

    global_iter = 0
    while True:
        active = [s for s in states.values() if not s.done and s.iteration < max_iterations]
        if not active:
            break
        global_iter += 1

        if parallel_claude:
            # Fire Claude API calls for ALL active tasks concurrently,
            # then run each sim sequentially (one Isaac Sim at a time).
            print(f"\n--- Round {global_iter}: parallel Claude for "
                  f"{[s.spec.task_name for s in active]} ---")

            codes: dict[str, str] = {}
            with ThreadPoolExecutor(max_workers=len(active)) as pool:
                futures = {}
                for state in active:
                    state.iteration += 1
                    fut = pool.submit(state.call_claude)
                    futures[fut] = state
                for fut in as_completed(futures):
                    state = futures[fut]
                    try:
                        text = fut.result()
                        state.last_assistant = text
                        state.generated_code = extract_code(text)
                        codes[state.spec.task_name] = state.generated_code
                        print(f"  [{state.spec.task_name}] Claude done: "
                              f"{len(state.generated_code):,} chars")
                    except Exception as e:
                        print(f"  [{state.spec.task_name}] Claude error: {e}")
                        codes[state.spec.task_name] = ""

            # Execute sims sequentially
            for state in active:
                code = codes.get(state.spec.task_name, "")
                if not code:
                    continue
                print(f"\n{'='*60}")
                print(f"TASK={state.spec.task_name}  ITER={state.iteration}"
                      f"  N_ENVS={state.n_envs}")
                print(f"{'='*60}")
                result = state._execute_and_record(code)
                if result["success"]:
                    state.done    = True
                    state.success = True
                    print(f"  *** {state.spec.task_name} COMPLETE ***")

        else:
            # Sequential: pick task with fewest iterations
            state = min(active, key=lambda s: s.iteration)
            state.run_one_iteration()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    results: dict[str, bool] = {}
    for name, state in states.items():
        status = ("SUCCESS" if state.success
                  else f"FAILED  best_rate={state.best_success_rate:.1%}"
                       f"  best_kp={state.best_min_kp:.4f}m"
                       f"  best_ee_obj={state.best_final_ee_obj_dist:.4f}m")
        print(f"  {name:25s}: {status}  (iters={state.iteration})")
        results[name] = state.success

    logger.finalize(all(results.values()), sum(s.iteration for s in states.values()))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_spec(task_name: str, n_envs: int, env_spacing: float,
                approach_standoff: Optional[float] = None,
                **phase_overrides) -> KeypointTaskSpec:
    """Load task defaults + CLI overrides → KeypointTaskSpec."""
    cfg = get_task_defaults(task_name)
    saved_poses      = cfg.pop("saved_poses")
    object_usd       = cfg.pop("object_usd")
    trajectory_json  = cfg.pop("trajectory_json", "")
    object_scale     = float(cfg.pop("object_scale",      1.0))
    object_mass      = float(cfg.pop("object_mass",       0.1))
    success_tol      = float(cfg.pop("success_tolerance", 0.05))
    task_description = cfg.pop("task_description", "")

    # Remaining cfg keys (e.g. object_orientation, object_z_settle_offset) go to spec
    return KeypointTaskSpec.from_saved_poses(
        saved_poses,
        object_usd=object_usd,
        trajectory_path=trajectory_json,
        object_scale=object_scale,
        object_mass=object_mass,
        success_tolerance=success_tol,
        n_envs=n_envs,
        env_spacing=env_spacing,
        task_description=task_description,
        task_name=task_name,
        approach_standoff=approach_standoff,
        **cfg,
        **phase_overrides,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-task parallel Isaac Sim pipeline.",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["bottle_lift"],
        metavar="TASK",
        help=(
            "Task names to run round-robin (or 'all'). "
            "Available: " + ", ".join(TASK_NAMES)
        ),
    )
    parser.add_argument("--n-envs",   type=int,   default=DEFAULT_N_ENVS,
                        help=f"Parallel envs per task iteration (default {DEFAULT_N_ENVS})")
    parser.add_argument("--env-spacing", type=float, default=2.0,
                        help="Metres between env origins in the grid (default 2.0)")
    parser.add_argument("--max-iter", type=int,   default=MAX_ITERATIONS,
                        help=f"Max iterations per task (default {MAX_ITERATIONS})")
    parser.add_argument("--parallel-claude", action="store_true",
                        help="Call Claude for all tasks simultaneously (I/O overlap)")
    parser.add_argument("--skip-smoke-test", action="store_true", default=True,
                        help="Skip the pre-flight smoke test (default: True)")
    parser.add_argument("--smoke-test", dest="skip_smoke_test", action="store_false",
                        help="Run the pre-flight smoke test before iterating")
    parser.add_argument("--t-settle",    type=int, default=None)
    parser.add_argument("--t-approach",  type=int, default=None)
    parser.add_argument("--t-grasp",     type=int, default=None)
    parser.add_argument("--t-transport", type=int, default=None)
    parser.add_argument("--t-hold",      type=int, default=None)
    parser.add_argument("--standoffs", nargs="+", type=float, default=None,
                        metavar="M",
                        help="Sweep _APPROACH_STANDOFF values (m). Each value runs the "
                             "full pipeline independently. Example: --standoffs 0.13 0.16 0.20")
    args = parser.parse_args()

    task_names = TASK_NAMES if "all" in args.tasks else args.tasks
    unknown = [t for t in task_names if t not in TASK_NAMES]
    if unknown:
        print(f"ERROR: unknown tasks: {unknown}.  Available: {TASK_NAMES}")
        return 1

    phase_overrides = {
        k: v for k, v in {
            "t_settle":    args.t_settle,
            "t_approach":  args.t_approach,
            "t_grasp":     args.t_grasp,
            "t_transport": args.t_transport,
            "t_hold":      args.t_hold,
        }.items() if v is not None
    }

    standoff_values: list[Optional[float]] = (
        [float(s) for s in args.standoffs] if args.standoffs else [None]
    )

    print(f"Tasks:           {task_names}")
    print(f"N_ENVS:          {args.n_envs}")
    print(f"ENV_SPACING:     {args.env_spacing}")
    print(f"MAX_ITER:        {args.max_iter}")
    print(f"PARALLEL_CLAUDE: {args.parallel_claude}")
    print(f"SKIP_SMOKE_TEST: {args.skip_smoke_test}")
    print(f"ISAAC_PYTHON:    {ISAAC_PYTHON}")
    if args.standoffs:
        print(f"STANDOFFS:       {standoff_values}")
        print(f"  (default when omitted: auto = clip(obj_radius + 0.10, 0.20, 0.30) m)")

    # Build one spec per (task, standoff) combination.  Each becomes an independent
    # "task" in the round-robin scheduler so the ordering is:
    #   task@s1 iter1 → task@s2 iter1 → task@s3 iter1 → task@s1 iter2 → …
    specs: list[KeypointTaskSpec] = []
    for task_name in task_names:
        for standoff in standoff_values:
            try:
                spec = _build_spec(
                    task_name, args.n_envs, args.env_spacing,
                    approach_standoff=standoff,
                    **phase_overrides,
                )
                if standoff is not None:
                    spec = _dc_replace(spec, task_name=f"{task_name}@{standoff:.3f}m")
                specs.append(spec)
                label = spec.task_name
                print(f"  [{label}] spec ready: "
                      f"obj={spec.object_usd}  traj={spec.trajectory_path or '(none)'}")
            except Exception as e:
                print(f"  ERROR building spec for {task_name}: {e}")
                return 1

    results = run_all_tasks(
        specs,
        max_iterations=args.max_iter,
        parallel_claude=args.parallel_claude,
        skip_smoke_test=args.skip_smoke_test,
    )
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
