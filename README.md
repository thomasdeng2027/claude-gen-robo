# claude-data-gen

Claude-driven data generation pipeline for dexterous manipulation tasks.
Uses Claude (Anthropic API) to generate Isaac Sim motion scripts, executes
them in parallel environments, and iterates based on success feedback.

## Repository layout

```
src/
  pipeline_parallel.py   # orchestrator — Claude API + subprocess launcher
  debug_grasp_close.py   # standalone Isaac Sim grasp debug script
  boilerplate_runtime.py # helpers injected into Claude-generated scripts
  solve_ik_curobo.py     # cuRobo IK solver (runs inside Isaac Sim)
  motion_planner_batch.py# cuRobo batch motion planner
  task_spec.py / task_configs.py / trajectory.py / logger.py / prompts.py
tasks/
  bottle_lift.py  drill_lift.py  cube_rotate.py  ...   # per-task DEFAULTS
  saved_poses_20260414_002010.py                        # scene init poses
assets/
  kinova_xhand/          # Kinova Gen3 + XHand URDF and meshes
  xhand/                 # standalone XHand meshes
  objects/blue_cube/     # blue cube USD
  gripper/ water_bottle/ # misc robot assets
```

## Prerequisites

- NVIDIA GPU with CUDA (Isaac Sim 5.0 requires an RTX-class GPU)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) at `/scr/jingyuny/miniconda3`
- cuRobo source at `/juno/u/jingyuny/curobo`
- `ANTHROPIC_API_KEY` set in your environment

## Installation

```bash
git clone git@github.com:yjy0625/claude-data-gen.git
cd claude-data-gen
bash setup.sh
```

`setup.sh` creates the `claude-data-gen` conda env (Python 3.11), installs
Isaac Sim 5.0 from NVIDIA's PyPI, and installs cuRobo as an editable install.

## Running

### Pipeline (Claude-driven, multi-env)

```bash
conda activate claude-data-gen
cd /scr/jingyuny/projects/p_r2s2r/claude-data-gen

export ANTHROPIC_API_KEY=sk-...

python src/pipeline_parallel.py --tasks bottle_lift --n-envs 4
python src/pipeline_parallel.py --tasks bottle_lift drill_lift --n-envs 8
python src/pipeline_parallel.py --tasks all --n-envs 16 --max-iter 20
```

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required. Anthropic API key. |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Claude model to use. |
| `ISAAC_PYTHON` | `/scr/jingyuny/miniconda3/envs/claude-data-gen/bin/python` | Python used to launch Isaac Sim subprocesses. |
| `KPT_RUNS_DIR` | `src/runs/keypoint_pipeline` | Output directory for logs and generated scripts. |
| `KPT_MAX_ITER` | `12` | Max Claude iterations per task. |
| `KPT_SUBPROC_TIMEOUT` | `1800` | Isaac Sim subprocess timeout (seconds). |

### Grasp debug script (Isaac Sim direct)

```bash
conda activate claude-data-gen
cd /scr/jingyuny/projects/p_r2s2r/claude-data-gen

python src/debug_grasp_close.py
```

Output frames are written to `/tmp/isaac_frames_debug_grasp/` and a video
`src/debug_grasp.mp4` is created on completion.

## SLURM job submission

### pipeline_parallel.py

```bash
#!/bin/bash
#SBATCH --job-name=claude-data-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%j_pipeline.out

source /scr/jingyuny/miniconda3/etc/profile.d/conda.sh
conda activate claude-data-gen
cd /scr/jingyuny/projects/p_r2s2r/claude-data-gen

export ANTHROPIC_API_KEY=sk-...

python src/pipeline_parallel.py --tasks bottle_lift --n-envs 4
```

### debug_grasp_close.py

```bash
#!/bin/bash
#SBATCH --job-name=debug-grasp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%j_debug_grasp.out

source /scr/jingyuny/miniconda3/etc/profile.d/conda.sh
conda activate claude-data-gen
cd /scr/jingyuny/projects/p_r2s2r/claude-data-gen

python src/debug_grasp_close.py
```

Submit with:
```bash
mkdir -p logs
sbatch jobs/pipeline.sh
sbatch jobs/debug_grasp.sh
```
