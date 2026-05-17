"""
generate_cup_usd.py — Generate a hollow open-top cup USD asset.

Creates an open-top plastic cup mesh (frustum shell + solid bottom, no top cap)
suitable for robot manipulation experiments.

Dimensions (all in metres, Isaac Sim stage unit = 1 m):
  height          = 0.10 m
  outer_r_bottom  = 0.035 m   (3.5 cm)
  outer_r_top     = 0.044 m   (4.4 cm) — slight taper
  wall_thickness  = 0.003 m   (3 mm)
  n_sides         = 32

Run with the Isaac Sim Python interpreter to write cup.usd next to this file:
    /path/to/isaac-python generate_cup_usd.py
"""
from __future__ import annotations
import math
import os
import numpy as np

# ── USD imports (available via pxr inside the Isaac Sim env) ──────────────────
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Vt

# ── Cup parameters ────────────────────────────────────────────────────────────
HEIGHT         = 0.10
OUTER_R_BOT    = 0.035
OUTER_R_TOP    = 0.044
WALL_T         = 0.003
INNER_R_BOT    = OUTER_R_BOT - WALL_T
INNER_R_TOP    = OUTER_R_TOP - WALL_T
N              = 32              # polygon sides


def _ring(n: int, radius: float, z: float) -> list[tuple[float, float, float]]:
    """Return N evenly-spaced points on a circle of given radius at height z."""
    pts = []
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        pts.append((radius * math.cos(theta), radius * math.sin(theta), z))
    return pts


def build_cup_mesh():
    """Return (points, face_vertex_counts, face_vertex_indices) for the cup."""
    # Four rings of N vertices each
    outer_bot = _ring(N, OUTER_R_BOT, 0.0)          # ring 0 — outer bottom edge
    outer_top = _ring(N, OUTER_R_TOP, HEIGHT)        # ring 1 — outer top (rim outer)
    inner_bot = _ring(N, INNER_R_BOT, WALL_T)        # ring 2 — inner bottom edge
    inner_top = _ring(N, INNER_R_TOP, HEIGHT)        # ring 3 — inner top (rim inner)

    # Centre of the outer bottom circle (for the bottom fan)
    bot_centre = (0.0, 0.0, 0.0)
    inner_bot_centre = (0.0, 0.0, WALL_T)

    all_pts = (outer_bot + outer_top + inner_bot + inner_top
               + [bot_centre, inner_bot_centre])

    # Vertex index offsets
    O_BOT = 0
    O_TOP = N
    I_BOT = 2 * N
    I_TOP = 3 * N
    BOT_C  = 4 * N          # outer bottom centre
    IBOT_C = 4 * N + 1      # inner bottom centre

    counts = []
    indices = []

    def quad(a, b, c, d):
        counts.append(4)
        indices.extend([a, b, c, d])

    def tri(a, b, c):
        counts.append(3)
        indices.extend([a, b, c])

    for i in range(N):
        j = (i + 1) % N

        # Outer wall (CCW when viewed from outside — normal points outward)
        quad(O_BOT + i, O_BOT + j, O_TOP + j, O_TOP + i)

        # Inner wall (CW when viewed from outside — normal points inward)
        quad(I_TOP + i, I_TOP + j, I_BOT + j, I_BOT + i)

        # Rim (top ring, thin strip connecting outer and inner top edges)
        quad(O_TOP + i, O_TOP + j, I_TOP + j, I_TOP + i)

        # Bottom annular ring (between outer-bottom and inner-bottom circles)
        quad(I_BOT + i, I_BOT + j, O_BOT + j, O_BOT + i)

        # Outer bottom disc (fan — connects outer_bot to centre)
        tri(BOT_C, O_BOT + j, O_BOT + i)

        # Inner bottom disc (fan — connects inner_bot to inner centre)
        tri(IBOT_C, I_BOT + i, I_BOT + j)

    return all_pts, counts, indices


def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cup.usd")

    stage = Usd.Stage.CreateNew(out_path)
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Root xform
    root = UsdGeom.Xform.Define(stage, "/Cup")
    stage.SetDefaultPrim(root.GetPrim())

    # Physics APIs on root
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr().Set(0.05)    # 50 g — typical plastic cup

    # Mesh prim
    mesh = UsdGeom.Mesh.Define(stage, "/Cup/CupMesh")

    pts, counts, indices = build_cup_mesh()

    mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices))

    # Orientation: open side faces +Z (up); no adjustment needed.
    mesh.GetOrientationAttr().Set(UsdGeom.Tokens.rightHanded)

    # Collision
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_col.CreateApproximationAttr().Set("convexDecomposition")

    stage.Save()
    print(f"Cup USD written to: {out_path}")
    print(f"  {len(pts)} vertices, {len(counts)} faces")
    print(f"  height={HEIGHT} m, outer_r_top={OUTER_R_TOP} m, "
          f"outer_r_bot={OUTER_R_BOT} m, wall_t={WALL_T} m")


if __name__ == "__main__":
    main()
