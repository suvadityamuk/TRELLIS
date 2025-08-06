import bpy, json, math, os, sys, numpy as np
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from tqdm import tqdm
import os, sys
sys.path.append(os.path.dirname(__file__))   # ← new
from render import main as render_one        # ← your existing render.py main()


job_file, out_root, nviews = sys.argv[-3:]
jobs = json.load(open(job_file))

def make_views(n):
    offs = (0.3, 0.7)
    yaws, pitchs = [], []
    for i in range(n):
        # simple deterministic sphere sampling
        u = (i + offs[0]) / n
        v = (i + offs[1]) / n
        yaws.append(2*math.pi*u)
        pitchs.append(math.acos(1-2*v)-math.pi/2)
    return [
        {"yaw": y, "pitch": p, "radius": 2, "fov": math.radians(40)}
        for y, p in zip(yaws, pitchs)
    ]

# initialise Cycles once
import argparse
for rec in tqdm(jobs, desc="Rendering objects", unit="obj"):
    # --- NEW: resolve to an absolute path ---
    obj_path = rec["local_path"]
    if not os.path.isabs(obj_path):               # relative? make it absolute
        obj_path = os.path.join(out_root, obj_path)
    obj_path = os.path.abspath(os.path.expanduser(obj_path))
    # ----------------------------------------

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        render_one(argparse.Namespace(
            views=json.dumps(make_views(int(nviews))),
            object=obj_path,
            output_folder=os.path.join(out_root, "renders", rec["sha256"]),
            resolution=512,
            engine="CYCLES",
            geo_mode=False,
            save_depth=False, save_normal=False,
            save_albedo=False, save_mist=False,
            split_normal=False, save_mesh=True
        ))
