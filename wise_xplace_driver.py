"""WiseSyn-owned seam over xplace's placement flow (ADR-0024, Phase 1).

Isolates the wisesyn C++ embedding from xplace's argparse / output-path internals: one clean
``place()`` entry, one file to update if xplace's option set drifts. Runs xplace in its own repo root
(its thirdparty paths are relative). Multi-LEF (tech + std-cell) is handed over through xplace's
``custom_json`` mode — ``custom_path`` keys tokens by name and so silently drops all but the last
``lef:`` entry, which loses the tech LEF (no SITE/LAYER → a divide-by-zero in the parser).

The caller (the embedded interpreter, or a test) is responsible for putting torch's site-packages on
``sys.path``; this module only guarantees the xplace repo root is importable.
"""
import json
import os
import sys
import glob
import shutil
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def place(lef_paths, in_def, out_def, util, site="", seed=0, deterministic=True):
    """Run xplace GPU global placement on ``in_def`` and copy the placed DEF to ``out_def``.

    Returns a dict: {"ok": bool, "gp_hpwl": float, "dp_hpwl": float, "error": str}.
    Never raises across the boundary — any failure is reported in ["error"] with ok=False.
    """
    result = {"ok": False, "gp_hpwl": -1.0, "dp_hpwl": -1.0, "error": ""}
    prev_cwd = os.getcwd()
    prev_argv = list(sys.argv)
    # Absolutize every caller path against the ORIGINAL cwd before chdir — the caller (the wisesyn CLI)
    # may pass relative LEF/DEF paths, which would break once we chdir into the xplace tree.
    lef_paths = [os.path.abspath(p) for p in lef_paths]
    in_def = os.path.abspath(in_def)
    out_def = os.path.abspath(out_def)
    try:
        os.chdir(_HERE)                      # xplace uses relative ./thirdparty paths
        sys.argv = ["wise_xplace_driver"]    # get_option() parses argv → defaults only

        import torch
        if not torch.cuda.is_available():
            result["error"] = "torch.cuda.is_available() is False (no GPU)"
            return result

        from main import get_option
        from utils import setup_logger
        from src import run_placement_main

        # Multi-LEF via custom_json: xplace's loader reorders the tech LEF (name has 'tech'/'.tlef')
        # to the front itself, so pass the LEF list in the caller's order.
        outdir = tempfile.mkdtemp(prefix="wise_xplace_")
        cfg = {
            "benchmark": "wise",
            "design_name": "wise_top",
            "lefs": list(lef_paths),
            "def": in_def,
        }
        cfg_path = os.path.join(outdir, "design.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        args = get_option()
        args.custom_path = ""
        args.custom_json = cfg_path
        args.load_from_raw = True
        args.deterministic = bool(deterministic)
        args.seed = int(seed)
        if hasattr(args, "target_density"):
            args.target_density = float(util)
        args.result_dir = outdir
        args.exp_id = "wise"
        args.output_dir = "output"
        args.output_prefix = "placement"
        args.design_name = "wise_top"
        args.write_placement = True
        args.write_global_placement = False

        run_placement_main(args, setup_logger(args, sys.argv))

        # xplace writes {result_dir}/{exp_id}/{output_dir}/{prefix}_{design}_<id>.def; the detailed-
        # placement result carries the '_dp' id and is the newest .def. Glob + newest is robust to id.
        pattern = os.path.join(outdir, "wise", "output", "*wise_top*.def")
        defs = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not defs:
            result["error"] = "xplace wrote no placement DEF (looked for %s)" % pattern
            return result
        shutil.copyfile(defs[-1], out_def)
        result["ok"] = os.path.exists(out_def)
        if not result["ok"]:
            result["error"] = "failed to copy placed DEF to %s" % out_def
        return result
    except BaseException as e:               # never leak across the C++ boundary
        import traceback
        result["error"] = "xplace driver: %s\n%s" % (e, traceback.format_exc())
        return result
    finally:
        os.chdir(prev_cwd)
        sys.argv = prev_argv
