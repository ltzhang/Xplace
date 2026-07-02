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


def place(lef_paths, in_def, out_def, util, site="", seed=0, deterministic=True, route=False):
    """Run xplace GPU global placement on ``in_def`` and copy the placed DEF to ``out_def``.

    When ``route`` is true, also run xplace's GPU global router (GGR) on the produced placement and
    report congestion (ADR-0024 P4). Returns a dict:
        {"ok": bool, "gp_hpwl": float, "dp_hpwl": float, "route": str, "error": str}.
    Never raises across the boundary — any failure is reported in ["error"] with ok=False.
    """
    result = {"ok": False, "gp_hpwl": -1.0, "dp_hpwl": -1.0, "route": "", "error": ""}
    prev_cwd = os.getcwd()
    prev_argv = list(sys.argv)
    outdir = None
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
        from src.run_placement import run_placement_single

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
        if route:
            # Enable GGR: enable_route = use_route_force or use_cell_inflate; final_route_eval triggers
            # the post-placement global-route evaluation that returns congestion metrics + a guide.
            args.use_route_force = True
            args.final_route_eval = True
            # Load the FLUTE lookup tables that GGR's Steiner router needs. main.py does this in main()
            # via Flute.register(); we bypass main() and call run_placement_single directly, so without
            # this the shared libflute LUT is never read and GGR segfaults in flutes_LD during routing.
            # Use ABSOLUTE paths (not the relative defaults) so the load is independent of cwd.
            from src.core import Flute
            Flute.register(
                args.num_threads,
                os.path.join(_HERE, "thirdparty", "flute", "POWV9.dat"),
                os.path.join(_HERE, "thirdparty", "flute", "POST9.dat"),
            )
            # Routing stage: route the placement in `in_def` AS GIVEN — do not re-place it. The wisesyn
            # CLI has already placed (via arrays or naive) and writes that placement here; the router
            # must report congestion for THAT placement, not a fresh xplace GP. global_placement=False
            # legalizes the input coords then runs the final GGR eval on them.
            args.global_placement = False

        # run_placement_single returns (place_metrics, route_metrics); run_placement_main discards them.
        metrics = run_placement_single(args, setup_logger(args, sys.argv))
        place_metrics = metrics[0] if isinstance(metrics, (tuple, list)) else None
        route_metrics = metrics[1] if isinstance(metrics, (tuple, list)) and len(metrics) > 1 else None
        if place_metrics is not None:
            try:  # columns: dp_hpwl, gp_hpwl, top5overflow, overflow, ...
                result["dp_hpwl"] = float(place_metrics[0])
                result["gp_hpwl"] = float(place_metrics[1])
            except (IndexError, TypeError, ValueError):
                pass
        if route and route_metrics is not None:
            try:  # columns: #OvflNets, GR WL, GR #Vias, GR EstShort, RC Hor, RC Ver
                result["route"] = ("ovfl_nets=%s routed_wl=%s vias=%s est_short=%s"
                                   % (route_metrics[0], route_metrics[1], route_metrics[2],
                                      route_metrics[3]))
            except (IndexError, TypeError):
                result["route"] = "routed (metrics unavailable)"

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
        if outdir is not None:
            shutil.rmtree(outdir, ignore_errors=True)  # DEF already copied to out_def


# --------------------------------------------------------------------------------------------------
# ADR-0024 P2: TRUE in-memory array netlist ingest (no DEF on disk).
#
# ``build_design_info_from_arrays`` reconstructs, from plain per-node / per-pin / per-net arrays, the
# exact ``design_info`` dict that ``utils/io_parser.py:preprocess_design_info`` produces from the C++
# ``gpdb`` — same keys, dtypes, and CSR/index tensor layout. ``place_arrays`` then feeds it straight
# into ``PlaceData`` and runs the *same* Nesterov global-placement + legalization/DP pipeline the
# file-parser path uses (via the shared ``run_nesterov_placement_on_data`` helper), so a placement can
# run with no LEF/DEF and no ``gpdb`` at all.
# --------------------------------------------------------------------------------------------------

# The seven canonical node categories, in the order the C++ GPDatabase emits them (see
# cpp_to_py/io_parser/gp/GPDatabase.cpp). Node arrays are (stably) reordered into this segmentation so
# the movable/connected/fixed index ranges are contiguous, exactly as the parser lays them out.
_NODE_CATEGORIES = ("Mov", "FloatMov", "Fix", "IOPin", "Blkg", "FloatIOPin", "FloatFix")
_NODE_CATEGORY_ALIASES = {
    "mov": "Mov", "movable": "Mov", "conn_mov": "Mov", "connmov": "Mov",
    "floatmov": "FloatMov", "float_mov": "FloatMov", "unconn_mov": "FloatMov",
    "fix": "Fix", "fixed": "Fix", "conn_fix": "Fix", "confix": "Fix", "macro": "Fix",
    "iopin": "IOPin", "io": "IOPin", "conn_iopin": "IOPin",
    "blkg": "Blkg", "blockage": "Blkg", "blockage_node": "Blkg",
    "floatiopin": "FloatIOPin", "float_iopin": "FloatIOPin", "unconn_iopin": "FloatIOPin",
    "floatfix": "FloatFix", "float_fix": "FloatFix", "unconn_fix": "FloatFix",
}


def _canon_category(name):
    if name in _NODE_CATEGORIES:
        return name
    key = str(name).strip().lower()
    if key in _NODE_CATEGORY_ALIASES:
        return _NODE_CATEGORY_ALIASES[key]
    raise ValueError(
        "unknown node_type %r; expected one of %s (or a documented alias)"
        % (name, list(_NODE_CATEGORIES)))


def build_design_info_from_arrays(arrays):
    """Build a ``design_info`` dict (the ``PlaceData(**design_info)`` interface) from plain arrays.

    ``arrays`` keys (tensors may be python lists / numpy / torch; converted here):

    Required
      * ``node_size``       (N,2) float — cell width,height in DB units.
      * ``node_lpos``       (N,2) float — lower-left initial position in DB units.
      * ``node_type``       len-N — per-node category: one of ``_NODE_CATEGORIES`` or an alias
                            (``movable``/``fixed``/``iopin``/...). Nodes are reordered into the
                            canonical segmentation; coordinates are returned in *input* order.
      * ``pin_id2node_id``  (P,) int — parent node (in INPUT node order) of each pin.
      * ``pin_rel_cpos``    (P,2) float — pin center offset relative to its node center.
      * net connectivity, one of:
            ``nets``          : list of lists of pin ids (per net), OR
            ``net2pin_list`` + ``net2pin_end`` : CSR (pin ids grouped by net; cumulative end index).
      * ``die_info``        (lx,hx,ly,hy) — die/core box in DB units.
      * ``site_info``       (site_w, site_h) — site width, row height in DB units.
      * ``microns``         int — DB units per micron.

    Optional
      * ``pin_size``        (P,2) float — pin box size (default 0).
      * ``pin_rel_lpos``    (P,2) float — pin lower-left offset within node; derived from
                            ``pin_rel_cpos``/``pin_size``/parent ``node_size`` when absent.
      * ``node_names``      len-N — per-node instance names (default ``node_i``).
      * ``celltype_names``  len-N — per-node cell/master names (drives ``node_special_type``;
                            default = ``node_names``).
      * ``pin_names`` / ``net_names`` — for logging/timing only (defaults synthesized).
      * ``node_id2region_id`` (N,) int — default 0 for connected nodes, -1 otherwise (single region).
      * ``benchmark`` / ``design_name`` — metadata strings (defaults ``wise`` / ``wise_top``).

    Also returns, under the private key ``__input_perm__``, the permutation that maps canonical node
    order back to input order, so callers can un-permute the placed coordinates.
    """
    import torch

    def _t(x, dtype=torch.float32):
        if torch.is_tensor(x):
            return x.to(dtype).contiguous()
        return torch.as_tensor(x, dtype=dtype).contiguous()

    node_size_in = _t(arrays["node_size"]).reshape(-1, 2)
    node_lpos_in = _t(arrays["node_lpos"]).reshape(-1, 2)
    num_nodes = node_size_in.shape[0]
    if node_lpos_in.shape[0] != num_nodes:
        raise ValueError("node_lpos rows (%d) != node_size rows (%d)"
                         % (node_lpos_in.shape[0], num_nodes))

    node_type_in = list(arrays["node_type"])
    if len(node_type_in) != num_nodes:
        raise ValueError("node_type length (%d) != num_nodes (%d)"
                         % (len(node_type_in), num_nodes))
    ranks = torch.tensor([_NODE_CATEGORIES.index(_canon_category(t)) for t in node_type_in],
                         dtype=torch.int64)
    # Stable sort into canonical segmentation; preserves caller's within-category order.
    perm = torch.argsort(ranks, stable=True)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(num_nodes, dtype=torch.int64)

    node_size = node_size_in[perm].contiguous()
    node_lpos = node_lpos_in[perm].contiguous()
    node_pos = (node_lpos + node_size / 2.0).contiguous()  # centers, matches getNodeCPosTensor
    ranks_sorted = ranks[perm]

    def _reorder_names(key, default_prefix):
        if key in arrays and arrays[key] is not None:
            names = list(arrays[key])
            if len(names) != num_nodes:
                raise ValueError("%s length (%d) != num_nodes (%d)" % (key, len(names), num_nodes))
        else:
            names = ["%s%d" % (default_prefix, i) for i in range(num_nodes)]
        return [names[i] for i in perm.tolist()]

    node_id2node_name = _reorder_names("node_names", "node_")
    node_id2celltype_name = (
        _reorder_names("celltype_names", "cell_")
        if ("celltype_names" in arrays and arrays["celltype_names"] is not None)
        else list(node_id2node_name)
    )

    # ---- node_type_indices: all seven (start, end, name) segments, even empty ones. ----
    counts = [int((ranks_sorted == r).sum().item()) for r in range(len(_NODE_CATEGORIES))]
    node_type_indices = []
    seg_end = {}
    cursor = 0
    for r, name in enumerate(_NODE_CATEGORIES):
        start = cursor
        cursor += counts[r]
        node_type_indices.append((start, cursor, name))
        seg_end[name] = cursor
    mov_end_idx = seg_end["FloatMov"]
    connected_end_idx = seg_end["IOPin"]
    fix_end_idx = seg_end["FloatFix"]
    movable_index = (0, mov_end_idx)
    connected_index = (0, connected_end_idx)
    fixed_index = (mov_end_idx, fix_end_idx)

    # ---- pins (NOT reordered; only parent-node ids are remapped through the node permutation). ----
    pin_id2node_id_in = _t(arrays["pin_id2node_id"], torch.int64).reshape(-1)
    num_pins = pin_id2node_id_in.shape[0]
    if int(pin_id2node_id_in.min()) < 0 or int(pin_id2node_id_in.max()) >= num_nodes:
        raise ValueError("pin_id2node_id out of range [0,%d)" % num_nodes)
    pin_id2node_id = inv_perm[pin_id2node_id_in].contiguous()

    pin_rel_cpos = _t(arrays["pin_rel_cpos"]).reshape(-1, 2)
    if pin_rel_cpos.shape[0] != num_pins:
        raise ValueError("pin_rel_cpos rows (%d) != num_pins (%d)" % (pin_rel_cpos.shape[0], num_pins))
    if "pin_size" in arrays and arrays["pin_size"] is not None:
        pin_size = _t(arrays["pin_size"]).reshape(-1, 2)
    else:
        pin_size = torch.zeros((num_pins, 2), dtype=torch.float32)
    if "pin_rel_lpos" in arrays and arrays["pin_rel_lpos"] is not None:
        pin_rel_lpos = _t(arrays["pin_rel_lpos"]).reshape(-1, 2)
    else:
        # Invert getPinRelCPosTensor: rel_cpos = rel_lx + pinW/2 - nodeW/2  =>
        # rel_lpos(=rel_lx,rel_ly) = rel_cpos - pin_size/2 + parent_node_size/2.
        parent_node_size = node_size[pin_id2node_id]
        pin_rel_lpos = (pin_rel_cpos - pin_size / 2.0 + parent_node_size / 2.0).contiguous()

    # ---- net -> pin CSR + hyperedge index (replicates GPDatabase::getHyperedgeInfoTensor). ----
    if "net2pin_list" in arrays and arrays["net2pin_list"] is not None:
        hyperedge_list = _t(arrays["net2pin_list"], torch.int64).reshape(-1)
        hyperedge_list_end = _t(arrays["net2pin_end"], torch.int64).reshape(-1)
        num_nets = hyperedge_list_end.shape[0]
        start = torch.zeros(num_nets, dtype=torch.int64)
        start[1:] = hyperedge_list_end[:-1]
        net_of_pos = torch.repeat_interleave(
            torch.arange(num_nets, dtype=torch.int64), hyperedge_list_end - start)
    else:
        nets = arrays["nets"]
        num_nets = len(nets)
        flat, helper, ends, last = [], [], [], 0
        for net_id, pins in enumerate(nets):
            for pin_id in pins:
                flat.append(int(pin_id))
                helper.append(net_id)
            last += len(pins)
            ends.append(last)
        hyperedge_list = torch.tensor(flat, dtype=torch.int64)
        net_of_pos = torch.tensor(helper, dtype=torch.int64)
        hyperedge_list_end = torch.tensor(ends, dtype=torch.int64)
    if hyperedge_list.shape[0] != num_pins:
        raise ValueError("net->pin connectivity covers %d pin slots but there are %d pins "
                         "(every pin must belong to exactly one net)"
                         % (hyperedge_list.shape[0], num_pins))
    if int(hyperedge_list_end[-1].item()) != num_pins:
        raise ValueError("net2pin_end[-1] (%d) != num_pins (%d)"
                         % (int(hyperedge_list_end[-1].item()), num_pins))
    _hi = torch.stack([hyperedge_list, net_of_pos], dim=0)
    _order = torch.argsort(_hi[0], dim=0)   # sort by pin id (distinct) -> deterministic
    hyperedge_index = _hi.index_select(1, _order).contiguous()

    # ---- node -> pin CSR + index (replicates GPDatabase::getNode2PinInfoTensor). ----
    # Group pins by (canonical) node id; within a node keep pin-id ascending order, which is exactly
    # the parser's node.pins() insertion order (pin ids grow monotonically with net iteration).
    order_by_node = torch.argsort(pin_id2node_id * num_pins
                                  + torch.arange(num_pins, dtype=torch.int64), stable=True)
    node2pin_list = order_by_node.contiguous()
    node_of_pos = pin_id2node_id[order_by_node]
    node2pin_list_end = torch.zeros(num_nodes, dtype=torch.int64)
    counts_per_node = torch.bincount(pin_id2node_id, minlength=num_nodes)
    node2pin_list_end = torch.cumsum(counts_per_node, dim=0)
    _n2p = torch.stack([node2pin_list, node_of_pos], dim=0)
    _order2 = torch.argsort(_n2p[0], dim=0)
    node2pin_index = _n2p.index_select(1, _order2).contiguous()

    # ---- die / site / region (one region = the whole die). ----
    die_info = _t(arrays["die_info"]).reshape(-1)
    if die_info.numel() != 4:
        raise ValueError("die_info must have 4 elements (lx,hx,ly,hy)")
    site_w, site_h = arrays["site_info"]
    site_info = (float(site_w), float(site_h))
    region_boxes = die_info.reshape(1, 4).clone().contiguous()
    region_boxes_end = torch.tensor([1], dtype=torch.int64)
    if "node_id2region_id" in arrays and arrays["node_id2region_id"] is not None:
        node_id2region_id = inv_perm.new_zeros(num_nodes)
        src = _t(arrays["node_id2region_id"], torch.int64).reshape(-1)
        node_id2region_id[inv_perm] = src  # provided in input order -> canonical order
    else:
        # Connected nodes (Mov+FloatMov+Fix+IOPin) -> region 0; unconnected -> -1 (parser convention).
        node_id2region_id = torch.full((num_nodes,), -1, dtype=torch.int64)
        node_id2region_id[:connected_end_idx] = 0

    benchmark = arrays.get("benchmark", "wise")
    design_name = arrays.get("design_name", "wise_top")
    # dataset_path carries a "def" key so PlaceData tags the format as "lefdef" (matches the file
    # path); no file is read because load_from_raw is False and every write-back step is skipped.
    dataset_path = {"benchmark": benchmark, "design_name": design_name, "def": "<in-memory>"}

    net_names = list(arrays["net_names"]) if arrays.get("net_names") is not None \
        else ["net_%d" % i for i in range(num_nets)]
    pin_names = list(arrays["pin_names"]) if arrays.get("pin_names") is not None \
        else ["pin_%d" % i for i in range(num_pins)]
    node_names_out = node_id2node_name  # movable-vs-all distinction is cosmetic for our flow

    design_info = {
        "benchmark": benchmark,
        "dataset_path": dataset_path,
        "node_names": node_names_out,
        "net_names": net_names,
        "pin_names": pin_names,
        "microns": int(arrays["microns"]),
        "node_type_indices": node_type_indices,
        "node_id2node_name": node_id2node_name,
        "node_id2celltype_name": node_id2celltype_name,
        "movable_index": movable_index,
        "connected_index": connected_index,
        "fixed_index": fixed_index,
        "site_info": site_info,
        "die_info": die_info.float().contiguous(),
        "node_pos": node_pos.contiguous(),
        "node_lpos": node_lpos.contiguous(),
        "node_size": node_size.contiguous(),
        "pin_rel_cpos": pin_rel_cpos.contiguous(),
        "pin_rel_lpos": pin_rel_lpos.contiguous(),
        "pin_size": pin_size.contiguous(),
        "pin_id2node_id": pin_id2node_id.long().contiguous(),
        "hyperedge_index": hyperedge_index.long().contiguous(),
        "hyperedge_list": hyperedge_list.long().contiguous(),
        "hyperedge_list_end": hyperedge_list_end.long().contiguous(),
        "node2pin_index": node2pin_index.long().contiguous(),
        "node2pin_list": node2pin_list.long().contiguous(),
        "node2pin_list_end": node2pin_list_end.long().contiguous(),
        "node_id2region_id": node_id2region_id.long().contiguous(),
        "region_boxes": region_boxes.contiguous(),
        "region_boxes_end": region_boxes_end.long().contiguous(),
        "__input_perm__": perm,  # canonical index k came from input node perm[k]
    }
    return design_info


def place_arrays(arrays, util, seed=0, deterministic=True):
    """Run xplace global placement + legalization/DP on an in-memory array netlist (no DEF on disk).

    Builds ``design_info`` via :func:`build_design_info_from_arrays`, constructs ``PlaceData``, and
    runs the SAME Nesterov pipeline the file path uses (``run_nesterov_placement_on_data`` with
    ``rawdb=gpdb=None``). Returns:
        {"ok": bool, "coords": [[x,y], ...]  # placed lower-left, DB units, in INPUT node order
         "gp_hpwl": float, "error": str}
    Never raises across the boundary — any failure is reported in ["error"] with ok=False and no
    fabricated coordinates (rule #7).
    """
    result = {"ok": False, "coords": [], "gp_hpwl": -1.0, "error": ""}
    prev_cwd = os.getcwd()
    prev_argv = list(sys.argv)
    outdir = None
    try:
        os.chdir(_HERE)
        sys.argv = ["wise_xplace_driver"]

        import torch
        if not torch.cuda.is_available():
            result["error"] = "torch.cuda.is_available() is False (no GPU)"
            return result

        from main import get_option
        from utils import setup_logger
        from utils.tools import set_random_seed
        from src.database import PlaceData
        from src.run_placement_nesterov import run_nesterov_placement_on_data

        design_info = build_design_info_from_arrays(arrays)
        input_perm = design_info.pop("__input_perm__")

        outdir = tempfile.mkdtemp(prefix="wise_xplace_arr_")
        args = get_option()
        args.custom_path = ""
        args.custom_json = ""
        args.load_from_raw = False          # no file parse; skips every gpdb write-back path
        args.deterministic = bool(deterministic)
        args.seed = int(seed)
        if hasattr(args, "target_density"):
            args.target_density = float(util)
        args.result_dir = outdir
        args.exp_id = "wise"
        args.output_dir = "output"
        args.output_prefix = "placement"
        args.design_name = "wise_top"       # not in setup_design_args table -> keep default bins
        args.write_placement = False
        args.write_global_placement = False

        logger = setup_logger(args, sys.argv)

        data = PlaceData(args, logger, **design_info)
        # Match run_placement_single's RNG setup so the array path and the file path consume the
        # random stream identically (the C++ parser never touches torch's RNG).
        set_random_seed(args)
        node_pos, place_metrics, _route_metrics = run_nesterov_placement_on_data(
            data, None, None, args, logger, params=data.dataset_path
        )

        # gp_hpwl is place_metrics[1] (columns: dp_hpwl, gp_hpwl, top5overflow, overflow, ...).
        try:
            result["gp_hpwl"] = float(place_metrics[1])
        except (IndexError, TypeError, ValueError):
            pass

        # Convert placed centers -> integer lower-left DB units, exactly as commit_node_pos_to_gpdb
        # would before writing a DEF, then un-permute back to the caller's input node order.
        exact_node_cpos = torch.round(node_pos * data.die_scale + data.die_shift)
        exact_node_lpos = torch.round(
            exact_node_cpos - torch.round(data.node_size * data.die_scale) / 2
        ).cpu()
        coords_canon = exact_node_lpos
        coords_input = torch.empty_like(coords_canon)
        coords_input[input_perm] = coords_canon   # canonical row k -> input row perm[k]
        result["coords"] = coords_input.tolist()
        result["ok"] = True
        return result
    except BaseException as e:
        import traceback
        result["error"] = "xplace array driver: %s\n%s" % (e, traceback.format_exc())
        result["coords"] = []   # never leak partial/fabricated coordinates on failure
        result["ok"] = False
        return result
    finally:
        os.chdir(prev_cwd)
        sys.argv = prev_argv
        if outdir is not None:
            shutil.rmtree(outdir, ignore_errors=True)
