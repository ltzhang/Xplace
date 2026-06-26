# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Xplace is a GPU-accelerated VLSI global placement framework built on PyTorch. The optimization core is Python (autograd-driven Nesterov placement), while performance-critical kernels (parsing, density maps, wirelength, DCT/FFT, detailed placement, global routing, timing) are C++/CUDA extensions compiled via PyBind11 and loaded as Python modules. Understanding the **Python optimization loop ↔ C++/CUDA kernel** split is essential to working here.

## Build

The C++/CUDA shared libraries **must** be built before running anything Python. They compile into `cpp_to_py/cpybin/` (gitignored) and are imported via `cpp_to_py/__init__.py`.

```bash
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=native -DPYTHON_EXECUTABLE=$(which python) ..
make -j40 && make install
```

- If `native` fails, pass your GPU compute capability explicitly, e.g. `-DCMAKE_CUDA_ARCHITECTURES=86` for RTX 3080/3090.
- Requires CMake >= 3.24, GCC >= 7.5, Boost >= 1.56, CUDA >= 11.3, PyTorch >= 1.12, Cairo. A CUDA-capable GPU is required at runtime (`torch.cuda.is_available()` is asserted).
- Clone with `--recursive`; third-party deps (pybind11, flute, lemon, lefdef, placers) live in `thirdparty/` as submodules/vendored code.

> **Verified toolchain (this workspace):** CUDA 13.2, PyTorch 2.12 (cu132, ABI=1), GCC 13, on an RTX 3080 (compute 8.6), using the venv at `/home/lintaoz/venv-cuda`. Building against CUDA 13 required two source changes that are already applied here: the build is bumped to **C++20** with `CMAKE_CXX_ABI` defaulting to **1** (match PyTorch ≥ 2.x), and `cpp_to_py/gpugr/taskflow/cuda/cuda_graph.hpp`/`cuda_device.hpp` were updated for CUDA 13's API (`cudaGraphGetEdges`/`cudaGraphAddDependencies` gained an `edgeData` argument; `clockRate`/`deviceOverlap`/`kernelExecTimeoutEnabled` were removed). NumPy 2.x also removed `np.round_`/`np.bool8`, which the mixed-size legalizer (`src/core/macro_legalization.py`) used — replaced with `np.round`/`np.bool_`. Verified end-to-end on GPU across `ispd2005/adaptec1`, `ispd2006/adaptec5`, `mms/adaptec1 --mixed_size`, `iccad2015/superblue1`, `iccad2019/ispd19_test1`.

### Adding a new C++/CUDA module
A module must be registered in **three** places: its own `cpp_to_py/<mod>/CMakeLists.txt`, the `add_subdirectory` list in `cpp_to_py/CMakeLists.txt`, and the import list in `cpp_to_py/__init__.py`.

## Running

There is no test suite; the project is exercised by running placement on benchmarks.

**Benchmark data (shared store).** Data is gitignored. In this workspace it lives in a single shared store one level up — `../benchmarks/<dataset>` — shared with the sibling **DREAMPlace** checkout so each dataset is downloaded only once. It is wired into `data/raw/` as per-dataset symlinks:

```
data/raw/ispd2005   -> ../../../benchmarks/ispd2005     # Bookshelf (.aux)
data/raw/ispd2006   -> ../../../benchmarks/ispd2006     # Bookshelf (.aux)
data/raw/mms        -> ../../../benchmarks/mms          # Bookshelf mixed-size (.aux)
data/raw/ispd2015   -> ../../../benchmarks/ispd2015     # LEF/DEF
data/raw/ispd2018   -> ../../../benchmarks/ispd2018     # LEF/DEF (routing)
data/raw/ispd2019   -> ../../../benchmarks/ispd2019     # LEF/DEF (routing)
data/raw/iccad2015  -> ../../../benchmarks/iccad2015    # LEF/DEF + timing (.lib/.sdc) — for --timing_opt
data/raw/iccad2019  -> ../../../benchmarks/iccad2019    # LEF/DEF (routing)
```

Available now: `ispd2005`, `ispd2006`, `mms`, `ispd2015`, `ispd2018`, `ispd2019`, `iccad2015`, `iccad2019`. `--dataset_root` defaults to `data/raw`, so the symlinks are picked up automatically. (`ispd2006`, `mms`, `iccad2015`, `iccad2019` were fetched 2026-06-26 from the refreshed OneDrive links upstream added to `data/README.md` — see the download note below.)

`data/download_data.sh` (synced with upstream 2026-06-26) is now **symlink-safe**: it no longer starts with `rm -rf raw/`, does `mkdir -p raw`, and **skips any `raw/<dataset>` that already exists** — so our shared-store symlinks (`ispd2005`, `ispd2015`, `ispd2018`, `ispd2019`) are left untouched. The OneDrive auto-download links are gone; instead the fresh OneDrive share links live in `data/README.md` (`ispd2005`, `ispd2006`, `mms`, `ispd2015`, `iccad2015`, `iccad2019`), and you place the `.tar.gz` files in `data/` before running the script (its skip-guards then extract them). The direct `ispd.cc` links (`ispd2018`, `ispd2019`) download non-interactively.

**Downloading the OneDrive sets headlessly** (no browser needed — the README's `1drv.ms/u/c/<driveid>/<TOKEN>?e=...` links do *not* resolve to a file via curl, but the underlying SharePoint endpoint does). Take the `IQ...` `<TOKEN>` from each link and fetch:
```bash
curl -A "Mozilla/5.0" -L \
  "https://my.microsoftpersonalcontent.com/personal/2c6e2bbdaffc31ad/_layouts/15/download.aspx?share=<TOKEN>" \
  -o <dataset>.tar.gz
```
(`2c6e2bbdaffc31ad` is the maintainer's drive id, also visible in the `1drv.ms` path.) This is how `ispd2006`/`mms`/`iccad2015`/`iccad2019` were fetched on 2026-06-26.

Caveats when running it here: (1) it runs `fix_ispd2015_route.py` unconditionally, generating the routability-only `ispd2015_fix` set (and `remove_fence_in_ispd19_test5.py` → `ispd2019_no_fence`) — both are pure Python, no Innovus needed; (2) the `ispd2018`/`ispd2019` blocks always re-`wget` from ispd.cc into `raw/ispd20xx/` (no skip guard), and since those are symlinks the data would be re-fetched into the shared store — comment out those blocks if you only need the OneDrive sets. To add just the missing sets, drop their tarballs in `data/` and the skip-guards handle the rest.

Main entry point is `main.py`. All behavior is controlled by CLI flags defined in `get_option()` (the authoritative parameter reference).

```bash
# single design
python main.py --dataset ispd2005 --design_name adaptec1
# whole dataset
python main.py --dataset ispd2005 --run_all True
# timing-driven
python main.py --dataset iccad2015 --design_name superblue4 --timing_opt True
# routability-driven
python main.py --dataset ispd2015_fix --run_all True --use_cell_inflate True
# mixed-size
python main.py --dataset mms --run_all True --mixed_size True
# place + GPU global route (GGR)
python main.py --dataset ispd2015_fix --run_all True --use_cell_inflate True --final_route_eval True
```

Outputs go to `result/<exp_id>/` (gitignored): `eval/` (curves + placement visualizations), `log/` (logs and CSV stats, including `run_all.csv` / `route.csv`), `output/` (placement DEF/solution).

- **Standalone timer**: configure design in `tool/timer.py` and run `python tool/timer.py`.
- **Custom designs**: `--custom_path lef:...,def:...,design_name:...,benchmark:...` or `--custom_json examples/examples.json`.

### Raw vs. pt mode
`--load_from_raw True` (default) parses LEF/DEF/bookshelf from scratch. For faster iteration on placement algorithms, preprocess designs into torch `.pt` files (`data/cad/`, gitignored) and use `--load_from_raw False`:
```bash
cd data && python convert_design_to_torch_data.py --dataset ispd2005
```
Caveats: always measure total runtime in raw mode; pt mode is **not** supported with routability-driven placement; new datasets for pt mode must be registered in `utils/get_design_params.py`.

### Determinism
Deterministic mode is on by default. For maximum speed when determinism isn't needed, pass `--deterministic False`.

## Architecture

### Flow control
`main.py` → `src/run_placement.py:run_placement_main` dispatches single vs. `--run_all`. Each design routes through `run_placement_main_nesterov` (`src/run_placement_nesterov.py`), the real driver:
1. `load_dataset` (`src/database.py`) parses via the C++ `IOParser` (or loads a `.pt`) into a `PlaceData` object — a PyG-style tensor container holding node positions/sizes, net/pin connectivity, and design metadata, all on GPU.
2. `global_placement_main` runs the Nesterov optimization loop.
3. Legalization → detail placement (`src/detail_placement.py`, backed by `gpudp`/`routedp` kernels) → evaluation.

### The optimization loop (`global_placement_main`)
This is the heart of Xplace and where most placement-algorithm work happens. The design intentionally separates concerns into pluggable modules — extend by swapping/adding these rather than editing the loop wholesale:
- **`NesterovOptimizer`** (`src/nesterov_optimizer.py`) — eplace-style Nesterov solver operating on a single `mov_node_pos` parameter tensor; gradients come from PyTorch autograd over the cost functions.
- **`ParamScheduler`** (`src/param_scheduler.py`) — dynamically adjusts density weight, learning rate, stopping (overflow), skip-update, sampling, etc. across iterations.
- **Cost functions** (`src/core/`) — `wa_wirelength_hpwl.py` (weighted-average wirelength), `electronic_density_layer.py` + `dct2_fft2.py`/`torch_dct.py` (electrostatic density via DCT/FFT), `route_force.py` (routability), `timing_opt.py` (timing). Each wraps a CUDA kernel from `cpp_to_py/` as an autograd `Function`.
- **Initialization / evaluation** — `src/initializer.py`, `src/evaluator.py`, `src/calculator.py`.

### Python ↔ C++/CUDA boundary
`cpp_to_py/` modules and their roles: `io_parser` (LEF/DEF/bookshelf parsing → database), `density_map_cuda` + `dct_cuda` (electrostatic density), `wa_wirelength_hpwl_cuda` + `hpwl_cuda` (wirelength), `gpudp`/`routedp` (legalization + detailed placement), `gpugr` (GGR GPU global router, see `cpp_to_py/gpugr/README.md`), `gputimer` + `wirelength_timing_cuda` (GPU static timing analysis), `flute_cpp` (RSMT), `draw_placement` (Cairo rendering), `common` (shared database/utilities). Each builds a `.so` in `cpp_to_py/cpybin/` exposed under matching names in `cpp_to_py/__init__.py`.

### Utilities (`utils/`)
`io_parser.py` (Python wrapper over the C++ parser), `get_design_params.py` (dataset/design path resolution — **edit here to register new datasets**), `setup_dataset.py`, `logger.py`, `visualization.py`, `tools.py` (incl. random seed setup).

## Conventions
- New placement parameters are added as flags in `main.py:get_option()` and threaded through `args`; there is no separate config system.
- Cost-function modules follow the pattern of subclassing `torch.autograd.Function` to bridge a CUDA forward/backward kernel into the autograd graph — mirror an existing one in `src/core/` when adding a new objective.

## GangSTA signoff timer (external STA comparison — `--signoff_timer`)

The [GangSTA](../../) STA engine is linked into the `gputimer` module so its WNS/TNS can be reported
**beside** Xplace's built-in GPU timer at signoff milestones (post-global-placement / post-detailed-
placement), for a side-by-side timer comparison on the same placement. GangSTA reads the static design
once (`.v` + `*_Early/_Late.lib` + `.sdc` — the iccad2015 inputs) and re-times under host-supplied
parasitics injected **in-memory** via the GangSTA C API (`gangsta_set_parasitics_inmem`, no SPEF file
round-trip).

- **Flags:** `--signoff_timer {gputimer|gangsta|both}` (default `gputimer` = unchanged legacy behavior;
  the GangSTA path is fully opt-in and never constructs the engine unless requested) and
  `--signoff_parasitics {none|load}`.
- **Code:** adapter `cpp_to_py/gputimer/core/gangsta_signoff.{h,cpp}` (C-API only, torch-free; safe
  no-op stub when built without GangSTA), bound in `PyBindCppMain.cpp` as `gputimer.GangstaSignoff`;
  Python wiring in `src/core/timing_opt.py` (`GPUTimer.log_gangsta_signoff`) and
  `src/run_placement_nesterov.py` (`timing_eval_func`).
- **Build:** GangSTA must be built first (`libgangsta.a`, PIC); the gputimer CMake auto-detects it
  (`GANGSTA_ROOT`, prints `ENABLED`/`DISABLED`) — a checkout without GangSTA builds exactly as before.
  Needs `libtcl8.6`.
- **✅ Validated end-to-end on ICCAD-2015 (`iccad2015/superblue4`).** A full
  `--timing_opt True --signoff_timer both --signoff_parasitics load` GPU placement run builds the
  GangSTA engine, reports WNS/TNS beside the GPU timer at the signoff milestone, and **never disturbs
  the existing gputimer flow** (gputimer numbers unaffected; the run completes normally). GangSTA
  `none`-mode late WNS = **−12241 ps**, identical to the standalone `gangsta run` CLI on the same
  `.v`/`.lib`/`.sdc` — confirming the in-memory netlist build + report path is correct at iccad2015
  scale (2.50M pins, ~8 s build).
- **Engine fix found via this integration:** on a no-parasitics design GangSTA's NLDM transition LUT
  used to *extrapolate* past `max_transition`, so long zero-load combinational chains diverged and
  setup WNS overflowed to a `-9e30` sentinel. Fixed in the gangsta repo (ADR-0019: clamp LUT lookups
  to table bounds; 214 golden tests still pass). After the fix, `superblue4` setup WNS is finite.
- **`load`-mode parasitics is experimental:** injecting Xplace's per-net loads makes
  `gangsta_report_wns_tns` return "no constrained endpoints" because Xplace's net/pin names don't all
  match GangSTA's Verilog-derived `inst:pin` names (a single correctly-named injection works — the
  bulk path needs name alignment). `none`-mode is the validated comparison path; the report failure
  now surfaces GangSTA's real error in the log (`report_gangsta_signoff`). Full details + the Mode-A
  (inner-loop timer replacement) plan: `cpp_to_py/gputimer/GANGSTA_SIGNOFF.md`.
