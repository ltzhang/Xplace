# GangSTA signoff-timer integration (Mode B)

Xplace can report an **external** static-timing engine, [GangSTA](../../../..), alongside its
built-in GPU timer so the two can be compared on the same placement. This is the "Mode B"
differential-comparison path: GangSTA is linked into the `gputimer` module and fed the design
in-memory (no SPEF file round-trip).

## What it does

At each timing **signoff milestone** (after global placement and after detailed placement), with
`--signoff_timer gangsta|both`, Xplace runs GangSTA on the current design and logs its WNS/TNS next
to the GPU timer's:

```
early WNS/TNS: .../...  (ns) | late WNS/TNS: .../...  (ns)        <- gputimer
[signoff] gangsta : late WNS/TNS: .../... | early WNS/TNS: .../... (gangsta time units, parasitics=none)
```

GangSTA reads the **static** design once (`design.v` + `*_Early.lib` + `*_Late.lib` + `*.sdc` — the
ICCAD-2015 / TAU-2015 input set already used by the timing-driven flow) and, on each report, re-times
under host-supplied parasitics injected through the GangSTA C API
(`gangsta_set_parasitics_inmem`, byte-identical to reading the equivalent SPEF).

## Flags

| Flag | Values | Meaning |
|------|--------|---------|
| `--signoff_timer` | `gputimer` (default), `gangsta`, `both` | Which engine reports signoff WNS/TNS. `gputimer` is unchanged legacy behavior. |
| `--signoff_parasitics` | `none` (default), `load` | Parasitics handed to GangSTA. `none`: its own lumped Liberty-cap model (validated). `load`: match the GPU timer's per-net capacitive load (experimental). |

The feature is fully opt-in: with the default `--signoff_timer gputimer` nothing in this path runs, so
the existing timing-driven flow is unchanged.

## Build

GangSTA must be built **first** (it is linked as a static library):

```bash
# in the gangsta repo root
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --target gangsta -j        # produces build/src/libgangsta.a (PIC)
```

Then (re)configure Xplace. The gputimer CMake auto-detects `libgangsta.a` + `tcl` and prints
`GangSTA signoff timer ENABLED`; if not found it prints `DISABLED` and builds exactly as before.
Override the location with `-DGANGSTA_ROOT=/path/to/gangsta` if it is not the repo two levels above
Xplace. Requires `libtcl8.6` (GangSTA's SDC parser dependency).

## Parasitic fidelity (`--signoff_parasitics load`)

`load` mode matches GangSTA's per-net **capacitive load** to the GPU timer's (wire cap only — GangSTA
adds each sink's Liberty pin cap itself, so pin caps are excluded to avoid double-counting), with
zero-ohm resistors forming a connected per-net tree (zero interconnect *delay*). Per-net interconnect
delay (a non-degenerate RC tree from the GPU timer's `pinRootDelay`) is a planned refinement; the
GangSTA C API already accepts resistor segments, so this is a localized extension of
`GPUTimer._build_signoff_parasitics`.

Cap-unit alignment between the two engines is not yet calibrated, so absolute `load`-mode numbers may
need a scale factor; load *ratios* across nets are faithful. `none` mode has no such caveat.

## Validation status

Validated:
- GangSTA in-memory parasitics injection is byte-identical to reading a SPEF
  (`gangsta` repo: `parasitics_inject_test`, `c_api_parasitics_test`).
- The adapter (`gangsta_signoff.{h,cpp}`) builds a TAU-2015 design and reports stable WNS/TNS
  (standalone) and through the Python binding (`gputimer.GangstaSignoff`), including non-empty
  name-keyed parasitics taking effect.
- gputimer.so links GangSTA and loads in Python with the feature enabled.

Pending (requires the ICCAD-2015 dataset, not in the local corpus — `data/raw` is empty and the
ispd2015 designs ship no `*_Early/_Late.lib`/`.sdc`): a full `--timing_opt --signoff_timer both`
placement run on a real timing design, and `load`-mode cap-unit calibration.

## Mode A (GangSTA as the inner-loop timer) — status

Mode A would replace the GPU timer *inside* the gradient loop (drive `timing_pin_weight` from
GangSTA's slacks). The timer does **not** need to be differentiable — Xplace uses net-weighting, so
the timer is a non-differentiable oracle; `wirelength_timing_cuda` supplies the gradient.

All **gangsta-side primitives** for this are implemented and unit-validated (in the gangsta repo):

- per-iteration RC injection with a **lean RC-only refresh** — `gangsta_set_parasitics_inmem`
  (`parasitics_inject_test`);
- one-call **bulk readback** of per-pin slack/arrival/slew/required as `[num_pins,4]` —
  `gangsta_read_pin_timing` (`bulk_readback_test`);
- **pin-id↔name** map for aligning gangsta's pin order to Xplace's — `gangsta_num_pins` /
  `gangsta_pin_name`;
- WNS/TNS. Per-pin **criticality** is host-computable from the bulk slack array (no API needed) for
  a first inner loop; a path-visit criticality is a later refinement.

Remaining (host-side, **blocked on the ICCAD-2015 dataset** for validation): a Python timer mirroring
the `GPUTimer` socket (`update_timing` → inject RC + update; `report_pin_slack` → bulk readback;
`step` → criticality from slack) behind the `--signoff_timer` switch, plus the perf reality that a CPU
full re-time per iteration is far slower than the GPU timer on large designs (GangSTA's GPU backend is
correct but not yet fast — ADR-0014/0015). Building this gradient-loop replacement without a placeable
timing design to validate against would risk silently wrong placements, so it is staged here rather
than shipped unverified.
