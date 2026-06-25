import logging
import torch
from cpp_to_py import gputimer, wirelength_timing_cuda
from src.param_scheduler import MetricRecorder
from utils import *

class GPUTimer():
    def __init__(self, data, rawdb, gpdb, params, args):
        self.metrics = [
            "wns",
            "tns",
        ]
        self.recorder = MetricRecorder(**{m: [] for m in self.metrics})
        
        self.data = data
        self.net_names = data.net_names
        self.pin_names = data.pin_names
        
        self.microns = data.microns
        self.wire_resistance_per_micron = args.wire_resistance_per_micron
        self.wire_capacitance_per_micron = args.wire_capacitance_per_micron

        self.node_size = data.node_size.detach().clone()
        node_lpos = data.node_pos.detach() - self.node_size / 2
        self.pin_rel_lpos = data.pin_rel_lpos.detach() + data.pin_size / 2

        die_info = data.die_info
        xl, xh, yl, yh = die_info.cpu().numpy()

        self.mov_lhs, self.mov_rhs = data.movable_index
        fix_lhs, fix_rhs = data.fixed_connected_index
        num_movable_nodes = self.mov_rhs - self.mov_lhs
        self.fix_conn_node_lpos = node_lpos[fix_lhs:fix_rhs]
        self.conn_node_lpos = torch.cat([
            node_lpos[self.mov_lhs:self.mov_rhs], self.fix_conn_node_lpos
        ], dim=0)

        scale_factor = 1.0 / data.site_width
        self.node_weight = data.node_special_type == 2
        
        self.timing_raw_db = gputimer.create_timing_rawdb(
            self.conn_node_lpos,
            data.node_size,
            self.pin_rel_lpos,
            data.pin_id2node_id,
            data.pin_id2net_id.int(),
            data.node2pin_list,
            data.node2pin_list_end,
            data.hyperedge_list.int(),
            data.hyperedge_list_end.int(),
            data.net_mask,
            num_movable_nodes,
            scale_factor,
            self.microns,
            self.wire_resistance_per_micron,
            self.wire_capacitance_per_micron
        )
        
        self.timer = gputimer.create_gputimer(params, rawdb, gpdb, self.timing_raw_db)
        
        ## Timing optimization
        self.timer.init()
        self.timer.levelize()
        self.pin_slack = torch.zeros(data.num_pins, dtype=torch.float32, device=data.device)
        self.timing_pin_weight = torch.ones(data.num_pins, dtype=torch.float32, device=data.device)
        self.history_x = None
        
        self.tns_record = []
        self.wns_record = []
        self.wns_max = []
        self.delay_K_max = []
        self.delay_1_max = []
        self.w_2 = None
        self.w_1 = None
        self.a_1 = None
        self.decay = args.decay_factor
        self.decay_boost = args.decay_boost
        self.init_alpha = 1.05
        self.beta = 0
        self.alpha = torch.ones(data.num_pins, dtype=torch.float32, device=data.device) * self.init_alpha
        self.global_weight = 1

        self.target_wns = 0

        # --- GangSTA signoff timer (Mode B differential comparison; --signoff_timer) ---
        # Reads the static design once and re-times under host-supplied parasitics so its WNS/TNS can be
        # logged beside the built-in GPU timer's. Entirely opt-in: when off (default) nothing here runs,
        # so the existing timing-driven flow is byte-for-byte unchanged.
        self.params = params
        self.signoff_mode = getattr(args, "signoff_timer", "gputimer")
        self.signoff_parasitics = getattr(args, "signoff_parasitics", "none")
        self.gangsta = None
        self.gangsta_err = ""
        if self.signoff_mode in ("gangsta", "both"):
            self._setup_gangsta_signoff(params)

    def update_timing(self, node_pos):
        node_lpos = (node_pos.detach() - self.node_size / 2).to(self.data.device)
        self.conn_node_lpos = torch.cat([
            node_lpos[self.mov_lhs:self.mov_rhs], self.fix_conn_node_lpos
        ], dim=0)
        
        self.timer.update_states()
        self.timer.update_rc(node_lpos, False, False, False)
        self.timer.update_timing()
    
    def update_timing_eval(self, node_pos):
        node_lpos = (node_pos.detach() - self.node_size / 2).to(self.data.device)
        self.conn_node_lpos = torch.cat([
            node_lpos[self.mov_lhs:self.mov_rhs], self.fix_conn_node_lpos
        ], dim=0)
        
        self.timer.update_states()
        self.timer.update_rc_flute(node_lpos, False)
        self.timer.update_timing()
    
    def update_timing_calibrated(self, node_pos, record=False):
        node_lpos = (node_pos.detach() - self.node_size / 2).to(self.data.device)
        self.conn_node_lpos = torch.cat([
            node_lpos[self.mov_lhs:self.mov_rhs], self.fix_conn_node_lpos
        ], dim=0)
        
        if record:
            self.timer.update_states()
            self.timer.update_rc_flute(node_lpos, True)
            self.timer.update_states()
            self.timer.update_rc(node_lpos, True, True, True)
        else:
            self.timer.update_states()
            self.timer.update_rc(node_lpos, False, True, True)
        self.timer.update_timing()
        
    def update_timing_spef(self):
        self.timer.update_states()
        self.timer.update_rc_spef()
        self.timer.update_timing()
        
    def report_timing_slack(self):
        time_unit = self.timer.time_unit()
        self.timer.update_endpoints()
        wns_early, tns_early, wns_late, tns_late = self.timer.report_wns_and_tns()
        wns_early = (wns_early.item() * (time_unit * 1e9))
        wns_late = (wns_late.item() * (time_unit * 1e9))
        tns_early = (tns_early.item() * (time_unit * 1e9))
        tns_late = (tns_late.item() * (time_unit * 1e9))
        self.push_metric(-wns_late, -tns_late)
        return wns_early, tns_early, wns_late, tns_late

    # ===== GangSTA signoff comparison (Mode B) =================================================
    def _setup_gangsta_signoff(self, params):
        log = logging.getLogger()
        if not gputimer.GangstaSignoff.available():
            self.gangsta_err = "Xplace was built without the gangsta library"
            log.warning("signoff_timer=%s requested but GangSTA is unavailable (%s); reporting "
                        "gputimer only" % (self.signoff_mode, self.gangsta_err))
            return
        need = ["verilog", "early_lib", "late_lib", "sdc"]
        missing = [k for k in need if not params.get(k)]
        if missing:
            self.gangsta_err = "design is missing inputs GangSTA needs: %s" % missing
            log.warning("signoff_timer=%s: %s; reporting gputimer only"
                        % (self.signoff_mode, self.gangsta_err))
            return
        g = gputimer.GangstaSignoff()
        ok = g.build(verilog=params["verilog"], early_lib=params["early_lib"],
                     late_lib=params["late_lib"], sdc=params["sdc"])
        if not ok:
            self.gangsta_err = g.error()
            log.warning("GangSTA build failed (%s); reporting gputimer only" % self.gangsta_err)
            return
        self.gangsta = g
        log.info("GangSTA signoff timer ready (parasitics=%s)" % self.signoff_parasitics)

    def _build_signoff_parasitics(self):
        # Returns the 9 CSR arrays gputimer.GangstaSignoff.report() expects.
        #   'none' -> empty: GangSTA uses its own lumped Liberty-cap model (the validated path).
        #   'load' -> match the GPU timer's per-net capacitive load. WIRE cap only (GangSTA adds each
        #             sink's Liberty pin cap itself, so we must not include pin caps -> no double count).
        #             EXPERIMENTAL: cap-unit alignment between the two engines is not yet calibrated, so
        #             absolute numbers may need a scale; the load *ratios* across nets are faithful.
        empty = ([], [], [0], [], [], [0], [], [], [])
        if self.signoff_parasitics != "load":
            return empty
        try:
            pin_load = self.timer.report_pin_load().detach().float().cpu()
            if pin_load.dim() > 1:
                pin_load = pin_load.reshape(pin_load.size(0), -1).max(dim=1).values
            pin_cap = self.timer.pin_capacitance().detach().float().cpu()
            he = self.data.hyperedge_list.long().cpu()
            he_end = self.data.hyperedge_list_end.long().cpu()
            net_mask = self.data.net_mask.cpu()
            net_names, net_total = [], []
            cap_start, cap_node, cap_val = [0], [], []
            res_start, res_a, res_b, res_val = [0], [], [], []
            start = 0
            for net_id in range(len(he_end)):
                end = int(he_end[net_id])
                pins = he[start:end].tolist()
                start = end
                if net_id >= len(net_mask) or not bool(net_mask[net_id]) or len(pins) == 0:
                    continue
                loads = pin_load[pins]
                driver_local = int(torch.argmax(loads))
                net_load = float(loads[driver_local])
                sink_lib = sum(float(pin_cap[p]) for i, p in enumerate(pins) if i != driver_local)
                wire_cap = max(net_load - sink_lib, 0.0)
                root = self.pin_names[pins[driver_local]]
                net_names.append(self.net_names[net_id])
                net_total.append(net_load)
                # Lump the net's wire cap at the driver node; GangSTA adds each sink's Liberty pin cap.
                cap_node.append(root)
                cap_val.append(wire_cap)
                cap_start.append(len(cap_val))
                # 0-ohm resistors root->each sink so the per-net RC graph is a connected tree (avoids a
                # "not a tree" warning); zero resistance => zero interconnect delay (matched-load mode).
                for i, p in enumerate(pins):
                    if i == driver_local:
                        continue
                    res_a.append(root)
                    res_b.append(self.pin_names[p])
                    res_val.append(0.0)
                res_start.append(len(res_val))
            return net_names, net_total, cap_start, cap_node, cap_val, res_start, res_a, res_b, res_val
        except Exception as e:  # never let signoff extraction disturb the placement loop
            logging.getLogger().warning("signoff_parasitics=load extraction failed (%s); using 'none'" % e)
            return empty

    def report_gangsta_signoff(self):
        """GangSTA WNS/TNS as a dict (raw gangsta time units), or None if unavailable. Never raises."""
        if self.gangsta is None:
            return None
        try:
            we, te, wl, tl, valid = self.gangsta.report(*self._build_signoff_parasitics())
            if not valid:
                return None
            return {"wns_early": we, "tns_early": te, "wns_late": wl, "tns_late": tl}
        except Exception as e:
            logging.getLogger().warning("GangSTA signoff report failed: %s" % e)
            return None

    def log_gangsta_signoff(self, logger):
        """Log GangSTA's WNS/TNS for side-by-side comparison with the GPU timer at a signoff milestone."""
        if self.signoff_mode not in ("gangsta", "both"):
            return
        gg = self.report_gangsta_signoff()
        if gg is None:
            logger.info("[signoff] gangsta : unavailable (%s)" % (self.gangsta_err or "no constrained endpoints"))
            return
        logger.info("[signoff] gangsta : late WNS/TNS: %.4f/%.4f | early WNS/TNS: %.4f/%.4f "
                    "(gangsta time units, parasitics=%s)"
                    % (gg["wns_late"], gg["tns_late"], gg["wns_early"], gg["tns_early"], self.signoff_parasitics))

    def report_pin_slack(self):
        self.pin_slack = self.timer.report_pin_slack()
        return self.pin_slack
    
    def report_path(self, ep_name=None, el = -1, verbose=False):
        if ep_name is not None:
            ep_idx = self.pin_names.index(ep_name)
            path, at, delay = self.timer.report_path(ep_idx, el, verbose)
        else:
            path, at, delay = self.timer.report_path(-1, el, verbose)
        return path, at, delay
    
    def report_arrival(self, pin_name):
        pin_idx = self.pin_names.index(pin_name)
        return self.timer.report_pin_at()[pin_idx]
    
    def report_slew(self, pin_name):
        pin_idx = self.pin_names.index(pin_name)
        return self.timer.report_pin_slew()[pin_idx]
    
    def report_load(self, pin_name):
        pin_idx = self.pin_names.index(pin_name)
        return self.timer.report_pin_load()[pin_idx]

    def report_required(self, pin_name):
        pin_idx = self.pin_names.index(pin_name)
        return self.timer.report_pin_rat()[pin_idx]
    
    def report_slack(self, pin_name):
        pin_idx = self.pin_names.index(pin_name)
        return self.timer.report_pin_slack()[pin_idx]

    def get_node_critocality(self):
        pin_slacks, _ = torch.min((torch.nan_to_num(self.report_pin_slack()) * (1e-9 / self.timer.time_unit())).clamp(max=0), 1)
        endpoints_index = self.timer.endpoints_index().long()
        endpoints_index = torch.unique(endpoints_index)
        ep_id2node_id = self.data.pin_id2node_id[endpoints_index]
        ep_slacks = pin_slacks[endpoints_index]
        node_slacks = torch.zeros(self.node_weight.size(0), dtype=torch.float32, device=self.data.device)
        node_slacks.scatter_add_(0, ep_id2node_id, ep_slacks)
        node_critocality = torch.abs(node_slacks) / (torch.abs(node_slacks)).max()
        return node_critocality

    def step(self, ps, node_pos, data):
        slacks, _ = torch.min(torch.nan_to_num(self.report_pin_slack()).clamp(max=0), 1)
        delay_k, _ = self.timer.report_criticality_threshold(0.75, False, True)
        delay_1, pin_visited = self.timer.report_criticality_threshold(0.99, False, True)
        
        self.tns_record.append(self.recorder.tns[-1])
        self.wns_record.append(self.recorder.wns[-1])
        self.wns_max.append(slacks.min().item())
        self.delay_K_max.append(delay_k.max().item())
        self.delay_1_max.append(delay_1.max().item())
        
        window = min(10, len(self.wns_max))
        x_wns = torch.tensor(self.wns_max[-window:], dtype=torch.float32)
        x_delay_k = torch.tensor(self.delay_K_max[-window:], dtype=torch.float32)
        x_delay_1 = torch.tensor(self.delay_1_max[-window:], dtype=torch.float32)

        wns_mean = x_wns[-1]
        delay_k_mean = x_delay_k[-1]
        delay_1_mean = x_delay_1[-1]
        
        pin_weight = slacks.abs() / (np.abs(wns_mean)) * self.beta
        pin_weight += (delay_k / delay_k_mean.clamp(min=1)) * self.beta * 2
        pin_weight += torch.pow(2, (delay_1 / delay_1_mean.clamp(min=1))) * pin_visited.clamp(max=1)
        
        w_0 = pin_weight
        delta_w_0 = None
        delta_w_1 = None
        if self.w_1 is not None:
            delta_w_0 = w_0 - self.w_1
        if self.w_2 is not None:
            delta_w_1 = self.w_1 - self.w_2

        if delta_w_0 is not None and delta_w_1 is not None:
            decay = (self.decay * torch.pow(5, delta_w_0.clamp(min=0)) / self.decay_boost).clamp(max=0.5)
        else:
            decay = 1
            
        self.w_2 = self.w_1
        self.w_1 = pin_weight.clone()
        if self.history_x is None:
            self.history_x = self.timing_pin_weight.clone()
        self.timing_pin_weight = pin_weight.clamp(min=ps.timing_wl_weight)
        self.timing_pin_weight = (decay * self.timing_pin_weight + (1 - decay) * self.history_x) * self.global_weight
        self.timing_pin_weight = self.timing_pin_weight.contiguous().to(node_pos.device)
        self.history_x = self.timing_pin_weight.clone()
        
    def push_metric(self, wns, tns):
        metrics_dict = {
            "wns": wns,
            "tns": tns,
        }
        self.recorder.push(**metrics_dict)
         
    def visualize(self, args):
        file_prefix = "%s_" % args.design_name
        res_root = os.path.join(args.result_dir, args.exp_id)
        prefix = os.path.join(res_root, args.eval_dir, file_prefix)
        if not os.path.exists(os.path.dirname(prefix)):
            os.makedirs(os.path.dirname(prefix))
        self.recorder.visualize(prefix, True)

def merged_wl_loss_grad_timing(
    node_pos,
    timing_pin_grads,
    pin_id2node_id,
    pin_rel_cpos,
    node2pin_list,
    node2pin_list_end,
    hyperedge_list,
    hyperedge_list_end,
    net_mask,
    net_weight,
    hpwl_scale,
    gamma,
    deterministic,
    cache_hpwl=True,
):
    (
        partial_wa_wl,
        node_grad,
        partial_hpwl,
    ) = wirelength_timing_cuda.merged_wl_loss_grad_timing(
        node_pos,
        timing_pin_grads,
        pin_id2node_id,
        pin_rel_cpos,
        node2pin_list,
        node2pin_list_end,
        hyperedge_list,
        hyperedge_list_end,
        net_mask,
        net_weight,
        hpwl_scale,
        gamma,
        deterministic,
    )
    return torch.sum(partial_wa_wl), node_grad
