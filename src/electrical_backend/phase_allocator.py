"""
Phase allocation and balancing for LV networks.

- Derives required phases per LV transformer territory (BFS).
- Balances single‑phase loads (split‑phase or 3‑phase) by complex addition.
- Applies consistent bus suffixes and line phase labels.
- Validates connectivity/consistency and reports balance metrics.
"""

from __future__ import annotations

import itertools
import logging
import random
from collections import deque
from typing import Optional

from .component_specs import BusSpec, ComponentSpec, LineSpec, LoadSpec, TransformerSpec


class PhaseImbalanceError(Exception):
    """Raised when phase imbalance in a transformer territory exceeds threshold."""


class PhaseConsistencyError(Exception):
    """Raised when a line's available phases are insufficient for downstream loads."""


class PhaseAllocator:
    """Deterministic phase allocator per LV transformer.

    Balances single‑phase loads, assigns line/transformer phases, and enforces
    minimal phase counts needed by downstream loads.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        max_imbalance_pct: float = 50.0,
        raise_on_imbalance: bool = True,
        optimize_retries: int = 0,
        retry_threshold_pct: float | None = None,
        random_seed: int | None = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.max_imbalance_pct = max_imbalance_pct
        self.raise_on_imbalance = raise_on_imbalance
        self._phase_loads: dict[str, complex] = {"A": 0 + 0j, "B": 0 + 0j, "C": 0 + 0j}
        self._territory_reports: dict[str, dict[str, object]] = {}
        self.optimize_retries = int(optimize_retries) if optimize_retries else 0
        self.retry_threshold_pct = retry_threshold_pct
        self.random_seed = random_seed

    def allocate(self, specs: list[ComponentSpec]) -> list[ComponentSpec]:
        """Allocate phases, balance loads, suffix buses, and validate.

        Returns an updated list of specs with phases and suffixes applied.
        """
        idx = self._index(specs)

        # Log service type classification for debugging
        split_phase_count = sum(1 for tx in idx.lv_transformers if getattr(tx, "service_type", None) == "split_phase")
        three_phase_count = len(idx.lv_transformers) - split_phase_count

        self.logger.debug(
            f"Phase allocation: {len(idx.lv_transformers)} LV transformers - "
            f"{split_phase_count} split-phase (A,B), "
            f"{three_phase_count} three-phase (A,B,C)"
        )

        # MV-level: assign MV primary phases for LV transformers to balance
        # feeder
        self._assign_mv_primary_phases(idx)

        # Determine required phase counts on LV by transformer territory
        for tx in idx.lv_transformers:
            self._propagate_required_phases(idx, tx)

        # Assign and balance single phase loads per transformer
        for tx in idx.lv_transformers:
            self._balance_single_phase_territory(idx, tx)

        # Finalize bus and line suffixes and phase strings
        self._apply_suffixes(idx)

        # Validate connectivity before mirroring
        self._validate_load_connectivity(specs)

        # Quick fix: Duplicate L1 trunk lines to L2 for split-phase coverage
        specs = self._duplicate_trunks_for_l2(specs)

        # Validate global consistency and split-phase rules (where modeled)
        self._validate(idx)

        # Compute global phase totals for reporting
        self._compute_global_phase_loads(specs)

        # Log summary statistics
        self._log_allocation_summary()
        self._log_mv_vs_lv_summary(idx)

        return specs

    # ---------------
    # Indexing / Topology
    # ---------------
    class _Idx:
        def __init__(self):
            self.loads_by_vertex: dict[int, list[LoadSpec]] = {}
            self.lines_from: dict[int, list[LineSpec]] = {}
            self.lines_to: dict[int, list[LineSpec]] = {}
            self.buses_by_name: dict[str, BusSpec] = {}
            self.buses_by_vertex: dict[int, BusSpec] = {}
            self.transformers_by_vertex: dict[int, TransformerSpec] = {}
            self.lv_transformers: list[TransformerSpec] = []
            self.mv_lines: list[LineSpec] = []
            self.lv_lines: list[LineSpec] = []

    def _index(self, specs: list[ComponentSpec]) -> _Idx:
        """Build minimal indices (buses, lines, loads, transformers)."""
        idx = self._Idx()

        # Buses
        for s in specs:
            if isinstance(s, BusSpec):
                idx.buses_by_name[s.name] = s
                if s.vertex_id is not None:
                    idx.buses_by_vertex[s.vertex_id] = s

        # Lines
        for s in specs:
            if isinstance(s, LineSpec):
                if s.from_vertex_id is not None:
                    idx.lines_from.setdefault(s.from_vertex_id, []).append(s)
                if s.to_vertex_id is not None:
                    idx.lines_to.setdefault(s.to_vertex_id, []).append(s)
                # Canonical integer phases for internal logic
                if not hasattr(s, "phases_int") or not isinstance(s.phases_int, int):
                    s.phases_int = 3

        # Loads
        for s in specs:
            if isinstance(s, LoadSpec) and getattr(s, "vertex_id", None) is not None:
                idx.loads_by_vertex.setdefault(s.vertex_id, []).append(s)

        # Transformers and classification
        for s in specs:
            if isinstance(s, TransformerSpec) and getattr(s, "vertex_id", None) is not None:
                idx.transformers_by_vertex[s.vertex_id] = s
                eq = getattr(s, "equipment", None)
                sec_kv = getattr(eq, "secondary_voltage_kv", None)
                if sec_kv is not None and sec_kv < 1.0:
                    # LV transformer
                    # Service type: split_phase for ~0.24kV, otherwise
                    # three_phase
                    if sec_kv <= 0.30 or getattr(eq, "n_phases", 1) == 1:
                        s.service_type = "split_phase"
                    else:
                        s.service_type = "three_phase"
                    idx.lv_transformers.append(s)
                else:
                    # MV/HV transformer; leave service_type unset
                    pass

        # Classify MV/LV lines (with robust fallbacks)
        for s in specs:
            if isinstance(s, LineSpec):
                if self._is_mv_line(idx, s):
                    idx.mv_lines.append(s)
                else:
                    idx.lv_lines.append(s)

        return idx

    def _is_mv_line(self, idx: _Idx, line: LineSpec) -> bool:
        """Classify a line as MV based on cable equipment metadata."""
        return line.cable_equipment.voltage_level == "MV"

    # ---------------
    # MV Primary Phase Assignment (Feeder-level balancing)
    # ---------------
    def _assign_mv_primary_phases(self, idx: _Idx) -> None:
        """Assign MV primary phase (A/B/C) to balance feeder loading.

        Split‑phase LV transformers are placed on the lightest MV phase by kVA;
        three‑phase LV transformers contribute equally to A/B/C.
        """
        mv_totals = {"A": 0.0, "B": 0.0, "C": 0.0}

        def _tx_size_kva(tx: TransformerSpec) -> float:
            eq = getattr(tx, "equipment", None)
            if eq is not None:
                s = getattr(eq, "s_max_kva", None)
                if s is not None:
                    try:
                        return float(s)
                    except Exception:
                        pass
            # fallback to tx.kva if set
            kva_attr = getattr(tx, "kva", None)
            if kva_attr is not None:
                try:
                    return float(kva_attr)
                except Exception:
                    pass
            # default minimal
            return 25.0

        # First account for three-phase LV transformers equally
        for tx in idx.lv_transformers:
            service = getattr(tx, "service_type", None) or "split_phase"
            size_kva = _tx_size_kva(tx)
            if service != "split_phase":
                share = size_kva / 3.0
                mv_totals["A"] += share
                mv_totals["B"] += share
                mv_totals["C"] += share
                tx.primary_phases = "ABC"

        # Then assign split-phase units greedily to the lightest MV phase
        split_txs = []
        for tx in idx.lv_transformers:
            if getattr(tx, "service_type", None) == "split_phase":
                size_kva = _tx_size_kva(tx)
                split_txs.append((size_kva, tx))

        # Largest first
        split_txs.sort(key=lambda t: t[0], reverse=True)
        for size_kva, tx in split_txs:
            # Choose lightest MV phase
            phase = min(mv_totals, key=mv_totals.get)
            mv_totals[phase] += size_kva
            tx.primary_phases = phase

        # Persist MV balance report
        self._mv_phase_totals = mv_totals
        self._mv_imbalance_pct = self._compute_imbalance_pct(list(mv_totals.values()))

    def _compute_imbalance_pct(self, mags: list[float]) -> float:
        if not mags:
            return 0.0
        total = sum(mags)
        if total == 0.0:
            return 0.0
        avg = total / len(mags)
        max_dev = max(abs(m - avg) for m in mags)
        return 100.0 * max_dev / avg

    def get_mv_balance(self) -> dict[str, object]:
        """Get MV-phase balance report: totals per phase and imbalance percentage."""
        totals = getattr(self, "_mv_phase_totals", {"A": 0.0, "B": 0.0, "C": 0.0})
        imb = getattr(self, "_mv_imbalance_pct", 0.0)
        return {"totals": dict(totals), "imbalance_pct": float(imb)}

    # ---------------
    # Propagation
    # ---------------
    def _propagate_required_phases(self, idx: _Idx, tx: TransformerSpec) -> None:
        """Derive required phase count per vertex and set LV line phases.

        Builds the LV territory from the transformer via BFS, computes required
        phase counts from attached loads, propagates upstream, and sets
        `phases_int` on LV trunks accordingly.
        """
        start_v = tx.vertex_id
        # Build LV territory reachable subgraph from transformer
        territory_vertices: set[int] = set()
        children: dict[int, list[int]] = {}
        parents: dict[int, list[int]] = {}

        q: deque[int] = deque([start_v])
        while q:
            v = q.popleft()
            if v in territory_vertices:
                continue
            territory_vertices.add(v)
            for ln in idx.lines_from.get(v, []):
                if ln in idx.lv_lines and getattr(ln, "to_vertex_id", None) is not None:
                    w = ln.to_vertex_id
                    children.setdefault(v, []).append(w)
                    parents.setdefault(w, []).append(v)
                    if w not in territory_vertices:
                        q.append(w)

        # Initialize required phases at each vertex from direct loads
        required_at_vertex: dict[int, int] = {}
        for v in territory_vertices:
            req = 1
            for ld in idx.loads_by_vertex.get(v, []):
                # Heuristic: large or non-residential loads are 3-phase
                nph = getattr(ld, "n_phases", None)
                if not isinstance(nph, int):
                    if (
                        getattr(ld, "load_type", "residential").lower() == "residential"
                        and float(getattr(ld, "kw", 0.0)) <= 10.0
                    ):
                        nph = 1
                    else:
                        nph = 3
                req = max(req, int(nph))
            required_at_vertex[v] = req

        # Post-order accumulation: propagate children's requirements to parents
        # Compute processing order via reverse BFS layering
        order: list[int] = []
        seen: set[int] = set()
        dq: deque[int] = deque([start_v])
        while dq:
            x = dq.popleft()
            if x in seen:
                continue
            seen.add(x)
            order.append(x)
            for y in children.get(x, []):
                if y not in seen:
                    dq.append(y)
        # Process in reverse so children first
        for v in reversed(order):
            for w in children.get(v, []):
                required_at_vertex[v] = max(required_at_vertex[v], required_at_vertex[w])

        # Set LV line phases to exactly satisfy downstream requirement (minimization);
        # lines will be upgraded upstream where children require more phases.
        for v in territory_vertices:
            for ln in idx.lines_from.get(v, []):
                if ln in idx.lv_lines and getattr(ln, "to_vertex_id", None) is not None:
                    req = required_at_vertex.get(ln.to_vertex_id, 1)
                    ln.phases_int = int(req)

        # Store per-transformer territory map for later balancing and
        # validation
        tx._territory_vertices = territory_vertices
        tx.required_at_vertex = required_at_vertex

    # ---------------
    # Balancing
    # ---------------
    def _balance_single_phase_territory(self, idx: _Idx, tx: TransformerSpec) -> None:
        """Greedy balance of single‑phase loads within one LV territory."""
        territory_vertices: set[int] = getattr(tx, "_territory_vertices", set())
        if not territory_vertices:
            return

        # Gather loads in this territory
        single_phase_items: list[tuple[float, LoadSpec, complex]] = []
        three_phase_total: complex = 0 + 0j
        for v in territory_vertices:
            for ld in idx.loads_by_vertex.get(v, []):
                # Decide load phase count: keep any preset value; else infer
                # from type/size
                nph = getattr(ld, "n_phases", None)
                if not isinstance(nph, int):
                    if (
                        getattr(ld, "load_type", "residential").lower() == "residential"
                        and float(getattr(ld, "kw", 0.0)) <= 10.0
                    ):
                        ld.n_phases = 1
                    else:
                        ld.n_phases = 3
                kw = float(getattr(ld, "kw", 0.0))
                kvar = float(getattr(ld, "kvar", 0.0))
                if int(ld.n_phases) == 1:
                    S = complex(kw, kvar)
                    single_phase_items.append((abs(S), ld, S))
                elif int(ld.n_phases) >= 3:
                    three_phase_total += complex(kw, kvar)

        if not single_phase_items:
            self.logger.debug(
                f"Territory {
                    getattr(
                        tx,
                        'name',
                        '?')}: No single-phase loads to balance"
            )
            return

        # Get service type and sort loads
        service = getattr(tx, "service_type", "split_phase")
        single_phase_items.sort(key=lambda t: t[0], reverse=True)

        # Debug: Log territory info
        tx_name = getattr(tx, "name", "?")
        self.logger.debug(
            f"Territory {tx_name}: service={service}, "
            f"single_phase_loads={len(single_phase_items)}, "
            f"three_phase_total={three_phase_total}"
        )

        # Helper: run one greedy attempt with specified phase order
        def run_attempt(phase_order: list[str]):
            if service == "split_phase":
                # Use L1, L2 for split-phase legs (NOT MV phases A, B)
                phases_local = phase_order[:2]
                totals_local = {"L1": 0 + 0j, "L2": 0 + 0j}
            else:
                # Use A, B, C for true three-phase service
                phases_local = phase_order[:3]
                per_phase_share = (three_phase_total / 3) if three_phase_total != 0 else 0 + 0j
                totals_local = {
                    "A": per_phase_share,
                    "B": per_phase_share,
                    "C": per_phase_share,
                }

            assign_list: list[tuple[LoadSpec, str]] = []
            for _, ld_, S_ in single_phase_items:
                # choose the phase with minimal |sum|, tie-broken by
                # phase_order
                best_phase = min(phases_local, key=lambda p: abs(totals_local[p]))
                assign_list.append((ld_, best_phase))
                totals_local[best_phase] += S_

            mags_local = [abs(totals_local[p]) for p in phases_local]
            if mags_local:
                if three_phase_total == 0 and any(m == 0 for m in mags_local):
                    considered = [m for m in mags_local if m > 1e-9]
                    if len(considered) <= 1:
                        pct_local = 100.0
                    else:
                        avg_local = sum(considered) / len(considered)
                        max_dev_local = max(abs(m - avg_local) for m in considered)
                        pct_local = 100.0 * max_dev_local / avg_local if avg_local > 0 else 0.0
                else:
                    avg_local = sum(mags_local) / len(phases_local)
                    max_dev_local = max(abs(m - avg_local) for m in mags_local)
                    pct_local = 100.0 * max_dev_local / avg_local if avg_local > 0 else 0.0
            else:
                pct_local = 0.0

            return assign_list, totals_local, pct_local

        # Base attempt with default phase ordering
        base_order = ["L1", "L2"] if service == "split_phase" else ["A", "B", "C"]
        best_assign, best_totals, best_pct = run_attempt(base_order)

        threshold = (
            float(self.retry_threshold_pct) if self.retry_threshold_pct is not None else float(self.max_imbalance_pct)
        )

        # Retry with multiple phase orderings if above threshold
        if best_pct > threshold and self.optimize_retries > 0:
            rng = random.Random(
                self.random_seed if self.random_seed is not None else (hash(getattr(tx, "name", "")) & 0xFFFFFFFF)
            )
            # Generate candidate phase orderings
            if service == "split_phase":
                candidates = [list(p) for p in itertools.permutations(["L1", "L2"], 2)]
            else:
                all_perms = [list(p) for p in itertools.permutations(["A", "B", "C"], 3)]
                rng.shuffle(all_perms)
                candidates = all_perms

            # Ensure base order is first, then try additional ones
            try_orders = [base_order] + [o for o in candidates if o != base_order]
            tries = min(1 + self.optimize_retries, len(try_orders))

            for order in try_orders[:tries]:
                assign_map, totals_try, pct_try = run_attempt(order)
                if pct_try < best_pct:
                    best_pct = pct_try
                    best_totals = totals_try
                    best_assign = assign_map

        # Apply best assignment
        for ld, ph in best_assign:
            ld.phase = ph

        # Debug: Log final assignment
        if service == "split_phase":
            phase_counts = {"L1": 0, "L2": 0}
        else:
            phase_counts = {"A": 0, "B": 0, "C": 0}

        for _ld, ph in best_assign:
            if ph in phase_counts:
                phase_counts[ph] += 1

        if service == "split_phase":
            self.logger.debug(
                f"Territory {tx_name}: Final assignment - "
                f"Leg 1: {phase_counts['L1']} loads, "
                f"Leg 2: {phase_counts['L2']} loads"
            )
        else:
            self.logger.debug(
                f"Territory {tx_name}: Final assignment - "
                f"Phase A: {phase_counts['A']} loads, "
                f"Phase B: {phase_counts['B']} loads, "
                f"Phase C: {phase_counts['C']} loads"
            )

        # Persist per-territory report
        self._territory_reports[getattr(tx, "name", str(tx))] = {
            "service": service,
            "totals": best_totals.copy(),
            "imbalance_pct": best_pct,
            "n_single_phase_loads": len(single_phase_items),
            "phase_counts": phase_counts.copy(),
        }

        # Special handling for single-load territories
        if len(single_phase_items) == 1:
            self.logger.debug(
                f"Territory {getattr(tx, 'name', '?')}: " f"Single load territory - 100% imbalance is expected"
            )
        elif best_pct > float(self.max_imbalance_pct):
            msg = (
                f"Territory around {getattr(tx, 'name', '?')} ({service}): "
                f"{len(single_phase_items)} loads, imbalance {best_pct:.1f}%, "
                f"totals {best_totals}"
            )
            if self.raise_on_imbalance:
                raise PhaseImbalanceError(msg)
            else:
                self.logger.warning(msg)

    # ---------------
    # Export / Suffixing (Pass D)
    # ---------------
    def _apply_suffixes(self, idx: _Idx) -> None:
        """Apply phase suffixes to loads, transformers, and 1‑phase lines."""
        # Suffix mapping for MV phases and split-phase legs
        mv_suffix = {"A": ".1", "B": ".2", "C": ".3"}
        lv_suffix = {"L1": ".1", "L2": ".2"}
        all_suffixes = {**mv_suffix, **lv_suffix}

        # Loads: single-phase wye bus suffix
        for _v, loads in idx.loads_by_vertex.items():
            for ld in loads:
                if int(getattr(ld, "n_phases", 1)) == 1 and getattr(ld, "conn", "wye") == "wye":
                    ph = getattr(ld, "phase", None)
                    if ph in all_suffixes and not any(ld.bus.endswith(suf) for suf in all_suffixes.values()):
                        ld.bus = ld.bus + all_suffixes[ph]

        # Transformers: MV primary bus suffix for split-phase units with
        # assigned primary phase (use MV suffix for primary side)
        for tx in idx.lv_transformers:
            primary_phase = getattr(tx, "primary_phases", None)
            if primary_phase in mv_suffix and getattr(tx, "bus1", None):
                # Avoid double suffixing
                base_bus = tx.bus1
                for s in mv_suffix.values():
                    if base_bus.endswith(s):
                        base_bus = base_bus[: -len(s)]
                        break
                tx.bus1 = base_bus + mv_suffix[primary_phase]

        # Lines: set display phases and suffix endpoints for single-phase lines
        service_line_phase_updates = {"L1": 0, "L2": 0, "A": 0, "B": 0, "C": 0}
        for ln in idx.lv_lines:
            if int(getattr(ln, "phases_int", 3)) == 1:
                # Choose phase by dominant load at downstream vertex
                chosen = self._choose_line_phase(idx, ln)
                ln.phases = chosen
                # Suffix both endpoints consistently using appropriate suffix
                # mapping
                ln.bus1 = self._with_phase_suffix(ln.bus1, chosen, all_suffixes)
                ln.bus2 = self._with_phase_suffix(ln.bus2, chosen, all_suffixes)

                # Track service line phase updates
                line_name = getattr(ln, "name", "")
                if "Consumer" in line_name or "Service" in line_name:
                    if chosen in service_line_phase_updates:
                        service_line_phase_updates[chosen] += 1
            else:
                ln.phases = "ABC"

        # Log service line phase distribution
        total_service = sum(service_line_phase_updates.values())
        if total_service > 0:
            self.logger.info(
                f"Service line phases: {service_line_phase_updates['L1']} L1, "
                f"{service_line_phase_updates['L2']} L2, "
                f"{service_line_phase_updates['A']} A, {service_line_phase_updates['B']} B, "
                f"{service_line_phase_updates['C']} C (total: {total_service})"
            )

    def _with_phase_suffix(self, bus_name: str, phase: str, suffix_map: dict[str, str]) -> str:
        """Return `bus_name` with exactly one matching phase suffix applied."""
        # Remove existing suffix if present then add
        for s in suffix_map.values():
            if bus_name.endswith(s):
                bus_name = bus_name[: -len(s)]
                break
        return bus_name + suffix_map.get(phase, "")

    def _choose_line_phase(self, idx: _Idx, ln: LineSpec) -> str:
        """Choose line phase by dominant downstream single‑phase load magnitude."""
        v = getattr(ln, "to_vertex_id", None)
        if v is None:
            return "L1"  # Default to L1 for split-phase

        # Count loads by phase (handles both MV phases A,B,C and split-phase
        # legs L1,L2)
        phase_tot = {"A": 0.0, "B": 0.0, "C": 0.0, "L1": 0.0, "L2": 0.0}
        for ld in idx.loads_by_vertex.get(v, []):
            ph = getattr(ld, "phase", None)
            if ph in phase_tot:
                S = complex(float(getattr(ld, "kw", 0.0)), float(getattr(ld, "kvar", 0.0)))
                phase_tot[ph] += abs(S)

        # Remove unused phases
        used_phases = {k: v for k, v in phase_tot.items() if v > 0.0}
        if not used_phases:
            return "L1"  # Default to L1 for split-phase

        return max(used_phases, key=used_phases.get)

    def _validate_load_connectivity(self, specs: list[ComponentSpec]) -> None:
        """Report loads not connected (via lines) to any transformer bus."""
        from .component_specs import LineSpec, LoadSpec, TransformerSpec

        # Build connectivity graph from lines
        connections = {}  # bus_name -> set of connected bus_names
        for spec in specs:
            if isinstance(spec, LineSpec):
                bus1 = getattr(spec, "bus1", "")
                bus2 = getattr(spec, "bus2", "")
                if bus1 and bus2:
                    connections.setdefault(bus1, set()).add(bus2)
                    connections.setdefault(bus2, set()).add(bus1)

        # Find transformer buses - distinguish MV (3φ unsuffixed) vs LV (split-phase .1/.2)
        mv_transformer_buses = set()  # MV: unsuffixed
        lv_transformer_buses = set()  # LV: .1 and .2 suffixed
        for spec in specs:
            if isinstance(spec, TransformerSpec):
                bus2 = getattr(spec, "bus2", "")  # Secondary side
                if not bus2:
                    continue

                # Classify as MV or LV based on voltage
                eq = getattr(spec, "equipment", None)
                sec_kv = getattr(eq, "secondary_voltage_kv", None)
                is_lv = sec_kv is not None and sec_kv < 1.0  # LV < 1kV

                if is_lv:
                    # LV split-phase: use .1 and .2 terminals, plus unsuffixed as fallback
                    lv_transformer_buses.add(f"{bus2}.1")
                    lv_transformer_buses.add(f"{bus2}.2")
                    lv_transformer_buses.add(bus2)  # Fallback for unsuffixed lines
                else:
                    # MV 3-phase: use unsuffixed bus
                    mv_transformer_buses.add(bus2)

        # Check each load for connectivity
        orphaned_loads = []
        orphan_diagnostics = []  # For detailed logging
        for spec in specs:
            if isinstance(spec, LoadSpec):
                load_bus = getattr(spec, "bus", "")
                load_name = getattr(spec, "name", "unknown")
                phase = getattr(spec, "phase", "unknown")

                if not load_bus:
                    continue

                # BFS to check connectivity
                reachable = self._bfs_reachable(load_bus, connections)

                # Check connectivity based on load phase
                if phase in ["L1", "L2", "A", "B", "C"]:
                    # LV load: check split-phase terminals
                    if phase == "L1":
                        target_terminals = [
                            t for t in lv_transformer_buses if t.endswith(".1") or not t.endswith((".1", ".2"))
                        ]
                    elif phase == "L2":
                        target_terminals = [
                            t for t in lv_transformer_buses if t.endswith(".2") or not t.endswith((".1", ".2"))
                        ]
                    else:
                        # Single-phase MV or 3-phase
                        target_terminals = list(lv_transformer_buses) + list(mv_transformer_buses)
                    connected = any(terminal in reachable for terminal in target_terminals)
                else:
                    # Unknown phase or 3-phase: check both MV and LV
                    all_transformer_buses = mv_transformer_buses | lv_transformer_buses
                    connected = any(bus in reachable for bus in all_transformer_buses)

                if not connected:
                    orphaned_loads.append(f"{load_name} (phase {phase}, bus {load_bus})")
                    # Diagnostic info for first few orphans
                    if len(orphan_diagnostics) < 5:
                        orphan_diagnostics.append(
                            {
                                "name": load_name,
                                "phase": phase,
                                "bus": load_bus,
                                "reachable_count": len(reachable),
                                "can_reach_mv": any(bus in reachable for bus in mv_transformer_buses),
                                "can_reach_lv": any(bus in reachable for bus in lv_transformer_buses),
                            }
                        )

        if orphaned_loads:
            self.logger.warning(f"Found {len(orphaned_loads)} disconnected loads:")
            # Log diagnostic info for first few orphans
            if orphan_diagnostics:
                self.logger.debug("Diagnostic info for first orphans:")
                for diag in orphan_diagnostics:
                    self.logger.debug(
                        f"  {diag['name']}: phase={diag['phase']}, bus={diag['bus']}, "
                        f"reachable_buses={diag['reachable_count']}, "
                        f"can_reach_mv_tx={diag['can_reach_mv']}, "
                        f"can_reach_lv_tx={diag['can_reach_lv']}"
                    )
            # Log all orphans
            for load in orphaned_loads[:20]:  # Limit output
                self.logger.warning(f"  - {load}")
            if len(orphaned_loads) > 20:
                self.logger.warning(f"  ... and {len(orphaned_loads) - 20} more")
        else:
            self.logger.info("All loads are connected to transformers")

    def _bfs_reachable(self, start_bus: str, connections: dict) -> set:
        """BFS to find all buses reachable from start_bus."""
        reachable = set()
        to_visit = [start_bus]

        while to_visit:
            current = to_visit.pop(0)
            if current in reachable:
                continue

            reachable.add(current)

            # Add all connected buses
            for connected_bus in connections.get(current, set()):
                if connected_bus not in reachable:
                    to_visit.append(connected_bus)

        return reachable

    def _duplicate_trunks_for_l2(self, specs: list[ComponentSpec]) -> list[ComponentSpec]:
        """
        Graph-based L2 trunk duplication: Find all paths from L2 loads to transformers
        and duplicate only the trunk segments needed for connectivity.
        """
        from .component_specs import LineSpec, LoadSpec, TransformerSpec

        # Find all L2 loads and their consumer vertices
        l2_load_vertices = set()
        l2_loads_info = []  # For diagnostic logging
        for spec in specs:
            if isinstance(spec, LoadSpec) and getattr(spec, "phase", None) == "L2":
                vertex_id = getattr(spec, "vertex_id", None)
                if vertex_id:
                    l2_load_vertices.add(vertex_id)
                    l2_loads_info.append((getattr(spec, "name", "?"), vertex_id))

        self.logger.debug(f"Found {len(l2_load_vertices)} L2 load consumer vertices")

        if not l2_load_vertices:
            self.logger.info("No L2 loads found - skipping L2 trunk duplication")
            return specs

        # Map consumer vertices to connection points via service lines (Line_Consumer_*)
        consumer_to_connection = {}  # consumer_vertex -> connection_vertex
        for spec in specs:
            if isinstance(spec, LineSpec):
                name = getattr(spec, "name", "")
                if "Line_Consumer_" in name or "Consumer" in name:
                    # Service line: from_vertex is connection point, to_vertex is consumer
                    from_v = getattr(spec, "from_vertex_id", None)
                    to_v = getattr(spec, "to_vertex_id", None)
                    if from_v and to_v:
                        consumer_to_connection[to_v] = from_v

        mapped_count = sum(1 for v in l2_load_vertices if v in consumer_to_connection)
        self.logger.info(f"Mapped {mapped_count}/{len(l2_load_vertices)} L2 consumers to connection points")

        # Build vertex connectivity graph from LV trunk lines only
        # MV trunks are always 3-phase and don't need L2 duplication
        vertex_to_trunks = {}  # vertex_id -> list of trunk lines connected to it
        trunk_count = 0
        for spec in specs:
            if isinstance(spec, LineSpec) and "Trunk" in getattr(spec, "name", ""):
                # Only include LV trunks (< 1 kV), not MV trunks
                cable = getattr(spec, "cable_equipment", None)
                if cable:
                    line_voltage = getattr(cable, "line_voltage", 0)
                    # Skip MV trunks (typically 12.47 kV or higher)
                    if line_voltage >= 1.0:
                        continue

                trunk_count += 1
                from_vertex = getattr(spec, "from_vertex_id", None)
                to_vertex = getattr(spec, "to_vertex_id", None)
                if from_vertex:
                    vertex_to_trunks.setdefault(from_vertex, []).append(spec)
                if to_vertex:
                    vertex_to_trunks.setdefault(to_vertex, []).append(spec)

        self.logger.debug(f"Found {trunk_count} trunk lines, {len(vertex_to_trunks)} vertices with trunks")

        # Find transformer vertices (endpoints)
        transformer_vertices = set()
        for spec in specs:
            if isinstance(spec, TransformerSpec):
                vertex_id = getattr(spec, "vertex_id", None)
                if vertex_id:
                    transformer_vertices.add(vertex_id)

        self.logger.debug(f"Found {len(transformer_vertices)} transformer vertices")

        # Trace paths from each L2 load's CONNECTION POINT back to transformer
        # Use dict to store unique trunks by name (LineSpec objects aren't hashable)
        required_l2_trunks = {}  # trunk_name -> LineSpec
        paths_found = 0
        for consumer_vertex in l2_load_vertices:
            # Start from connection point, not consumer vertex
            connection_vertex = consumer_to_connection.get(consumer_vertex)
            if connection_vertex is None:
                # Fallback: try consumer vertex directly (shouldn't happen but be safe)
                connection_vertex = consumer_vertex

            path_trunks = self._trace_trunk_path_to_transformer(
                connection_vertex, transformer_vertices, vertex_to_trunks
            )
            if path_trunks:
                paths_found += 1
                # Store by name to ensure uniqueness
                for trunk in path_trunks:
                    trunk_name = getattr(trunk, "name", str(id(trunk)))
                    required_l2_trunks[trunk_name] = trunk

        self.logger.info(f"Traced {paths_found} successful paths from {len(l2_load_vertices)} L2 loads")
        self.logger.info(f"Total unique trunk segments required for L2: {len(required_l2_trunks)}")

        # Diagnostic logging for first few L2 loads
        if l2_loads_info[:3]:
            self.logger.info("Sample L2 load mappings:")
            for name, consumer_v in l2_loads_info[:3]:
                conn_v = consumer_to_connection.get(consumer_v, "NOT_MAPPED")
                has_path = (
                    "yes"
                    if consumer_v
                    in [cv for cv in l2_load_vertices if consumer_to_connection.get(cv) in vertex_to_trunks]
                    else "unknown"
                )
                self.logger.info(f"  {name}: consumer_v={consumer_v}, connection_v={conn_v}, path={has_path}")

        # Create L2 duplicates for required trunk segments
        new_specs = []
        for trunk_spec in required_l2_trunks.values():
            l2_spec = self._create_l2_trunk_duplicate(trunk_spec)
            if l2_spec:
                new_specs.append(l2_spec)

        self.logger.info(
            f"Added {
                len(new_specs)} targeted L2 trunk segments for L2 load connectivity"
        )
        return specs + new_specs

    def _trace_trunk_path_to_transformer(
        self, start_vertex: int, transformer_vertices: set, vertex_to_trunks: dict
    ) -> list:
        """Trace LV trunk path from a connection point back to transformer.

        Returns the list of trunk `LineSpec`s that require an L2 duplicate.
        """
        visited = set()
        path_trunks = []
        to_visit = [(start_vertex, [])]  # (vertex, path_so_far)

        while to_visit:
            current_vertex, current_path = to_visit.pop(0)

            if current_vertex in visited:
                continue
            visited.add(current_vertex)

            # If we reached a transformer, save this path
            if current_vertex in transformer_vertices:
                path_trunks.extend(current_path)

                # CRITICAL FIX: Include trunks directly connected to transformer
                # that connect FROM visited vertices (the "last mile" to transformer)
                for trunk in vertex_to_trunks.get(current_vertex, []):
                    from_v = getattr(trunk, "from_vertex_id", None)
                    to_v = getattr(trunk, "to_vertex_id", None)

                    # Check if this trunk connects transformer to an already-visited vertex
                    other_end = None
                    if from_v == current_vertex and to_v in visited:
                        other_end = to_v
                    elif to_v == current_vertex and from_v in visited:
                        other_end = from_v

                    # Add this trunk if it connects to our path and isn't already included
                    if other_end is not None and trunk not in path_trunks:
                        path_trunks.append(trunk)

                break

            # Explore connected trunk segments
            for trunk in vertex_to_trunks.get(current_vertex, []):
                # Find the other end of this trunk
                from_v = getattr(trunk, "from_vertex_id", None)
                to_v = getattr(trunk, "to_vertex_id", None)

                next_vertex = None
                if from_v == current_vertex and to_v is not None:
                    next_vertex = to_v
                elif to_v == current_vertex and from_v is not None:
                    next_vertex = from_v

                if next_vertex and next_vertex not in visited:
                    new_path = current_path + [trunk]
                    to_visit.append((next_vertex, new_path))

        return path_trunks

    def _create_l2_trunk_duplicate(self, l1_trunk) -> Optional:
        """Create an L2 duplicate of a given L1 trunk (bus suffix adjusted)."""
        from .component_specs import LineSpec

        bus1 = getattr(l1_trunk, "bus1", "")
        bus2 = getattr(l1_trunk, "bus2", "")

        # Convert to L2 buses: add .2 if no suffix, or change .1 to .2
        l2_bus1 = self._convert_to_l2_bus(bus1)
        l2_bus2 = self._convert_to_l2_bus(bus2)

        l2_name = getattr(l1_trunk, "name", "").replace("LV_Trunk", "LV_Trunk_L2")

        return LineSpec(
            name=l2_name,
            cable_equipment=getattr(l1_trunk, "cable_equipment", None),
            bus1=l2_bus1,
            bus2=l2_bus2,
            length_km=getattr(l1_trunk, "length_km", 0.0),
            parallel=getattr(l1_trunk, "parallel", 1),
            coordinates=getattr(l1_trunk, "coordinates", None),
            phases="L2",
            from_vertex_id=getattr(l1_trunk, "from_vertex_id", None),
            to_vertex_id=getattr(l1_trunk, "to_vertex_id", None),
        )

    def _convert_to_l2_bus(self, bus_name: str) -> str:
        """Return L2 version of `bus_name` (replace .1→.2 or append .2)."""
        if ".1" in bus_name:
            return bus_name.replace(".1", ".2")
        elif ".2" not in bus_name:
            return bus_name + ".2"
        else:
            return bus_name

    # ---------------
    # Validation
    # ---------------
    def _validate(self, idx: _Idx) -> None:
        """Ensure each LV line's `phases_int` meets downstream load needs."""
        for ln in idx.lv_lines:
            v = getattr(ln, "to_vertex_id", None)
            if v is None:
                continue
            for ld in idx.loads_by_vertex.get(v, []):
                if int(getattr(ld, "n_phases", 1)) > int(getattr(ln, "phases_int", 3)):
                    raise PhaseConsistencyError(
                        f"Line {
                            getattr(
                                ln,
                                'name',
                                '?')} with {
                            ln.phases_int} phases feeds load {
                            ld.name} needing {
                            ld.n_phases}"
                    )

    # ---------------
    # Reporting helpers
    # ---------------

    def _compute_global_phase_loads(self, specs: list[ComponentSpec]) -> None:
        """Aggregate complex load per phase for imbalance reporting."""
        totals: dict[str, complex] = {"A": 0 + 0j, "B": 0 + 0j, "C": 0 + 0j}
        loads = [s for s in specs if isinstance(s, LoadSpec)]
        for ld in loads:
            kw = float(getattr(ld, "kw", 0.0))
            kvar = float(getattr(ld, "kvar", 0.0))
            S = complex(kw, kvar)
            nph = int(getattr(ld, "n_phases", 1))
            if nph == 1 and getattr(ld, "phase", None) in totals:
                totals[ld.phase] += S
            else:
                # Spread 3-phase load equally across A,B,C for reporting
                share = S / 3
                totals["A"] += share
                totals["B"] += share
                totals["C"] += share
        self._phase_loads = totals

    def get_phase_imbalance(self) -> float:
        """Return LV single‑phase imbalance percentage across A/B/C totals."""
        mags = {ph: abs(val) for ph, val in self._phase_loads.items()}
        total = mags["A"] + mags["B"] + mags["C"]
        if total == 0:
            return 0.0
        avg = total / 3.0
        max_dev = max(abs(m - avg) for m in mags.values())
        return 100.0 * max_dev / avg

    def _log_allocation_summary(self) -> None:
        """Log summary statistics of phase allocation results."""
        if not self._territory_reports:
            return

        # Count territories by service type and imbalance levels
        split_phase_count = 0
        three_phase_count = 0
        high_imbalance_count = 0
        single_load_count = 0
        total_loads = 0

        for report in self._territory_reports.values():
            if report["service"] == "split_phase":
                split_phase_count += 1
            else:
                three_phase_count += 1

            n_loads = report["n_single_phase_loads"]
            total_loads += n_loads

            if n_loads == 1:
                single_load_count += 1
            elif report["imbalance_pct"] > self.max_imbalance_pct:
                high_imbalance_count += 1

        avg_loads_per_territory = total_loads / len(self._territory_reports) if self._territory_reports else 0

        self.logger.info(
            f"Phase allocation summary: {len(self._territory_reports)} territories, "
            f"{total_loads} total loads, {avg_loads_per_territory:.1f} avg loads/territory"
        )
        self.logger.info(f"Service types: {split_phase_count} split-phase, {three_phase_count} three-phase")
        self.logger.info(
            f"Balance issues: {single_load_count} single-load territories, "
            f"{high_imbalance_count} high-imbalance territories"
        )

    def _log_mv_vs_lv_summary(self, idx: _Idx) -> None:
        """Log clear summary distinguishing MV feeder balancing vs LV split-phase balancing."""
        # MV feeder balance summary
        mv_report = self.get_mv_balance()
        mv_totals = mv_report["totals"]
        mv_imbalance = mv_report["imbalance_pct"]

        self.logger.info("=" * 60)
        self.logger.info("MV FEEDER BALANCING (Distribution Transformer Primary Connections):")
        self.logger.info(f"  Phase A feeders: {mv_totals['A']:.1f} kVA")
        self.logger.info(f"  Phase B feeders: {mv_totals['B']:.1f} kVA")
        self.logger.info(f"  Phase C feeders: {mv_totals['C']:.1f} kVA")
        self.logger.info(f"  MV Imbalance: {mv_imbalance:.1f}%")

        # LV split-phase balance summary
        lv_l1_total = 0.0
        lv_l2_total = 0.0
        split_territories = 0

        for report in self._territory_reports.values():
            if report["service"] == "split_phase":
                split_territories += 1
                totals = report["totals"]
                if "L1" in totals:
                    lv_l1_total += abs(totals["L1"])
                if "L2" in totals:
                    lv_l2_total += abs(totals["L2"])

        self.logger.info("LV SPLIT-PHASE BALANCING (Within Transformer Territories):")
        self.logger.info(
            f"  Leg 1 total: {
                lv_l1_total:.1f} kVA across {split_territories} territories"
        )
        self.logger.info(
            f"  Leg 2 total: {
                lv_l2_total:.1f} kVA across {split_territories} territories"
        )
        lv_total = lv_l1_total + lv_l2_total
        if lv_total > 0:
            lv_imbalance = 100.0 * abs(lv_l1_total - lv_l2_total) / lv_total
            self.logger.info(f"  Overall LV Imbalance: {lv_imbalance:.1f}%")
        self.logger.info("=" * 60)
