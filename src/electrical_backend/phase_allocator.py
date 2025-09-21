"""
Simplified BFS-based Phase Allocator for pylovo-usa

Goals:
- Assign different phases and balance by complex addition per transformer
- Guarantee consistency between required phases of loads and available phases on lines and transformers

This allocator avoids brittle heuristics by:
- Building minimal topology indices from vertex adjacency in LineSpec
- Classifying LV transformers and performing a single BFS per LV transformer
- Keeping a single canonical integer phase count on lines (phases_int) and deriving strings at export time
"""

from __future__ import annotations

import itertools
import logging
import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .component_specs import (BusSpec, ComponentSpec, LineSpec, LoadSpec,
                              TransformerSpec)


class PhaseImbalanceError(Exception):
    """Raised when phase imbalance in a transformer territory exceeds threshold."""


class PhaseConsistencyError(Exception):
    """Raised when a line's available phases are insufficient for downstream loads."""


class PhaseAllocator:
    """Minimal, deterministic phase allocator per LV transformer."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        max_imbalance_pct: float = 50.0,
        raise_on_imbalance: bool = True,
        optimize_retries: int = 0,
        retry_threshold_pct: Optional[float] = None,
        random_seed: Optional[int] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.max_imbalance_pct = max_imbalance_pct
        self.raise_on_imbalance = raise_on_imbalance
        self._phase_loads: Dict[str, complex] = {
            "A": 0 + 0j, "B": 0 + 0j, "C": 0 + 0j}
        self._territory_reports: Dict[str, Dict[str, object]] = {}
        self.optimize_retries = int(
            optimize_retries) if optimize_retries else 0
        self.retry_threshold_pct = retry_threshold_pct
        self.random_seed = random_seed

    def allocate(self, specs: List[ComponentSpec]) -> List[ComponentSpec]:
        idx = self._index(specs)

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

        # Validate global consistency and split-phase rules (where modeled)
        self._validate(idx)

        # Compute global phase totals for reporting
        self._compute_global_phase_loads(specs)

        return specs

    # ---------------
    # Indexing / Topology
    # ---------------
    class _Idx:
        def __init__(self):
            self.loads_by_vertex: Dict[int, List[LoadSpec]] = {}
            self.lines_from: Dict[int, List[LineSpec]] = {}
            self.lines_to: Dict[int, List[LineSpec]] = {}
            self.buses_by_name: Dict[str, BusSpec] = {}
            self.buses_by_vertex: Dict[int, BusSpec] = {}
            self.transformers_by_vertex: Dict[int, TransformerSpec] = {}
            self.lv_transformers: List[TransformerSpec] = []
            self.mv_lines: List[LineSpec] = []
            self.lv_lines: List[LineSpec] = []
            # Cache for line MV/LV classification
            self._mv_cache: Dict[str, bool] = {}

    def _index(self, specs: List[ComponentSpec]) -> _Idx:
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
                if not hasattr(s, "phases_int") or not isinstance(
                    getattr(s, "phases_int"), int
                ):
                    setattr(s, "phases_int", 3)

        # Loads
        for s in specs:
            if isinstance(s, LoadSpec) and getattr(
                    s, "vertex_id", None) is not None:
                idx.loads_by_vertex.setdefault(s.vertex_id, []).append(s)

        # Transformers and classification
        for s in specs:
            if (
                isinstance(s, TransformerSpec)
                and getattr(s, "vertex_id", None) is not None
            ):
                idx.transformers_by_vertex[s.vertex_id] = s
                eq = getattr(s, "equipment", None)
                sec_kv = getattr(eq, "secondary_voltage_kv", None)
                if sec_kv is not None and sec_kv < 1.0:
                    # LV transformer
                    # Service type: split_phase for ~0.24kV, otherwise
                    # three_phase
                    if sec_kv <= 0.30 or getattr(eq, "n_phases", 1) == 1:
                        setattr(s, "service_type", "split_phase")
                    else:
                        setattr(s, "service_type", "three_phase")
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
        return line.cable_equipment.voltage_level == "MV"

    # ---------------
    # MV Primary Phase Assignment (Feeder-level balancing)
    # ---------------
    def _assign_mv_primary_phases(self, idx: _Idx) -> None:
        """Assign MV primary phase (A/B/C) for split-phase LV transformers to balance MV feeder.

        Uses transformer kVA (equipment.s_max_kva or tx.kva) as size proxy.
        Three-phase LV transformers contribute equally to A/B/C.
        Stores aggregate MV totals and exposes via get_mv_balance().
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
                setattr(tx, "primary_phase", "ABC")

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
            setattr(tx, "primary_phase", phase)

        # Persist MV balance report
        self._mv_phase_totals = mv_totals
        self._mv_imbalance_pct = self._compute_imbalance_pct(
            list(mv_totals.values()))

    def _compute_imbalance_pct(self, mags: List[float]) -> float:
        if not mags:
            return 0.0
        total = sum(mags)
        if total == 0.0:
            return 0.0
        avg = total / len(mags)
        max_dev = max(abs(m - avg) for m in mags)
        return 100.0 * max_dev / avg

    def get_mv_balance(self) -> Dict[str, object]:
        """Get MV-phase balance report: totals per phase and imbalance percentage."""
        totals = getattr(
            self, "_mv_phase_totals", {
                "A": 0.0, "B": 0.0, "C": 0.0})
        imb = getattr(self, "_mv_imbalance_pct", 0.0)
        return {"totals": dict(totals), "imbalance_pct": float(imb)}

    # ---------------
    # Propagation
    # ---------------
    def _propagate_required_phases(
            self, idx: _Idx, tx: TransformerSpec) -> None:
        start_v = tx.vertex_id
        # Build LV territory reachable subgraph from transformer
        territory_vertices: Set[int] = set()
        children: Dict[int, List[int]] = {}
        parents: Dict[int, List[int]] = {}

        q: deque[int] = deque([start_v])
        while q:
            v = q.popleft()
            if v in territory_vertices:
                continue
            territory_vertices.add(v)
            for ln in idx.lines_from.get(v, []):
                if ln in idx.lv_lines and getattr(
                        ln, "to_vertex_id", None) is not None:
                    w = ln.to_vertex_id
                    children.setdefault(v, []).append(w)
                    parents.setdefault(w, []).append(v)
                    if w not in territory_vertices:
                        q.append(w)

        # Initialize required phases at each vertex from direct loads
        required_at_vertex: Dict[int, int] = {}
        for v in territory_vertices:
            req = 1
            for ld in idx.loads_by_vertex.get(v, []):
                # Heuristic: large or non-residential loads are 3-phase
                nph = getattr(ld, "n_phases", None)
                if not isinstance(nph, int):
                    if (
                        getattr(
                            ld, "load_type", "residential").lower() == "residential"
                        and float(getattr(ld, "kw", 0.0)) <= 10.0
                    ):
                        nph = 1
                    else:
                        nph = 3
                req = max(req, int(nph))
            required_at_vertex[v] = req

        # Post-order accumulation: propagate children's requirements to parents
        # Compute processing order via reverse BFS layering
        order: List[int] = []
        seen: Set[int] = set()
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
                required_at_vertex[v] = max(
                    required_at_vertex[v], required_at_vertex[w]
                )

        # Set LV line phases to exactly satisfy downstream requirement (minimization);
        # lines will be upgraded upstream where children require more phases.
        for v in territory_vertices:
            for ln in idx.lines_from.get(v, []):
                if ln in idx.lv_lines and getattr(
                        ln, "to_vertex_id", None) is not None:
                    req = required_at_vertex.get(ln.to_vertex_id, 1)
                    ln.phases_int = int(req)

        # Store per-transformer territory map for later balancing and
        # validation
        setattr(tx, "_territory_vertices", territory_vertices)
        setattr(tx, "_children", children)
        setattr(tx, "_parents", parents)
        setattr(tx, "required_at_vertex", required_at_vertex)

    # ---------------
    # Balancing
    # ---------------
    def _balance_single_phase_territory(
            self, idx: _Idx, tx: TransformerSpec) -> None:
        territory_vertices: Set[int] = getattr(
            tx, "_territory_vertices", set())
        if not territory_vertices:
            return

        # Gather loads in this territory
        single_phase_items: List[Tuple[float, LoadSpec, complex]] = []
        three_phase_total: complex = 0 + 0j
        for v in territory_vertices:
            for ld in idx.loads_by_vertex.get(v, []):
                # Decide load phase count: keep any preset value; else infer
                # from type/size
                nph = getattr(ld, "n_phases", None)
                if not isinstance(nph, int):
                    if (
                        getattr(
                            ld, "load_type", "residential").lower() == "residential"
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
            return

        # Sort loads by descending apparent power magnitude
        single_phase_items.sort(key=lambda t: t[0], reverse=True)

        service = getattr(tx, "service_type", "split_phase")

        # Helper: run one greedy attempt with specified phase order
        def run_attempt(phase_order: List[str]):
            if service == "split_phase":
                phases_local = phase_order[:2]
                totals_local = {"A": 0 + 0j, "B": 0 + 0j}
            else:
                phases_local = phase_order[:3]
                per_phase_share = (
                    (three_phase_total / 3) if three_phase_total != 0 else 0 + 0j
                )
                totals_local = {
                    "A": per_phase_share,
                    "B": per_phase_share,
                    "C": per_phase_share,
                }

            assign_list: List[Tuple[LoadSpec, str]] = []
            for _, ld_, S_ in single_phase_items:
                # choose the phase with minimal |sum|, tie-broken by
                # phase_order
                best_phase = min(
                    phases_local, key=lambda p: abs(
                        totals_local[p]))
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
                        max_dev_local = max(abs(m - avg_local)
                                            for m in considered)
                        pct_local = (
                            100.0 * max_dev_local / avg_local if avg_local > 0 else 0.0
                        )
                else:
                    avg_local = sum(mags_local) / len(phases_local)
                    max_dev_local = max(abs(m - avg_local) for m in mags_local)
                    pct_local = (
                        100.0 * max_dev_local / avg_local if avg_local > 0 else 0.0
                    )
            else:
                pct_local = 0.0

            return assign_list, totals_local, pct_local

        # Base attempt with default phase ordering
        base_order = [
            "A", "B"] if service == "split_phase" else [
            "A", "B", "C"]
        best_assign, best_totals, best_pct = run_attempt(base_order)

        threshold = (
            float(self.retry_threshold_pct)
            if self.retry_threshold_pct is not None
            else float(self.max_imbalance_pct)
        )

        # Retry with multiple phase orderings if above threshold
        if best_pct > threshold and self.optimize_retries > 0:
            rng = random.Random(
                self.random_seed
                if self.random_seed is not None
                else (hash(getattr(tx, "name", "")) & 0xFFFFFFFF)
            )
            # Generate candidate phase orderings
            if service == "split_phase":
                candidates = [list(p)
                              for p in itertools.permutations(["A", "B"], 2)]
            else:
                all_perms = [
                    list(p) for p in itertools.permutations(["A", "B", "C"], 3)
                ]
                rng.shuffle(all_perms)
                candidates = all_perms

            # Ensure base order is first, then try additional ones
            try_orders = [base_order] + \
                [o for o in candidates if o != base_order]
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

        # Persist per-territory report
        self._territory_reports[getattr(tx, "name", str(tx))] = {
            "service": service,
            "totals": best_totals.copy(),
            "imbalance_pct": best_pct,
            "n_single_phase_loads": len(single_phase_items),
        }

        if best_pct > float(self.max_imbalance_pct):
            msg = f"Territory around {
                getattr(
                    tx, 'name', '?')} imbalance {
                best_pct:.1f}% totals {best_totals}"
            if self.raise_on_imbalance:
                raise PhaseImbalanceError(msg)
            else:
                self.logger.warning(msg)

    # ---------------
    # Export / Suffixing (Pass D)
    # ---------------
    def _apply_suffixes(self, idx: _Idx) -> None:
        suffix = {"A": ".1", "B": ".2", "C": ".3"}

        # Loads: single-phase wye bus suffix
        for v, loads in idx.loads_by_vertex.items():
            for ld in loads:
                if (
                    int(getattr(ld, "n_phases", 1)) == 1
                    and getattr(ld, "conn", "wye") == "wye"
                ):
                    ph = getattr(ld, "phase", None)
                    if ph in suffix:
                        if not any(ld.bus.endswith(suf)
                                   for suf in suffix.values()):
                            ld.bus = ld.bus + suffix[ph]

        # Transformers: MV primary bus suffix for split-phase units with
        # assigned primary phase
        for tx in idx.lv_transformers:
            if getattr(tx, "primary_phase", None) in suffix and getattr(
                tx, "bus1", None
            ):
                # Avoid double suffixing
                base_bus = tx.bus1
                for s in suffix.values():
                    if base_bus.endswith(s):
                        base_bus = base_bus[: -len(s)]
                        break
                tx.bus1 = base_bus + suffix[getattr(tx, "primary_phase")]

        # Lines: set display phases and suffix endpoints for single-phase lines
        for ln in idx.lv_lines:
            if int(getattr(ln, "phases_int", 3)) == 1:
                # Choose phase by dominant load at downstream vertex
                chosen = self._choose_line_phase(idx, ln)
                ln.phases = chosen
                # Suffix both endpoints consistently
                ln.bus1 = self._with_phase_suffix(ln.bus1, chosen, suffix)
                ln.bus2 = self._with_phase_suffix(ln.bus2, chosen, suffix)
            else:
                ln.phases = "ABC"

    def _with_phase_suffix(
        self, bus_name: str, phase: str, suffix_map: Dict[str, str]
    ) -> str:
        # Remove existing suffix if present then add
        for s in suffix_map.values():
            if bus_name.endswith(s):
                bus_name = bus_name[: -len(s)]
                break
        return bus_name + suffix_map.get(phase, "")

    def _choose_line_phase(self, idx: _Idx, ln: LineSpec) -> str:
        v = getattr(ln, "to_vertex_id", None)
        if v is None:
            return "A"
        phase_tot = {"A": 0.0, "B": 0.0, "C": 0.0}
        for ld in idx.loads_by_vertex.get(v, []):
            ph = getattr(ld, "phase", None)
            if ph in phase_tot:
                S = complex(
                    float(
                        getattr(
                            ld, "kw", 0.0)), float(
                        getattr(
                            ld, "kvar", 0.0))
                )
                phase_tot[ph] += abs(S)
        if sum(phase_tot.values()) == 0.0:
            return "A"
        return max(phase_tot, key=phase_tot.get)

    # ---------------
    # Validation
    # ---------------
    def _validate(self, idx: _Idx) -> None:
        # Ensure lines with phases_int feed adequate phase counts
        for ln in idx.lv_lines:
            v = getattr(ln, "to_vertex_id", None)
            if v is None:
                continue
            for ld in idx.loads_by_vertex.get(v, []):
                if int(getattr(ld, "n_phases", 1)) > int(
                        getattr(ln, "phases_int", 3)):
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

        # Split-phase 240 V two-wire delta modeling note:
        # If such loads are modeled explicitly, ensure they are not exported with a single node suffix.
        # Current LoadSpec schema does not capture two-node delta endpoints;
        # enforce at export time if added later.

    # ---------------
    # Reporting helpers (compatibility with legacy builder)
    # ---------------
    def _compute_global_phase_loads(self, specs: List[ComponentSpec]) -> None:
        totals: Dict[str, complex] = {"A": 0 + 0j, "B": 0 + 0j, "C": 0 + 0j}
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
        mags = {ph: abs(val) for ph, val in self._phase_loads.items()}
        total = mags["A"] + mags["B"] + mags["C"]
        if total == 0:
            return 0.0
        avg = total / 3.0
        max_dev = max(abs(m - avg) for m in mags.values())
        return 100.0 * max_dev / avg

    def get_territory_reports(self) -> Dict[str, Dict[str, object]]:
        """Return per-transformer (territory) imbalance reports.

        Structure: { tx_name: { 'service': 'split_phase'|'three_phase',
                                 'totals': {'A': S_A, 'B': S_B, 'C': S_C},
                                 'imbalance_pct': float,
                                 'n_single_phase_loads': int } }
        """
        return self._territory_reports.copy()
