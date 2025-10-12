"""
Grid statistics calculator and plotter for component specs.

This module provides statistics and visualization for electrical grids
generated with OpenDss backend.

MV/LV Classification Logic:
- Loads: Uses load.kv (≥1.0 kV → MV), falls back to bus naming patterns
- Lines: Uses cable_equipment.line_voltage or voltage_level, falls back to bus patterns
- Buses: Pattern matching on name ("MV_", "SOURCE", "SubTx", etc.)

Cable Lengths:
- All cable lengths account for parallel conductors (length_km × parallel)

Plots Generated:
1. Transformer Distribution (pie + bar)
2. Cable Distribution by Type (top-N bar, no unit mixing)
3. MV vs LV Comparison (3 clean bar charts: cable km, load count, load kW)
4. Network Overview (component counts)
"""

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import seaborn as sns

from ..electrical_backend.component_specs import BusSpec, ComponentSpec, LineSpec, LoadSpec, TransformerSpec


class GridStats(TypedDict, total=False):
    """Type definition for grid statistics dictionary."""

    buses: int
    lines: int
    mv_buses: int
    lv_buses: int
    substation_transformers: int
    distribution_transformers: int
    mv_loads: int
    lv_loads: int
    total_loads: int
    transformer_kvas: list[float]
    kva_distribution: dict[int, int]
    max_kva: float
    avg_kva: float
    total_kva: float
    avg_lv_loads_per_dist_tx: float
    total_cable_km: float
    mv_cable_km: float
    lv_cable_km: float
    cable_type_lengths: dict[str, float]
    cable_type_lengths_mv: dict[str, float]
    cable_type_lengths_lv: dict[str, float]
    cable_per_consumer: float
    lv_cable_per_consumer: float
    mv_cable_per_consumer: float
    total_load_kw: float
    mv_load_kw: float
    lv_load_kw: float
    avg_load_per_consumer: float


def _is_mv_bus(name: str) -> bool:
    """
    Check if a bus name indicates MV level.

    Args:
        name: Bus name to check

    Returns:
        True if bus is likely MV level
    """
    n = (name or "").upper()
    return any(
        x in n
        for x in ["MV_", "MV-", "MV MAIN", "MV_MAIN", "SOURCE", "SUBTX", "SUB", "MAIN", "MV_NODE", "TRAFO_", "_MV"]
    )


def _classify_load(load: LoadSpec) -> str:
    """
    Classify load as MV or LV based on voltage and naming.

    Args:
        load: Load specification to classify

    Returns:
        "MV" or "LV"
    """
    # Prefer voltage
    try:
        kv = getattr(load, "kv", None)
        if kv is not None and float(kv) >= 1.0:
            return "MV"
    except (ValueError, TypeError):
        pass

    # Fallback to name/bus patterns
    if _is_mv_bus(load.bus) or "MV_" in load.name.upper():
        return "MV"

    return "LV"


def _classify_line(line: LineSpec) -> str:
    """
    Classify line as MV or LV based on cable equipment and naming.

    Args:
        line: Line specification to classify

    Returns:
        "MV" or "LV"
    """
    # Prefer equipment voltage metadata
    ce = getattr(line, "cable_equipment", None)
    if ce:
        try:
            # CableEquipment.line_voltage is in kV
            line_voltage = getattr(ce, "line_voltage", None)
            if line_voltage is not None and float(line_voltage) >= 1.0:
                return "MV"

            # Fallback: use voltage_level field
            vl = (getattr(ce, "voltage_level", "") or "").upper()
            if "MV" in vl:
                return "MV"
        except (ValueError, TypeError):
            pass

    # Fallback to bus patterns
    bus1 = getattr(line, "bus1", "")
    bus2 = getattr(line, "bus2", "")
    if _is_mv_bus(bus1) or _is_mv_bus(bus2):
        return "MV"

    return "LV"


def _eff_length(line: LineSpec) -> float:
    """
    Calculate effective cable length accounting for parallel conductors.

    Args:
        line: Line specification

    Returns:
        Effective length in km (length_km × parallel)
    """
    length_km = float(getattr(line, "length_km", 0.0) or 0.0)
    parallel = int(getattr(line, "parallel", 1) or 1)
    return length_km * parallel


def _savefig_safe(path: Path, logger: logging.Logger, fig=None, save_svg: bool = True) -> None:
    """
    Safely save figure with error handling. Saves both PNG and SVG formats.

    Args:
        path: Output file path (PNG)
        logger: Logger instance
        fig: Matplotlib figure (None to use current figure)
        save_svg: Whether to also save as SVG (default: True)
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save PNG
        if fig is not None:
            fig.savefig(path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig(path, bbox_inches="tight", dpi=300)

        # Save SVG
        if save_svg:
            svg_path = path.with_suffix(".svg")
            if fig is not None:
                fig.savefig(svg_path, bbox_inches="tight", format="svg")
            else:
                plt.savefig(svg_path, bbox_inches="tight", format="svg")

    except Exception as e:
        logger.error(f"Failed to save figure {path}: {e}", exc_info=False)
    finally:
        if fig is not None:
            plt.close(fig)
        else:
            plt.close()


def extract_statistics(component_specs: list[ComponentSpec], logger: logging.Logger | None = None) -> dict[str, Any]:
    """
    Extract statistics from component specs for analysis and plotting.

    Args:
        component_specs: All component specifications
        logger: Optional logger instance

    Returns:
        Dictionary with all extracted statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Filter components by type
    buses = [s for s in component_specs if isinstance(s, BusSpec)]
    lines = [s for s in component_specs if isinstance(s, LineSpec)]
    transformers = [s for s in component_specs if isinstance(s, TransformerSpec)]
    loads = [s for s in component_specs if isinstance(s, LoadSpec)]

    # Separate substation transformer from distribution transformers
    substation_transformers = [t for t in transformers if "SubTx" in t.name or "Source" in getattr(t, "bus1", "")]
    distribution_transformers = [t for t in transformers if t not in substation_transformers]

    # Classify MV/LV using robust logic (voltage first, then naming)
    mv_loads = [l for l in loads if _classify_load(l) == "MV"]
    lv_loads = [l for l in loads if _classify_load(l) == "LV"]

    mv_lines = [ln for ln in lines if _classify_line(ln) == "MV"]
    lv_lines = [ln for ln in lines if _classify_line(ln) == "LV"]

    mv_buses = [b for b in buses if _is_mv_bus(b.name)]
    lv_buses = [b for b in buses if b not in mv_buses]

    # Transformer statistics
    transformer_kvas = [
        t.equipment.s_max_kva if t.equipment else (getattr(t, "kva", None) or 0) for t in distribution_transformers
    ]

    # Group transformers by kVA rating
    kva_distribution = defaultdict(int)
    for kva in transformer_kvas:
        kva_rounded = round(kva / 25) * 25  # Round to 25 kVA increments
        kva_distribution[kva_rounded] += 1

    # Cable statistics by type (accounting for parallel conductors)
    cable_type_lengths = defaultdict(float)
    cable_type_lengths_mv = defaultdict(float)
    cable_type_lengths_lv = defaultdict(float)

    for line in lines:
        if line.cable_equipment:
            cable_name = line.cable_equipment.name
            length_eff = _eff_length(line)
            cable_type_lengths[cable_name] += length_eff

            if _classify_line(line) == "MV":
                cable_type_lengths_mv[cable_name] += length_eff
            else:
                cable_type_lengths_lv[cable_name] += length_eff

    # Calculate metrics (with parallel conductor adjustment)
    total_cable_km = sum(_eff_length(line) for line in lines)
    mv_cable_km = sum(_eff_length(line) for line in mv_lines)
    lv_cable_km = sum(_eff_length(line) for line in lv_lines)

    total_load_kw = sum(l.kw for l in loads)
    mv_load_kw = sum(l.kw for l in mv_loads)
    lv_load_kw = sum(l.kw for l in lv_loads)

    return {
        # Component counts
        "buses": len(buses),
        "mv_buses": len(mv_buses),
        "lv_buses": len(lv_buses),
        "lines": len(lines),
        "substation_transformers": len(substation_transformers),
        "distribution_transformers": len(distribution_transformers),
        "mv_loads": len(mv_loads),
        "lv_loads": len(lv_loads),
        "total_loads": len(loads),
        # Transformer stats
        "transformer_kvas": transformer_kvas,
        "kva_distribution": dict(kva_distribution),
        "max_kva": max(transformer_kvas) if transformer_kvas else 0,
        "avg_kva": sum(transformer_kvas) / len(transformer_kvas) if transformer_kvas else 0,
        "total_kva": sum(transformer_kvas),
        "avg_lv_loads_per_dist_tx": len(lv_loads) / len(distribution_transformers) if distribution_transformers else 0,
        # Cable stats (with parallel conductor accounting)
        "total_cable_km": total_cable_km,
        "mv_cable_km": mv_cable_km,
        "lv_cable_km": lv_cable_km,
        "cable_type_lengths": dict(cable_type_lengths),
        "cable_type_lengths_mv": dict(cable_type_lengths_mv),
        "cable_type_lengths_lv": dict(cable_type_lengths_lv),
        "cable_per_consumer": total_cable_km / len(loads) if loads else 0,
        "lv_cable_per_consumer": lv_cable_km / len(lv_loads) if lv_loads else 0,
        "mv_cable_per_consumer": mv_cable_km / max(1, len(mv_loads)) if mv_loads else 0,
        # Load stats
        "total_load_kw": total_load_kw,
        "mv_load_kw": mv_load_kw,
        "lv_load_kw": lv_load_kw,
        "avg_load_per_consumer": total_load_kw / len(loads) if loads else 0,
    }


def save_statistics_report(
    stats: dict[str, Any], kcid: int, scid: int, output_dir: str = ".", logger: logging.Logger | None = None
) -> None:
    """
    Save statistics as text report.

    Args:
        stats: Statistics dictionary from extract_statistics()
        kcid: K-means cluster ID
        scid: Substation cluster ID
        output_dir: Directory to save report
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    stats_text = f"""
Grid Statistics Report
======================
Cluster: K{kcid}_S{scid}
Generated: {timestamp}

Component Counts
----------------
Buses: {stats['buses']}
Lines: {stats['lines']}
Substation Transformers: {stats['substation_transformers']}
Distribution Transformers: {stats['distribution_transformers']}
MV Consumers (Loads): {stats['mv_loads']}
LV Consumers (Loads): {stats['lv_loads']}
Total Consumers: {stats['total_loads']}

Transformer Statistics (Distribution Transformers Only)
-------------------------------------------------------
Count: {stats['distribution_transformers']}
Maximum Capacity: {stats['max_kva']:.2f} kVA
Average Capacity: {stats['avg_kva']:.2f} kVA
Total Capacity: {stats['total_kva']:.2f} kVA
Average LV Loads per Dist. Transformer: {stats['avg_lv_loads_per_dist_tx']:.2f}

Network Statistics
------------------
Total Cable Length: {stats['total_cable_km']:.3f} km
  MV Cable: {stats['mv_cable_km']:.3f} km
  LV Cable: {stats['lv_cable_km']:.3f} km
Cable Length per Consumer: {stats['cable_per_consumer']:.3f} km
Total Load Demand: {stats['total_load_kw']:.2f} kW
Average Load per Consumer: {stats['avg_load_per_consumer']:.2f} kW

MV Network
----------
MV Consumers: {stats['mv_loads']}
MV Cable Length: {stats['mv_cable_km']:.3f} km
MV Total Load: {stats['mv_load_kw']:.2f} kW

LV Network
----------
LV Consumers: {stats['lv_loads']}
LV Cable Length: {stats['lv_cable_km']:.3f} km
LV Total Load: {stats['lv_load_kw']:.2f} kW
"""

    output_file = Path(output_dir) / f"grid_statistics_K{kcid}_S{scid}.txt"
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(stats_text)
        logger.info(f"✓ Saved grid statistics to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save statistics file: {str(e)}")


def plot_grid_statistics(
    stats: dict[str, Any],
    kcid: int,
    scid: int,
    output_dir: str = "statistics",
    show_plots: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """
    Generate visualization plots for grid statistics.

    Plots are saved to: {output_dir}/K{kcid}_S{scid}/

    Args:
        stats: Statistics dictionary from extract_statistics()
        kcid: K-means cluster ID
        scid: Substation cluster ID
        output_dir: Root directory (default: "statistics")
        show_plots: Whether to display plots interactively
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    plot_dir = Path(output_dir) / f"K{kcid}_S{scid}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 100

    # Plot 1: Transformer Distribution (Pie + Bar)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if stats.get("kva_distribution"):
        # Pie chart
        kva_dist = stats["kva_distribution"]
        labels = [f"{k} kVA" for k in sorted(kva_dist.keys())]
        sizes = [kva_dist[k] for k in sorted(kva_dist.keys())]

        ax1.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Transformer Size Distribution", fontsize=14, fontweight="bold")

        # Bar chart
        ax2.bar(labels, sizes, color="steelblue", edgecolor="black")
        ax2.set_xlabel("Transformer Size")
        ax2.set_ylabel("Count")
        ax2.set_title("Transformer Count by Size", fontsize=14, fontweight="bold")
        ax2.tick_params(axis="x", rotation=45)
    else:
        ax1.text(0.5, 0.5, "No transformer data available", ha="center", va="center", transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "No transformer data available", ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    _savefig_safe(plot_dir / f"transformer_distribution_K{kcid}_S{scid}.png", logger, fig)

    # Plot 2: Cable Distribution by Type (Top-N Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6))

    data = stats.get("cable_type_lengths", {})
    if data:
        # Sort by length descending, take top 10
        items = sorted(data.items(), key=lambda kv: kv[1], reverse=True)
        N = 10
        top = items[:N]
        other_sum = sum(v for _, v in items[N:])
        if other_sum > 0:
            top.append(("Other", other_sum))

        labels, values = zip(*top, strict=False) if top else ([], [])
        ax.barh(labels, values, color="steelblue", edgecolor="black")
        ax.set_xlabel("Cable Length (km)", fontsize=12)
        ax.set_title("Cable Length by Type (Top 10)", fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No cable data available", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    _savefig_safe(plot_dir / f"cable_distribution_K{kcid}_S{scid}.png", logger, fig)

    # Plot 3: MV vs LV Comparison (Clean 3-panel, no mixed units)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    categories = ["MV", "LV"]
    colors = ["#E37222", "#0065BD"]  # TUM Orange and Blue

    # Cable length comparison
    cable_lengths = [stats.get("mv_cable_km", 0), stats.get("lv_cable_km", 0)]
    bars1 = ax1.bar(categories, cable_lengths, color=colors, edgecolor="black")
    ax1.set_ylabel("Cable Length (km)", fontsize=12)
    ax1.set_title("Cable Length: MV vs LV", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom", fontsize=10)

    # Load count comparison
    load_counts = [stats.get("mv_loads", 0), stats.get("lv_loads", 0)]
    bars2 = ax2.bar(categories, load_counts, color=colors, edgecolor="black")
    ax2.set_ylabel("Number of Loads", fontsize=12)
    ax2.set_title("Load Count: MV vs LV", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom", fontsize=10)

    # Total load demand comparison
    load_kws = [stats.get("mv_load_kw", 0), stats.get("lv_load_kw", 0)]
    bars3 = ax3.bar(categories, load_kws, color=colors, edgecolor="black")
    ax3.set_ylabel("Total Load (kW)", fontsize=12)
    ax3.set_title("Total Load: MV vs LV", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}", ha="center", va="bottom", fontsize=10)

    plt.suptitle(f"MV vs LV Network Comparison - K{kcid}_S{scid}", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    _savefig_safe(plot_dir / f"mv_lv_comparison_K{kcid}_S{scid}.png", logger, fig)

    # Plot 4: Network Overview
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = [
        "Buses",
        "Lines",
        "Dist. Trafos",
        "Loads",
        f'Cable\n({stats.get("total_cable_km", 0):.2f} km)',
        f'Load\n({stats.get("total_load_kw", 0):.1f} kW)',
    ]
    values = [
        stats.get("buses", 0),
        stats.get("lines", 0),
        stats.get("distribution_transformers", 0),
        stats.get("total_loads", 0),
        stats.get("total_cable_km", 0) * 10,  # Scale for visibility
        stats.get("total_load_kw", 0) / 10,  # Scale for visibility
    ]

    colors = ["#0065BD", "#A2AD00", "#E37222", "#98C6EA", "#DAD7CB", "#7F7F7F"]
    bars = ax.bar(metrics, values, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Count (scaled for cables/loads)", fontsize=12)
    ax.set_title(f"Network Overview - K{kcid}_S{scid}", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)

    plt.tight_layout()
    _savefig_safe(plot_dir / f"network_overview_K{kcid}_S{scid}.png", logger, fig)

    logger.info(f"✓ Saved plots to {plot_dir}")


def calculate_and_save_statistics(
    component_specs: list[ComponentSpec],
    kcid: int,
    scid: int,
    output_dir: str = "statistics",
    generate_plots: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Calculate statistics, save report, and optionally generate plots.

    All outputs are saved to centralized statistics directory:
    - Text report: {output_dir}/grid_statistics_K{kcid}_S{scid}.txt
    - Plots: {output_dir}/K{kcid}_S{scid}/*.png and *.svg

    Args:
        component_specs: All component specifications
        kcid: K-means cluster ID
        scid: Substation cluster ID
        output_dir: Root output directory (default: "statistics")
        generate_plots: Whether to generate visualization plots
        logger: Optional logger instance

    Returns:
        Dictionary with all statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Extract statistics
    stats = extract_statistics(component_specs, logger)

    # Save text report
    save_statistics_report(stats, kcid, scid, output_dir, logger)

    # Generate plots if requested
    if generate_plots:
        plot_grid_statistics(stats, kcid, scid, output_dir, show_plots=False, logger=logger)

    return stats
