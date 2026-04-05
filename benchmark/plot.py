"""
benchmark/plot.py — Ablation results visualisation

Reads benchmark JSON files from outputs/results/ and produces
publication-ready figures for the paper (IEEE two-column format).

Usage:
    # Auto-discover all JSON files in dated subdirectory:
    python benchmark/plot.py --results-dir outputs/results/2026-04-10/

    # Explicit files:
    python benchmark/plot.py \\
        --baseline  outputs/results/.../baseline_v0_M1_*.json \\
                    outputs/results/.../baseline_v0_M2_*.json \\
                    outputs/results/.../baseline_v0_M3_*.json \\
        --pipeline  outputs/results/.../benchmark_V1_*.json \\
                    outputs/results/.../benchmark_V2_*.json \\
                    outputs/results/.../benchmark_V3_*.json

    # Save figures to a directory:
    python benchmark/plot.py --results-dir outputs/results/2026-04-10/ \\
        --out outputs/figures/

Outputs (saved as PNG + PDF):
    fig1_ablation_tsr.png/pdf     — Grouped bar: V0–V3 × easy/medium/hard
    fig2_model_comparison.png/pdf — Grouped bar: M1/M2/M3 V0 vs V3
    fig3_score_progression.png/pdf — Line: avg score per iteration (LOOP cases)
    fig4_status_breakdown.png/pdf  — Stacked bar: PASS/STOP_MAX/STOP_HIGH per variant
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from benchmark.analysis import compute_tsr, compute_tsr_with_ci

# ── Style ─────────────────────────────────────────────────────────────────────
# IEEE two-column: text width ≈ 3.5 in per column; full width ≈ 7.16 in
COL_W   = 3.5    # single-column figure width (inches)
FULL_W  = 7.16   # double-column figure width (inches)
FIG_H   = 2.6    # default height

PALETTE = {
    "V0": "#6c757d",   # grey  — baseline
    "V1": "#4a90d9",   # blue  — execution + judgement
    "V2": "#e07b39",   # orange — static + judgement
    "V3": "#2a9d5c",   # green — full pipeline
}
DIFF_COLOURS = {
    "easy":   "#5b8dd9",
    "medium": "#f0a500",
    "hard":   "#d94f4f",
}
VARIANT_LABELS = {
    "V0": "V0 Temel",
    "V1": "V1 Yür.+Yargı",
    "V2": "V2 Stat.+Yargı",
    "V3": "V3 Tam Hat",
}

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       8,
    "axes.titlesize":  9,
    "axes.labelsize":  8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi":      150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_results_dir(results_dir: str) -> tuple[dict, dict]:
    """
    Auto-discovers JSON files in results_dir.

    Returns:
        variant_data : {"V0": [result, ...], "V1": [...], ...}
        baseline_by_model: {"M1": [result, ...], "M2": [...], "M3": [...]}
    """
    files = glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)
    variant_data: dict[str, list] = {}
    baseline_by_model: dict[str, list] = {}

    for path in files:
        try:
            data = _load(path)
        except Exception:
            continue

        variant = data.get("variant", "")
        if variant == "V0":
            model = data.get("model", "unknown")
            baseline_by_model[model] = data.get("results", [])
        elif variant in ("V1", "V2", "V3"):
            # Keep the most recent file if multiple exist for the same variant
            if variant not in variant_data:
                variant_data[variant] = data.get("results", [])

    # Aggregate V0 across all models for overall baseline
    if baseline_by_model:
        merged: list = []
        for rows in baseline_by_model.values():
            merged.extend(rows)
        variant_data["V0"] = merged

    return variant_data, baseline_by_model


def load_iteration_jsons(results_dir: str) -> list[dict]:
    """Loads all per-iteration JSON files for score-progression analysis."""
    pattern = os.path.join(results_dir, "**", "iterations", "**", "*.json")
    rows = []
    for path in glob.glob(pattern, recursive=True):
        try:
            rows.append(_load(path))
        except Exception:
            continue
    return rows


# ── Figure 1 — Ablation TSR grouped bar ───────────────────────────────────────

def fig_ablation_tsr(variant_data: dict, out_dir: str) -> None:
    """
    Grouped bar chart: V0–V3 × easy/medium/hard TSR.
    One group per difficulty level, one bar per variant.
    Error bars: 95% Wilson CI.
    """
    difficulties = ["easy", "medium", "hard"]
    diff_labels  = ["Kolay", "Orta", "Zor"]
    variants     = [v for v in ("V0", "V1", "V2", "V3") if v in variant_data]

    if not variants:
        print("  [fig1] Veri yok — atlanıyor")
        return

    n_diff    = len(difficulties)
    n_var     = len(variants)
    bar_w     = 0.18
    x         = np.arange(n_diff)
    offsets   = np.linspace(-(n_var - 1) / 2, (n_var - 1) / 2, n_var) * bar_w

    fig, ax = plt.subplots(figsize=(COL_W, FIG_H))

    for i, variant in enumerate(variants):
        results = variant_data[variant]
        counts  = defaultdict(lambda: [0, 0])
        for r in results:
            diff = r.get("difficulty", "medium")
            if "results" in r:
                passed = any(m["status"] == "PASS" for m in r["results"])
            else:
                passed = r.get("passed", False)
            counts[diff][1] += 1
            if passed:
                counts[diff][0] += 1

        tsr_vals = []
        ci_errs  = []
        for d in difficulties:
            p, t = counts[d]
            ci   = compute_tsr_with_ci(p, t)
            tsr_vals.append(ci["tsr"])
            ci_errs.append(ci["tsr"] - ci["ci_lo"])   # symmetric approx

        bars = ax.bar(
            x + offsets[i],
            tsr_vals,
            width=bar_w,
            color=PALETTE[variant],
            label=VARIANT_LABELS[variant],
            alpha=0.88,
            yerr=ci_errs,
            capsize=2,
            error_kw={"linewidth": 0.8, "ecolor": "black", "alpha": 0.6},
        )

        # Value labels on top of bars
        for bar, val in zip(bars, tsr_vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=5.5, color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(diff_labels)
    ax.set_ylabel("Görev Başarı Oranı (TSR)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Ablasyon Varyantlarına Göre TSR\n(hata çubukları: %95 Wilson GA)")
    ax.legend(loc="upper right", framealpha=0.85, ncol=2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))

    fig.tight_layout()
    _save(fig, out_dir, "fig1_ablation_tsr")
    print(f"  [fig1] Kaydedildi → {out_dir}/fig1_ablation_tsr")


# ── Figure 2 — Per-model V0 vs V3 comparison ─────────────────────────────────

def fig_model_comparison(variant_data: dict, baseline_by_model: dict,
                          out_dir: str) -> None:
    """
    Grouped bar chart: for each model (M1/M2/M3),
    show V0 TSR and V3 TSR side by side.
    """
    models = [m for m in ("M1", "M2", "M3") if m in baseline_by_model]
    if not models or "V3" not in variant_data:
        print("  [fig2] Veri yok — atlanıyor")
        return

    # V3 results split by active_model
    v3_by_model: dict[str, list] = defaultdict(list)
    for r in variant_data["V3"]:
        for m_result in r.get("results", []):
            model_id = m_result.get("model", "")
            if model_id in models:
                v3_by_model[model_id].append({
                    "difficulty": r.get("difficulty", "medium"),
                    "passed":     m_result["status"] == "PASS",
                })

    x       = np.arange(len(models))
    bar_w   = 0.32
    fig, ax = plt.subplots(figsize=(COL_W, FIG_H))

    v0_tsrs, v3_tsrs = [], []
    v0_cis,  v3_cis  = [], []

    for model in models:
        # V0
        b_results = baseline_by_model[model]
        passed_b  = sum(1 for r in b_results if r.get("passed", False))
        ci_b      = compute_tsr_with_ci(passed_b, len(b_results))
        v0_tsrs.append(ci_b["tsr"])
        v0_cis.append(ci_b["tsr"] - ci_b["ci_lo"])

        # V3 — model contribution
        m_rows    = v3_by_model.get(model, [])
        passed_v3 = sum(1 for r in m_rows if r["passed"])
        ci_v3     = compute_tsr_with_ci(passed_v3, len(m_rows)) if m_rows else {"tsr": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
        v3_tsrs.append(ci_v3["tsr"])
        v3_cis.append(ci_v3["tsr"] - ci_v3.get("ci_lo", 0.0))

    bars_v0 = ax.bar(x - bar_w / 2, v0_tsrs, width=bar_w,
                     color=PALETTE["V0"], label="V0 Temel",
                     alpha=0.88, yerr=v0_cis, capsize=2,
                     error_kw={"linewidth": 0.8, "ecolor": "black", "alpha": 0.6})
    bars_v3 = ax.bar(x + bar_w / 2, v3_tsrs, width=bar_w,
                     color=PALETTE["V3"], label="V3 Tam Hat",
                     alpha=0.88, yerr=v3_cis, capsize=2,
                     error_kw={"linewidth": 0.8, "ecolor": "black", "alpha": 0.6})

    # Delta annotations
    for xi, (v0, v3) in enumerate(zip(v0_tsrs, v3_tsrs)):
        delta = v3 - v0
        sign  = "+" if delta >= 0 else ""
        colour = "#2a9d5c" if delta >= 0 else "#d94f4f"
        ax.annotate(
            f"{sign}{delta:.2f}",
            xy=(xi + bar_w / 2, v3 + (v3_cis[xi] if v3_cis else 0) + 0.02),
            ha="center", va="bottom",
            fontsize=6.5, color=colour, fontweight="bold",
        )

    model_names = {
        "M1": "M1\ngpt-oss-20b",
        "M2": "M2\nQwen3-Coder",
        "M3": "M3\nGLM-4.7-Flash",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([model_names.get(m, m) for m in models])
    ax.set_ylabel("Görev Başarı Oranı (TSR)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Bazlı V0 → V3 Karşılaştırması\n(yeşil delta = tam hat kazanımı)")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))

    fig.tight_layout()
    _save(fig, out_dir, "fig2_model_comparison")
    print(f"  [fig2] Kaydedildi → {out_dir}/fig2_model_comparison")


# ── Figure 3 — Score progression across iterations ───────────────────────────

def fig_score_progression(iter_rows: list[dict], out_dir: str) -> None:
    """
    Line chart: average node3_total_score per iteration number.
    Only LOOP rows are used (PASS/STOP rows bias the last iteration).
    One line per difficulty level.
    """
    if not iter_rows:
        print("  [fig3] İterasyon verisi yok — atlanıyor")
        return

    # Collect (iteration, score, difficulty) for LOOP rows
    data: dict[str, dict[int, list]] = {
        "easy": defaultdict(list),
        "medium": defaultdict(list),
        "hard": defaultdict(list),
    }

    for row in iter_rows:
        status = row.get("status", "")
        diff   = row.get("difficulty", "medium")
        iter_n = row.get("iteration", row.get("iter", 1))
        score  = row.get("node3", {}).get("total_score",
                 row.get("node3_total_score", None))
        if score is None or diff not in data:
            continue
        data[diff][iter_n].append(score)

    max_iter = max(
        (max(iters.keys()) for iters in data.values() if iters),
        default=0,
    )
    if max_iter == 0:
        print("  [fig3] İterasyon verisi boş — atlanıyor")
        return

    fig, ax = plt.subplots(figsize=(COL_W, FIG_H))

    for diff, colour in DIFF_COLOURS.items():
        iters_data = data[diff]
        if not iters_data:
            continue
        xs  = sorted(iters_data)
        ys  = [np.mean(iters_data[x]) for x in xs]
        sds = [np.std(iters_data[x])  for x in xs]

        ax.plot(xs, ys, marker="o", markersize=3.5,
                color=colour, linewidth=1.5,
                label={"easy": "Kolay", "medium": "Orta", "hard": "Zor"}[diff])
        ax.fill_between(
            xs,
            [y - s for y, s in zip(ys, sds)],
            [y + s for y, s in zip(ys, sds)],
            alpha=0.12, color=colour,
        )

    ax.set_xlabel("Yineleme")
    ax.set_ylabel("Ortalama Toplam Puan")
    ax.set_title("Yineleme Boyunca Puan Gelişimi\n(gölge alan: ±1 std)")
    ax.set_xlim(0.5, max_iter + 0.5)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(1, max_iter + 1))
    ax.legend(loc="lower right", framealpha=0.85)
    ax.axhline(80, color="#2a9d5c", linewidth=0.7, linestyle="--", alpha=0.5, label="Kolay eşik (80)")
    ax.axhline(70, color="#f0a500", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.axhline(60, color="#d94f4f", linewidth=0.7, linestyle="--", alpha=0.5)

    fig.tight_layout()
    _save(fig, out_dir, "fig3_score_progression")
    print(f"  [fig3] Kaydedildi → {out_dir}/fig3_score_progression")


# ── Figure 4 — Status breakdown per variant ───────────────────────────────────

def fig_status_breakdown(variant_data: dict, out_dir: str) -> None:
    """
    Stacked horizontal bar: PASS / STOP_MAX / STOP_HIGH ratio per variant.
    """
    variants = [v for v in ("V0", "V1", "V2", "V3") if v in variant_data]
    if not variants:
        print("  [fig4] Veri yok — atlanıyor")
        return

    status_colours = {
        "PASS":      "#2a9d5c",
        "STOP_MAX":  "#f0a500",
        "STOP_HIGH": "#d94f4f",
        "PENDING":   "#cccccc",
    }

    fig, ax = plt.subplots(figsize=(COL_W, FIG_H * 0.85))

    for i, variant in enumerate(reversed(variants)):
        results = variant_data[variant]
        counts: dict[str, int] = defaultdict(int)
        total = 0
        for r in results:
            model_results = r.get("results", [{"status": r.get("final_status", "PENDING")}])
            for m in model_results:
                s = m.get("status", "PENDING")
                counts[s] += 1
                total += 1

        left = 0.0
        for status in ("PASS", "STOP_MAX", "STOP_HIGH"):
            val = counts.get(status, 0) / total if total else 0
            ax.barh(
                i, val, left=left, height=0.55,
                color=status_colours[status],
                label=status if i == 0 else "_nolegend_",
            )
            if val > 0.04:
                ax.text(left + val / 2, i, f"{val:.0%}",
                        ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold")
            left += val

    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([VARIANT_LABELS[v] for v in reversed(variants)])
    ax.set_xlabel("Oran")
    ax.set_xlim(0, 1)
    ax.set_title("Varyant Başına Durum Dağılımı")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))

    legend_patches = [
        mpatches.Patch(color=status_colours["PASS"],      label="PASS"),
        mpatches.Patch(color=status_colours["STOP_MAX"],  label="STOP_MAX"),
        mpatches.Patch(color=status_colours["STOP_HIGH"], label="STOP_HIGH"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              framealpha=0.85, ncol=3, fontsize=6)

    fig.tight_layout()
    _save(fig, out_dir, "fig4_status_breakdown")
    print(f"  [fig4] Kaydedildi → {out_dir}/fig4_status_breakdown")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge — Ablation Results Visualisation"
    )
    parser.add_argument(
        "--results-dir",
        help="Directory containing benchmark JSON files (auto-discovers all)",
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        help="Baseline JSON file(s) (V0, one per model)",
    )
    parser.add_argument(
        "--pipeline",
        nargs="+",
        help="Pipeline benchmark JSON file(s) (V1/V2/V3)",
    )
    parser.add_argument(
        "--out",
        default="outputs/figures",
        help="Output directory for figures (default: outputs/figures)",
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.results_dir:
        variant_data, baseline_by_model = load_results_dir(args.results_dir)
        iter_rows = load_iteration_jsons(args.results_dir)
    else:
        if not args.baseline or not args.pipeline:
            parser.error("--results-dir or both --baseline and --pipeline are required")

        variant_data: dict = {}
        baseline_by_model: dict = {}
        iter_rows: list = []

        for path in (args.baseline or []):
            data = _load(path)
            model = data.get("model", "unknown")
            baseline_by_model[model] = data.get("results", [])

        if baseline_by_model:
            merged = []
            for rows in baseline_by_model.values():
                merged.extend(rows)
            variant_data["V0"] = merged

        for path in (args.pipeline or []):
            data = _load(path)
            v = data.get("variant", "")
            if v in ("V1", "V2", "V3"):
                variant_data[v] = data.get("results", [])

    print(f"Yüklendi: {list(variant_data)} — model: {list(baseline_by_model)}")
    print(f"İterasyon verisi: {len(iter_rows)} kayıt")
    print(f"Çıktı dizini: {args.out}")
    print()

    # ── Generate figures ──────────────────────────────────────────────────────
    fig_ablation_tsr(variant_data, args.out)
    fig_model_comparison(variant_data, baseline_by_model, args.out)
    fig_score_progression(iter_rows, args.out)
    fig_status_breakdown(variant_data, args.out)

    print("\nTüm grafikler oluşturuldu.")
