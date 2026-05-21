"""
Kabukicho Urban Pulse — Analysis & Visualisation
=================================================
Generates 6 presentation-ready charts from the collected data.

Charts
------
  01_dashboard.png          — KPI cards + crowd timeline + activity pie
  03_tourist_local.png      — tourist vs local breakdown + accessory signals
  04_demographics.png       — adult / child split + solo vs group behaviour
  05_crowd_comparison.png   — crowd metrics daytime vs evening comparison
  06_gender_comparison.png  — gender distribution daytime vs evening comparison
  07_correlations.png       — metric correlation matrix (Big Data context)

Usage
-----
    python src/analysis.py          # today's session only
    python src/analysis.py --all    # all data in the database
"""

import argparse, sqlite3, sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
import config

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 140, "font.size": 10})

# Colour palette
C = {
    "crowd":      "#2ecc71",
    "tourists":   "#e74c3c",
    "locals":     "#3498db",
    "children":   "#9b59b6",
    "adults":     "#1abc9c",
    "waving":     "#f39c12",
    "direction":  "#00bcd4",
    "speed":      "#ff9800",
    "brightness": "#ffd700",
    "suitcase":   "#c0392b",
    "backpack":   "#e67e22",
    "umbrella":   "#2980b9",
    "bag":        "#8e44ad",
    "QUIET":          "#2ecc71",
    "MODERATE":       "#f39c12",
    "BUSY":           "#e67e22",
    "CROWDED":        "#e74c3c",
    "NIGHT-QUIET":    "#1a237e",
    "NIGHT-BUSY":     "#283593",
    "NIGHT-PEAK":     "#1565c0",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load(db_path: str, real_only: bool) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql("SELECT * FROM observations ORDER BY window_start", conn)
    conn.close()
    if df.empty:
        raise SystemExit("[analysis] No data yet — run main.py first.")

    df["window_start"] = pd.to_datetime(df["window_start"], format="ISO8601", utc=True)
    df["local_time"]   = df["window_start"].dt.tz_convert("Asia/Tokyo")
    df["time_label"]   = df["local_time"].dt.strftime("%H:%M:%S")
    df["hour"]         = df["local_time"].dt.hour
    df["date"]         = df["local_time"].dt.date

    if real_only:
        df = df[df["date"] == df["date"].max()].copy()

    df["elapsed_min"] = (
        (df["window_start"] - df["window_start"].iloc[0]).dt.total_seconds() / 60
    ).round(2)
    return df.reset_index(drop=True)


def save(fig, name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, bbox_inches="tight")
    print(f"[analysis] Saved  {p.name}")
    plt.close(fig)


# ── Chart 1: Dashboard ────────────────────────────────────────────────────────

def chart_dashboard(df: pd.DataFrame, out_dir: Path):
    """KPI cards, crowd+tourist timeline, activity pie."""
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("#1a1a2e")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.40,
                            left=0.06, right=0.97, top=0.88, bottom=0.08)

    is_night = df["brightness_avg"].mean() < 90
    session  = "[Night session]" if is_night else "[Day session]"
    fig.suptitle(
        f"Kabukicho, Shinjuku — Urban Pulse Dashboard\n"
        f"{session}  |  {df['time_label'].iloc[0]} to {df['time_label'].iloc[-1]} JST  "
        f"({len(df)} windows x {config.AGGREGATION_WINDOW_SECONDS}s)",
        fontsize=13, fontweight="bold", color="#e0e0e0",
    )

    male_r = df["male_ratio"].mean() if "male_ratio" in df.columns else 0.5
    kpis = [
        ("Avg Crowd\nper Frame",    f"{df['crowd_avg'].mean():.1f}",         "#27ae60"),
        ("Tourist Index",           f"{df['tourist_index'].mean():.0%}",      "#c0392b"),
        ("Gender Split\n(est.)",    f"M {male_r:.0%} / F {1-male_r:.0%}",    "#1565c0"),
        ("Friendliness\n(waving)",  f"{df['friendliness_idx'].mean():.1%}",   "#f39c12"),
    ]
    for col, (lbl, val, bg) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(bg)
        ax.text(0.5, 0.60, val, transform=ax.transAxes,
                ha="center", va="center", fontsize=26, fontweight="bold", color="white")
        ax.text(0.5, 0.88, lbl, transform=ax.transAxes,
                ha="center", va="center", fontsize=9,  fontweight="bold", color="white")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

    # Crowd + tourists over time
    ax_t = fig.add_subplot(gs[1, :3])
    ax_t.set_facecolor("#16213e")
    ax_t.plot(df["elapsed_min"], df["crowd_avg"],    color=C["crowd"],    lw=2,   label="All people")
    ax_t.plot(df["elapsed_min"], df["tourists_avg"], color=C["tourists"], lw=2,   linestyle="--", label="Tourists")
    ax_t.plot(df["elapsed_min"], df["children_avg"], color=C["children"], lw=1.5, linestyle=":",  label="Children")
    ax_t.fill_between(df["elapsed_min"], df["crowd_avg"], alpha=0.15, color=C["crowd"])
    ax_t.set_xlabel("Minutes into recording", color="#e0e0e0")
    ax_t.set_ylabel("Avg count / frame",      color="#e0e0e0")
    ax_t.set_title("Crowd composition over time", color="#e0e0e0")
    ax_t.tick_params(colors="#e0e0e0")
    ax_t.legend(fontsize=9, facecolor="#16213e", labelcolor="#e0e0e0")

    # Activity pie
    ax_p = fig.add_subplot(gs[1, 3])
    ax_p.set_facecolor("#16213e")
    counts = df["activity"].value_counts()
    ax_p.pie(counts.values, labels=counts.index, autopct="%1.0f%%",
             colors=[C.get(k, "#888") for k in counts.index],
             startangle=140,
             wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 1.5},
             textprops={"fontsize": 8, "color": "#e0e0e0"})
    ax_p.set_title("Activity breakdown", color="#e0e0e0")

    save(fig, "01_dashboard.png", out_dir)


# ── Chart 3: Tourist vs Local ─────────────────────────────────────────────────

def chart_tourist(df: pd.DataFrame, out_dir: Path):
    """Stacked tourist/local timeline, accessory bar, tourist-index trend."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Tourist vs Local Analysis — Kabukicho, Shinjuku", fontsize=13)

    # Stacked bar: tourists vs locals
    ax = axes[0]
    ax.bar(df["elapsed_min"], df["tourists_avg"], width=0.5,
           color=C["tourists"], label="Tourists (carry luggage / bags)", alpha=0.85)
    ax.bar(df["elapsed_min"], df["locals_avg"],   width=0.5,
           bottom=df["tourists_avg"], color=C["locals"], label="Locals", alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(df["elapsed_min"], df["tourist_index"] * 100,
             color="black", lw=1.5, linestyle="--", alpha=0.5, label="Tourist %")
    ax2.set_ylabel("Tourist index (%)")
    ax.set_xlabel("Minutes"); ax.set_ylabel("Avg count / frame")
    ax.set_title("Tourist vs Local over time")
    ax.legend(fontsize=8, loc="upper left")

    # Accessory breakdown
    ax = axes[1]
    labels = ["Suitcases\n(strong tourist)", "Backpacks\n(moderate)",
              "Shopping bags\n(retail activity)", "Umbrellas\n(weather)"]
    vals   = [df["suitcases_avg"].mean(), df["backpacks_avg"].mean(),
              df["shopping_bags_avg"].mean(), df["umbrellas_avg"].mean()]
    colors = [C["suitcase"], C["backpack"], C["bag"], C["umbrella"]]
    bars   = ax.bar(range(4), vals, color=colors, edgecolor="white", width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Avg detected / frame")
    ax.set_title("Accessories detected\n(tourist classification signals)")

    # Tourist index trend
    ax = axes[2]
    ax.fill_between(df["elapsed_min"], df["tourist_index"] * 100,
                    color=C["tourists"], alpha=0.55)
    ax.plot(df["elapsed_min"], df["tourist_index"] * 100,
            color=C["tourists"], lw=1.5)
    ax.axhline(df["tourist_index"].mean() * 100, color="black", lw=1.5,
               linestyle="--", label=f"Session avg: {df['tourist_index'].mean():.0%}")
    ax.set_xlabel("Minutes"); ax.set_ylabel("Tourist index (%)")
    ax.set_title("Tourist density trend\n(% of people classified as tourists)")
    ax.set_ylim(0, 100); ax.legend(fontsize=9)

    insight = (
        f"{'Tourist-heavy' if df['tourist_index'].mean() > 0.4 else 'Mixed tourists & locals'}\n"
        f"Avg tourist index : {df['tourist_index'].mean():.0%}\n"
        f"Suitcases / frame : {df['suitcases_avg'].mean():.2f}\n"
        f"Backpacks / frame : {df['backpacks_avg'].mean():.2f}"
    )
    axes[2].text(0.97, 0.97, insight, transform=axes[2].transAxes,
                 fontsize=8, va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e1"))

    plt.tight_layout()
    save(fig, "03_tourist_local.png", out_dir)


# ── Chart 4: Demographics & Social Behaviour ──────────────────────────────────

def chart_demographics(df: pd.DataFrame, out_dir: Path):
    """Adult/child timeline + pie, solo vs group bar."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Crowd Demographics & Social Behaviour", fontsize=13)

    # Adult vs child over time
    ax = axes[0]
    ax.plot(df["elapsed_min"], df["adults_avg"],   color=C["adults"],   lw=2, label="Adults")
    ax.plot(df["elapsed_min"], df["children_avg"], color=C["children"], lw=2, label="Children")
    ax.fill_between(df["elapsed_min"], df["adults_avg"],   alpha=0.15, color=C["adults"])
    ax.fill_between(df["elapsed_min"], df["children_avg"], alpha=0.15, color=C["children"])
    ax.set_xlabel("Minutes"); ax.set_ylabel("Avg count / frame")
    ax.set_title("Adults vs Children over time\n(height-based estimation)")
    ax.legend()

    # Age pie
    ax = axes[1]
    ta = df["adults_avg"].sum();  tc = df["children_avg"].sum()
    tot = ta + tc
    if tot > 0:
        ax.pie([ta, tc],
               labels=[f"Adults ({ta/tot:.0%})", f"Children ({tc/tot:.0%})"],
               colors=[C["adults"], C["children"]], autopct="%1.1f%%",
               startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("Age split\n(Kabukicho = entertainment district,\nmostly adults)")

    # Solo vs group
    ax = axes[2]
    ax.bar(df["elapsed_min"] - 0.18, df["solo_avg"],        width=0.34,
           color=C["locals"],   label="Solo people",    alpha=0.85)
    ax.bar(df["elapsed_min"] + 0.18, df["group_count_avg"], width=0.34,
           color=C["tourists"], label="Groups (3+ ppl)", alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(df["elapsed_min"], df["largest_group"], color="black", lw=1.2,
             linestyle=":", marker="^", markersize=3, label="Largest group")
    ax2.set_ylabel("Largest group size")
    ax2.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("Minutes"); ax.set_ylabel("Avg count / frame")
    ax.set_title("Solo vs Group behaviour\n(tourist groups tend to be larger)")
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    save(fig, "04_demographics.png", out_dir)


# ── Chart 5: Crowd Comparison — Daytime vs Evening ────────────────────────────

def chart_crowd_comparison(df_day: pd.DataFrame, df_eve: pd.DataFrame, out_dir: Path):
    """Crowd comparison — Daytime session vs Evening session."""
    DAY_COL = "#f9a825"
    EVE_COL = "#1565c0"

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        "Crowd Comparison — Daytime vs Evening\n"
        "Kabukicho, Shinjuku  |  Same location, different time of day",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                           left=0.06, right=0.97, top=0.85, bottom=0.12)

    # ── Panel 1: Key metrics bar chart ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    metrics      = ["Avg Crowd", "Peak Crowd", "Avg Adults", "Avg Children"]
    day_vals     = [df_day["crowd_avg"].mean(), df_day["crowd_max"].max(),
                    df_day["adults_avg"].mean(), df_day["children_avg"].mean()]
    eve_vals     = [df_eve["crowd_avg"].mean(), df_eve["crowd_max"].max(),
                    df_eve["adults_avg"].mean(), df_eve["children_avg"].mean()]
    x = np.arange(len(metrics))
    bars_d = ax1.bar(x - 0.22, day_vals, width=0.40, color=DAY_COL,
                     label="Daytime (12:39 JST)", alpha=0.88)
    bars_e = ax1.bar(x + 0.22, eve_vals, width=0.40, color=EVE_COL,
                     label="Evening (18:57 JST)", alpha=0.88)
    for bar in list(bars_d) + list(bars_e):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.15,
                 f"{bar.get_height():.1f}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=9)
    ax1.set_ylabel("People per frame")
    ax1.set_title("Crowd Metrics", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)

    # ── Panel 2: Crowd timeline — both sessions on same axis ─────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df_day["elapsed_min"], df_day["crowd_avg"],
             color=DAY_COL, lw=2.5, marker="o", markersize=3, label="Daytime")
    ax2.fill_between(df_day["elapsed_min"], df_day["crowd_avg"],
                     color=DAY_COL, alpha=0.18)
    ax2.plot(df_eve["elapsed_min"], df_eve["crowd_avg"],
             color=EVE_COL, lw=2.5, marker="s", markersize=3, label="Evening")
    ax2.fill_between(df_eve["elapsed_min"], df_eve["crowd_avg"],
                     color=EVE_COL, alpha=0.18)
    ax2.axhline(df_day["crowd_avg"].mean(), color=DAY_COL,
                lw=1.2, linestyle="--", alpha=0.7,
                label=f"Day avg ({df_day['crowd_avg'].mean():.1f})")
    ax2.axhline(df_eve["crowd_avg"].mean(), color=EVE_COL,
                lw=1.2, linestyle="--", alpha=0.7,
                label=f"Eve avg ({df_eve['crowd_avg'].mean():.1f})")
    ax2.set_xlabel("Minutes into recording")
    ax2.set_ylabel("Avg crowd / frame")
    ax2.set_title("Crowd Timeline\n(both sessions, 0–10 min)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8.5)

    # ── Panel 3: Activity breakdown comparison ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    all_labels = sorted(set(df_day["activity"].unique()) | set(df_eve["activity"].unique()))
    day_counts = [df_day["activity"].value_counts().get(a, 0) for a in all_labels]
    eve_counts = [df_eve["activity"].value_counts().get(a, 0) for a in all_labels]
    x3 = np.arange(len(all_labels))
    ax3.bar(x3 - 0.22, day_counts, width=0.40, color=DAY_COL,
            label="Daytime", alpha=0.88)
    ax3.bar(x3 + 0.22, eve_counts, width=0.40, color=EVE_COL,
            label="Evening", alpha=0.88)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(all_labels, fontsize=8.5, rotation=15, ha="right")
    ax3.set_ylabel("Number of windows")
    ax3.set_title("Activity Level Breakdown\n(how many 30-sec windows per label)",
                  fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)

    save(fig, "05_crowd_comparison.png", out_dir)


# ── Chart 6: Gender Comparison — Daytime vs Evening ──────────────────────────

def chart_gender_comparison(df_day: pd.DataFrame, df_eve: pd.DataFrame, out_dir: Path):
    """Gender comparison — Daytime session vs Evening session."""
    MALE_COL   = "#1565c0"
    FEMALE_COL = "#e91e63"
    DAY_COL    = "#f9a825"
    EVE_COL    = "#1a237e"

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(
        "Gender Distribution Comparison — Daytime vs Evening\n"
        "Estimated from upper-body appearance (DeepFace) — statistical approximation at crowd level",
        fontsize=12, fontweight="bold",
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40,
                           left=0.06, right=0.97, top=0.83, bottom=0.12)

    # ── Panel 1: Pie charts side by side ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    ax1.set_title("Overall Gender Split", fontsize=11, fontweight="bold", pad=12)

    # draw two pies manually using inset axes
    ax_day = fig.add_axes([0.07, 0.18, 0.22, 0.58])
    ax_eve = fig.add_axes([0.30, 0.18, 0.22, 0.58])

    for ax, df, label, edge in [(ax_day, df_day, "Daytime\n12:39 JST", DAY_COL),
                                 (ax_eve, df_eve, "Evening\n18:57 JST", EVE_COL)]:
        m = df["male_avg"].sum()
        f = df["female_avg"].sum()
        t = m + f
        if t > 0:
            ax.pie(
                [m, f],
                labels=[f"Male\n{m/t:.0%}", f"Female\n{f/t:.0%}"],
                colors=[MALE_COL, FEMALE_COL],
                autopct="%1.0f%%",
                startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
                textprops={"fontsize": 10},
            )
        ax.set_title(label, fontsize=10, fontweight="bold",
                     color=edge, pad=6)

    # ── Panel 2: Male ratio bar comparison ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sessions  = ["Daytime\n12:39 JST", "Evening\n18:57 JST"]
    male_r    = [df_day["male_ratio"].mean() * 100,
                 df_eve["male_ratio"].mean() * 100]
    female_r  = [100 - r for r in male_r]
    x = np.arange(2)

    bars_m = ax2.bar(x, male_r,   width=0.45, color=MALE_COL,   label="Male %",   alpha=0.88)
    bars_f = ax2.bar(x, female_r, width=0.45, bottom=male_r,
                     color=FEMALE_COL, label="Female %", alpha=0.88)

    for bar, val in zip(bars_m, male_r):
        ax2.text(bar.get_x() + bar.get_width() / 2, val / 2,
                 f"{val:.0f}%", ha="center", va="center",
                 fontsize=13, fontweight="bold", color="white")
    for bar, val, base in zip(bars_f, female_r, male_r):
        ax2.text(bar.get_x() + bar.get_width() / 2, base + val / 2,
                 f"{val:.0f}%", ha="center", va="center",
                 fontsize=13, fontweight="bold", color="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels(sessions, fontsize=10)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_ylim(0, 110)
    ax2.set_title("Male vs Female Ratio\n(100% stacked)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)

    # ── Panel 3: Avg counts per frame ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    vals_d = [df_day["male_avg"].mean(), df_day["female_avg"].mean()]
    vals_e = [df_eve["male_avg"].mean(), df_eve["female_avg"].mean()]
    xlabels = ["Male", "Female"]
    x3 = np.arange(2)

    bars_d = ax3.bar(x3 - 0.22, vals_d, width=0.40, color=[MALE_COL, FEMALE_COL],
                     alpha=0.60, label="Daytime", edgecolor=DAY_COL, linewidth=2)
    bars_e = ax3.bar(x3 + 0.22, vals_e, width=0.40, color=[MALE_COL, FEMALE_COL],
                     alpha=0.95, label="Evening", edgecolor=EVE_COL, linewidth=2)

    for bar in list(bars_d) + list(bars_e):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.03,
                 f"{bar.get_height():.1f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax3.set_xticks(x3)
    ax3.set_xticklabels(xlabels, fontsize=11)
    ax3.set_ylabel("Avg people per frame")
    ax3.set_title("Avg Count per Frame\n(lighter = daytime, darker = evening)",
                  fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)

    save(fig, "06_gender_comparison.png", out_dir)


# ── Chart 7: Correlation Matrix ───────────────────────────────────────────────

def chart_correlation(df: pd.DataFrame, out_dir: Path):
    """Correlation heatmap — shows which metrics are linked (Big Data insight)."""
    rename = {
        "crowd_avg":          "Crowd count",
        "tourist_index":      "Tourist index",
        "children_avg":       "Children",
        "friendliness_idx":   "Friendliness",
        "direction_balance":  "Flow direction",
        "movement_speed_avg": "Movement speed",
        "suitcases_avg":      "Suitcases",
        "backpacks_avg":      "Backpacks",
        "solo_avg":           "Solo people",
        "group_count_avg":    "Groups",
        "brightness_avg":     "Brightness (day/night)",
        "male_ratio":         "Male ratio",
    }
    available = {k: v for k, v in rename.items()
                 if k in df.columns and df[k].std() > 0}
    corr = df[list(available.keys())].rename(columns=available).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, ax=ax, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title(
        "Metric Correlation Matrix\n"
        "Green = metrics that rise together   |   Red = inverse relationship\n"
        "Used for Big Data pattern discovery across thousands of windows",
        fontsize=11,
    )
    plt.tight_layout()
    save(fig, "07_correlations.png", out_dir)


# ── text report ───────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame):
    dur = len(df) * config.AGGREGATION_WINDOW_SECONDS
    w   = 64
    print("\n" + "=" * w)
    print("  KABUKICHO URBAN PULSE — SESSION REPORT")
    print("=" * w)
    print(f"  Location : Kabukicho 1-chome, Shinjuku, Tokyo")
    print(f"  Recorded : {df['time_label'].iloc[0]} to {df['time_label'].iloc[-1]} JST")
    print(f"  Duration : {dur // 60} min {dur % 60} sec  ({len(df)} windows)")
    print()
    print("  CROWD")
    print(f"    Avg / frame        : {df['crowd_avg'].mean():.1f}")
    print(f"    Peak in one frame  : {int(df['crowd_max'].max())}")
    print(f"    Frame coverage     : {df['crowd_density_avg'].mean():.1%}")
    print()
    print("  TOURIST vs LOCAL")
    print(f"    Tourist index      : {df['tourist_index'].mean():.1%}")
    print(f"    Suitcases / frame  : {df['suitcases_avg'].mean():.2f}")
    print(f"    Backpacks / frame  : {df['backpacks_avg'].mean():.2f}")
    print()
    print("  DEMOGRAPHICS")
    print(f"    Adults avg         : {df['adults_avg'].mean():.2f}")
    print(f"    Children avg       : {df['children_avg'].mean():.2f}")
    print()
    print("  GENDER (estimated)")
    if "male_avg" in df.columns and df["male_avg"].sum() > 0:
        print(f"    Avg male / frame   : {df['male_avg'].mean():.2f}")
        print(f"    Avg female / frame : {df['female_avg'].mean():.2f}")
        print(f"    Male ratio         : {df['male_ratio'].mean():.0%}")
        cov = (df["male_avg"] + df["female_avg"]) / df["crowd_avg"].clip(0.01) * 100
        print(f"    Coverage           : {cov.mean():.0f}% of crowd estimated")
    else:
        print("    No gender data yet")
    print()
    print("  CROWD FLOW")
    right = (df["direction_balance"] > 0.1).mean() * 100
    left  = (df["direction_balance"] < -0.1).mean() * 100
    print(f"    Rightward windows  : {right:.0f}%")
    print(f"    Leftward windows   : {left:.0f}%")
    print(f"    Avg movement speed : {df['movement_speed_avg'].mean():.2f}")
    print(f"    Slow (browsing)    : {df['speed_slow_pct'].mean() * 100:.0f}% of frames")
    print(f"    Fast (rushing)     : {df['speed_fast_pct'].mean() * 100:.0f}% of frames")
    print()
    print("  FRIENDLINESS")
    print(f"    Avg waving / frame : {df['waving_avg'].mean():.2f}")
    print(f"    Friendliness index : {df['friendliness_idx'].mean():.1%}")
    print()
    print("  ACTIVITY BREAKDOWN")
    for lvl, cnt in df["activity"].value_counts().items():
        print(f"    {lvl:<16} : {cnt:>4} windows ({cnt / len(df):.0%})")
    print("=" * w + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",         default=str(config.DB_PATH))
    parser.add_argument("--compare-db", default=str(config.DATA_DIR / "kabukicho_evening.db"))
    parser.add_argument("--out",        default=str(config.OUTPUT_DIR))
    parser.add_argument("--all",        action="store_true",
                        help="Use all data in the DB (default: today only)")
    args = parser.parse_args()

    df = load(args.db, real_only=not args.all)
    print(f"[analysis] {len(df)} observations loaded from {Path(args.db).name}\n")

    # try to load the comparison (evening) database
    compare_path = Path(args.compare_db)
    df_compare   = None
    if compare_path.exists():
        try:
            df_compare = load(str(compare_path), real_only=False)
            print(f"[analysis] {len(df_compare)} observations loaded from {compare_path.name} (comparison)\n")
        except Exception:
            df_compare = None

    out = Path(args.out)
    print_report(df)
    chart_dashboard(df, out)
    chart_tourist(df, out)
    chart_demographics(df, out)

    # comparison charts — only when both sessions exist
    if df_compare is not None:
        chart_crowd_comparison(df, df_compare, out)
        chart_gender_comparison(df, df_compare, out)
    else:
        print("[analysis] Only one session found — skipping comparison charts.")

    chart_correlation(df, out)
    print(f"\n[analysis] All charts saved to {out}/")


if __name__ == "__main__":
    main()
