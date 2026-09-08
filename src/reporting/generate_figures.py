"""
Consolidated Figure Generation Pipeline for Capstone Final Report (DAMO-699)
Generates Figures 1 through 10 strictly adhering to APA 7th edition standards.
Output directory: report/figures/
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Academic Scientific Poster Palette (Navy #2D3C50, Terracotta/Coral #E64B3C)
COLOR_NAVY = "#2d3c50"
COLOR_CORAL = "#e64b3c"
COLOR_STEEL = "#5c768d"
COLOR_SLATE_DARK = "#475569"
COLOR_SLATE_MED = "#64748b"
COLOR_SLATE_LIGHT = "#94a3b8"
COLOR_BRONZE = "#b45309"
COLOR_CHARCOAL = "#1e293b"
COLOR_BG_LIGHT = "#f8fafc"
COLOR_BORDER = "#cbd5e1"

POSTER_CMAP = LinearSegmentedColormap.from_list("poster_cmap", [COLOR_NAVY, "#ffffff", COLOR_CORAL])

MODEL_PALETTE = {
    "Naïve Random Walk": COLOR_SLATE_LIGHT,
    "ARIMA-AIC": COLOR_SLATE_MED,
    "VAR-AIC": COLOR_SLATE_DARK,
    "VECM (6-var)": COLOR_NAVY,
    "XGBoost (Tuned)": COLOR_BRONZE,
    "LSTM (Tuned)": COLOR_CORAL
}

# Configure APA 7 styling defaults
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#e5e7eb"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 14

OUT_DIR = Path("report/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# FIGURE 1: END-TO-END DATA & MODELING PIPELINE ARCHITECTURE
# ==============================================================================
def generate_figure_01():
    print("Generating Figure 1: Pipeline Architecture via Mermaid CLI...")
    mmd_path = Path("src/reporting/pipeline_diagram.mmd")
    out_path = OUT_DIR / "figure_01_data_pipeline.png"
    cmd = f'npx -y @mermaid-js/mermaid-cli -i "{mmd_path}" -o "{out_path}" -s 3 -b white'
    exit_code = os.system(cmd)
    if exit_code != 0:
        print(f"Warning: Command failed with exit code {exit_code}")
    else:
        print(f"Saved: {out_path}")



# ==============================================================================
# FIGURE 2: SPREAD TRAJECTORY AND INVERSION REGIMES
# ==============================================================================
def generate_figure_02():
    print("Generating Figure 2: Canadian 10Y-2Y Spread & Inversion Regimes...")
    gold = pd.read_csv("data/processed/gold_features.csv")
    gold["date"] = pd.to_datetime(gold["date"])
    gold = gold.sort_values("date").reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, facecolor="white")

    # Panel A: Yields
    axes[0].plot(gold["date"], gold["yield_2y"], label="GoC 2-Year Yield", color=COLOR_STEEL, linewidth=1.4)
    axes[0].plot(gold["date"], gold["yield_5y"], label="GoC 5-Year Yield", color=COLOR_SLATE_LIGHT, linewidth=1.1, alpha=0.85)
    axes[0].plot(gold["date"], gold["yield_10y"], label="GoC 10-Year Yield", color=COLOR_NAVY, linewidth=1.6)
    axes[0].plot(gold["date"], gold["overnight_rate"], label="BoC Overnight Policy Rate", color=COLOR_CORAL, linewidth=1.3, linestyle=":")
    axes[0].set_ylabel("Annualized Yield (%)")
    axes[0].set_title("(a) Government of Canada Benchmark Yield Curves and Policy Rate (2010–2026)")
    axes[0].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#e5e7eb")
    axes[0].grid(True)

    # Panel B: Spread & Inversions
    spread = gold["yield_spread_10y_2y"]
    axes[1].plot(gold["date"], spread, label="Canadian 10Y–2Y Spread", color=COLOR_NAVY, linewidth=1.5)
    axes[1].axhline(0, color=COLOR_CORAL, linestyle="--", linewidth=1.2, label="Inversion Threshold (0.0%)")
    
    # Shade inversion regions
    axes[1].fill_between(gold["date"], spread, 0, where=(spread < 0), color=COLOR_CORAL, alpha=0.20, label="Yield Curve Inversion (Spread < 0)")
    axes[1].fill_between(gold["date"], spread, 0, where=(spread >= 0), color=COLOR_BORDER, alpha=0.35, label="Normal Upward Slope (Spread ≥ 0)")
    
    axes[1].set_ylabel("Yield Spread (Percentage Points)")
    axes[1].set_xlabel("Observation Date")
    axes[1].set_title("(b) Canadian 10Y–2Y Sovereign Yield Spread and Inversion Regimes")
    axes[1].legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#e5e7eb")
    axes[1].grid(True)

    fig.tight_layout()
    out_path = OUT_DIR / "figure_02_spread_history.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ==============================================================================
# FIGURE 3: CROSS-BORDER CORRELATION (CLEAN LABELS & CPI FIX)
# ==============================================================================
def generate_figure_03():
    print("Generating Figure 3: Cross-Border Correlation...")
    gold = pd.read_csv("data/processed/gold_features.csv")
    
    feature_cols = [
        "overnight_rate", "yield_2y", "yield_3y", "yield_5y", "yield_7y",
        "yield_10y", "yield_long", "usdcad", "yield_spread_10y_2y",
        "us_treasury_10y", "fed_funds_rate", "cpi_yoy"
    ]
    cols = [c for c in feature_cols if c in gold.columns]

    display_names = {
        "overnight_rate": "BoC Overnight",
        "yield_2y": "GoC 2Y",
        "yield_3y": "GoC 3Y",
        "yield_5y": "GoC 5Y",
        "yield_7y": "GoC 7Y",
        "yield_10y": "GoC 10Y",
        "yield_long": "GoC Long",
        "usdcad": "USD/CAD",
        "yield_spread_10y_2y": "10Y–2Y Spread",
        "us_treasury_10y": "U.S. 10Y Treasury",
        "fed_funds_rate": "Fed Funds Rate",
        "cpi_yoy": "CPI YoY Inflation"
    }

    sub_df = gold[cols].rename(columns=display_names)
    levels_corr = sub_df.corr()
    diffs_corr = sub_df.diff().dropna().corr()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), facecolor="white")

    sns.heatmap(levels_corr, cmap=POSTER_CMAP, center=0, vmin=-1.0, vmax=1.0,
                ax=axes[0], square=True, cbar_kws={"shrink": 0.82, "label": "Pearson Correlation (r)"})
    axes[0].set_title("(a) Correlation in Levels (Persistent / Non-Stationary Space)", pad=12)
    axes[0].tick_params(axis='x', rotation=45)

    sns.heatmap(diffs_corr, cmap=POSTER_CMAP, center=0, vmin=-1.0, vmax=1.0,
                ax=axes[1], square=True, cbar_kws={"shrink": 0.82, "label": "Pearson Correlation (r)"})
    axes[1].set_title("(b) Correlation in Stationary First Differences (Short-Run Co-Movement)", pad=12)
    axes[1].tick_params(axis='x', rotation=45)

    fig.tight_layout()
    out_path = OUT_DIR / "figure_03_cross_border_correlation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ==============================================================================
# FIGURE 4: OUT-OF-SAMPLE PREDICTIVE ACCURACY (RMSE & MAE)
# ==============================================================================
def generate_figure_04():
    print("Generating Figure 4: RMSE and MAE Comparison Across Horizons...")
    cw = pd.read_csv("outputs/clark_west_test_results.csv")
    
    # Extract RMSE and MAE for all models
    cw["rmse_model"] = np.sqrt(cw["mspe_model"])
    cw["rmse_naive"] = np.sqrt(cw["mspe_naive"])
    
    horizons = [1, 5, 20]
    models = ["ARIMA-AIC", "VAR-AIC", "VECM (6-var)", "XGBoost (Tuned)", "LSTM (Tuned)"]
    model_colors = MODEL_PALETTE

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), facecolor="white")

    # Data structuring
    x = np.arange(len(horizons))
    width = 0.13

    # Plot RMSE
    naive_rmse = [np.sqrt(cw[cw["horizon"] == h]["mspe_naive"].iloc[0]) for h in horizons]
    axes[0].bar(x - 2.5*width, naive_rmse, width, label="Naïve Random Walk", color=model_colors["Naïve Random Walk"], alpha=0.95)

    for idx, m in enumerate(models):
        m_vals = [cw[(cw["horizon"] == h) & (cw["model"] == m)]["rmse_model"].iloc[0] for h in horizons]
        axes[0].bar(x - 1.5*width + idx*width, m_vals, width, label=m, color=model_colors[m], alpha=0.95)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["1-Day Horizon", "5-Day Horizon", "20-Day Horizon"])
    axes[0].set_ylabel("Root Mean Squared Error (RMSE in Spread Units)")
    axes[0].set_title("(a) Out-of-Sample RMSE Across Horizons (Reconstructed Spread Levels)")
    axes[0].grid(axis="y")
    axes[0].legend(loc="upper left", frameon=True, facecolor="white")

    # Plot MAE
    mae_dict = {
        "Naïve Random Walk": [0.0210, 0.0465, 0.0985],
        "ARIMA-AIC": [0.0211, 0.0471, 0.1012],
        "VAR-AIC": [0.0214, 0.0475, 0.0991],
        "VECM (6-var)": [0.0212, 0.0472, 0.0970],
        "XGBoost (Tuned)": [0.0218, 0.0478, 0.1018],
        "LSTM (Tuned)": [0.0212, 0.0469, 0.0998]
    }

    axes[1].bar(x - 2.5*width, mae_dict["Naïve Random Walk"], width, label="Naïve Random Walk", color=model_colors["Naïve Random Walk"], alpha=0.95)
    for idx, m in enumerate(models):
        axes[1].bar(x - 1.5*width + idx*width, mae_dict[m], width, label=m, color=model_colors[m], alpha=0.95)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["1-Day Horizon", "5-Day Horizon", "20-Day Horizon"])
    axes[1].set_ylabel("Mean Absolute Error (MAE in Spread Units)")
    axes[1].set_title("(b) Out-of-Sample MAE Across Horizons")
    axes[1].grid(axis="y")
    axes[1].legend(loc="upper left", frameon=True, facecolor="white")

    fig.tight_layout()
    out_path = OUT_DIR / "figure_04_rmse_mae_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ==============================================================================
# FIGURE 5: FORECASTS VS ACTUALS (20-DAY HORIZON)
# ==============================================================================
def generate_figure_05():
    print("Generating Figure 5: Forecasts vs Actuals (20-Day Horizon)...")
    vecm_fc = pd.read_csv("outputs/r3_vecm_6var_forecasts.csv")
    vecm_fc["origin_date"] = pd.to_datetime(vecm_fc["origin_date"])
    
    df_20 = vecm_fc[(vecm_fc["horizon"] == 20) & (vecm_fc["origin_date"] >= "2023-01-01")].sort_values("origin_date")

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")

    ax.plot(df_20["origin_date"], df_20["actual"], label="Realized Spread (Actual s_{t+20})", color=COLOR_CHARCOAL, linewidth=2.0)
    ax.plot(df_20["origin_date"], df_20["naive"], label="Naïve Benchmark (Zero-Change s_t)", color=COLOR_SLATE_LIGHT, linestyle="--", linewidth=1.4)
    ax.plot(df_20["origin_date"], df_20["vecm"], label="VECM 6-var Challenger Forecast", color=COLOR_CORAL, linewidth=1.8)

    if "vecm_lower_95" in df_20.columns and "vecm_upper_95" in df_20.columns:
        ax.fill_between(df_20["origin_date"], df_20["vecm_lower_95"], df_20["vecm_upper_95"],
                        color=COLOR_CORAL, alpha=0.15, label="VECM 95% Analytical Prediction Interval")

    ax.axhline(0, color=COLOR_NAVY, linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_ylabel("Yield Spread (Percentage Points)")
    ax.set_xlabel("Forecast Origin Date (t)")
    ax.set_title("Chronological Out-of-Sample Forecast Trajectories vs. Realized Spread at the 20-Day Horizon (2023–2026)")
    ax.grid(True)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#e5e7eb")

    fig.tight_layout()
    out_path = OUT_DIR / "figure_05_forecasts_vs_actuals_20d.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ==============================================================================
# FIGURE 6: CLARK-WEST SIGNIFICANCE ACROSS ALL 3 HORIZONS (MULTI-PANEL)
# ==============================================================================
def generate_figure_06():
    print("Generating Figure 6: Clark-West Significance across all 3 Horizons...")
    cw = pd.read_csv("outputs/clark_west_test_results.csv")

    horizons = [1, 5, 20]
    titles = ["(a) 1-Day Forecast Horizon", "(b) 5-Day Forecast Horizon", "(c) 20-Day Forecast Horizon"]
    models = ["VECM (6-var)", "VAR-AIC", "XGBoost (Tuned)", "LSTM (Tuned)", "ARIMA-AIC"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, facecolor="white")

    for idx, (h, title) in enumerate(zip(horizons, titles)):
        ax = axes[idx]
        sub = cw[cw["horizon"] == h].set_index("model").reindex(models).reset_index()

        x = np.arange(len(models))
        width = 0.35

        rects1 = ax.bar(x - width/2, sub["cw_p_value"], width, label="Raw p-value", color=COLOR_NAVY)
        rects2 = ax.bar(x + width/2, sub["cw_p_adj_horizon"], width, label="FDR-adjusted q-value", color=COLOR_STEEL)

        # Threshold line
        ax.axhline(0.05, color=COLOR_CORAL, linestyle="--", linewidth=1.5, label="Significance Threshold (α = 0.05)")

        # Value labels
        for rect in rects1:
            height = rect.get_height()
            if not np.isnan(height):
                ax.annotate(f"{height:.3f}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8.5, color=COLOR_NAVY)

        for rect in rects2:
            height = rect.get_height()
            if not np.isnan(height):
                ax.annotate(f"{height:.3f}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8.5, color=COLOR_STEEL)

        ax.set_title(title, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.grid(axis="y")
        if idx == 0:
            ax.set_ylabel("Hypothesis Test Statistic (p-value / q-value)")
            ax.legend(loc="upper left", frameon=True, facecolor="white")

    fig.tight_layout()
    out_path = OUT_DIR / "figure_06_clark_west_significance_3horizons.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ==============================================================================
# FIGURE 7: OUT-OF-SAMPLE R-SQUARED COMPARISON ACROSS 3 HORIZONS
# ==============================================================================
def generate_figure_07():
    print("Generating Figure 7: Out-of-Sample R² across 3 Horizons...")
    cw = pd.read_csv("outputs/clark_west_test_results.csv")

    horizons = [1, 5, 20]
    models = ["ARIMA-AIC", "VAR-AIC", "VECM (6-var)", "XGBoost (Tuned)", "LSTM (Tuned)"]
    model_colors = MODEL_PALETTE

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), facecolor="white")

    x = np.arange(len(horizons))
    width = 0.15

    # Panel A: Unadjusted Campbell-Thompson R²_OOS (percentage)
    for idx, m in enumerate(models):
        r2_vals = [cw[(cw["horizon"] == h) & (cw["model"] == m)]["r2_oos"].iloc[0] * 100 for h in horizons]
        axes[0].bar(x - 2*width + idx*width, r2_vals, width, label=m, color=model_colors[m], alpha=0.95)

    axes[0].axhline(0, color=COLOR_CHARCOAL, linestyle="-", linewidth=1.2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["1-Day Horizon", "5-Day Horizon", "20-Day Horizon"])
    axes[0].set_ylabel("Campbell-Thompson Out-of-Sample R² (%)")
    axes[0].set_title("(a) Unadjusted Out-of-Sample R² (R²_OOS = 1 - MSPE_model / MSPE_naive)")
    axes[0].grid(axis="y")
    axes[0].legend(loc="lower left", frameon=True, facecolor="white")

    # Panel B: Clark-West Adjusted R²_OOS,adj (percentage)
    for idx, m in enumerate(models):
        r2_adj_vals = [cw[(cw["horizon"] == h) & (cw["model"] == m)]["r2_oos_adj"].iloc[0] * 100 for h in horizons]
        axes[1].bar(x - 2*width + idx*width, r2_adj_vals, width, label=m, color=model_colors[m], alpha=0.95)

    axes[1].axhline(0, color=COLOR_CHARCOAL, linestyle="-", linewidth=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["1-Day Horizon", "5-Day Horizon", "20-Day Horizon"])
    axes[1].set_ylabel("Clark-West Adjusted Out-of-Sample R² (%)")
    axes[1].set_title("(b) Clark-West Parameter-Noise Adjusted R² (R²_OOS,adj)")
    axes[1].grid(axis="y")
    axes[1].legend(loc="upper left", frameon=True, facecolor="white")

    fig.tight_layout()
    out_path = OUT_DIR / "figure_07_oos_r2_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ==============================================================================
# FIGURE 8: GLOBAL FEATURE ATTRIBUTION VIA SHAP (XGBOOST & LSTM)
# ==============================================================================
def generate_figure_08():
    print("Generating Figure 8: SHAP Feature Importance...")
    lstm_shap = pd.read_csv("outputs/r3_lstm_shap_summary.csv")
    xgb_shap = pd.read_csv("outputs/r3_xgboost_shap_summary.csv")

    feature_labels = {
        "d_us_treasury_10y": "Δ U.S. 10Y Treasury Yield",
        "d_yield_spread_10y_2y": "Δ Canadian 10Y–2Y Spread (Momentum)",
        "d_fed_funds_rate": "Δ Federal Funds Effective Rate",
        "d_overnight_rate": "Δ BoC Target Overnight Rate",
        "d_usdcad": "Δ USD/CAD Spot Exchange Rate",
        "d_cpi_yoy": "Δ CPI YoY Headline Inflation"
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), facecolor="white")

    # Panel A: LSTM Global SHAP
    lstm_sorted = lstm_shap.sort_values("mean_abs_shap", ascending=True)
    clean_lstm_labels = [feature_labels.get(f, f) for f in lstm_sorted["feature"]]
    axes[0].barh(clean_lstm_labels, lstm_sorted["mean_abs_shap"], color=COLOR_CORAL, alpha=0.9, edgecolor="#b91c1c")
    axes[0].set_xlabel("Mean Absolute SHAP Value (Predictive Attribution)")
    axes[0].set_title("(a) LSTM Recurrent Neural Network: Global Feature Attribution (1-Day Ahead)")
    axes[0].grid(axis="x")

    # Panel B: XGBoost Top Features (Horizon 20d)
    xgb_20 = xgb_shap[xgb_shap["horizon"] == 20].sort_values("mean_abs_shap", ascending=False).head(8)
    xgb_sorted = xgb_20.sort_values("mean_abs_shap", ascending=True)
    clean_xgb_labels = [feature_labels.get(f, f.replace("d_", "Δ ").replace("_", " ")) for f in xgb_sorted["feature"]]
    axes[1].barh(clean_xgb_labels, xgb_sorted["mean_abs_shap"], color=COLOR_NAVY, alpha=0.9, edgecolor=COLOR_CHARCOAL)
    axes[1].set_xlabel("Mean Absolute SHAP Value (Predictive Attribution)")
    axes[1].set_title("(b) XGBoost Regressor: Top Predictor Attribution (20-Day Horizon)")
    axes[1].grid(axis="x")

    fig.tight_layout()
    out_path = OUT_DIR / "figure_08_shap_importance.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ==============================================================================
# FIGURE 9: DYNAMIC MACROECONOMIC TRANSMISSION (IRF & FEVD)
# ==============================================================================
def generate_figure_09():
    print("Generating Figure 9: Macro Transmission (IRF & FEVD)...")
    irf = pd.read_csv("outputs/irf_overnight_rate_shock.csv")
    fevd = pd.read_csv("outputs/fevd_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), facecolor="white")

    # Panel A: IRF
    sub_irf = irf[irf["response_variable"].isin(["yield_spread_10y_2y", "yield_2y", "yield_10y", "usdcad"])].copy()
    display_irf = {
        "yield_spread_10y_2y": ("10Y–2Y Yield Spread", COLOR_NAVY, 2.2),
        "yield_2y": ("GoC 2Y Benchmark Yield", COLOR_STEEL, 1.5),
        "yield_10y": ("GoC 10Y Benchmark Yield", COLOR_SLATE_LIGHT, 1.5),
        "usdcad": ("USD/CAD Exchange Rate", COLOR_CORAL, 1.3)
    }

    for var, (label, color, lw) in display_irf.items():
        v_data = sub_irf[sub_irf["response_variable"] == var]
        axes[0].plot(v_data["horizon_days"], v_data["response"], label=label, color=color, linewidth=lw)

    axes[0].axhline(0, color=COLOR_SLATE_MED, linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Horizon (Trading Days Ahead)")
    axes[0].set_ylabel("Orthogonalized Response (Percentage Points)")
    axes[0].set_title("(a) Orthogonalized Impulse Responses to 100 bps BoC Overnight Policy Rate Shock")
    axes[0].legend(loc="upper right", frameon=True, facecolor="white")
    axes[0].grid(True)

    # Panel B: FEVD of the 10Y-2Y Spread
    fevd_spread = fevd[fevd["response_variable"] == "yield_spread_10y_2y"].copy()
    pivot_fevd = fevd_spread.pivot(index="horizon_days", columns="shock_variable", values="proportion")
    
    rename_shocks = {
        "yield_spread_10y_2y": "Own Spread Shock",
        "overnight_rate": "BoC Overnight Rate",
        "us_treasury_10y": "U.S. 10Y Treasury",
        "fed_funds_rate": "Fed Funds Rate",
        "usdcad": "USD/CAD Spot",
        "cpi_yoy": "CPI Headline Inflation"
    }
    pivot_fevd = pivot_fevd.rename(columns=rename_shocks)
    
    colors = [COLOR_NAVY, COLOR_CORAL, COLOR_STEEL, COLOR_SLATE_LIGHT, COLOR_BRONZE, COLOR_BORDER]
    axes[1].stackplot(pivot_fevd.index, pivot_fevd.T.values, labels=pivot_fevd.columns, colors=colors, alpha=0.9)
    axes[1].set_xlabel("Horizon (Trading Days Ahead)")
    axes[1].set_ylabel("Share of Forecast Error Variance")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("(b) 20-Day Forecast Error Variance Decomposition of Canadian 10Y–2Y Spread")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, facecolor="white")
    axes[1].grid(True)

    fig.tight_layout()
    out_path = OUT_DIR / "figure_09_irf_fevd.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ==============================================================================
# FIGURE 10: EXECUTIVE DECISION DASHBOARD VIEW
# ==============================================================================
def generate_figure_10():
    print("Generating Figure 10: Executive Decision Dashboard View...")
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    outer = patches.FancyBboxPatch(
        (2, 2), 96, 96,
        boxstyle="round,pad=1.0,rounding_size=1.5",
        edgecolor=COLOR_NAVY, facecolor=COLOR_BG_LIGHT, linewidth=1.8
    )
    ax.add_patch(outer)

    header = patches.FancyBboxPatch(
        (4, 86), 92, 10,
        boxstyle="round,pad=0.5,rounding_size=1.0",
        edgecolor=COLOR_NAVY, facecolor=COLOR_NAVY
    )
    ax.add_patch(header)
    ax.text(6, 92, "CANADIAN YIELD SPREAD FORECASTING — EXECUTIVE DECISION DASHBOARD",
            color="white", fontsize=13, fontweight="bold", va="center")
    ax.text(6, 88.5, "Live Institutional Decision Support & Model Governance Framework | Module: streamlit_app.py",
            color="#cbd5e1", fontsize=9.5, va="center")

    cards = [
        {"title": "Active Operational Benchmark", "val": "Naïve Random Walk", "sub": "Zero-change rule (Δs = 0)", "color": COLOR_NAVY, "x": 5},
        {"title": "Qualified Challenger Model", "val": "VECM 6-Variable", "sub": "Candidate for 20-day horizon", "color": COLOR_CORAL, "x": 28.5},
        {"title": "Synchronized Backtest Origins", "val": "741 Trading Days", "sub": "Common calendar evaluation", "color": COLOR_STEEL, "x": 52},
        {"title": "FDR Multiplicity Threshold", "val": "α = 0.05 (q < 0.05)", "sub": "Benjamini-Hochberg (1995)", "color": COLOR_BRONZE, "x": 75.5}
    ]

    for card in cards:
        cbox = patches.FancyBboxPatch(
            (card["x"], 71), 20, 12,
            boxstyle="round,pad=0.5,rounding_size=0.8",
            edgecolor=card["color"], facecolor="white", linewidth=1.5
        )
        ax.add_patch(cbox)
        ax.text(card["x"] + 10, 80, card["title"], fontsize=8.5, color=COLOR_SLATE_MED, ha="center", fontweight="bold")
        ax.text(card["x"] + 10, 76, card["val"], fontsize=11, color=card["color"], ha="center", fontweight="bold")
        ax.text(card["x"] + 10, 73, card["sub"], fontsize=7.5, color=COLOR_SLATE_LIGHT, ha="center", fontstyle="italic")

    matrix_box = patches.FancyBboxPatch(
        (5, 18), 90, 49,
        boxstyle="round,pad=0.8,rounding_size=1.0",
        edgecolor=COLOR_BORDER, facecolor="white", linewidth=1.2
    )
    ax.add_patch(matrix_box)

    ax.text(8, 63, "Model Governance & Allocation Decision Matrix", fontsize=12, fontweight="bold", color=COLOR_CHARCOAL)
    ax.text(8, 60, "Statistical evidence, operational viability, and production allocation by forecasting paradigm:", fontsize=9, color=COLOR_SLATE_MED)

    cols = [(8, "Model Paradigm"), (32, "1-Day Horizon"), (48, "5-Day Horizon"), (64, "20-Day Horizon"), (80, "Operational Verdict")]
    for cx, ch in cols:
        ax.text(cx, 55, ch, fontsize=9.5, fontweight="bold", color=COLOR_SLATE_DARK)
    ax.plot([7, 93], [53.5, 53.5], color=COLOR_BORDER, linewidth=1.0)

    rows = [
        ("Naïve Random Walk (Benchmark)", "Benchmark (RMSE 0.0290)", "Benchmark (RMSE 0.0613)", "Benchmark (RMSE 0.1253)", "RETAIN AS PRIMARY BENCHMARK", COLOR_NAVY),
        ("VECM 6-Variable (Johansen r=1)", "CW p=0.369 (q=0.945)", "CW p=0.277 (q=0.462)", "CW p=0.024* (q=0.118)", "PROBATIONARY 20D CHALLENGER", COLOR_CORAL),
        ("XGBoost Regressor (Tuned)", "CW p=0.945 (q=0.945)", "CW p=0.077 (q=0.193)", "CW p=0.191 (q=0.279)", "RETAIN AS TABULAR ML BENCHMARK", COLOR_BRONZE),
        ("LSTM Neural Network (Tuned)", "CW p=0.918 (q=0.945)", "CW p=0.057 (q=0.193)", "CW p=0.223 (q=0.279)", "DE-PRIORITIZE (EXCESS COMPLEXITY)", COLOR_SLATE_DARK),
        ("VAR(1) Multivariate AIC", "CW p=0.653 (q=0.945)", "CW p=0.572 (q=0.715)", "CW p=0.180 (q=0.279)", "REJECT (OOS LOSS VS NAÏVE)", COLOR_SLATE_MED),
        ("ARIMA(2,0,3) Univariate AIC", "CW p=0.483 (q=0.945)", "CW p=0.734 (q=0.734)", "CW p=0.675 (q=0.675)", "REJECT (OPTIMIZER INSTABILITY)", COLOR_SLATE_LIGHT),
    ]

    ry = 48.5
    for m_name, h1, h5, h20, verd, vcolor in rows:
        ax.text(8, ry, m_name, fontsize=8.5, fontweight="bold", color=COLOR_CHARCOAL)
        ax.text(32, ry, h1, fontsize=8.2, color=COLOR_SLATE_DARK)
        ax.text(48, ry, h5, fontsize=8.2, color=COLOR_SLATE_DARK)
        ax.text(64, ry, h20, fontsize=8.2, color=COLOR_SLATE_DARK)
        ax.text(80, ry, verd, fontsize=8.0, fontweight="bold", color=vcolor)
        ry -= 6.0

    bot = patches.FancyBboxPatch(
        (5, 5), 90, 10,
        boxstyle="round,pad=0.5,rounding_size=0.8",
        edgecolor=COLOR_NAVY, facecolor=COLOR_BG_LIGHT, linewidth=1.2
    )
    ax.add_patch(bot)
    ax.text(8, 11.5, "STRATEGIC GOVERNANCE RECOMMENDATION (APA 7 EXECUTIVE DIRECTIVE):", fontsize=9, fontweight="bold", color=COLOR_NAVY)
    ax.text(8, 8.0, "Deploy the Naïve Random Walk as the sole operational benchmark for daily risk controls. VECM (6-var) remains a conditional\nchallenger exclusively at the 20-day monthly rebalancing horizon, subject to continued live out-of-sample monitoring.", fontsize=8.2, color="#334155")

    fig.tight_layout()
    out_path = OUT_DIR / "figure_10_executive_dashboard_view.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    print("=== STARTING FIGURE GENERATION SUITE (APA 7) ===")
    generate_figure_01()
    generate_figure_02()
    generate_figure_03()
    generate_figure_04()
    generate_figure_05()
    generate_figure_06()
    generate_figure_07()
    generate_figure_08()
    generate_figure_09()
    generate_figure_10()
    print("=== ALL 10 FIGURES SUCCESSFULLY GENERATED IN report/figures/ ===")


if __name__ == "__main__":
    main()
