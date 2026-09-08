import json
from pathlib import Path

nb = {
    'cells': [],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.11'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

def add_md(text):
    nb['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.strip().split('\n')]
    })

def add_code(code):
    nb['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in code.strip().split('\n')]
    })

# Cell 0: Header
add_md("""# Consolidated Capstone Report Visual Exhibits Generator (APA 7th Edition)
**DAMO-699 Capstone Project | Group 5**  
**Repository Reference:** Issue #125 (`feat(report): build consolidated summary report notebook to generate APA 7 publication-ready figures`)  
**Target Document:** `docs/course/Final Report Capstone Draft.docx.md`  
**Output Destination:** `report/figures/` (High-resolution 300 DPI PNGs)

---

### Purpose & Scope
This notebook serves as the single source of truth for generating, styling, and exporting all 10 visual exhibits required for the graduate capstone final report. All charts conform strictly to **APA 7th edition** presentation standards:
* **No internal raster titles:** Main figure titles are excluded from graphics and formatted in the document prose.
* **Typographic hierarchy:** Sans-serif base font >= 10 pt for tick labels and >= 12 pt for axis titles and panel tags.
* **Production-grade outputs:** Rendered at **300 DPI** with white backgrounds (`facecolor="white"`) and tight boundaries.
* **Key Enhancements:**
  1. **Figure 1:** Vector pipeline architecture with explicit **EDA diagnostic bridge** between Silver and Gold.
  2. **Figure 3:** Corrected CPI series using `cpi_yoy` with clean financial labels.
  3. **Figure 6:** Multi-horizon 3-panel facet ($h=1, 5, 20$) comparing Clark-West raw $p$-values vs. FDR $q$-values.
  4. **Figure 7:** Multi-horizon Campbell-Thompson and Clark-West Out-of-Sample $R^2$ ($R^2_{OOS}$) comparison across all 5 candidate models.
""")

# Cell 1: Setup & Styling
add_code("""import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from IPython.display import Image, display

# Set working directory to project root if executed from notebooks/05_reporting/
if Path.cwd().name == "05_reporting":
    os.chdir("../..")

OUT_DIR = Path("report/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

print("Setup complete. Output figures will be saved to:", OUT_DIR.resolve())
""")

# Cell 2: Figure 1 Markdown
add_md("""## Figure 1: End-to-End Macro-Financial Ingestion, Feature Engineering, and Modeling Pipeline Architecture
* **Report Section:** Chapter 3 (§3.1 Overview of Data Pipeline, approx. line 145)  
* **Methodological Role:** Visualizes the full Medallion lifecycle, highlighting the vital **EDA bridge** (stationarity tests, calendar harmonization, and CPI vintage mapping) between Silver and Gold.
""")

# Cell 3: Figure 1 Code
add_code("""from src.reporting.generate_figures import generate_figure_01
generate_figure_01()
""")

# Cell 4: Figure 2 Markdown
add_md("""## Figure 2: Historical Trajectory of the Canadian 10Y–2Y Sovereign Yield Spread and Inversion Regimes (2009–2026)
* **Report Section:** Chapter 4 (§4.2 Spread Behavior & §4.3 Inversion Regimes, approx. line 245)  
* **Methodological Role:** Illustrates the benchmark sovereign yield curves and identifies historical inversion regimes (Spread < 0), especially the 2022–2024 monetary tightening episode.
""")

# Cell 5: Figure 2 Code
add_code("""from src.reporting.generate_figures import generate_figure_02
generate_figure_02()
""")

# Cell 6: Figure 3 Markdown
add_md("""## Figure 3: Cross-Border Correlation and Co-Movement of Canadian and U.S. Interest Rates in Levels and First Differences
* **Report Section:** Chapter 4 (§4.4 Co-movement Between Canada & U.S., approx. line 285)  
* **Methodological Role:** Compares persistent correlation in non-stationary levels versus short-run co-movement in first differences. Solves the missing CPI row/column defect using `cpi_yoy` with clean financial labels.
""")

# Cell 7: Figure 3 Code
add_code("""from src.reporting.generate_figures import generate_figure_03
generate_figure_03()
""")

# Cell 8: Figure 4 Markdown
add_md("""## Figure 4: Out-of-Sample Predictive Accuracy (RMSE and MAE) Across 1-, 5-, and 20-Day Forecast Horizons
* **Report Section:** Chapter 7 (§7.1 Overview of Predictive Accuracy Across Horizons, approx. line 735)  
* **Methodological Role:** Grouped bar chart comparing out-of-sample forecast errors across all 5 candidate models and the Naïve Random Walk benchmark.
""")

# Cell 9: Figure 4 Code
add_code("""from src.reporting.generate_figures import generate_figure_04
generate_figure_04()
""")

# Cell 10: Figure 5 Markdown
add_md("""## Figure 5: Chronological Forecast Trajectories Versus Realized Canadian 10Y–2Y Yield Spread at the 20-Day Horizon
* **Report Section:** Chapter 7 (§7.4 VECM Performance at 20 Days, approx. line 815)  
* **Methodological Role:** Chronological trajectory of out-of-sample forecasts comparing realized spread against VECM 6-var and the Naïve benchmark across evaluation origins (2023–2026).
""")

# Cell 11: Figure 5 Code
add_code("""from src.reporting.generate_figures import generate_figure_05
generate_figure_05()
""")

# Cell 12: Figure 6 Markdown
add_md("""## Figure 6: Clark-West Predictive Accuracy Significance: Raw p-Values Versus False Discovery Rate q-Values Across All Horizons
* **Report Section:** Chapter 7 (§7.7 Clark-West Tests & FDR Control, approx. line 895)  
* **Methodological Role:** Three-panel facet ($h = 1, 5, 20$) displaying Clark-West test statistics against the Naïve benchmark, contrasting raw $p$-values against Benjamini-Hochberg FDR $q$-values relative to alpha = 0.05.
""")

# Cell 13: Figure 6 Code
add_code("""from src.reporting.generate_figures import generate_figure_06
generate_figure_06()
""")

# Cell 14: Figure 7 Markdown
add_md("""## Figure 7: Out-of-Sample Predictive R² and Adjusted R² Comparison Across 1-, 5-, and 20-Day Horizons
* **Report Section:** Chapter 7 (§7.6 Out-of-Sample R² & Predictive Gains, replaces former error distributions)  
* **Methodological Role:** Contrasts unadjusted Campbell-Thompson $R^2_{OOS}$ with Clark-West parameter-noise adjusted $R^2_{OOS,adj}$ across all 5 candidate models and 3 forecast horizons.
""")

# Cell 15: Figure 7 Code
add_code("""from src.reporting.generate_figures import generate_figure_07
generate_figure_07()
""")

# Cell 16: Figure 8 Markdown
add_md("""## Figure 8: Global Feature Attribution and Predictor Importance Ranking via Mean Absolute SHAP Values
* **Report Section:** Chapter 7 (§7.8 Machine Learning Interpretability and Feature Attribution, approx. line 925)  
* **Methodological Role:** Compares global feature attribution for the LSTM recurrent network against top predictors in the regularized XGBoost model.
""")

# Cell 17: Figure 8 Code
add_code("""from src.reporting.generate_figures import generate_figure_08
generate_figure_08()
""")

# Cell 18: Figure 9 Markdown
add_md("""## Figure 9: Dynamic Macroeconomic Transmission: Orthogonalized Impulse Response Functions and Forecast Error Variance Decomposition
* **Report Section:** Chapter 7 (§7.9 Econometric Diagnostics, approx. line 1107)  
* **Methodological Role:** Two-panel exhibit presenting orthogonalized impulse responses to a 100 bps BoC overnight policy rate shock and 20-day FEVD for the Canadian 10Y–2Y spread.
""")

# Cell 19: Figure 9 Code
add_code("""from src.reporting.generate_figures import generate_figure_09
generate_figure_09()
""")

# Cell 20: Figure 10 Markdown
add_md("""## Figure 10: Executive Decision-Support Dashboard: Operational Model Monitoring and Governance Interface
* **Report Section:** Chapter 8 (§8.9 Executive Decision View, line 1121) and Chapter 11 (§11.3 Monitoring Strategy, line 1405)  
* **Methodological Role:** Visualizes the model allocation governance matrix, live monitoring triggers, and institutional decision translation implemented in `streamlit_app.py`.
""")

# Cell 21: Figure 10 Code
add_code("""from src.reporting.generate_figures import generate_figure_10
generate_figure_10()
""")

# Save notebook
out_nb_path = Path('notebooks/05_reporting/generate_report_figures.ipynb')
with open(out_nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Notebook successfully written to:', out_nb_path)

# Execute notebook so visual exhibits are immediately rendered and visible
print('Executing notebook to pre-render visual exhibits...')
try:
    import nbformat
    from nbclient import NotebookClient

    with open(out_nb_path, 'r', encoding='utf-8') as f:
        nb_obj = nbformat.read(f, as_version=4)

    client = NotebookClient(nb_obj, timeout=120, kernel_name='python3')
    client.execute()

    with open(out_nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb_obj, f)

    print('Notebook successfully executed and all 10 visual exhibits pre-rendered!')
except Exception as e:
    print(f'Warning: Could not pre-render notebook exhibits automatically: {e}')
