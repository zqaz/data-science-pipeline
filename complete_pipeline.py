"""
Completion script: generates model comparison plots + SHAP analysis
using already-trained models. Run after train_models.py saves pkl files.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
import json
import warnings
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "covid.csv")
OUT_DIR    = os.path.join(BASE_DIR, "pipeline_outputs")
MODELS_DIR = os.path.join(OUT_DIR, "models")
PLOTS_DIR  = os.path.join(OUT_DIR, "plots")
SHAP_DIR   = os.path.join(OUT_DIR, "shap")

SAMPLE_SIZE  = 200_000
SHAP_SAMPLES = 1_000
TARGET       = "DEATH"
RANDOM_STATE = 42

ALL_FEATURES = [
    "SEX","HOSPITALIZED","PNEUMONIA","AGE","PREGNANT","DIABETES",
    "COPD","ASTHMA","IMMUNOSUPPRESSION","HYPERTENSION","OTHER_DISEASE",
    "CARDIOVASCULAR","OBESITY","RENAL_CHRONIC","TOBACCO","COVID_POSITIVE",
]

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

# ── Load data sample ──────────────────────────────────────────────────────────
print("Loading data sample...")
df = pd.read_csv(DATA_PATH)
df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
X = df_sample[ALL_FEATURES]
y = df_sample[TARGET]
print(f"  Sample: {len(df_sample):,} rows, mortality rate: {y.mean()*100:.2f}%")

# ── Load models ───────────────────────────────────────────────────────────────
print("\nLoading pre-trained models...")
model_names = ["Linear Regression","Lasso","Ridge","CART","Random Forest","LightGBM"]
trained_models = {}
for name in model_names:
    path = os.path.join(MODELS_DIR, f"{name.replace(' ','_')}.pkl")
    if os.path.exists(path):
        trained_models[name] = joblib.load(path)
        print(f"  Loaded: {name}")
    else:
        print(f"  MISSING: {name}")

# ── Load CV results ───────────────────────────────────────────────────────────
cv_df = pd.read_csv(os.path.join(OUT_DIR, "cv_results.csv"), index_col=0)
print(f"\nCV Results:\n{cv_df[['CV RMSE','CV R2','AUC']].round(4).to_string()}")

# ── Model Comparison Plots ────────────────────────────────────────────────────
print("\n[Step 1] Generating model comparison plots...")

MODEL_COLORS_LIST = ["#3498DB","#2ECC71","#9B59B6","#E74C3C","#F39C12","#1ABC9C"]
clrs_m = {n: MODEL_COLORS_LIST[i] for i, n in enumerate(cv_df.index)}

fig, axes = plt.subplots(2, 2, figsize=(18, 13))

def model_bar_h(ax, values, errors, title, xlabel, better="low"):
    clrs_v = [clrs_m.get(n, "#888") for n in cv_df.index]
    ax.barh(cv_df.index, values, xerr=errors, color=clrs_v,
            edgecolor="white", capsize=4, alpha=0.87)
    best = values.idxmin() if better == "low" else values.idxmax()
    for i, (idx, v, e) in enumerate(zip(cv_df.index, values, errors)):
        ax.text(v + e + 0.0005, i, f"{v:.4f}", va="center", fontsize=9,
                fontweight="bold" if idx == best else "normal")
    ax.set_title(title, fontweight="bold"); ax.set_xlabel(xlabel)

model_bar_h(axes[0,0], cv_df["CV RMSE"], cv_df["CV RMSE Std"],
            "CV RMSE (lower = better)", "RMSE", better="low")
model_bar_h(axes[0,1], cv_df["CV R2"], cv_df["CV R2 Std"],
            "CV R2 (higher = better)", "R2", better="high")
model_bar_h(axes[1,0], cv_df["AUC"], pd.Series(np.zeros(len(cv_df)), index=cv_df.index),
            "AUC-ROC (higher = better)", "AUC", better="high")
axes[1,0].axvline(0.5, color="gray", linestyle="--", linewidth=1)

x = np.arange(len(cv_df)); w = 0.35
axes[1,1].bar(x - w/2, cv_df["Train RMSE"], w, label="Train RMSE",
              color="steelblue", alpha=0.8, edgecolor="white")
axes[1,1].bar(x + w/2, cv_df["CV RMSE"], w, label="CV RMSE",
              color="crimson",  alpha=0.8, edgecolor="white")
axes[1,1].set_xticks(x)
axes[1,1].set_xticklabels(cv_df.index, rotation=20, ha="right")
axes[1,1].set_title("Overfitting: Train vs CV RMSE", fontweight="bold")
axes[1,1].set_ylabel("RMSE"); axes[1,1].legend()

plt.suptitle("Model Comparison Dashboard", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "13_model_comparison.png"), bbox_inches="tight")
plt.close("all")
print("  Saved: 13_model_comparison.png")

# ── SHAP Analysis ─────────────────────────────────────────────────────────────
print("\n[Step 2] Computing SHAP values...")
shap_bg = X.sample(n=SHAP_SAMPLES, random_state=RANDOM_STATE)
shap_data = {}

for name, model in trained_models.items():
    print(f"  SHAP -> [{name}]...", flush=True)
    try:
        if name in ("CART", "Random Forest", "LightGBM"):
            explainer = shap.TreeExplainer(model, shap_bg)
        else:
            explainer = shap.LinearExplainer(model, shap_bg)

        shap_vals = explainer.shap_values(shap_bg)
        ev = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0
        shap_data[name] = {
            "shap_values": shap_vals,
            "data": shap_bg.copy(),
            "expected_value": ev,
        }

        # Beeswarm
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_vals, shap_bg, show=False, plot_type="dot", alpha=0.5)
        plt.title(f"SHAP Beeswarm - {name}", fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, f"shap_beeswarm_{name.replace(' ','_')}.png"),
                    bbox_inches="tight")
        plt.close("all")

        # Bar
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, shap_bg, show=False, plot_type="bar")
        plt.title(f"SHAP Feature Importance - {name}", fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, f"shap_bar_{name.replace(' ','_')}.png"),
                    bbox_inches="tight")
        plt.close("all")

        # Dependence plots for top-3
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top3 = np.argsort(mean_abs)[::-1][:3]
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        for j, fi in enumerate(top3):
            shap.dependence_plot(fi, shap_vals, shap_bg, ax=axs[j], show=False, alpha=0.35)
            axs[j].set_title(f"SHAP Dependence: {ALL_FEATURES[fi]}", fontweight="bold")
        plt.suptitle(f"Top-3 SHAP Dependence Plots - {name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, f"shap_dependence_{name.replace(' ','_')}.png"),
                    bbox_inches="tight")
        plt.close("all")

        print(f"     Done. EV={ev:.4f}")
    except Exception as e:
        print(f"     WARNING: {e}")

joblib.dump(shap_data, os.path.join(SHAP_DIR, "shap_values.pkl"))
print("  Saved: shap_values.pkl")

# ── Cross-model SHAP comparison ───────────────────────────────────────────────
print("\n[Step 3] Cross-model SHAP comparison plots...")
imp_all = {}
for name, d in shap_data.items():
    mean_abs = np.abs(d["shap_values"]).mean(axis=0)
    imp_all[name] = pd.Series(mean_abs, index=ALL_FEATURES)

imp_df = pd.DataFrame(imp_all)
if not imp_df.empty:
    imp_norm = imp_df.div(imp_df.max(axis=0) + 1e-12, axis=1)

    # Bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    sort_col = "LightGBM" if "LightGBM" in imp_norm.columns else imp_norm.columns[-1]
    imp_norm.sort_values(sort_col, ascending=False).plot(
        kind="bar", ax=ax, colormap="Set2", edgecolor="white", width=0.8)
    ax.set_title("Normalized SHAP Feature Importance Across All Models",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalized Mean |SHAP|"); ax.set_xlabel("Feature")
    ax.legend(loc="upper right"); plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, "shap_importance_all_models.png"), bbox_inches="tight")
    plt.close("all")

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(imp_norm.T, annot=True, fmt=".2f", cmap="YlOrRd",
                ax=ax, linewidths=0.3, annot_kws={"size": 8})
    ax.set_title("SHAP Feature Importance Heatmap (All Models)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Feature"); ax.set_ylabel("Model"); plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, "shap_heatmap_all_models.png"), bbox_inches="tight")
    plt.close("all")
    print("  Cross-model SHAP plots saved.")

print("\n" + "=" * 50)
print("PIPELINE COMPLETE!")
print("=" * 50)
print(f"\nAll outputs saved to: {OUT_DIR}")
print("\nLaunch dashboard with:")
print("  streamlit run app.py")
