"""
COVID-19 Mortality Prediction -- Full ML Pipeline
Trains: Linear Regression, Lasso, Ridge, CART, Random Forest, LightGBM
Generates: EDA plots, CV results, SHAP analysis
Run once before launching the Streamlit app.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import roc_auc_score
import shap
import joblib
import os
import json
import warnings

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "covid.csv")
OUT_DIR    = os.path.join(BASE_DIR, "pipeline_outputs")
MODELS_DIR = os.path.join(OUT_DIR, "models")
PLOTS_DIR  = os.path.join(OUT_DIR, "plots")
SHAP_DIR   = os.path.join(OUT_DIR, "shap")

for d in [OUT_DIR, MODELS_DIR, PLOTS_DIR, SHAP_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
SAMPLE_SIZE  = 200_000
SHAP_SAMPLES = 1_000
CV_FOLDS     = 5
TARGET       = "DEATH"
RANDOM_STATE = 42

plt.rcParams.update({"figure.dpi": 150, "font.size": 11})
sns.set_style("whitegrid")
BLUE, RED = "#2196F3", "#F44336"

# ── 1. Load & Sample ───────────────────────────────────────────────────────────
print("=" * 60)
print("COVID-19 Mortality Prediction Pipeline")
print("=" * 60)
print("\n[1/5] Loading data...")

df = pd.read_csv(DATA_PATH)
print(f"  Full dataset  : {len(df):,} rows × {df.shape[1]} columns")

df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
FEATURES = [c for c in df.columns if c != TARGET]
binary_feats = [c for c in FEATURES if c != "AGE"]
X = df_sample[FEATURES]
y = df_sample[TARGET]

stats = {
    "total_rows":  int(len(df)),
    "sample_rows": int(len(df_sample)),
    "n_features":  len(FEATURES),
    "target":      TARGET,
    "features":    FEATURES,
    "death_rate":  float(df[TARGET].mean()),
    "binary_features": binary_feats,
}
with open(os.path.join(OUT_DIR, "dataset_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

print(f"  Training sample: {len(df_sample):,} rows")
print(f"  Features ({len(FEATURES)}): {FEATURES}")
print(f"  Mortality rate : {df[TARGET].mean()*100:.2f}%")

# ── 2. EDA Plots ───────────────────────────────────────────────────────────────
print("\n[2/5] Generating EDA plots...")
plot_meta = []

def save_plot(fname, title, desc):
    plt.savefig(os.path.join(PLOTS_DIR, fname), bbox_inches="tight")
    plt.close("all")
    plot_meta.append({"file": fname, "title": title, "desc": desc})
    print(f"  Saved: {fname}")

# ── Plot 1: Class balance
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
vc = df[TARGET].value_counts().sort_index()
labels = ["Survived (0)", "Died (1)"]
clrs = [BLUE, RED]
axes[0].bar(labels, vc.values, color=clrs, edgecolor="white", linewidth=0.8)
for i, v in enumerate(vc.values):
    axes[0].text(i, v + vc.values.max() * 0.01, f"{v:,}\n({v/len(df)*100:.1f}%)",
                 ha="center", fontweight="bold", fontsize=10)
axes[0].set_title("Target Class Distribution", fontweight="bold")
axes[0].set_ylabel("Count")
axes[1].pie(vc.values, labels=labels, colors=clrs, autopct="%1.1f%%",
            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Mortality Rate", fontweight="bold")
plt.suptitle("COVID-19 Mortality: Class Balance", fontsize=14, fontweight="bold")
plt.tight_layout()
save_plot("01_class_balance.png", "Class Balance", "Distribution of DEATH target variable")

# ── Plot 2: Age distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["AGE"], bins=60, color=BLUE, edgecolor="white", alpha=0.85)
axes[0].set_title("Age Distribution (All Patients)", fontweight="bold")
axes[0].set_xlabel("Age"); axes[0].set_ylabel("Count")
mu, med = df["AGE"].mean(), df["AGE"].median()
axes[0].axvline(mu,  color="red",    linestyle="--", label=f"Mean: {mu:.1f}")
axes[0].axvline(med, color="orange", linestyle="--", label=f"Median: {med:.1f}")
axes[0].legend()
axes[1].hist(df[df[TARGET]==0]["AGE"], bins=60, alpha=0.6, color=BLUE, label="Survived", density=True)
axes[1].hist(df[df[TARGET]==1]["AGE"], bins=60, alpha=0.6, color=RED,  label="Died",     density=True)
axes[1].set_title("Age Distribution by Outcome", fontweight="bold")
axes[1].set_xlabel("Age"); axes[1].set_ylabel("Density"); axes[1].legend()
plt.suptitle("Age Analysis", fontsize=14, fontweight="bold"); plt.tight_layout()
save_plot("02_age_distribution.png", "Age Distribution", "Age by survival outcome")

# ── Plot 3: Death rate by age group
df_tmp = df.copy()
df_tmp["age_group"] = pd.cut(df_tmp["AGE"], bins=[0,20,40,60,80,120],
                              labels=["0-20","21-40","41-60","61-80","80+"])
ag = df_tmp.groupby("age_group", observed=True)[TARGET].agg(["mean","count"]).reset_index()
ag["mean"] *= 100
fig, ax1 = plt.subplots(figsize=(10, 5))
cmap_vals = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(ag)))
bars = ax1.bar(ag["age_group"], ag["mean"], color=cmap_vals, edgecolor="white")
ax1.set_xlabel("Age Group"); ax1.set_ylabel("Death Rate (%)", color="#c0392b")
ax1.tick_params(axis="y", labelcolor="#c0392b")
ax1.set_title("Death Rate & Patient Count by Age Group", fontsize=13, fontweight="bold")
for bar, val in zip(bars, ag["mean"]):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{val:.1f}%", ha="center", fontweight="bold", fontsize=10)
ax2 = ax1.twinx()
ax2.plot(ag["age_group"], ag["count"], "b-o", linewidth=2, markersize=8)
ax2.set_ylabel("Patient Count", color="blue"); ax2.tick_params(axis="y", labelcolor="blue")
plt.tight_layout()
save_plot("03_age_group_death.png", "Age Group Death Rate", "Mortality rate by age group")

# ── Plot 4: Feature prevalence
prevs = df[binary_feats].mean().sort_values(ascending=True) * 100
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(prevs.index, prevs.values, color=BLUE, edgecolor="white", alpha=0.85)
for bar, val in zip(bars, prevs.values):
    ax.text(val+0.4, bar.get_y()+bar.get_height()/2, f"{val:.1f}%", va="center", fontsize=9)
ax.set_title("Prevalence of Conditions (%)", fontsize=13, fontweight="bold")
ax.set_xlabel("Prevalence (%)"); plt.tight_layout()
save_plot("04_feature_prevalence.png", "Feature Prevalence", "How common each condition is")

# ── Plot 5: Death rate by condition
overall_dr = df[TARGET].mean() * 100
dr = {c: df[df[c]==1][TARGET].mean()*100 for c in binary_feats}
dr_s = pd.Series(dr).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(11, 6))
clrs_dr = [RED if v > overall_dr else BLUE for v in dr_s.values]
bars = ax.barh(dr_s.index, dr_s.values, color=clrs_dr, edgecolor="white", alpha=0.85)
ax.axvline(overall_dr, color="black", linestyle="--", linewidth=1.5,
           label=f"Overall rate: {overall_dr:.1f}%")
ax.set_title("Death Rate Among Patients WITH Each Condition", fontsize=13, fontweight="bold")
ax.set_xlabel("Death Rate (%)"); ax.legend()
for bar, val in zip(bars, dr_s.values):
    ax.text(val+0.2, bar.get_y()+bar.get_height()/2, f"{val:.1f}%", va="center", fontsize=9)
plt.tight_layout()
save_plot("05_death_rate_by_condition.png", "Death Rate by Condition", "Risk per condition")

# ── Plot 6: Correlation heatmap
corr = df[FEATURES + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(15, 13))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            ax=ax, linewidths=0.3, annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold"); plt.tight_layout()
save_plot("06_correlation_heatmap.png", "Correlation Heatmap", "Feature correlations")

# ── Plot 7: Death rate by feature value grid
n_cols = 4
n_rows = int(np.ceil(len(binary_feats) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*3.5))
axes = axes.flatten()
for i, col in enumerate(binary_feats):
    ax = axes[i]
    vals = df.groupby(col)[TARGET].mean() * 100
    ax.bar(["No (0)", "Yes (1)"], vals.values,
           color=[BLUE if idx==0 else RED for idx in vals.index], edgecolor="white")
    ax.axhline(overall_dr, color="gray", linestyle="--", linewidth=1)
    ax.set_title(col, fontweight="bold", fontsize=10)
    ax.set_ylabel("Death Rate (%)", fontsize=8)
    for j, v in enumerate(vals.values):
        ax.text(j, v+0.5, f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
for i in range(len(binary_feats), len(axes)):
    axes[i].axis("off")
plt.suptitle("Death Rate: Feature OFF (0) vs ON (1)", fontsize=14, fontweight="bold")
plt.tight_layout()
save_plot("07_death_rate_grid.png", "Death Rate by Feature", "Binary feature impact on mortality")

# ── Plot 8: Sex analysis
sex = df.groupby("SEX")[TARGET].agg(["mean","count"]).reset_index()
sex["mean"] *= 100; sex["label"] = sex["SEX"].map({0:"Female",1:"Male"})
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
clrs_sex = ["#E91E63", "#2196F3"]
axes[0].bar(sex["label"], sex["mean"], color=clrs_sex, edgecolor="white")
axes[0].set_title("Death Rate by Sex", fontweight="bold"); axes[0].set_ylabel("Death Rate (%)")
for i, (_, row) in enumerate(sex.iterrows()):
    axes[0].text(i, row["mean"]+0.3, f"{row['mean']:.1f}%", ha="center", fontweight="bold")
axes[1].bar(sex["label"], sex["count"], color=clrs_sex, edgecolor="white")
axes[1].set_title("Patient Count by Sex", fontweight="bold"); axes[1].set_ylabel("Count")
for i, (_, row) in enumerate(sex.iterrows()):
    axes[1].text(i, row["count"]+1000, f"{row['count']:,}", ha="center", fontweight="bold")
plt.suptitle("Sex Analysis", fontsize=14, fontweight="bold"); plt.tight_layout()
save_plot("08_sex_analysis.png", "Sex Analysis", "Mortality breakdown by sex")

# ── Plot 9: Boxplot age by top comorbidities
top_cond = ["PNEUMONIA","DIABETES","HYPERTENSION","OBESITY","RENAL_CHRONIC"]
fig, axes = plt.subplots(1, len(top_cond), figsize=(18, 5))
for i, cond in enumerate(top_cond):
    axes[i].boxplot([df[df[cond]==0]["AGE"], df[df[cond]==1]["AGE"]],
                    labels=["No","Yes"], patch_artist=True,
                    boxprops=dict(facecolor="lightblue", color="navy"),
                    medianprops=dict(color="red", linewidth=2))
    axes[i].set_title(cond, fontweight="bold"); axes[i].set_xlabel("Has Condition")
    if i == 0: axes[i].set_ylabel("Age")
plt.suptitle("Age Distribution by Comorbidity", fontsize=14, fontweight="bold"); plt.tight_layout()
save_plot("09_age_by_condition.png", "Age by Condition", "Age distribution for key comorbidities")

# ── Plot 10: Comorbidity co-occurrence
comorb = ["PNEUMONIA","DIABETES","COPD","ASTHMA","HYPERTENSION","CARDIOVASCULAR","OBESITY","RENAL_CHRONIC"]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[comorb].corr(), annot=True, fmt=".2f", cmap="YlOrRd",
            ax=ax, linewidths=0.5, annot_kws={"size": 9})
ax.set_title("Comorbidity Co-occurrence Correlation", fontsize=13, fontweight="bold"); plt.tight_layout()
save_plot("10_comorbidity_cooccurrence.png", "Comorbidity Co-occurrence", "How conditions cluster")

# ── Plot 11: Hospitalized vs not -- breakdown
hosp = df.groupby(["HOSPITALIZED","SEX"])[TARGET].mean().reset_index()
hosp["Death Rate (%)"] = hosp[TARGET]*100
hosp["Hospitalized"] = hosp["HOSPITALIZED"].map({0:"Not Hospitalized",1:"Hospitalized"})
hosp["Sex"] = hosp["SEX"].map({0:"Female",1:"Male"})
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(2); w = 0.35
for j, sex_label in enumerate(["Female","Male"]):
    sub = hosp[hosp["Sex"]==sex_label]
    ax.bar(x + j*w - w/2, sub["Death Rate (%)"].values, w,
           label=sex_label, color=clrs_sex[j], alpha=0.85, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(["Not Hospitalized","Hospitalized"])
ax.set_title("Death Rate by Hospitalization & Sex", fontsize=13, fontweight="bold")
ax.set_ylabel("Death Rate (%)"); ax.legend(); plt.tight_layout()
save_plot("11_hospitalized_sex_death.png", "Hospitalization & Sex", "Death rate breakdown")

# ── Plot 12: COVID positive vs negative
cov = df.groupby("COVID_POSITIVE")[TARGET].mean() * 100
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(["COVID Negative","COVID Positive"], cov.values, color=[BLUE, RED], edgecolor="white")
ax.set_title("Death Rate: COVID Positive vs Negative", fontsize=13, fontweight="bold")
ax.set_ylabel("Death Rate (%)")
for i, v in enumerate(cov.values):
    ax.text(i, v+0.3, f"{v:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
save_plot("12_covid_positive_death.png", "COVID Status vs Death", "Mortality by COVID status")

with open(os.path.join(OUT_DIR, "plot_meta.json"), "w") as f:
    json.dump(plot_meta, f, indent=2)
print(f"  Total EDA plots: {len(plot_meta)}")

# ── 3. Model Training ──────────────────────────────────────────────────────────
print("\n[3/5] Training models with 5-fold CV...")

models_def = {
    "Linear Regression": LinearRegression(),
    "Lasso":             Lasso(alpha=0.0005, max_iter=5000),
    "Ridge":             Ridge(alpha=1.0),
    "CART":              DecisionTreeRegressor(max_depth=8, min_samples_leaf=50, random_state=RANDOM_STATE),
    "Random Forest":     RandomForestRegressor(n_estimators=150, max_depth=12,
                                               min_samples_leaf=50, n_jobs=-1, random_state=RANDOM_STATE),
    "LightGBM":          LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                       num_leaves=63, n_jobs=-1, random_state=RANDOM_STATE, verbose=-1),
}

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_results  = {}
trained_models = {}

for name, model in models_def.items():
    print(f"  Training [{name}]...", flush=True)
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring={
            "neg_rmse": "neg_root_mean_squared_error",
            "r2":       "r2",
            "neg_mae":  "neg_mean_absolute_error",
        },
        return_train_score=True, n_jobs=1,
    )
    model.fit(X, y)
    trained_models[name] = model
    preds = np.clip(model.predict(X), 0, 1)
    auc   = roc_auc_score(y, preds)

    cv_results[name] = {
        "CV RMSE":      float(-scores["test_neg_rmse"].mean()),
        "CV RMSE Std":  float( scores["test_neg_rmse"].std()),
        "CV R2":        float( scores["test_r2"].mean()),
        "CV R2 Std":    float( scores["test_r2"].std()),
        "CV MAE":       float(-scores["test_neg_mae"].mean()),
        "CV MAE Std":   float( scores["test_neg_mae"].std()),
        "Train RMSE":   float(-scores["train_neg_rmse"].mean()),
        "Train R2":     float( scores["train_r2"].mean()),
        "AUC":          float(auc),
    }
    print(f"     RMSE={cv_results[name]['CV RMSE']:.4f}  "
          f"R²={cv_results[name]['CV R2']:.4f}  AUC={auc:.4f}")
    joblib.dump(model, os.path.join(MODELS_DIR, f"{name.replace(' ','_')}.pkl"))

cv_df = pd.DataFrame(cv_results).T
cv_df.to_csv(os.path.join(OUT_DIR, "cv_results.csv"))
print("  CV results saved -> cv_results.csv")

# ── 4. Model Comparison Plots ──────────────────────────────────────────────────
print("\n[4/5] Generating model comparison plots...")

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
            "CV RMSE (↓ better)", "RMSE", better="low")
model_bar_h(axes[0,1], cv_df["CV R2"], cv_df["CV R2 Std"],
            "CV R² (↑ better)", "R²", better="high")
model_bar_h(axes[1,0], cv_df["AUC"], pd.Series(np.zeros(len(cv_df)), index=cv_df.index),
            "AUC-ROC (↑ better)", "AUC", better="high")
axes[1,0].axvline(0.5, color="gray", linestyle="--", linewidth=1)

x = np.arange(len(cv_df)); w = 0.35
axes[1,1].bar(x - w/2, cv_df["Train RMSE"], w, label="Train RMSE",
              color="steelblue", alpha=0.8, edgecolor="white")
axes[1,1].bar(x + w/2, cv_df["CV RMSE"], w, label="CV RMSE",
              color="crimson",  alpha=0.8, edgecolor="white")
axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(cv_df.index, rotation=20, ha="right")
axes[1,1].set_title("Overfitting: Train vs CV RMSE", fontweight="bold")
axes[1,1].set_ylabel("RMSE"); axes[1,1].legend()

plt.suptitle("Model Comparison Dashboard", fontsize=16, fontweight="bold"); plt.tight_layout()
save_plot("13_model_comparison.png", "Model Comparison", "Cross-validation across all models")
print("  Model comparison plot saved.")

# ── 5. SHAP Analysis ──────────────────────────────────────────────────────────
print("\n[5/5] Computing SHAP values...")
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
        shap_data[name] = {"shap_values": shap_vals, "data": shap_bg.copy(), "expected_value": ev}

        # Beeswarm
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_vals, shap_bg, show=False, plot_type="dot", alpha=0.5)
        plt.title(f"SHAP Beeswarm - {name}", fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, f"shap_beeswarm_{name.replace(' ','_')}.png"),
                    bbox_inches="tight"); plt.close("all")

        # Bar
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, shap_bg, show=False, plot_type="bar")
        plt.title(f"SHAP Feature Importance - {name}", fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, f"shap_bar_{name.replace(' ','_')}.png"),
                    bbox_inches="tight"); plt.close("all")

        # Dependence plots for top-3 features
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top3 = np.argsort(mean_abs)[::-1][:3]
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        for j, fi in enumerate(top3):
            shap.dependence_plot(fi, shap_vals, shap_bg, ax=axs[j], show=False, alpha=0.35)
            axs[j].set_title(f"SHAP Dependence: {FEATURES[fi]}", fontweight="bold")
        plt.suptitle(f"Top-3 SHAP Dependence Plots - {name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, f"shap_dependence_{name.replace(' ','_')}.png"),
                    bbox_inches="tight"); plt.close("all")

        print(f"     Done. Expected value: {ev:.4f}")
    except Exception as e:
        print(f"     WARNING: {e}")

joblib.dump(shap_data, os.path.join(SHAP_DIR, "shap_values.pkl"))

# ── SHAP Cross-model comparison ────────────────────────────────────────────────
imp_all = {}
for name, d in shap_data.items():
    mean_abs = np.abs(d["shap_values"]).mean(axis=0)
    imp_all[name] = pd.Series(mean_abs, index=FEATURES)
imp_df = pd.DataFrame(imp_all)
if not imp_df.empty:
    imp_norm = imp_df.div(imp_df.max(axis=0) + 1e-12, axis=1)

    # Bar chart across models
    fig, ax = plt.subplots(figsize=(14, 8))
    sort_col = "LightGBM" if "LightGBM" in imp_norm.columns else imp_norm.columns[-1]
    imp_norm.sort_values(sort_col, ascending=False).plot(kind="bar", ax=ax,
                                                          colormap="Set2", edgecolor="white", width=0.8)
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
    ax.set_title("SHAP Feature Importance Heatmap (All Models)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Feature"); ax.set_ylabel("Model"); plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, "shap_heatmap_all_models.png"), bbox_inches="tight")
    plt.close("all")
    print("  Cross-model SHAP comparison saved.")

# ── Final Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"\nOutputs saved to: {OUT_DIR}")
print("\nCross-Validation Summary:")
print(cv_df[["CV RMSE","CV R2","AUC"]].round(4).to_string())
print("\nRun the dashboard with:")
print("  streamlit run app.py")
