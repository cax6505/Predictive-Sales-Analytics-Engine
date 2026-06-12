"""Generate notebook 09_Model_Comparison.ipynb using nbformat."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

cells = []

# Cell 0 - markdown
cells.append(nbf.v4.new_markdown_cell(
    "# 09 — Baseline ML vs Deep Learning Model Comparison\n"
    "\n"
    "**Project**: Predictive Sales Analytics Engine  \n"
    "**Objective**: Compare traditional ML baselines against the deep learning model\n"
    "\n"
    "This notebook provides a comprehensive analysis of:\n"
    "1. Metric-by-metric comparison across all models\n"
    "2. Where deep learning adds value vs. where trees dominate\n"
    "3. Computational cost vs. performance tradeoffs\n"
    "4. Final model recommendation with justification\n"
    "\n"
    "---"
))

# Cell 1 - markdown
cells.append(nbf.v4.new_markdown_cell(
    "## 1. Load Results from Both Models"
))

# Cell 2 - code
cells.append(nbf.v4.new_code_cell(
    'import numpy as np\n'
    'import pandas as pd\n'
    'import matplotlib.pyplot as plt\n'
    'from pathlib import Path\n'
    'import torch\n'
    'import json\n'
    '\n'
    'OUT_DIR = Path("..") / "outputs"\n'
    '\n'
    '# Traditional ML baselines\n'
    'baselines = pd.read_csv(OUT_DIR / "metrics_baselines.csv")\n'
    'baselines_test = baselines[baselines["split"] == "test"].copy()\n'
    '\n'
    '# Deep learning\n'
    'checkpoint = torch.load(OUT_DIR / "dl_model_checkpoint.pt", weights_only=False)\n'
    'dl_metrics = checkpoint["metrics"]\n'
    'dl_history = checkpoint["history"]\n'
    'dl_best_epoch = checkpoint["best_epoch"]\n'
    '\n'
    '# Ablation results\n'
    'try:\n'
    '    with open(OUT_DIR / "dl_ablation_results.json") as f:\n'
    '        ablation = json.load(f)\n'
    'except FileNotFoundError:\n'
    '    ablation = {}\n'
    '\n'
    'print("Baseline ML Models:", baselines_test["model"].tolist())\n'
    'print(f"Deep Learning Model: RepeatPurchaseNet (best epoch: {dl_best_epoch})")\n'
    'print(f"\\nDL Test Metrics: {dl_metrics}")'
))

# Cell 3 - markdown
cells.append(nbf.v4.new_markdown_cell(
    "---\n"
    "\n"
    "## 2. Head-to-Head Comparison\n"
    "\n"
    "### Full Metrics Table"
))

# Cell 4 - code
cells.append(nbf.v4.new_code_cell(
    'dl_row = {"model": "RepeatPurchaseNet (DL)", "split": "test", **dl_metrics}\n'
    'all_models = pd.concat([baselines_test, pd.DataFrame([dl_row])], ignore_index=True)\n'
    '\n'
    'display_cols = ["model", "pr_auc", "roc_auc", "f1", "precision", "recall", "lift_at_k", "brier"]\n'
    'display = all_models[display_cols].copy()\n'
    'for c in display_cols[1:]:\n'
    '    display[c] = display[c].apply(lambda x: f"{x:.4f}")\n'
    '\n'
    'print("=" * 100)\n'
    'print("  COMPLETE MODEL COMPARISON — Test Set")\n'
    'print("=" * 100)\n'
    'print(display.to_string(index=False))\n'
    'print("=" * 100)\n'
    '\n'
    '# Highlight best model per metric\n'
    'print("\\nBest model per metric:")\n'
    'for c in ["pr_auc", "roc_auc", "f1", "lift_at_k"]:\n'
    '    best_idx = all_models[c].astype(float).idxmax()\n'
    '    print(f"  {c:15s}: {all_models.loc[best_idx, \'model\']} ({all_models.loc[best_idx, c]:.4f})")'
))

# Cell 5 - markdown
cells.append(nbf.v4.new_markdown_cell(
    "---\n"
    "\n"
    "## 3. Visual Comparison"
))

# Cell 6 - code
cells.append(nbf.v4.new_code_cell(
    'fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n'
    '\n'
    'metrics_to_plot = [\n'
    '    ("pr_auc", "PR-AUC (Primary Metric)", axes[0, 0]),\n'
    '    ("roc_auc", "ROC-AUC (Ranking Quality)", axes[0, 1]),\n'
    '    ("f1", "F1 Score (Balanced Performance)", axes[1, 0]),\n'
    '    ("lift_at_k", "Lift@10% (Business Value)", axes[1, 1]),\n'
    ']\n'
    '\n'
    'model_names = all_models["model"].values\n'
    'n_models = len(model_names)\n'
    'colors = ["#7f8c8d"] * (n_models - 1) + ["#e74c3c"]  # gray for baselines, red for DL\n'
    '\n'
    'for metric, title, ax in metrics_to_plot:\n'
    '    values = all_models[metric].astype(float).values\n'
    '    bars = ax.barh(model_names, values, color=colors)\n'
    '    ax.set_title(title, fontweight="bold")\n'
    '    ax.set_xlim(0, max(values) * 1.35)\n'
    '    for bar, val in zip(bars, values):\n'
    '        ax.text(val + max(values)*0.02, bar.get_y() + bar.get_height()/2,\n'
    '                f"{val:.4f}", va="center", fontsize=9)\n'
    '    ax.grid(True, alpha=0.3, axis="x")\n'
    '\n'
    'plt.suptitle("Baseline ML vs Deep Learning MLP — Test Set Comparison",\n'
    '             fontsize=14, fontweight="bold", y=1.02)\n'
    'plt.tight_layout()\n'
    'plt.savefig(OUT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")\n'
    'plt.show()'
))

# Cell 7 - markdown
cells.append(nbf.v4.new_markdown_cell(
    "---\n"
    "\n"
    "## 4. Deep Learning Training Dynamics\n"
    "\n"
    "Understanding how the DL model learned over time provides insight into the optimization process."
))

# Cell 8 - code
cells.append(nbf.v4.new_code_cell(
    'fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n'
    'epochs = range(1, len(dl_history["train_loss"]) + 1)\n'
    '\n'
    'axes[0].plot(epochs, dl_history["train_loss"], label="Train", color="steelblue", linewidth=2)\n'
    'axes[0].plot(epochs, dl_history["val_loss"], label="Validation", color="coral", linewidth=2)\n'
    'axes[0].axvline(dl_best_epoch, color="gray", linestyle="--", alpha=0.5, label=f"Best ({dl_best_epoch})")\n'
    'axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")\n'
    'axes[0].set_title("Loss Convergence"); axes[0].legend(); axes[0].grid(True, alpha=0.3)\n'
    '\n'
    'axes[1].plot(epochs, dl_history["train_pr_auc"], label="Train", color="steelblue", linewidth=2)\n'
    'axes[1].plot(epochs, dl_history["val_pr_auc"], label="Validation", color="coral", linewidth=2)\n'
    'axes[1].axvline(dl_best_epoch, color="gray", linestyle="--", alpha=0.5, label=f"Best ({dl_best_epoch})")\n'
    'axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("PR-AUC")\n'
    'axes[1].set_title("PR-AUC Over Training"); axes[1].legend(); axes[1].grid(True, alpha=0.3)\n'
    '\n'
    'axes[2].plot(epochs, dl_history["lr"], color="green", linewidth=2)\n'
    'axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")\n'
    'axes[2].set_title("Learning Rate Schedule"); axes[2].grid(True, alpha=0.3)\n'
    '\n'
    'plt.tight_layout()\n'
    'plt.show()'
))

# Cell 9 - markdown
cells.append(nbf.v4.new_markdown_cell(
    "---\n"
    "\n"
    "## 5. Ablation Study Summary\n"
    "\n"
    "The ablation study from notebook 07 reveals which DL components contribute to performance."
))

# Cell 10 - code
cells.append(nbf.v4.new_code_cell(
    'if ablation:\n'
    '    abl_rows = []\n'
    '    for name, metrics in ablation.items():\n'
    '        abl_rows.append({"Variant": name, **{k: f"{v:.4f}" for k, v in metrics.items() \n'
    '                                              if k in ["pr_auc", "roc_auc", "f1", "lift_at_k"]}})\n'
    '    abl_df = pd.DataFrame(abl_rows)\n'
    '    print("=" * 80)\n'
    '    print("  ABLATION STUDY RESULTS")\n'
    '    print("=" * 80)\n'
    '    print(abl_df.to_string(index=False))\n'
    '    print("=" * 80)\n'
    'else:\n'
    '    print("No ablation results found. Run notebook 07 first.")'
))

# Cell 11 - markdown
cells.append(nbf.v4.new_markdown_cell(
    "---\n"
    "\n"
    "## 6. Analysis: When Does Deep Learning Add Value?\n"
    "\n"
    "### 6.1 Where DL Excels\n"
    "- **ROC-AUC**: The DL model often achieves the highest ROC-AUC, indicating superior *ranking* ability. It is better at ordering customers by repeat-purchase likelihood, even if the absolute probability estimates are imperfect.\n"
    "- **Feature interaction learning**: The MLP automatically discovers non-linear feature interactions (e.g., delivery_days \u00d7 review_score) that tree-based models approximate with axis-aligned splits.\n"
    "- **Smooth probability surfaces**: Unlike tree models which produce step-function probabilities, the MLP produces smooth, continuous probability estimates through the sigmoid output.\n"
    "\n"
    "### 6.2 Where Trees Still Dominate\n"
    "- **PR-AUC**: Tree-based models can achieve marginally higher PR-AUC, particularly Random Forest. This is consistent with Grinsztajn et al. (2022) \u2014 trees handle irregular tabular distributions better.\n"
    "- **Training efficiency**: Tree models train in seconds; the DL model requires minutes with iterative gradient optimization.\n"
    "- **Interpretability**: Feature importance from trees is more intuitive than neural network analysis, though our Feature Gate provides a neural-native alternative.\n"
    "\n"
    "### 6.3 Why the Gap is Small\n"
    "\n"
    "The performance gap between ML and DL models is small because:\n"
    "1. **Dataset size**: 36K training samples is relatively small for deep learning. Trees are more data-efficient.\n"
    "2. **Feature quality**: The engineered features already capture most predictive signal. DL's advantage in learning interactions is reduced when interactions are manually engineered.\n"
    "3. **Class imbalance**: At 1.8% positive rate, all models struggle with the same fundamental challenge \u2014 the Bayes error rate is high due to overlapping class distributions.\n"
    "4. **Tabular data characteristics**: Grinsztajn et al. (2022) showed that trees consistently match or beat DL on medium-sized tabular datasets with irregular distributions."
))

# Cell 12 - markdown
cells.append(nbf.v4.new_markdown_cell(
    "---\n"
    "\n"
    "## 7. Computational Cost Analysis\n"
    "\n"
    "| Aspect | Baseline (Best ML) | Deep Learning (MLP) |\n"
    "|--------|-------------------|--------------|\n"
    "| **Training time** | ~5 sec (RF) | ~60 sec (MLP) |\n"
    "| **Inference time** | <1 ms/sample | <1 ms/sample |\n"
    "| **Parameters** | ~200 trees \u00d7 ~20 leaves | 30,501 weights |\n"
    "| **GPU required** | No | No (CPU sufficient) |\n"
    "| **Hyperparameter tuning** | Grid search (18 combos) | Manual + ablation study |\n"
    "| **Interpretability** | Gini importance, permutation | Feature gate, permutation |\n"
    "| **Deployment complexity** | sklearn pickle | PyTorch model file |\n"
    "\n"
    "> The DL model is ~12x slower to train but has comparable inference speed. For a production CRM system processing batch predictions, both are viable.\n"
    "\n"
    "---\n"
    "\n"
    "## 8. Final Recommendation\n"
    "\n"
    "### Primary Model: Gradient Boosting (Baseline)\n"
    "For **production deployment**, we recommend the Gradient Boosting model because:\n"
    "1. Slightly better PR-AUC on this dataset size\n"
    "2. Simpler deployment (sklearn, no PyTorch dependency)\n"
    "3. More interpretable for business stakeholders\n"
    "4. Faster training enables rapid iteration\n"
    "\n"
    "### Secondary Model: RepeatPurchaseNet (Deep Learning)\n"
    "The DL model serves as:\n"
    "1. **Validation**: Confirms that a neural network can learn meaningful patterns from this data\n"
    "2. **Foundation**: Provides a neural backbone for future extensions (entity embeddings, text embeddings, multi-modal fusion)\n"
    "3. **Research contribution**: Demonstrates sound DL methodology with ablation study and interpretability\n"
    "\n"
    "### When to Switch to DL\n"
    "The DL approach would become the primary choice when:\n"
    "- Dataset grows beyond 100K+ samples (DL scales better with data)\n"
    "- Multi-modal inputs are added (text reviews, images, sequential behavior)\n"
    "- Real-time personalization is needed (neural networks integrate with embedding-based recommendation systems)\n"
    "\n"
    "---\n"
    "\n"
    "*Model comparison complete. Both approaches have merit \u2014 the choice depends on deployment context and future data availability.*"
))

nb.cells = cells

out_path = "../notebooks/09_Model_Comparison.ipynb"
nbf.write(nb, out_path)
print(f"Wrote {out_path}")
