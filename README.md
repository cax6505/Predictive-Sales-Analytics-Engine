<h1 align="center">Predictive Sales Analytics Engine</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-orange.svg" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/MLOps-Pipeline-brightgreen.svg" alt="MLOps">
  <img src="https://img.shields.io/badge/Status-Production_Ready-success.svg" alt="Status">
</p>

## Executive Summary
Customer acquisition costs are inherently high. This engine predicts whether a customer will make a **repeat purchase within 180 days** after their first delivered order. By framing the architecture around proactive CRM retention rather than retroactive sentiment analysis, this project delivers actionable, quantifiable business intelligence.

Using the Brazilian E-Commerce Public Dataset by Olist, this repository pipelines raw relational databases into a rigorous, leakage-safe machine learning architecture that targets the massive class imbalance inherent in e-commerce retention.

---

## System Architecture & MLOps Pipeline
This structure diverges from standard monolithic Jupyter experiments. Logic has been modularized into `src/` libraries, making it fully reproducible, testable, and automated via chronologically strict `scripts/`.

### 1. Leakage-Safe Data Engineering
* **Temporal Splitting:** Most models fail in production because they train on future artifacts (`train_test_split`). This pipeline enforces strict **time-based chronological cutoffs**—the model trains strictly on the past to predict the out-of-time future.
* **Target Encoding & Sparsity Control:** Categorical high-cardinality features (like 70+ product categories) are compressed using smoothed Target Encoders, optimized exclusively on the training matrix to prevent data bleeding.

### 2. Feature Synthesis
Raw inputs are mathematically augmented to capture human psychology:
* **`freight_ratio`**: Quantifying "shipping shock" (freight cost relative to total order value).
* **`delivery_delay_days`**: Measuring the physical delta between estimated and actual delivery, isolating the root cause of post-purchase dissonance.
* **TF-IDF NLP:** Extracting weighted vector embeddings from raw customer text reviews.

### 3. Baseline Modeling & Evaluation
To combat a severe `~97:3` Class Imbalance, traditional Accuracy metrics are discarded. 
* **Loss Function Modification:** Algorithms utilize `class_weight='balanced'` to actively penalize minority-class miss-classifications.
* **Hyperparameter Thresholding:** We dynamically calculate the probability threshold on the Validation set that maximizes **F1 / PR-AUC**, porting that exact threshold to the held-out Test strict.

#### Final Phase-1 Metrics (Tabular Random Forest)
| Metric | Score | Interpretation |
|---|---|---|
| **PR-AUC** | `0.0229` | Area under the precision-recall curve. |
| **ROC-AUC** | `0.5698` | Ability to distinguish between churned/retained. |
| **Precision@10%** | `0.0255` | Performance isolated to the top 10% most likely repeaters. |
| **Lift@10%** | `1.5756` | Model predicts returning users **1.57x** better than random guessing. |

---

## Repository Structure

```tree
.
├── configs/                  # YAML configurations preventing hardcoded variables
├── data/
│   ├── raw/                  # Immutable source CSVs
│   └── processed/            # Serialized train/val/test splits
├── notebooks/                # R&D Jupyter environments (EDA, Prototyping)
├── scripts/                  # Automated CI/CD Execution Pipeline
├── src/sales_analytics/      # Core ML library (features, encoding, models)
├── outputs/            # Generated explainability plots and metrics
├── submission/               # Final reports and architecture alignment
└── README.md
```

---

## Execution & Deployment

This project natively supports automated end-to-end execution. Ensure your virtual environment satisfies `requirements.txt`.

### 1. Run the Full ML Pipeline
Execute the modular pipeline sequentially to rebuild splits, train the algorithms, and export interpretability plots:
```bash
python scripts/01_build_dataset.py
python scripts/02_train_baselines.py
python scripts/04_explain.py
```

### 2. Run CI/CD Integrity Validation
To guarantee compilation integrity before proposing merges, invoke the AST (Abstract Syntax Tree) validator:
```bash
python scripts/05_validate_submission.py
```
*This asserts zero structural syntax errors, 100% data layout integrity, and zero cross-set leakage.*

---

## Model Explainability
Black-box models are unacceptable for enterprise integration. Utilizing **Permutation Feature Importance** and **Partial Dependence Plots (PDP)**, the engine physically extracts and ranks the mathematical rules driving its predictions. 

View the exported `outputs/` directory to trace exactly how features like `review_score` and `delivery_delay` alter outcome probabilities.
