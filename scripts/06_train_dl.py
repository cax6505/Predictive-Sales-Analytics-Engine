"""
Phase 2 Deep Learning Training Script
Single model: RepeatPurchaseNet (MLP with Feature Gating + Residual + Entity Embeddings)
Handles heterogeneous data: numeric features + categorical entity embeddings
"""
import sys, os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.sales_analytics.metrics import compute_metrics

# ── Reproducibility ──────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_DIR = Path(__file__).parent.parent / "outputs"

# ── Load data ────────────────────────────────────────────────────
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

TARGET_COL = "target_repeat_within_180d"

# ── Feature Engineering (same as baseline notebook 05) ───────────
def engineer_features(df):
    df = df.copy()
    df["price_per_item"] = df["total_price"] / df["item_count"].clip(lower=1)
    df["freight_pct"] = df["total_freight"] / (df["total_price"] + 1e-8) * 100
    df["payment_overhead"] = df["payment_value_total"] - df["total_price"] - df["total_freight"]
    df["delivery_efficiency"] = df["delivery_days"] / (df["freight_ratio"].clip(lower=0.01))
    df["emotional_intensity"] = df["exclamation_count"] + df["question_count"]
    df["word_density"] = df["text_word_count"] / (df["text_char_len"].clip(lower=1))
    df["volume_weight_ratio"] = df["package_volume_cm3_mean"] / (df["product_weight_g_mean"].clip(lower=1))
    df["desc_per_photo"] = df["product_description_lenght_mean"] / (df["product_photos_qty_mean"].clip(lower=1))
    df["multi_seller"] = (df["seller_count"] > 1).astype(int)
    df["multi_product"] = (df["product_count"] > 1).astype(int)
    df["multi_payment"] = (df["payment_type_nunique"] > 1).astype(int)
    df["score_x_delivery"] = df["review_score"] * (1 / (df["delivery_days"].clip(lower=1)))
    df["text_x_score"] = df["text_present"] * df["review_score"]
    df["price_x_late"] = df["log1p_total_price"] * df["late_delivery_flag"]
    return df

train_eng = engineer_features(train_df)
val_eng = engineer_features(val_df)
test_eng = engineer_features(test_df)

# ── Numeric features ─────────────────────────────────────────────
NUM_FEATURES = [
    "review_score", "text_present", "text_char_len", "text_word_count",
    "exclamation_count", "question_count",
    "log1p_total_price", "log1p_total_freight", "log1p_payment_value_total",
    "payment_installments_max", "payment_records", "payment_type_nunique",
    "item_count", "seller_count", "product_count", "same_state_seller_customer",
    "log1p_approval_lag_hours", "delivery_days",
    "delivery_delay_days_clipped", "late_delivery_flag",
    "freight_ratio", "payment_gap", "log1p_product_weight_g_mean",
    "log1p_package_volume_cm3_mean", "product_photos_qty_mean",
    "product_description_lenght_mean", "purchase_month",
    "purchase_quarter", "weekend_purchase_flag",
    "price_per_item", "freight_pct", "payment_overhead",
    "delivery_efficiency", "emotional_intensity", "word_density",
    "volume_weight_ratio", "desc_per_photo",
    "multi_seller", "multi_product", "multi_payment",
    "score_x_delivery", "text_x_score", "price_x_late",
]
NUM_FEATURES = [f for f in NUM_FEATURES if f in train_eng.columns]
NUM_DIM = len(NUM_FEATURES)

# ── Categorical features (for entity embeddings) ─────────────────
CAT_FEATURES = ["payment_type_mode", "product_category_main", "seller_state_mode", "customer_state"]
CAT_FEATURES = [f for f in CAT_FEATURES if f in train_eng.columns]

# Encode categoricals as integers (fit on train, transform all)
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
cat_train = ord_enc.fit_transform(train_eng[CAT_FEATURES].fillna("__MISSING__"))
cat_val = ord_enc.transform(val_eng[CAT_FEATURES].fillna("__MISSING__"))
cat_test = ord_enc.transform(test_eng[CAT_FEATURES].fillna("__MISSING__"))

# Compute vocabulary sizes and embedding dimensions
cat_vocab_sizes = []
cat_embed_dims = []
for i, feat in enumerate(CAT_FEATURES):
    n_unique = int(cat_train[:, i].max()) + 2  # +1 for 0-index, +1 for unknown
    emb_dim = min(50, (n_unique + 1) // 2)  # rule of thumb: emb_dim ~ cardinality/2
    emb_dim = max(emb_dim, 2)  # minimum 2
    cat_vocab_sizes.append(n_unique)
    cat_embed_dims.append(emb_dim)
    print(f"  {feat}: {n_unique} categories → {emb_dim}D embedding")

TOTAL_EMB_DIM = sum(cat_embed_dims)
INPUT_DIM = NUM_DIM + TOTAL_EMB_DIM
print(f"\nNumeric features: {NUM_DIM}")
print(f"Categorical embeddings: {TOTAL_EMB_DIM}D (from {len(CAT_FEATURES)} features)")
print(f"Total input dim after embedding: {INPUT_DIM}")

# ── Impute + Scale numeric ───────────────────────────────────────
imputer = SimpleImputer(strategy="median")
X_train_num = imputer.fit_transform(train_eng[NUM_FEATURES].replace([np.inf, -np.inf], np.nan).values)
X_val_num = imputer.transform(val_eng[NUM_FEATURES].replace([np.inf, -np.inf], np.nan).values)
X_test_num = imputer.transform(test_eng[NUM_FEATURES].replace([np.inf, -np.inf], np.nan).values)

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_num).astype(np.float32)
X_val_num = scaler.transform(X_val_num).astype(np.float32)
X_test_num = scaler.transform(X_test_num).astype(np.float32)

# Categorical as int64 tensors (shift -1 to 0 for unknown)
cat_train = np.clip(cat_train, 0, None).astype(np.int64)
cat_val = np.clip(cat_val, 0, None).astype(np.int64)
cat_test = np.clip(cat_test, 0, None).astype(np.int64)

y_train = train_eng[TARGET_COL].values.astype(np.float32)
y_val = val_eng[TARGET_COL].values.astype(np.float32)
y_test = test_eng[TARGET_COL].values.astype(np.float32)

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
print(f"\nPositive: {int(n_pos)}, Negative: {int(n_neg)}, Ratio: 1:{n_neg/n_pos:.0f}")

# ── DataLoaders (multi-input: numeric + categorical) ─────────────
BATCH_SIZE = 512

def make_loader(X_num, X_cat, y, shuffle=False):
    ds = TensorDataset(
        torch.tensor(X_num, dtype=torch.float32),
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_train_num, cat_train, y_train, shuffle=True)
val_loader = make_loader(X_val_num, cat_val, y_val)
test_loader = make_loader(X_test_num, cat_test, y_test)

print(f"Batch size: {BATCH_SIZE}, Train batches: {len(train_loader)}")


# ══════════════════════════════════════════════════════════════════
# MODEL: RepeatPurchaseNet (with Entity Embeddings)
# ══════════════════════════════════════════════════════════════════

class FeatureGate(nn.Module):
    """Learnable soft feature selection via element-wise gating."""
    def __init__(self, input_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        gates = self.gate_net(x)
        return x * gates, gates


class ResidualBlock(nn.Module):
    """MLP block with residual (skip) connection."""
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        return self.norm(self.block(x) + self.skip(x))


class RepeatPurchaseNet(nn.Module):
    """Custom MLP with Entity Embeddings, Feature Gating, and Residual Connections.

    Handles heterogeneous data:
      - Numeric features → StandardScaled → direct input
      - Categorical features → Entity Embeddings → concatenated with numeric
      - Combined → FeatureGate → ResidualBlocks → Output
    """
    def __init__(self, num_dim, cat_vocab_sizes, cat_embed_dims,
                 hidden_dims=(128, 64), dropout=0.3):
        super().__init__()

        # Entity embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim)
            for vocab_size, emb_dim in zip(cat_vocab_sizes, cat_embed_dims)
        ])
        self.emb_dropout = nn.Dropout(0.2)

        total_input = num_dim + sum(cat_embed_dims)
        self.feature_gate = FeatureGate(total_input)

        blocks = []
        prev_dim = total_input
        for h_dim in hidden_dims:
            blocks.append(ResidualBlock(prev_dim, h_dim, dropout))
            prev_dim = h_dim
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(prev_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x_num, x_cat):
        # Embed categoricals
        emb_list = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        emb_concat = self.emb_dropout(torch.cat(emb_list, dim=1))

        # Concatenate numeric + embeddings → heterogeneous fusion
        x = torch.cat([x_num, emb_concat], dim=1)

        # Feature gate → residual blocks → output
        x, gates = self.feature_gate(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1), gates


# ── Training ─────────────────────────────────────────────────────

model = RepeatPurchaseNet(
    num_dim=NUM_DIM,
    cat_vocab_sizes=cat_vocab_sizes,
    cat_embed_dims=cat_embed_dims,
    hidden_dims=(128, 64),
    dropout=0.3,
)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel:\n{model}")
print(f"Total parameters: {total_params:,}")

pos_weight = torch.tensor([15.0])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

EPOCHS = 200
PATIENCE = 30
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
)


def evaluate(model, loader):
    model.eval()
    all_logits, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for x_num, x_cat, y_batch in loader:
            logits, _ = model(x_num, x_cat)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            all_logits.append(logits)
            all_labels.append(y_batch)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(all_labels)
    probs = torch.sigmoid(all_logits).numpy()
    pr_auc = average_precision_score(all_labels.numpy(), probs)
    return avg_loss, pr_auc


# ── Training loop ────────────────────────────────────────────────
history = {"train_loss": [], "val_loss": [], "train_pr_auc": [], "val_pr_auc": [], "lr": []}
best_val_loss = float("inf")
best_epoch = 0
best_state = None

print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE})...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    for x_num, x_cat, y_batch in train_loader:
        optimizer.zero_grad()
        logits, _ = model(x_num, x_cat)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    train_loss, train_pr_auc = evaluate(model, train_loader)
    val_loss, val_pr_auc = evaluate(model, val_loader)
    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_pr_auc"].append(train_pr_auc)
    history["val_pr_auc"].append(val_pr_auc)
    history["lr"].append(current_lr)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    elif epoch - best_epoch >= PATIENCE:
        print(f"Early stopping at epoch {epoch} (best: {best_epoch})")
        break

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | TrLoss: {train_loss:.4f} | VaLoss: {val_loss:.4f} | "
              f"TrPR: {train_pr_auc:.4f} | VaPR: {val_pr_auc:.4f} | LR: {current_lr:.6f}")

model.load_state_dict(best_state)
print(f"\nRestored best model from epoch {best_epoch}")

# ── Test evaluation ──────────────────────────────────────────────
model.eval()
all_logits = []
with torch.no_grad():
    for x_num, x_cat, _ in test_loader:
        logits, _ = model(x_num, x_cat)
        all_logits.append(logits)
test_logits = torch.cat(all_logits)
test_probs = torch.sigmoid(test_logits).numpy()

dl_metrics = compute_metrics(y_test, test_probs, top_fraction=0.10)

print("\n" + "=" * 55)
print("  RepeatPurchaseNet — Test Set Results")
print("=" * 55)
for k, v in dl_metrics.items():
    print(f"  {k:20s}: {v:.4f}")
print("=" * 55)

# ── Save model + results ─────────────────────────────────────────
torch.save({
    "model_state": best_state,
    "history": history,
    "best_epoch": best_epoch,
    "num_dim": NUM_DIM,
    "num_features": NUM_FEATURES,
    "cat_features": CAT_FEATURES,
    "cat_vocab_sizes": cat_vocab_sizes,
    "cat_embed_dims": cat_embed_dims,
    "input_dim": INPUT_DIM,
    "metrics": dl_metrics,
}, OUT_DIR / "dl_model_checkpoint.pt")

print(f"\nSaved checkpoint to {OUT_DIR / 'dl_model_checkpoint.pt'}")

# ── Ablation Study ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ABLATION STUDY")
print("=" * 55)

ablation_results = {"full_model": dl_metrics}


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


def train_ablation(model_factory, name, loss_fn=None, epochs=100):
    torch.manual_seed(SEED)
    m = model_factory()
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=7)
    crit = loss_fn if loss_fn else nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]))
    best_vl, best_st, best_ep = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        m.train()
        for x_num, x_cat, yb in train_loader:
            opt.zero_grad()
            logits, _ = m(x_num, x_cat)
            l = crit(logits, yb)
            l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            vl_logits = torch.cat([m(xn, xc)[0] for xn, xc, _ in val_loader])
        vl_loss = crit(vl_logits, torch.tensor(y_val)).item()
        sch.step(vl_loss)
        if vl_loss < best_vl:
            best_vl = vl_loss; best_ep = ep
            best_st = {k: v.clone() for k, v in m.state_dict().items()}
        elif ep - best_ep >= 20:
            break
    m.load_state_dict(best_st)
    m.eval()
    with torch.no_grad():
        te_probs = torch.sigmoid(torch.cat([m(xn, xc)[0] for xn, xc, _ in test_loader])).numpy()
    metrics = compute_metrics(y_test, te_probs)
    print(f"  {name:35s} | PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f} (best ep: {best_ep})")
    return metrics


# Ablation 1: No Feature Gate
class NoGateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(v, e) for v, e in zip(cat_vocab_sizes, cat_embed_dims)])
        self.emb_dropout = nn.Dropout(0.2)
        blocks = []
        prev = INPUT_DIM
        for h in (128, 64):
            blocks.append(ResidualBlock(prev, h, 0.3))
            prev = h
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(prev, 1)
    def forward(self, x_num, x_cat):
        embs = self.emb_dropout(torch.cat([e(x_cat[:, i]) for i, e in enumerate(self.embeddings)], dim=1))
        x = torch.cat([x_num, embs], dim=1)
        for b in self.blocks:
            x = b(x)
        return self.head(x).squeeze(-1), None

ablation_results["no_feature_gate"] = train_ablation(NoGateNet, "Without Feature Gate")

# Ablation 2: No Residual (plain MLP)
class PlainMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(v, e) for v, e in zip(cat_vocab_sizes, cat_embed_dims)])
        self.emb_dropout = nn.Dropout(0.2)
        layers = []
        prev = INPUT_DIM
        for h in (128, 64):
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x_num, x_cat):
        embs = self.emb_dropout(torch.cat([e(x_cat[:, i]) for i, e in enumerate(self.embeddings)], dim=1))
        x = torch.cat([x_num, embs], dim=1)
        return self.net(x).squeeze(-1), None

ablation_results["no_residual"] = train_ablation(PlainMLP, "Without Residual Connections")

# Ablation 3: No Entity Embeddings (numeric only)
class NumericOnlyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_gate = FeatureGate(NUM_DIM)
        blocks = []
        prev = NUM_DIM
        for h in (128, 64):
            blocks.append(ResidualBlock(prev, h, 0.3))
            prev = h
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(prev, 1)
    def forward(self, x_num, x_cat):
        x, gates = self.feature_gate(x_num)  # ignore categoricals
        for b in self.blocks:
            x = b(x)
        return self.head(x).squeeze(-1), gates

ablation_results["no_embeddings"] = train_ablation(NumericOnlyNet, "Without Entity Embeddings")

# Ablation 4: Focal Loss
def make_full_model():
    return RepeatPurchaseNet(NUM_DIM, cat_vocab_sizes, cat_embed_dims, (128, 64), 0.3)

ablation_results["focal_loss"] = train_ablation(make_full_model, "With Focal Loss", loss_fn=FocalLoss())

# Save ablation results
def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    return obj

with open(OUT_DIR / "dl_ablation_results.json", "w") as f:
    json.dump(to_serializable(ablation_results), f, indent=2)
print(f"\nAblation results saved to {OUT_DIR / 'dl_ablation_results.json'}")

# ── Feature Gate Analysis ────────────────────────────────────────
print("\n" + "=" * 55)
print("  FEATURE GATE WEIGHTS (learned importance)")
print("=" * 55)
model.eval()
with torch.no_grad():
    x_num_s = torch.tensor(X_train_num[:1000])
    x_cat_s = torch.tensor(cat_train[:1000])
    embs = torch.cat([e(x_cat_s[:, i]) for i, e in enumerate(model.embeddings)], dim=1)
    combined = torch.cat([x_num_s, embs], dim=1)
    _, gates = model.feature_gate(combined)
    mean_gates = gates.mean(dim=0).numpy()

# Label the gate weights
gate_labels = NUM_FEATURES.copy()
for feat, edim in zip(CAT_FEATURES, cat_embed_dims):
    for j in range(edim):
        gate_labels.append(f"{feat}_emb{j}")

gate_importance = pd.Series(mean_gates, index=gate_labels[:len(mean_gates)]).sort_values(ascending=False)
print(gate_importance.head(20).to_string())
gate_importance.to_csv(OUT_DIR / "dl_feature_gate_weights.csv")

# ── Permutation Importance ───────────────────────────────────────
print("\n" + "=" * 55)
print("  PERMUTATION IMPORTANCE (top 15)")
print("=" * 55)

def get_pr_auc_full(model, X_num, X_cat, y):
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.tensor(X_num), torch.tensor(X_cat))
    probs = torch.sigmoid(logits).numpy()
    return average_precision_score(y, probs)

baseline_score = get_pr_auc_full(model, X_test_num, cat_test, y_test)
perm_importance = {}

# Numeric feature importance
for i, feat in enumerate(NUM_FEATURES):
    X_perm = X_test_num.copy()
    rng = np.random.RandomState(SEED + i)
    rng.shuffle(X_perm[:, i])
    perm_score = get_pr_auc_full(model, X_perm, cat_test, y_test)
    perm_importance[feat] = baseline_score - perm_score

# Categorical feature importance
for i, feat in enumerate(CAT_FEATURES):
    X_cat_perm = cat_test.copy()
    rng = np.random.RandomState(SEED + NUM_DIM + i)
    rng.shuffle(X_cat_perm[:, i])
    perm_score = get_pr_auc_full(model, X_test_num, X_cat_perm, y_test)
    perm_importance[feat] = baseline_score - perm_score

perm_df = pd.Series(perm_importance).sort_values(ascending=False)
print(perm_df.head(15).to_string())
perm_df.to_csv(OUT_DIR / "dl_permutation_importance.csv")

print("\nDone!")
