"""
scorer.py — PosterIQ Scoring Module
Extracted from poster_topsis_improved.ipynb

Public API:
    load_model(state_path)                        → loads .pkl into memory
    is_model_loaded()                             → bool
    evaluate_new_poster(image_path, state_path)   → {score, issues, features}
    detect_issues(row_dict)                       → list[str]
    extract_features(img_path)                    → dict

Training (run from Colab / locally, not needed for inference):
    train_and_save(dataset_path, state_path)      → saves .pkl
"""

import os
import pickle
import logging

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
from skimage.measure import shannon_entropy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

log = logging.getLogger("posteriq.scorer")

# ── Constants ────────────────────────────────────────────────────────────────
NON_CNN_FEATS = [
    "brightness", "contrast", "entropy", "edge_density",
    "colorfulness", "lr_balance", "tb_balance",
    "whitespace", "saliency", "text_density", "word_count",
]
PCA_N         = 20
OPTIMAL_FEATS = ["contrast", "entropy", "colorfulness", "whitespace", "text_density"]

# ── Module-level singletons ───────────────────────────────────────────────────
_resnet    = None
_transform = None
_state     = None          # loaded .pkl dict


# ── ResNet loader (lazy, thread-safe enough for single-worker Flask) ──────────
def _get_resnet():
    global _resnet, _transform
    if _resnet is None:
        log.info("Loading ResNet50 feature extractor…")
        net = models.resnet50(pretrained=True)
        net = torch.nn.Sequential(*list(net.children())[:-1])
        net.eval()
        _resnet = net
        _transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            ),
        ])
    return _resnet, _transform


# ── Model state ──────────────────────────────────────────────────────────────
def load_model(state_path: str = "poster_scorer_state.pkl"):
    """Load trained TOPSIS state from .pkl file into memory."""
    global _state
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"Model state file not found: '{state_path}'\n"
            "Run the training notebook (Cells 1-13) and copy poster_scorer_state.pkl here."
        )
    with open(state_path, "rb") as f:
        _state = pickle.load(f)
    log.info(f"Model state loaded from '{state_path}'")
    return _state


def is_model_loaded() -> bool:
    return _state is not None


def _require_state():
    if _state is None:
        raise RuntimeError(
            "Model not loaded. Call load_model('poster_scorer_state.pkl') first."
        )
    return _state


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(img_path: str) -> dict:
    """
    Extract all features from a poster image.
    Returns a flat dict with 11 hand-crafted + 2048 CNN keys.
    """
    resnet, tfm = _get_resnet()

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # ── Low-level features ──────────────────────────────────────────
    brightness  = float(np.mean(gray))
    contrast    = float(np.std(gray))
    entropy_val = float(shannon_entropy(gray))

    edges     = cv2.Canny(gray, 100, 200)
    edge_dens = float(np.sum(edges > 0) / edges.size)

    B, G, R   = cv2.split(img.astype("float"))
    colorf    = float(np.mean(np.abs(R - G)) + np.mean(np.abs(0.5 * (R + G) - B)))

    # ── Layout balance ───────────────────────────────────────────────
    gf         = gray.astype(np.float64)
    left, right = gf[:, :w // 2].sum(), gf[:, w // 2:].sum()
    top, bottom = gf[:h // 2, :].sum(), gf[h // 2:, :].sum()
    lr_balance  = abs(left  - right)  / (left  + right  + 1e-6)
    tb_balance  = abs(top   - bottom) / (top   + bottom + 1e-6)

    # ── Whitespace ───────────────────────────────────────────────────
    _, thresh  = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    whitespace = float(np.sum(thresh == 255) / thresh.size)

    # ── Saliency (Laplacian proxy) ───────────────────────────────────
    saliency = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_64F))))

    # ── Text density via OCR ─────────────────────────────────────────
    try:
        import pytesseract
        text         = pytesseract.image_to_string(Image.fromarray(gray))
        words        = text.split()
        word_count   = len(words)
        text_density = word_count / (h * w / 1000.0 + 1e-6)
    except Exception:
        word_count, text_density = 0, 0.0

    # ── CNN features ─────────────────────────────────────────────────
    tensor   = tfm(Image.fromarray(img)).unsqueeze(0)
    with torch.no_grad():
        cnn_feat = resnet(tensor).squeeze().numpy()   # (2048,)

    row = {
        "brightness":   brightness,
        "contrast":     contrast,
        "entropy":      entropy_val,
        "edge_density": edge_dens,
        "colorfulness": colorf,
        "lr_balance":   lr_balance,
        "tb_balance":   tb_balance,
        "whitespace":   whitespace,
        "saliency":     saliency,
        "text_density": text_density,
        "word_count":   word_count,
    }
    for i, v in enumerate(cnn_feat):
        row[f"cnn_{i}"] = float(v)
    return row


# ── Issue detection ───────────────────────────────────────────────────────────
def detect_issues(row_dict: dict) -> list[str]:
    """
    Return list of human-readable design issues for a feature dict.
    Uses z-score thresholds vs training distribution statistics.
    """
    st     = _require_state()
    stats  = st["stats"]
    issues = []

    def z(col):
        return (row_dict[col] - stats[col]["mean"]) / (stats[col]["std"] + 1e-6)

    if z("contrast")     < -0.7:  issues.append("Low contrast – text may be hard to read")
    if z("contrast")     >  2.0:  issues.append("Overly harsh contrast – soften slightly")
    if z("entropy")      >  0.7:  issues.append("Too cluttered – reduce visual elements for clarity")
    if z("entropy")      < -0.7:  issues.append("Too simple – lacks visual interest and engagement")
    if z("edge_density") >  1.0:  issues.append("Too many edges – layout appears busy and cluttered")
    if z("whitespace")   < -0.7:  issues.append("Not enough whitespace – design feels cramped")
    if z("colorfulness") < -0.7:  issues.append("Dull color palette – boost vibrancy for visual appeal")
    if z("colorfulness") >  1.0:  issues.append("Too many colors – simplify palette for brand consistency")
    if z("text_density") >  0.7:  issues.append("Too much text – trim copy to key messages only")
    if z("text_density") < -0.7:  issues.append("Too little text – add more informative content")
    if z("lr_balance")   >  1.0 or z("tb_balance") > 1.0:
        issues.append("Layout imbalance – elements are unevenly distributed across the poster")

    return issues


# ── Main inference function ───────────────────────────────────────────────────
def evaluate_new_poster(
    image_path: str,
    state_path: str = "poster_scorer_state.pkl"
) -> dict:
    """
    Score a single new poster image against the trained TOPSIS model.

    Parameters
    ----------
    image_path : str  – path to the poster image
    state_path : str  – path to saved state (.pkl from notebook Cell 13)

    Returns
    -------
    dict:
        score    (float 0-100)
        issues   (list[str])
        features (dict of 11 hand-crafted feature values)
    """
    # Ensure model is loaded
    global _state
    if _state is None:
        load_model(state_path)
    st = _state

    # 1. Raw features
    feat = extract_features(image_path)

    # 2. Non-CNN feature vector
    X_non = pd.DataFrame([{c: feat[c] for c in NON_CNN_FEATS}])

    # 3. CNN → PCA  (use TRAINED pca, not fit_transform!)
    cnn_arr = np.array([feat[f"cnn_{i}"] for i in range(2048)]).reshape(1, -1)
    cnn_red = st["pca_cnn"].transform(cnn_arr)

    X_single = X_non.copy()
    for i in range(cnn_red.shape[1]):
        X_single[f"cnn_pca_{i}"] = cnn_red[0, i]

    # 4. Ideal-based transform  (use TRAINING medians)
    X_adj = X_single.copy()
    for col, med in st["optimal_medians"].items():
        X_adj[col] = -abs(float(X_single[col].iloc[0]) - med)

    # 5. Scale  (use TRAINING scaler)
    X_sc = st["scaler"].transform(X_adj)

    # 6. Weight
    X_wt = X_sc * st["weights"]

    # 7. Distances to TRAINING ideal best / worst reference points
    d_best  = float(np.linalg.norm(X_wt - st["ideal_best_ref"]))
    d_worst = float(np.linalg.norm(X_wt - st["ideal_worst_ref"]))
    raw     = d_worst / (d_best + d_worst + 1e-9)

    # 8. Normalise with TRAINING range → score in ~[0, 100]
    score = 100.0 * (raw - st["score_min"]) / (st["score_max"] - st["score_min"] + 1e-9)
    score = float(np.clip(score, 0, 100))

    # 9. Issue detection
    issues = detect_issues({c: feat[c] for c in NON_CNN_FEATS})

    return {
        "score":    round(score, 2),
        "issues":   issues,
        "features": {k: round(feat[k], 4) for k in NON_CNN_FEATS},
    }


# ── Training helper (call from notebook or CLI, not needed for serving) ───────
def train_and_save(dataset_path: str, state_path: str = "poster_scorer_state.pkl"):
    """
    Re-train TOPSIS model on a folder of poster images and save state.
    Mirrors Cells 5-13 of the notebook.
    """
    rows = []
    for fname in os.listdir(dataset_path):
        fpath = os.path.join(dataset_path, fname)
        try:
            feat = extract_features(fpath)
            feat["file"] = fname
            rows.append(feat)
            log.info(f"  ✓ {fname}")
        except Exception as e:
            log.warning(f"  ✗ {fname}: {e}")

    df       = pd.DataFrame(rows)
    cnn_cols = [c for c in df.columns if c.startswith("cnn_")]

    X_non_cnn = df[NON_CNN_FEATS].copy().astype(float)
    X_cnn     = df[cnn_cols].values.astype(float)

    pca_cnn   = PCA(n_components=PCA_N)
    cnn_red   = pca_cnn.fit_transform(X_cnn)

    X_new = X_non_cnn.copy()
    for i in range(PCA_N):
        X_new[f"cnn_pca_{i}"] = cnn_red[:, i]

    optimal_medians = {c: float(X_new[c].median()) for c in OPTIMAL_FEATS}
    X_adj = X_new.copy()
    for col, med in optimal_medians.items():
        X_adj[col] = -np.abs(X_new[col] - med)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_adj)

    w0  = np.ones(X_scaled.shape[1]) / X_scaled.shape[1]
    Xw0 = X_scaled * w0
    ib0, iw0 = Xw0.max(axis=0), Xw0.min(axis=0)
    db0 = np.linalg.norm(Xw0 - ib0, axis=1)
    dw0 = np.linalg.norm(Xw0 - iw0, axis=1)
    y_proxy = dw0 / (db0 + dw0 + 1e-9)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y_proxy)
    weights = np.abs(ridge.coef_)
    weights = weights / (weights.sum() + 1e-9)

    X_weighted      = X_scaled * weights
    ideal_best_ref  = X_weighted.max(axis=0)
    ideal_worst_ref = X_weighted.min(axis=0)

    d_b = np.linalg.norm(X_weighted - ideal_best_ref, axis=1)
    d_w = np.linalg.norm(X_weighted - ideal_worst_ref, axis=1)
    raw = d_w / (d_b + d_w + 1e-9)
    score_min, score_max = raw.min(), raw.max()

    stats = {c: {"mean": float(X_new[c].mean()), "std": float(X_new[c].std())}
             for c in NON_CNN_FEATS}

    state = {
        "pca_cnn":          pca_cnn,
        "scaler":           scaler,
        "weights":          weights,
        "ideal_best_ref":   ideal_best_ref,
        "ideal_worst_ref":  ideal_worst_ref,
        "score_min":        score_min,
        "score_max":        score_max,
        "optimal_medians":  optimal_medians,
        "stats":            stats,
        "feat_names":       list(X_adj.columns),
    }
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    log.info(f"✅ Saved: {state_path}")
    return state
