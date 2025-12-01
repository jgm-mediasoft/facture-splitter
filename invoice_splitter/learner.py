from __future__ import annotations
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------
# Rendu PDF → image BGR
# ----------------------------------------------------------
def _render_pdf_to_bgr_array(pdf_path: Path, page_index: int = 0, dpi: int = 300) -> np.ndarray:
    """Rend une page PDF en image BGR (numpy)."""
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)  # RGB
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------
# Features
# ----------------------------------------------------------
def extract_features_from_pdf(pdf_path: Path, dpi: int = 300) -> np.ndarray:
    """
    Features visuelles :
    - ORB (moyenne des descripteurs, 32 dims)
    - Histogrammes couleurs 8x8x8 (512 dims)
    - Moments Hu (7 dims)
    Total: 551 dimensions.
    """
    img = _render_pdf_to_bgr_array(pdf_path, page_index=0, dpi=dpi)

    # ORB
    orb = cv2.ORB_create()
    _kp, des = orb.detectAndCompute(img, None)
    orb_feat = des.mean(axis=0) if des is not None else np.zeros(32, dtype=np.float32)

    # Histogramme couleurs
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)

    # Moments Hu (log)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = cv2.moments(gray)
    hu = cv2.HuMoments(m).flatten()
    hu = (-np.sign(hu) * np.log(np.abs(hu) + 1e-12)).astype(np.float32)

    return np.hstack([orb_feat, hist, hu]).astype(np.float32)


# ----------------------------------------------------------
# Chargement dataset étiqueté
# ----------------------------------------------------------
def load_labeled_dataset(pages_dir: Path, csv_path: Path) -> pd.DataFrame:
    """
    CSV attendu :
    - 'nom' : nom de fichier sans .pdf (ex: page_001)
    - 'tampon' (0/1) et/ou 'sans' (0/1)
    Ajoute :
    - 'chemin_pdf'
    - 'label' (=1 si tampon)
    """
    df = pd.read_csv(csv_path)
    if "nom" not in df.columns:
        raise ValueError("Le CSV doit contenir la colonne 'nom'.")

    df["nom"] = df["nom"].astype(str).str.strip()
    df["chemin_pdf"] = df["nom"].apply(lambda n: str((pages_dir / f"{n}.pdf").resolve()))

    if "tampon" in df.columns:
        df["label"] = df["tampon"].astype(int)
    elif "sans" in df.columns:
        df["label"] = (1 - df["sans"].astype(int)).astype(int)
    else:
        raise ValueError("Le CSV doit contenir 'tampon' ou 'sans'.")

    return df


def build_feature_matrix(df: pd.DataFrame, dpi: int = 300) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Construit la matrice X (features), y (labels) et la liste des noms alignée."""
    X_list, y_list, names = [], [], []
    for _, row in df.iterrows():
        pdf_path = Path(row["chemin_pdf"])
        if not pdf_path.exists():
            continue
        try:
            feats = extract_features_from_pdf(pdf_path, dpi=dpi)
        except Exception:
            continue
        X_list.append(feats)
        y_list.append(int(row["label"]))
        names.append(pdf_path.stem)

    if not X_list:
        raise ValueError("Aucune feature extraite — vérifie PDF/CSV.")

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, names


# ----------------------------------------------------------
# Entraînement / évaluation
# ----------------------------------------------------------
def train_and_evaluate(
    pages_dir: Path,
    csv_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    dpi: int = 300,
):
    """
    Entraîne un RandomForest (class_weight='balanced') avec split train/test stratifié.
    Retourne : metrics, df_pred, model, feature_info, df_all.
    """
    df_all = load_labeled_dataset(pages_dir, csv_path)
    X, y, names = build_feature_matrix(df_all, dpi=dpi)

    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, names, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
    else:
        y_proba = np.zeros_like(y_pred, dtype=float)

    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["sans_tampon", "tampon"],
    )

    df_pred = pd.DataFrame(
        {
            "nom": names_test,
            "y_reel": y_test,
            "y_pred": y_pred,
            "proba_tampon": y_proba,
        }
    ).sort_values("nom")

    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_distribution_total": {
            "0_sans_tampon": int((y == 0).sum()),
            "1_tampon": int((y == 1).sum()),
        },
    }
    feature_info = {"n_samples": int(len(y)), "n_features": int(X.shape[1]), "dpi": dpi}
    return metrics, df_pred, clf, feature_info, df_all


# ----------------------------------------------------------
# Sauvegarde / chargement du modèle
# ----------------------------------------------------------
def save_model(model, feature_info: dict):
    with open(MODEL_DIR / "tampon_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "tampon_model.meta.json", "w", encoding="utf-8") as f:
        json.dump(feature_info, f, indent=2)


def load_model():
    with open(MODEL_DIR / "tampon_model.pkl", "rb") as f:
        model = pickle.load(f)
    meta = {}
    mp = MODEL_DIR / "tampon_model.meta.json"
    if mp.exists():
        with open(mp, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return model, meta
