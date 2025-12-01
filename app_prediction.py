
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
import cv2

from invoice_splitter.learner import load_model

st.set_page_config(page_title="Prédiction – Tampon sur PDF multipages", layout="wide")
st.title("🔎 Prédire les pages avec tampon (PDF multipages)")

st.markdown(
    """
Dépose un **PDF multipages**. Pour chaque page, on calcule :
- **tampon_pred** : 1 si tampon, 0 sinon  
- **proba_tampon** : probabilité tampon  
- **fiabilité** : proba si prédiction=1, sinon 1-proba  

➡️ **Quick-fix** : seuil par défaut **0.30** pour privilégier la détection de tampons.
"""
)


# ------------------------------------------------------------
# Rendu + features (doivent matcher ceux de learner.py)
# ------------------------------------------------------------
def render_pdf_page_to_bgr(pdf_path: Path, page_index: int = 0, dpi: int = 300):
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def extract_features_from_bgr(img: np.ndarray) -> np.ndarray:
    orb = cv2.ORB_create()
    _kp, des = orb.detectAndCompute(img, None)
    orb_feat = des.mean(axis=0) if des is not None else np.zeros(32, dtype=np.float32)

    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = cv2.moments(gray)
    hu = cv2.HuMoments(m).flatten()
    hu = (-np.sign(hu) * np.log(np.abs(hu) + 1e-12)).astype(np.float32)

    return np.hstack([orb_feat, hist, hu]).astype(np.float32)


# ------------------------------------------------------------
# Chargement du modèle
# ------------------------------------------------------------
try:
    model, meta = load_model()
    st.success(f"Modèle chargé depuis ./models ✅  (meta: {meta})")
except Exception as e:
    st.error(
        "Aucun modèle trouvé. Entraîne et sauvegarde d'abord via l'app d'apprentissage."
    )
    st.stop()

default_dpi = int(meta.get("dpi", 300))

c1, c2, c3 = st.columns(3)
with c1:
    dpi = st.select_slider(
        "DPI rendu PDF",
        options=[100, 150, 200, 250, 300],
        value=default_dpi,
    )
with c2:
    threshold = st.slider(
        "Seuil (plus bas = plus de tampons détectés)",
        0.05,
        0.90,
        0.30,  # quick-fix : favorise le rappel tampon
        0.01,
    )
with c3:
    show_thumbs = st.checkbox(
        "Afficher miniatures pages tampon=1 (optionnel)", value=False
    )

uploaded_pdf = st.file_uploader("📄 Dépose un PDF multipages", type=["pdf"])

# ------------------------------------------------------------
# Prédiction par page
# ------------------------------------------------------------
if uploaded_pdf:
    tmp_dir = Path("data/tmp_pred")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / uploaded_pdf.name
    pdf_path.write_bytes(uploaded_pdf.getbuffer())
    st.info(f"Fichier enregistré : {pdf_path.resolve()}")

    try:
        with fitz.open(str(pdf_path)) as doc:
            n_pages = len(doc)
    except Exception as e:
        st.error(f"Impossible d'ouvrir le PDF: {e}")
        st.stop()

    st.write(f"Pages détectées : **{n_pages}**")
    progress = st.progress(0, text="Extraction des features…")

    feats, pages = [], []
    for i in range(n_pages):
        img = render_pdf_page_to_bgr(pdf_path, page_index=i, dpi=dpi)
        feats.append(extract_features_from_bgr(img))
        pages.append(i + 1)
        if (i + 1) % max(1, n_pages // 20) == 0 or i == n_pages - 1:
            progress.progress((i + 1) / n_pages, text=f"Extraction page {i+1}/{n_pages}")

    X = np.vstack(feats).astype(np.float32)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        raw = (
            model.decision_function(X)
            if hasattr(model, "decision_function")
            else model.predict(X)
        )
        proba = 1 / (1 + np.exp(-np.clip(raw, -10, 10))).astype(float)

    pred = (proba >= threshold).astype(int)
    fiab = np.where(pred == 1, proba, 1.0 - proba)

    df = pd.DataFrame(
        {
            "page": pages,
            "tampon_pred": pred.astype(int),
            "proba_tampon": np.round(proba, 4),
            "fiabilite": np.round(fiab, 4),
        }
    )

    st.subheader("Résultats par page")
    st.dataframe(df, width="stretch")

    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        st.metric("Pages tampon (=1)", int(df["tampon_pred"].sum()))
    with cB:
        st.metric("Pages sans (=0)", int((1 - df["tampon_pred"]).sum()))
    with cC:
        st.download_button(
            "📥 Télécharger le CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            mime="text/csv",
            file_name=f"predictions_{pdf_path.stem}.csv",
        )

    # Miniatures optionnelles
    if show_thumbs and df["tampon_pred"].any():
        st.subheader("Aperçu des pages prédites **tampon=1**")
        idx_pos = df.index[df["tampon_pred"] == 1].tolist()
        with fitz.open(str(pdf_path)) as doc:
            i = 0
            per_row = 4
            while i < len(idx_pos):
                cols = st.columns(per_row)
                for col in cols:
                    if i >= len(idx_pos):
                        break
                    page_no = int(df.loc[idx_pos[i], "page"]) - 1
                    page = doc.load_page(page_no)
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(100 / 72, 100 / 72), alpha=False
                    )
                    col.image(
                        pix.tobytes("png"),
                        caption=f"Page {page_no+1} (p={df.loc[idx_pos[i],'proba_tampon']:.2f})",
                    )
                    i += 1
else:
    st.info("Charge un modèle sauvegardé puis dépose un PDF multipages pour prédire.")
