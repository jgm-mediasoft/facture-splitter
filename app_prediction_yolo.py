from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF

from ultralytics import YOLO

# ------------------------------------------------------------
# OpenCV pour les QR-codes (BV)
# ------------------------------------------------------------
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

def decode_qr_data(img_rgb: np.ndarray):
    """
    Décode un éventuel QR-code sur l'image.
    Retourne la chaîne décodée ou None si aucun QR.
    Un BV = présence d'un QR-code sur la page.
    """
    if not HAS_CV2:
        return None

    try:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(img_gray)
        if points is not None and len(points) > 0 and data:
            return data
    except Exception:
        return None

    return None


# ------------------------------------------------------------
# Rendu PDF -> image RGB (pour YOLO)
# ------------------------------------------------------------
def render_pdf_page_to_rgb(pdf_path: Path, page_index: int = 0, dpi: int = 300):
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        return img


# ------------------------------------------------------------
# Chargement du modèle YOLO
# ------------------------------------------------------------
@st.cache_resource
def load_yolo_model(model_path: str):
    return YOLO(model_path)


# ------------------------------------------------------------
# Config Streamlit
# ------------------------------------------------------------
st.set_page_config(page_title="YOLO – Tampons, factures & BV (v21)", layout="wide")
st.title("🧭 Détection de tampons, découpe des factures & BV (v21 – sans OCR montant)")

bv_status = "BV via QR-code ✅" if HAS_CV2 else "BV via QR-code ❌ (OpenCV non disponible)"
st.caption(bv_status)

st.markdown(
    """
Cette application :

1. Utilise un modèle **YOLOv8** pour détecter les **tampons** sur chaque page d’un PDF.  
2. Déduit les **factures** à partir des tampons :
   - une facture commence sur une page qui contient un tampon ;
   - les pages suivantes **sans tampon** appartiennent à la même facture
     jusqu’au **prochain tampon**.
3. Détecte les **Bulletins de versement (BV)** via la présence d’un **QR-code** sur la page.  
4. Affiche :
   - un tableau **par page** (tampon / probas / index de facture / présence BV) ;
   - un tableau **par facture** (page début, page fin, nb de pages, présence BV) ;
   - le **nombre total de factures** détectées.
"""
)

# ------------------------------------------------------------
# Paramètres modèle + interface
# ------------------------------------------------------------
MODEL_DEFAULT = "runs_tampon/yolov8s_tampon/weights/best.pt"

model_path_text = st.text_input(
    "Chemin du modèle YOLO (.pt)",
    value=str(Path(MODEL_DEFAULT).resolve()),
)

conf_thres = st.slider(
    "Seuil de confiance (conf tampons)",
    min_value=0.10,
    max_value=1.00,
    value=0.94,
    step=0.01,
)

iou_thres = st.slider("Seuil IoU (NMS)", 0.1, 0.9, 0.45, 0.05)
dpi = st.select_slider("DPI rendu PDF", options=[150, 200, 250, 300], value=300)
show_images = st.checkbox("Afficher les pages avec boîtes de détection", value=False)

uploaded_pdf = st.file_uploader("📄 Dépose un PDF multipages", type=["pdf"])


# ------------------------------------------------------------
# Pipeline principal
# ------------------------------------------------------------
if uploaded_pdf:
    # Sauvegarde du PDF dans un répertoire temporaire
    tmp_dir = Path("data/tmp_pred_yolo")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / uploaded_pdf.name
    pdf_path.write_bytes(uploaded_pdf.getbuffer())
    st.info(f"PDF enregistré : {pdf_path.resolve()}")

    # Chargement du modèle YOLO
    try:
        model = load_yolo_model(model_path_text)
        st.success("Modèle YOLO chargé ✅")
    except Exception as e:
        st.error(f"Impossible de charger le modèle : {e}")
        st.stop()

    # Ouverture du PDF pour compter les pages
    try:
        with fitz.open(str(pdf_path)) as doc:
            n_pages = len(doc)
    except Exception as e:
        st.error(f"Impossible d'ouvrir le PDF : {e}")
        st.stop()

    st.write(f"Pages détectées : **{n_pages}**")
    progress = st.progress(0, text="Analyse des pages…")

    rows = []
    images_to_show = []

    # Boucle sur les pages : rendu + prédictions
    for i in range(n_pages):
        img_rgb = render_pdf_page_to_rgb(pdf_path, page_index=i, dpi=dpi)

        # 1) Prédiction YOLO pour les tampons
        results = model(img_rgb, conf=conf_thres, iou=iou_thres, verbose=False)
        r = results[0]
        boxes = r.boxes
        n_det = len(boxes)

        if n_det > 0:
            confs = boxes.conf.cpu().numpy()
            max_conf = float(confs.max())
            tampon_pred = 1
            proba_tampon = max_conf
        else:
            tampon_pred = 0
            proba_tampon = 0.0
            max_conf = 0.0

        # 2) Détection BV via QR-code
        qr_data = decode_qr_data(img_rgb)
        has_bv_page = qr_data is not None

        rows.append(
            {
                "page": i + 1,
                "tampon_pred": tampon_pred,
                "proba_tampon": round(proba_tampon, 4),
                "n_detections": int(n_det),
                "max_conf": round(max_conf, 4),
                "has_bv_page": has_bv_page,
            }
        )

        # Optionnel : affichage des images avec boîtes
        if show_images:
            im_plot = r.plot()
            images_to_show.append((i + 1, im_plot))

        # Progression
        if (i + 1) % max(1, n_pages // 20) == 0 or i == n_pages - 1:
            progress.progress((i + 1) / n_pages, text=f"Page {i+1}/{n_pages}")

    # DataFrame des pages
    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Découpage factures par tampons
    # --------------------------------------------------------
    facture_index = 0
    facture_index_per_page = []
    for _, row in df.iterrows():
        if row["tampon_pred"] == 1:
            # Nouveau tampon -> nouvelle facture
            facture_index += 1
            facture_index_per_page.append(facture_index)
        else:
            # Pas de tampon
            if facture_index > 0:
                # On est dans une facture en cours
                facture_index_per_page.append(facture_index)
            else:
                # Pages avant le premier tampon : pas de facture
                facture_index_per_page.append(None)

    df["facture_index"] = facture_index_per_page
    nb_factures = facture_index

    # --------------------------------------------------------
    # Résumé par facture (page début / fin / nb pages / BV)
    # --------------------------------------------------------
    data_invoices = []
    if nb_factures > 0:
        for idx in range(1, nb_factures + 1):
            sub = df[df["facture_index"] == idx]
            if sub.empty:
                continue
            start_page = int(sub["page"].min())
            end_page = int(sub["page"].max())
            has_bv_facture = bool(sub["has_bv_page"].any())
            data_invoices.append(
                {
                    "facture_index": idx,
                    "page_debut": start_page,
                    "page_fin": end_page,
                    "nb_pages": end_page - start_page + 1,
                    "has_bv_facture": has_bv_facture,
                }
            )

    df_factures = (
        pd.DataFrame(data_invoices)
        if data_invoices
        else pd.DataFrame(
            columns=["facture_index", "page_debut", "page_fin", "nb_pages", "has_bv_facture"]
        )
    )

    # --------------------------------------------------------
    # Affichage
    # --------------------------------------------------------
    st.subheader("📊 Résultats par page")
    st.dataframe(df, use_container_width=True)

    st.subheader("🧾 Factures détectées (basées sur les tampons)")
    st.metric("Nombre de factures détectées", nb_factures)

    if not df_factures.empty:
        st.dataframe(df_factures, use_container_width=True)
    else:
        st.info("Aucune facture détectée (aucun tampon).")

    # Téléchargement des résultats (optionnel)
    st.download_button(
        "📥 Télécharger les résultats par page (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        mime="text/csv",
        file_name=f"yolo_tampons_factures_v21_pages_{pdf_path.stem}.csv",
    )

    st.download_button(
        "📥 Télécharger le résumé par facture (CSV)",
        data=df_factures.to_csv(index=False).encode("utf-8"),
        mime="text/csv",
        file_name=f"yolo_tampons_factures_v21_factures_{pdf_path.stem}.csv",
    )

    # Aperçu visuel YOLO
    if show_images and images_to_show:
        st.subheader("Aperçu des pages avec boîtes de détection")
        per_row = 2
        i = 0
        while i < len(images_to_show):
            cols = st.columns(per_row)
            for col in cols:
                if i >= len(images_to_show):
                    break
                page_no, im_plot = images_to_show[i]
                col.image(im_plot, caption=f"Page {page_no}")
                i += 1
else:
    st.info("Charge un modèle YOLO entraîné, puis dépose un PDF multipages pour prédire.")

