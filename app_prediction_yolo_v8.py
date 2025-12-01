from pathlib import Path 

import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF

from ultralytics import YOLO


st.set_page_config(page_title="YOLO – Détection de tampons", layout="wide")
st.title("🧭 Détection de tampons avec YOLOv8")


st.markdown(
    """
Cette app utilise un modèle **YOLOv8** entraîné sur des exemples de tampons (.png)  
et des pages PDF synthétiques pour détecter les **tampons** sur chaque page d’un PDF multipages.

- Classe unique : `tampon`  
- `tampon_pred = 1` si au moins une détection au-dessus du seuil de confiance.
"""
)


@st.cache_resource
def load_yolo_model(model_path: str):
    return YOLO(model_path)


def render_pdf_page_to_rgb(pdf_path: Path, page_index: int = 0, dpi: int = 300):
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        return img  # RGB


# >>>----------------- Fonction de calcul des factures -----------------
def compute_invoices_from_tampons(tampon_flags):
    """
    tampon_flags : liste de 0/1 ou bool indiquant si chaque page contient un tampon.

    Règle :
    - Une facture commence sur une page qui contient un tampon (1)
    - Elle inclut toutes les pages suivantes tant qu'elles ne contiennent pas de tampon (0)
    - Elle se termine juste avant la prochaine page avec tampon (1)
    - La dernière facture va jusqu'à la fin du document.
    """
    n_pages = len(tampon_flags)
    invoices = []
    current_start = None  # index 0-based de la page de début de la facture en cours

    for i in range(n_pages):
        has_stamp = bool(tampon_flags[i])

        if has_stamp:
            # Si on a déjà une facture en cours, on la clôture à la page précédente
            if current_start is not None:
                invoices.append((current_start, i - 1))
            # On démarre une nouvelle facture à cette page
            current_start = i
        else:
            # Page sans tampon : si une facture est en cours, elle en fait partie,
            # sinon c'est juste une page hors facture (avant le premier tampon).
            pass

    # Si une facture est encore ouverte à la fin, on la clôture à la dernière page
    if current_start is not None:
        invoices.append((current_start, n_pages - 1))

    return invoices
# <<<----------------------------------------------------


# ----------------- Paramètres modèle -----------------
MODEL_DEFAULT = "runs_tampon/yolov8n_tampon/weights/best.pt"

model_path_text = st.text_input(
    "Chemin du modèle YOLO (.pt)",
    value=str(Path(MODEL_DEFAULT).resolve()),
)

conf_thres = st.slider(
    "Seuil de confiance (conf)",
    min_value=0.10,
    max_value=0.94,
    value=0.25,
    step=0.01,
)

iou_thres = st.slider("Seuil IoU (NMS)", 0.1, 0.9, 0.45, 0.05)
dpi = st.select_slider("DPI rendu PDF", options=[150, 200, 250, 300], value=300)
show_images = st.checkbox("Afficher les pages avec boîtes détection", value=False)

uploaded_pdf = st.file_uploader("📄 Dépose un PDF multipages", type=["pdf"])

if uploaded_pdf:
    tmp_dir = Path("data/tmp_pred_yolo")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / uploaded_pdf.name
    pdf_path.write_bytes(uploaded_pdf.getbuffer())
    st.info(f"PDF enregistré : {pdf_path.resolve()}")

    # Charge le modèle
    try:
        model = load_yolo_model(model_path_text)
        st.success("Modèle YOLO chargé ✅")
    except Exception as e:
        st.error(f"Impossible de charger le modèle : {e}")
        st.stop()

    try:
        with fitz.open(str(pdf_path)) as doc:
            n_pages = len(doc)
    except Exception as e:
        st.error(f"Impossible d'ouvrir le PDF: {e}")
        st.stop()

    st.write(f"Pages détectées : **{n_pages}**")
    progress = st.progress(0, text="Analyse des pages…")

    rows = []
    images_to_show = []

    for i in range(n_pages):
        img_rgb = render_pdf_page_to_rgb(pdf_path, page_index=i, dpi=dpi)

        results = model(
            img_rgb,
            conf=conf_thres,
            iou=iou_thres,
            verbose=False,
        )

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

        rows.append(
            {
                "page": i + 1,
                "tampon_pred": tampon_pred,
                "proba_tampon": round(proba_tampon, 4),
                "n_detections": int(n_det),
            }
        )

        if show_images:
            im_plot = r.plot()  # image avec boîtes dessinées (RGB)
            images_to_show.append((i + 1, im_plot))

        if (i + 1) % max(1, n_pages // 20) == 0 or i == n_pages - 1:
            progress.progress((i + 1) / n_pages, text=f"Page {i+1}/{n_pages}")

    df = pd.DataFrame(rows)

    st.subheader("Résultats par page")
    st.dataframe(df, width="stretch")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.metric("Pages avec tampon (=1)", int((df["tampon_pred"] == 1).sum()))
    with c2:
        st.metric("Pages sans tampon (=0)", int((df["tampon_pred"] == 0).sum()))
    with c3:
        st.download_button(
            "📥 Télécharger les résultats (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            mime="text/csv",
            file_name=f"yolo_predictions_{pdf_path.stem}.csv",
        )

    # >>>----------------- Nouvelle section : Comptage de factures -----------------
    st.subheader("🧾 Comptage de factures (défini par les tampons)")

    tampon_flags = df["tampon_pred"].tolist()
    invoices = compute_invoices_from_tampons(tampon_flags)

    if not invoices:
        st.warning(
            "Aucune facture détectée : aucune page ne contient de tampon (tampon_pred = 1)."
        )
    else:
        nb_invoices = len(invoices)
        st.success(f"Nombre de factures détectées : **{nb_invoices}**")

        data_invoices = []
        for idx, (start_idx, end_idx) in enumerate(invoices, start=1):
            # Conversion en numérotation humaine (1-based)
            start_page = start_idx + 1
            end_page = end_idx + 1
            data_invoices.append(
                {
                    "facture": idx,
                    "page_debut": start_page,
                    "page_fin": end_page,
                    "nb_pages": end_page - start_page + 1,
                }
            )

        df_invoices = pd.DataFrame(data_invoices)
        st.markdown("**Factures détectées (début / fin / nombre de pages)**")
        st.dataframe(df_invoices, use_container_width=True)
    # <<<-------------------------------------------------------------

    if show_images and images_to_show:
        st.subheader("Aperçu des pages détectées")
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
