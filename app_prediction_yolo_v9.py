from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF

from ultralytics import YOLO


st.set_page_config(page_title="YOLO – Détection de tampons + Infos facture", layout="wide")
st.title("🧭 Détection de tampons + extraction infos facture (par facture)")


st.markdown(
    """
Cette app utilise un modèle **YOLOv8** pour détecter les **tampons** sur chaque page d’un PDF multipages,  
puis regroupe les pages en **factures** selon la règle :

> Une facture débute sur la page qui contient un tampon et inclut toutes les suivantes si elles ne contiennent pas de tampons, jusqu’à la prochaine qui contient un tampon.

Pour **chaque facture détectée**, l’app extrait (via le texte des pages de la facture) :

- le **n° de facture**  
- le **montant total estimé** (Total TTC / Net à payer / Montant total …)
"""
)


# ------------------------------------------------------------
# Rendu + utils YOLO
# ------------------------------------------------------------
@st.cache_resource
def load_yolo_model(model_path: str):
    return YOLO(model_path)


def render_pdf_page_to_rgb(pdf_path: Path, page_index: int = 0, dpi: int = 300):
    """Rend une page PDF en image RGB (numpy)."""
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        return img  # RGB


# ------------------------------------------------------------
# Comptage de factures via tampons
# ------------------------------------------------------------
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
            # sinon c'est une page hors facture (avant la 1ère facture ou entre deux).
            pass

    # Si une facture est encore ouverte à la fin, on la clôture à la dernière page
    if current_start is not None:
        invoices.append((current_start, n_pages - 1))

    return invoices


# ------------------------------------------------------------
# Extraction infos facture depuis le texte PDF (par facture)
# ------------------------------------------------------------
def extract_invoice_metadata_for_pages(
    pdf_path: Path,
    start_idx: int,
    end_idx: int,
    invoice_index: int,
) -> dict:
    """
    Extrait les infos pour UNE facture à partir du texte des pages [start_idx, end_idx] (0-based).

    Retourne :
      - invoice_index : index interne de la facture (1, 2, 3, ...)
      - n_pages       : nb de pages de cette facture
      - invoice_number: n° de facture trouvé dans ce bloc
      - invoice_amount: montant total estimé dans ce bloc
    """
    try:
        with fitz.open(str(pdf_path)) as doc:
            texts = []
            for page_no in range(start_idx, end_idx + 1):
                page = doc.load_page(page_no)
                texts.append(page.get_text("text"))
    except Exception as e:
        return {
            "invoice_index": invoice_index,
            "n_pages": end_idx - start_idx + 1,
            "invoice_number": None,
            "invoice_amount": None,
            "error": str(e),
        }

    full_text = "\n".join(texts)

    # --- extraction n° de facture ---
    invoice_number = None
    number_patterns = [
        r"facture\s*(?:n[°o]|no|num[eé]ro)?\s*[:\-]?\s*([A-Z0-9\-\/]+)",
        r"invoice\s*(?:n[°o]|no|#)?\s*[:\-]?\s*([A-Z0-9\-\/]+)",
    ]
    for pat in number_patterns:
        m = re.search(pat, full_text, flags=re.IGNORECASE)
        if m:
            invoice_number = m.group(1).strip()
            break

    # --- extraction montant total ---
    amount_candidates = []
    amount_patterns = [
        r"(total\s+(?:ttc|g[eé]n[eé]ral|facture|à\s+payer)[^0-9]{0,30}([0-9][0-9\s.,]*))",
        r"(montant\s+total[^0-9]{0,30}([0-9][0-9\s.,]*))",
        r"(net\s+à\s+payer[^0-9]{0,30}([0-9][0-9\s.,]*))",
    ]

    for pat in amount_patterns:
        for m in re.finditer(pat, full_text, flags=re.IGNORECASE):
            raw_val = m.group(2)
            # Nettoyage grossier : garder chiffres, espaces, , et .
            cleaned = re.sub(r"[^\d,\.]", "", raw_val)
            cleaned = cleaned.replace(" ", "").replace(",", ".")
            try:
                val = float(cleaned)
                amount_candidates.append(val)
            except ValueError:
                continue

    invoice_amount = max(amount_candidates) if amount_candidates else None

    return {
        "invoice_index": invoice_index,
        "n_pages": end_idx - start_idx + 1,
        "invoice_number": invoice_number,
        "invoice_amount": invoice_amount,
        "error": None,
    }


# ------------------------------------------------------------
# Contrôles Streamlit
# ------------------------------------------------------------
MODEL_DEFAULT = "runs_tampon/yolov8s_tampon/weights/best.pt"

model_path_text = st.text_input(
    "Chemin du modèle YOLO (.pt)",
    value=str(Path(MODEL_DEFAULT).resolve()),
)

conf_thres = st.slider(
    "Seuil de confiance (conf tampons)",
    min_value=0.10,
    max_value=0.98,
    value=0.25,
    step=0.01,
)

iou_thres = st.slider("Seuil IoU (NMS)", 0.1, 0.9, 0.45, 0.05)
dpi = st.select_slider("DPI rendu PDF", options=[150, 200, 250, 300], value=300)
show_images = st.checkbox("Afficher les pages avec boîtes détection", value=False)

uploaded_pdf = st.file_uploader("📄 Dépose un PDF multipages", type=["pdf"])


# ------------------------------------------------------------
# Pipeline prédiction
# ------------------------------------------------------------
if uploaded_pdf:
    tmp_dir = Path("data/tmp_pred_yolo")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / uploaded_pdf.name
    pdf_path.write_bytes(uploaded_pdf.getbuffer())
    st.info(f"PDF enregistré : {pdf_path.resolve()}")

    # Charge modèle YOLO
    try:
        model = load_yolo_model(model_path_text)
        st.success("Modèle YOLO chargé ✅")
    except Exception as e:
        st.error(f"Impossible de charger le modèle : {e}")
        st.stop()

    # Ouvrir le PDF pour compter les pages
    try:
        with fitz.open(str(pdf_path)) as doc:
            n_pages = len(doc)
    except Exception as e:
        st.error(f"Impossible d'ouvrir le PDF: {e}")
        st.stop()

    st.write(f"Pages détectées : **{n_pages}**")

    # Boucle YOLO page par page
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

    # --------------------------------------------------------
    # Comptage de factures via les tampons
    # --------------------------------------------------------
    st.subheader("🧾 Comptage de factures (via tampons)")

    tampon_flags = df["tampon_pred"].tolist()
    invoices = compute_invoices_from_tampons(tampon_flags)

    # Préparation des colonnes par page
    facture_index_per_page = [None] * n_pages
    invoice_number_per_page = [None] * n_pages
    invoice_amount_per_page = [None] * n_pages

    if not invoices:
        st.warning(
            "Aucune facture détectée selon la règle : aucune page ne contient de tampon (tampon_pred = 1)."
        )
        df_invoices = pd.DataFrame(
            columns=[
                "facture_index",
                "page_debut",
                "page_fin",
                "nb_pages",
                "invoice_number",
                "invoice_amount",
            ]
        )
    else:
        nb_invoices = len(invoices)
        st.success(f"Nombre de factures détectées : **{nb_invoices}**")

        data_invoices = []

        for idx, (start_idx, end_idx) in enumerate(invoices, start=1):
            start_page = start_idx + 1  # 1-based pour l'affichage
            end_page = end_idx + 1

            # Extraction n° de facture + montant uniquement sur ces pages
            meta = extract_invoice_metadata_for_pages(
                pdf_path=pdf_path,
                start_idx=start_idx,
                end_idx=end_idx,
                invoice_index=idx,
            )

            # Renseigner pour les pages de cette facture
            for p in range(start_idx, end_idx + 1):
                facture_index_per_page[p] = idx
                invoice_number_per_page[p] = meta.get("invoice_number")
                invoice_amount_per_page[p] = meta.get("invoice_amount")

            data_invoices.append(
                {
                    "facture_index": idx,
                    "page_debut": start_page,
                    "page_fin": end_page,
                    "nb_pages": end_page - start_page + 1,
                    "invoice_number": meta.get("invoice_number"),
                    "invoice_amount": meta.get("invoice_amount"),
                }
            )

        df_invoices = pd.DataFrame(data_invoices)
        st.markdown("**Factures détectées (début / fin / nb pages / n° / montant)**")
        st.dataframe(df_invoices, use_container_width=True)

    # Ajouter les colonnes par page dans df
    df["facture_index"] = facture_index_per_page
    df["invoice_number"] = invoice_number_per_page
    df["invoice_amount"] = invoice_amount_per_page

    # --------------------------------------------------------
    # Résultats par page + export
    # --------------------------------------------------------
    st.subheader("Résultats par page + informations facture")
    st.dataframe(df, use_container_width=True)

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

    # Miniatures optionnelles
    if show_images and images_to_show:
        st.subheader("Aperçu des pages (boîtes détection)")
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
