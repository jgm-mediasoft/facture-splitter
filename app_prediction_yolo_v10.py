from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF

from ultralytics import YOLO


st.set_page_config(page_title="YOLO – Détection de tampons + BV", layout="wide")
st.title("🧭 Détection de tampons + extraction infos facture & BV (v10)")


st.markdown(
    """
Cette app utilise un modèle **YOLOv8** pour détecter les **tampons** sur chaque page d’un PDF multipages,  
puis regroupe les pages en **factures** selon la règle :

> Une facture débute sur la page qui contient un tampon et inclut toutes les suivantes si elles ne contiennent pas de tampons, jusqu’à la prochaine qui contient un tampon.

Pour **chaque facture détectée**, l’app :

- identifie un éventuel **bulletin de versement (BV)** sur la **dernière page** de la facture ;
- extrait la **Référence BV** (format obligatoire `XX XXXXX XXXXX XXXXX XXXXX XXXXX`) ;
- extrait le **montant total** :
  - en priorité sur le BV (`CHF` …),
  - sinon en priorité sur **`Total TTC`** dans le texte de la facture,
  - sinon via d’autres libellés (Montant total, Net à payer…) ;
- récupère un **n° de facture** : **valeur numérique** à droite des intitulés :
  - `Facture ...`
  - `Facture vente N° ...`
  - `Facture No.: ...`
  - `N° Facture ...`
  - ou `Numéro ...` / `Numéro` suivi de la valeur sur la ligne suivante.
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
    current_start = None  # index 0-based

    for i in range(n_pages):
        has_stamp = bool(tampon_flags[i])

        if has_stamp:
            if current_start is not None:
                invoices.append((current_start, i - 1))
            current_start = i
        else:
            pass

    if current_start is not None:
        invoices.append((current_start, n_pages - 1))

    return invoices


# ------------------------------------------------------------
# Extraction montants dans du texte (général ou BV)
# ------------------------------------------------------------
def _parse_amount_candidates(text: str):
    """Retourne une liste de montants possibles trouvés dans un texte."""
    amount_candidates = []
    amount_patterns = [
        r"(total\s+(?:ttc|g[eé]n[eé]ral|facture|à\s+payer)[^0-9]{0,30}([0-9][0-9\s.,']*))",
        r"(montant\s+total[^0-9]{0,30}([0-9][0-9\s.,']*))",
        r"(net\s+à\s+payer[^0-9]{0,30}([0-9][0-9\s.,']*))",
        r"(montant[^0-9]{0,30}([0-9][0-9\s.,']*))",
    ]

    for pat in amount_patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            raw_val = m.group(2)
            cleaned = re.sub(r"[^\d,\.']", "", raw_val)
            cleaned = cleaned.replace(" ", "").replace("'", "")
            cleaned = cleaned.replace(",", ".")
            try:
                val = float(cleaned)
                amount_candidates.append(val)
            except ValueError:
                continue

    return amount_candidates


def _find_total_ttc(text: str):
    """
    Cherche spécifiquement 'Total TTC' et retourne le dernier montant trouvé,
    ou None si rien n'est trouvé.
    """
    pattern = r"total\s+ttc[^0-9]{0,40}([0-9][0-9\s.,']*)"
    candidates = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        raw_val = m.group(1)
        cleaned = re.sub(r"[^\d,\.']", "", raw_val)
        cleaned = cleaned.replace(" ", "").replace("'", "")
        cleaned = cleaned.replace(",", ".")
        try:
            val = float(cleaned)
            candidates.append(val)
        except ValueError:
            continue

    if not candidates:
        return None
    return candidates[-1]


# ------------------------------------------------------------
# Extraction du N° de facture (valeur numérique à droite)
# ------------------------------------------------------------
def _extract_invoice_number(text: str):
    """
    Extrait un n° de facture à partir de plusieurs formes possibles dans le texte.

    On retourne uniquement les CHIFFRES (lettres, espaces, apostrophes supprimés).
    """
    patterns = [
        # N° Facture 422'040   ou  N° Facture: 422'040
        r"n[°o]\s*facture[^\n\r0-9A-Z]{0,20}([A-Z0-9']{3,})",
        r"n[°o]\s*facture\s*[:\-]?\s*([A-Z0-9']{3,})",

        # Facture        1741046168
        r"\bfacture[^\n\r0-9A-Z]{0,40}([A-Z0-9']{3,})",

        # Facture ... N° FA25001716 / Facture vente N° ...
        r"\bfacture[^\n\r]{0,40}n[°o]\s*[:\.]?\s*([A-Z0-9']{3,})",

        # Facture ... No.: 453129
        r"\bfacture[^\n\r]{0,40}no\.?\s*[:\-]?\s*([A-Z0-9']{3,})",

        # Facture ... Numéro 9250625565 (même ligne)
        r"\bfacture[^\n\r]{0,40}num[eé]ro[^\n\r0-9A-Z]{0,20}([A-Z0-9']{3,})",

        # Numéro 9250625565  (sans le mot Facture)
        r"\bnum[eé]ro\b[^\n\r0-9A-Z]{0,20}([A-Z0-9']{3,})",

        # Numéro\n9250625565 (valeur sur la ligne suivante)
        r"\bnum[eé]ro\b[^\n\r0-9A-Z]{0,20}\n\s*([A-Z0-9']{3,})",
    ]

    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            raw = m.group(1)
            digits = re.sub(r"[^\d]", "", raw)  # garde uniquement les chiffres
            if digits:
                return digits

    return None


# ------------------------------------------------------------
# Extraction infos facture + BV pour un bloc de pages
# ------------------------------------------------------------
def extract_invoice_metadata_for_pages(
    pdf_path: Path,
    start_idx: int,
    end_idx: int,
    invoice_index: int,
) -> dict:
    """
    Extrait les infos pour UNE facture à partir du texte des pages [start_idx, end_idx] (0-based).
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
            "reference": None,
            "has_bv": False,
            "error": str(e),
        }

    full_text = "\n".join(texts)
    last_page_text = texts[-1] if texts else ""

    # --- N° de facture ---
    invoice_number = _extract_invoice_number(full_text)

    # --- Montant général : priorité à 'Total TTC' ---
    amount_ttc = _find_total_ttc(full_text)
    if amount_ttc is not None:
        amount_general = amount_ttc
    else:
        amount_candidates_general = _parse_amount_candidates(full_text)
        amount_general = max(amount_candidates_general) if amount_candidates_general else None

    # --- Détection BV via Référence sur la dernière page ---
    ref_regex = r"\b\d{2}\s\d{5}\s\d{5}\s\d{5}\s\d{5}\s\d{5}\b"
    ref_match = re.search(ref_regex, last_page_text)
    reference = None
    has_bv = False

    if ref_match:
        raw_ref = ref_match.group(0)
        ref_clean = re.sub(r"\s+", " ", raw_ref).strip()
        reference = ref_clean
        has_bv = True

    # --- Montant sur le BV (prioritaire si BV) ---
    amount_bv = None
    if has_bv:
        chf_candidates = []
        chf_pattern = r"CHF[^0-9]{0,10}([0-9][0-9\s.,']*)"
        for m in re.finditer(chf_pattern, last_page_text, flags=re.IGNORECASE):
            raw_val = m.group(1)
            cleaned = re.sub(r"[^\d,\.']", "", raw_val)
            cleaned = cleaned.replace(" ", "").replace("'", "")
            cleaned = cleaned.replace(",", ".")
            try:
                val = float(cleaned)
                chf_candidates.append(val)
            except ValueError:
                continue

        if chf_candidates:
            amount_bv = max(chf_candidates)
        else:
            amount_candidates_bv = _parse_amount_candidates(last_page_text)
            amount_bv = max(amount_candidates_bv) if amount_candidates_bv else None

    # --- Choix final du montant ---
    if amount_bv is not None:
        invoice_amount = amount_bv
    else:
        invoice_amount = amount_general

    return {
        "invoice_index": invoice_index,
        "n_pages": end_idx - start_idx + 1,
        "invoice_number": invoice_number,
        "invoice_amount": invoice_amount,
        "reference": reference,
        "has_bv": has_bv,
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
    max_value=1.00,     # <<< max à 1.0
    value=0.94,
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

    facture_index_per_page = [None] * n_pages
    invoice_number_per_page = [None] * n_pages
    invoice_amount_per_page = [None] * n_pages
    reference_per_page = [None] * n_pages
    has_bv_per_page = [False] * n_pages

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
                "reference",
                "has_bv",
            ]
        )
    else:
        nb_invoices = len(invoices)
        st.success(f"Nombre de factures détectées : **{nb_invoices}**")

        data_invoices = []

        for idx, (start_idx, end_idx) in enumerate(invoices, start=1):
            start_page = start_idx + 1
            end_page = end_idx + 1

            meta = extract_invoice_metadata_for_pages(
                pdf_path=pdf_path,
                start_idx=start_idx,
                end_idx=end_idx,
                invoice_index=idx,
            )

            for p in range(start_idx, end_idx + 1):
                facture_index_per_page[p] = idx
                invoice_number_per_page[p] = meta.get("invoice_number")
                invoice_amount_per_page[p] = meta.get("invoice_amount")
                reference_per_page[p] = meta.get("reference")
                has_bv_per_page[p] = meta.get("has_bv", False)

            data_invoices.append(
                {
                    "facture_index": idx,
                    "page_debut": start_page,
                    "page_fin": end_page,
                    "nb_pages": end_page - start_page + 1,
                    "invoice_number": meta.get("invoice_number"),
                    "invoice_amount": meta.get("invoice_amount"),
                    "reference": meta.get("reference"),
                    "has_bv": meta.get("has_bv", False),
                }
            )

        df_invoices = pd.DataFrame(data_invoices)
        st.markdown(
            "**Factures détectées (début / fin / nb pages / n° / montant / Référence BV / BV présent ?)**"
        )
        st.dataframe(df_invoices, use_container_width=True)

    df["facture_index"] = facture_index_per_page
    df["invoice_number"] = invoice_number_per_page
    df["invoice_amount"] = invoice_amount_per_page
    df["reference"] = reference_per_page
    df["has_bv"] = has_bv_per_page

    # --------------------------------------------------------
    # Résultats par page + export
    # --------------------------------------------------------
    st.subheader("Résultats par page + informations facture & BV")
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
            file_name=f"yolo_predictions_v10_{pdf_path.stem}.csv",
        )

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
