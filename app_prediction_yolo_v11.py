from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF

from ultralytics import YOLO


# ------------------------------------------------------------
# Config Streamlit
# ------------------------------------------------------------
st.set_page_config(page_title="YOLO – Détection de tampons + BV", layout="wide")
st.title("🧭 Détection de tampons + infos facture & BV (v11)")

st.markdown(
    """
**Définition d'une facture dans cette app :**

1. **Priorité au BV**  
   - Si une facture a un **bulletin de versement (BV)**, la page qui contient la
     **Référence** au format `XX XXXXX XXXXX XXXXX XXXXX XXXXX` est la **dernière page**
     de la facture.

2. **Factures sans BV**  
   - On utilise les **tampons** détectés par YOLO (1 tampon par facture) pour séparer :
     chaque tampon marque le **début** d’une facture,
     la fin est juste avant le tampon suivant (ou la fin du document).

Le **slider de confiance** contrôle seulement `tampon_pred` affiché dans le tableau.  
La **séparation réelle des factures** utilise un seuil interne fixe (`0.5`) pour ne pas perdre de factures.
"""
)


# ------------------------------------------------------------
# Constantes / utils
# ------------------------------------------------------------
BV_REF_REGEX = r"\b\d{2}\s\d{5}\s\d{5}\s\d{5}\s\d{5}\s\d{5}\b"
SEGMENTATION_CONF = 0.5  # seuil interne pour décider "tampon présent" pour la séparation


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
# Séparation des factures (BV priorité, tampons sinon)
# ------------------------------------------------------------
def compute_invoices(stamp_flags_seg, has_bv_page_flags, n_pages: int):
    """
    stamp_flags_seg : booléen par page (max_conf >= SEGMENTATION_CONF)
    has_bv_page_flags : booléen par page (Référence BV trouvée)
    n_pages : nombre total de pages

    Règle :
    - On utilise les **tampons** comme points de départ de factures.
    - Pour chaque tampon à la page s :
        * si un BV est trouvé entre s et le tampon suivant (ou fin de doc),
          la facture se termine à la page du BV (priorité BV).
        * sinon, la facture se termine juste avant le tampon suivant,
          ou à la dernière page s'il n'y a pas de tampon suivant.
    - S'il n'y a **aucun tampon**, on découpe uniquement avec les BV,
      sinon une seule facture pour tout le document.
    """
    stamp_indices = [i for i, f in enumerate(stamp_flags_seg) if f]
    bv_indices = [i for i, f in enumerate(has_bv_page_flags) if f]

    # Cas 1 : aucun tampon -> fallback BV
    if not stamp_indices:
        if not bv_indices:
            return [(0, n_pages - 1)] if n_pages > 0 else []
        bv_indices = sorted(set(bv_indices))
        invoices = []
        start = 0
        for bv in bv_indices:
            invoices.append((start, bv))
            start = bv + 1
        if start <= n_pages - 1:
            invoices.append((start, n_pages - 1))
        return invoices

    stamp_indices = sorted(set(stamp_indices))
    bv_indices = sorted(set(bv_indices))
    invoices = []

    # Pour chaque tampon comme début de facture
    for idx, s in enumerate(stamp_indices):
        # borne supérieure par défaut : avant le prochain tampon ou fin doc
        if idx < len(stamp_indices) - 1:
            next_stamp = stamp_indices[idx + 1]
            end_candidate = next_stamp - 1
        else:
            end_candidate = n_pages - 1

        # Chercher un BV entre s et end_candidate
        bv_in_range = [b for b in bv_indices if s <= b <= end_candidate]
        if bv_in_range:
            end = min(bv_in_range)  # le BV est la dernière page de la facture
        else:
            end = end_candidate

        invoices.append((s, end))

    # Pages avant le 1er tampon (en-tête, etc.)
    first_start = invoices[0][0]
    if first_start > 0:
        bvs_before = [b for b in bv_indices if b < first_start]
        if bvs_before:
            invoices.insert(0, (0, max(bvs_before)))
        else:
            invoices.insert(0, (0, first_start - 1))

    # Fusion / nettoyage final
    merged = []
    for s, e in sorted(invoices):
        if not merged:
            merged.append([s, e])
        else:
            ls, le = merged[-1]
            if s <= le + 1:
                merged[-1][1] = max(le, e)
            else:
                merged.append([s, e])

    return [(s, e) for s, e in merged]


# ------------------------------------------------------------
# Montants dans le texte
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
    """Cherche spécifiquement 'Total TTC' et retourne le dernier montant trouvé."""
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
    On retourne uniquement les CHIFFRES.

    Correction spéciale pour :
      - "N° Facture 422'038" ou "N° Facture 422' 038"
    """

    # 1) Cas spécifique : "N° Facture ...."
    nf_pattern = r"n[°o]\s*facture[^\n\r]*"
    for m in re.finditer(nf_pattern, text, flags=re.IGNORECASE):
        segment = m.group(0)  # ex: "N° Facture 422'038"
        m_num = re.search(r"([0-9][0-9' ]{2,})", segment)
        if m_num:
            raw = m_num.group(1)
            digits = re.sub(r"[^\d]", "", raw)  # enlève espaces, apostrophes, etc.
            if digits:
                return digits

    # 2) Autres formes plus générales
    patterns = [
        # Facture        1741046168
        r"\bfacture[^\n\r0-9A-Z]{0,40}([A-Z0-9']{3,})",

        # Facture ... N° FA25001716 / Facture vente N°
        r"\bfacture[^\n\r]{0,40}n[°o]\s*[:\.]?\s*([A-Z0-9']{3,})",

        # Facture ... No.: 453129
        r"\bfacture[^\n\r]{0,40}no\.?\s*[:\-]?\s*([A-Z0-9']{3,})",

        # Facture ... Numéro 9250625565
        r"\bfacture[^\n\r]{0,40}num[eé]ro[^\n\r0-9A-Z]{0,20}([A-Z0-9']{3,})",

        # Numéro 9250625565
        r"\bnum[eé]ro\b[^\n\r0-9A-Z]{0,20}([A-Z0-9']{3,})",

        # Numéro\n9250625565
        r"\bnum[eé]ro\b[^\n\r0-9A-Z]{0,20}\n\s*([A-Z0-9']{3,})",
    ]

    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            raw = m.group(1)
            digits = re.sub(r"[^\d]", "", raw)
            if digits:
                return digits

    return None


# ------------------------------------------------------------
# Extraction infos facture + BV pour un bloc de pages
# ------------------------------------------------------------
def extract_invoice_metadata_for_pages(
    page_texts,
    start_idx: int,
    end_idx: int,
    invoice_index: int,
) -> dict:
    """
    Extrait les infos pour UNE facture à partir du texte des pages [start_idx, end_idx].
    """
    texts = page_texts[start_idx : end_idx + 1]
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
    ref_match = re.search(BV_REF_REGEX, last_page_text)
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
    "Seuil de confiance (affichage tampons)",
    min_value=0.10,
    max_value=1.00,   # max 1.0
    value=0.94,       # valeur initiale 0.94
    step=0.01,
    help="Contrôle `tampon_pred` affiché. La séparation des factures utilise un seuil interne fixe (0.5).",
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

    # Ouvrir le PDF pour compter les pages + récupérer le texte
    try:
        with fitz.open(str(pdf_path)) as doc:
            n_pages = len(doc)
            page_texts = [doc.load_page(i).get_text("text") for i in range(n_pages)]
    except Exception as e:
        st.error(f"Impossible d'ouvrir le PDF: {e}")
        st.stop()

    st.write(f"Pages détectées : **{n_pages}**")

    # Détection BV page par page
    has_bv_page_flags = [
        bool(re.search(BV_REF_REGEX, txt)) for txt in page_texts
    ]

    # Boucle YOLO page par page (tampons)
    progress = st.progress(0, text="Analyse des pages…")
    rows = []
    images_to_show = []

    stamp_flags_seg = []  # pour la séparation (seuil interne 0.5)

    for i in range(n_pages):
        img_rgb = render_pdf_page_to_rgb(pdf_path, page_index=i, dpi=dpi)

        # conf bas pour ne rien rater ; tri ensuite
        results = model(
            img_rgb,
            conf=0.10,
            iou=iou_thres,
            verbose=False,
        )

        r = results[0]
        boxes = r.boxes
        n_det = len(boxes)
        if n_det > 0:
            confs = boxes.conf.cpu().numpy()
            max_conf = float(confs.max())
        else:
            max_conf = 0.0

        # tampon affiché (slider)
        tampon_pred = 1 if max_conf >= conf_thres else 0
        proba_tampon = max_conf if tampon_pred == 1 else 0.0

        # tampon pour séparation (seuil fixe)
        stamp_flags_seg.append(max_conf >= SEGMENTATION_CONF)

        rows.append(
            {
                "page": i + 1,
                "tampon_pred": tampon_pred,
                "proba_tampon": round(proba_tampon, 4),
                "n_detections": int(n_det),
                "max_conf": round(max_conf, 4),
                "has_bv_page": has_bv_page_flags[i],
            }
        )

        if show_images:
            im_plot = r.plot()  # image avec boîtes dessinées (RGB)
            images_to_show.append((i + 1, im_plot))

        if (i + 1) % max(1, n_pages // 20) == 0 or i == n_pages - 1:
            progress.progress((i + 1) / n_pages, text=f"Page {i+1}/{n_pages}")

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Comptage de factures via BV + tampons
    # --------------------------------------------------------
    st.subheader("🧾 Factures détectées (BV prioritaire, tampons ensuite)")

    invoices = compute_invoices(stamp_flags_seg, has_bv_page_flags, n_pages)

    facture_index_per_page = [None] * n_pages
    invoice_number_per_page = [None] * n_pages
    invoice_amount_per_page = [None] * n_pages
    reference_per_page = [None] * n_pages
    has_bv_per_page = [False] * n_pages

    if not invoices:
        st.warning("Aucune facture détectée.")
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
        data_invoices = []
        for idx, (start_idx, end_idx) in enumerate(invoices, start=1):
            start_page = start_idx + 1
            end_page = end_idx + 1

            meta = extract_invoice_metadata_for_pages(
                page_texts=page_texts,
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
        st.dataframe(df_invoices, use_container_width=True)

    # Colonnes par page
    df["facture_index"] = facture_index_per_page
    df["invoice_number"] = invoice_number_per_page
    df["invoice_amount"] = invoice_amount_per_page
    df["reference"] = reference_per_page
    df["has_bv_facture"] = has_bv_per_page

    # --------------------------------------------------------
    # Résultats par page + export
    # --------------------------------------------------------
    st.subheader("Résultats par page + infos facture & BV")
    st.dataframe(df, use_container_width=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.metric("Pages avec tampon (=1 affiché)", int((df["tampon_pred"] == 1).sum()))
    with c2:
        st.metric("Pages sans tampon (=0 affiché)", int((df["tampon_pred"] == 0).sum()))
    with c3:
        st.download_button(
            "📥 Télécharger les résultats (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            mime="text/csv",
            file_name=f"yolo_predictions_v11_{pdf_path.stem}.csv",
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



