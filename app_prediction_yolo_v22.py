# app_prediction_yolo_v22.py
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF

from ultralytics import YOLO

# ------------------------------------------------------------
# OpenCV pour le QR-code
# ------------------------------------------------------------
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# ------------------------------------------------------------
# Tesseract pour OCR (texte à droite du QR)
# ------------------------------------------------------------
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    Image = None
    TESSERACT_AVAILABLE = False


# ------------------------------------------------------------
# PDF -> image RGB
# ------------------------------------------------------------
def render_pdf_page_to_rgb(pdf_path: Path, page_index: int = 0, dpi: int = 300):
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        return img


# ------------------------------------------------------------
# YOLO – chargement (facultatif)
# ------------------------------------------------------------
@st.cache_resource
def load_yolo_model(model_path: str):
    return YOLO(model_path)


# ------------------------------------------------------------
# Détection du QR BV (focus bas de page)
# ------------------------------------------------------------
def detect_bv_qr(img_rgb: np.ndarray):
    """
    Détecte le QR du Bulletin de versement (BV) et retourne (data, points).
    Stratégie:
      - ROI bas de page (>= 55% hauteur)
      - detectAndDecodeMulti puis detectAndDecode
      - choisir QR de plus grande surface
    Retour:
      - data : texte décodé (None si vide)
      - points: (4,2) dans coordonnées page entière
    """
    if not HAS_CV2:
        return None, None

    detector = cv2.QRCodeDetector()

    try:
        h, w, _ = img_rgb.shape
        y_start = int(h * 0.55)
        roi = img_rgb[y_start:, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    except Exception:
        return None, None

    candidates = []

    # 1) Multi
    try:
        retval, decoded_infos, points, _ = detector.detectAndDecodeMulti(gray)
    except Exception:
        retval, decoded_infos, points = False, None, None

    if retval and points is not None and len(points) > 0:
        for i, pts in enumerate(points):
            data_i = decoded_infos[i] if decoded_infos is not None else ""
            pts_full = pts.copy()
            pts_full[:, 1] += y_start
            xs, ys = pts_full[:, 0], pts_full[:, 1]
            area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
            candidates.append((area, data_i, pts_full))

    # 2) Single fallback
    try:
        data_single, pts_single, _ = detector.detectAndDecode(gray)
    except Exception:
        data_single, pts_single = "", None

    if pts_single is not None:
        pts_full = pts_single[0] if pts_single.ndim == 3 else pts_single
        pts_full = pts_full.copy()
        pts_full[:, 1] += y_start
        xs, ys = pts_full[:, 0], pts_full[:, 1]
        area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
        candidates.append((area, data_single or "", pts_full))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, data_best, pts_best = candidates[0]
    return (data_best if data_best else None), pts_best


# ------------------------------------------------------------
# Crop : zone à droite du QR (pour Référence)
# ------------------------------------------------------------
def crop_region_right_of_qr(img_rgb: np.ndarray, qr_points: np.ndarray):
    h, w, _ = img_rgb.shape
    xs = qr_points[:, 0]
    ys = qr_points[:, 1]

    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    margin_x = int(0.02 * w)
    margin_y = int(0.02 * h)

    left = min(w - 1, x_max + margin_x)
    right = w
    top = max(0, y_min - margin_y)
    bottom = min(h, y_max + margin_y)

    if left >= right or top >= bottom:
        return None
    return img_rgb[top:bottom, left:right]


# ------------------------------------------------------------
# OCR Tesseract d'une image RGB
# ------------------------------------------------------------
def ocr_image_rgb(img_rgb: np.ndarray) -> str:
    if not (TESSERACT_AVAILABLE and Image is not None):
        return ""
    try:
        pil_img = Image.fromarray(img_rgb)
        txt = pytesseract.image_to_string(pil_img, lang="fra+deu+ita+eng")
        return txt or ""
    except Exception:
        return ""


# ------------------------------------------------------------
# Parsing Référence dans texte OCR (droite du QR)
# ------------------------------------------------------------
REF_PATTERN_STRICT = r"\b\d{2}(?:\s+\d{5}){5}\b"
REF_LABELS = ["référence", "reference", "referenz"]


def extract_reference_from_text(text: str):
    """
    Cherche la référence dans le texte OCR (zone à droite du QR).
    D'abord autour d'un label "Référence/Referenz/...",
    puis pattern XX XXXXX XXXXX XXXXX XXXXX XXXXX.
    """
    norm = (
        text.replace("\xa0", " ")
        .replace("\u202f", " ")
        .replace("\u00a0", " ")
    )
    norm_low = norm.lower()

    idx_label = -1
    for lab in REF_LABELS:
        pos = norm_low.find(lab)
        if pos != -1 and (idx_label == -1 or pos < idx_label):
            idx_label = pos

    if idx_label != -1:
        start = max(0, idx_label - 50)
        end = min(len(norm), idx_label + 250)
        zone = norm[start:end]
    else:
        zone = norm

    zone_one_space = re.sub(r"\s+", " ", zone)
    m = re.search(REF_PATTERN_STRICT, zone_one_space)
    if m:
        return m.group(0).strip()

    digit_sequences = []
    for m2 in re.finditer(r"[0-9\D]{20,}", zone):
        raw = m2.group(0)
        digits = re.sub(r"\D", "", raw)
        if len(digits) >= 27:
            digit_sequences.append(digits)

    if not digit_sequences:
        return None

    best = max(digit_sequences, key=len)[:27]
    if len(best) < 27:
        return None

    blocks = [best[0:2], best[2:7], best[7:12], best[12:17], best[17:22], best[22:27]]
    return " ".join(blocks)


# ------------------------------------------------------------
# Fallback : Référence dans le payload du QR (QRR / SCOR)
# ------------------------------------------------------------
def extract_reference_from_qr_payload(txt: str):
    """
    Swiss QR-bill payload:
      ... QRR\n210000000003139471430009017\n...
    ou SCOR + ligne suivante.
    Retour formaté comme imprimé sur BV (pour QRR).
    """
    norm = txt.replace("\r", "\n")
    lines = [l.strip() for l in norm.splitlines() if l.strip()]

    for i, line in enumerate(lines):
        if line in ("QRR", "SCOR"):
            if i + 1 >= len(lines):
                break
            ref_line = lines[i + 1].strip()

            if line == "QRR":
                digits = re.sub(r"\D", "", ref_line)
                if len(digits) >= 27:
                    digits = digits[:27]
                    blocks = [digits[0:2], digits[2:7], digits[7:12], digits[12:17], digits[17:22], digits[22:27]]
                    return " ".join(blocks)
                return ref_line or None
            else:  # SCOR
                return ref_line or None

    return None


# ------------------------------------------------------------
# Parsing Monnaie + Montant dans le texte du QR
# ------------------------------------------------------------
def parse_amount(raw: str):
    cleaned = re.sub(r"[^\d,\.']", "", raw)
    cleaned = cleaned.replace(" ", "").replace("'", "").replace("’", "")
    cleaned = cleaned.replace(",", ".")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_currency_amount_from_qr_text(txt: str):
    norm = txt.replace("\r", "\n")
    m_curr = re.search(r"\b(CHF|EUR)\b", norm, flags=re.IGNORECASE)
    currency = m_curr.group(1).upper() if m_curr else None

    amount = None
    for line in norm.splitlines():
        ln = line.strip().replace(" ", "")
        if re.fullmatch(r"\d{1,9}[.,]\d{2}", ln):
            amount = parse_amount(ln)
            if amount is not None:
                break

    return currency, amount


# ------------------------------------------------------------
# V22 : Comptage factures par BV "fin de facture"
# ------------------------------------------------------------
def assign_invoices_by_bv(df_pages: pd.DataFrame):
    """
    Hypothèse v22:
      - Si une page contient un BV, alors c'est la DERNIÈRE page de la facture.
    Donc:
      - invoice_id s'incrémente à chaque page has_bv == True
      - toutes les pages depuis la dernière coupure appartiennent à la même facture

    Retour:
      df_pages enrichi (invoice_id, invoice_end_by_bv, is_orphan_after_last_bv)
      df_invoices (1 ligne par facture "close" par BV, et éventuellement une "INCOMPLETE" si pages restantes)
    """
    df = df_pages.copy()
    invoice_ids = []
    cur = 1
    last_closed_invoice = 0

    for _, r in df.iterrows():
        invoice_ids.append(cur)
        if bool(r.get("has_bv", False)):
            last_closed_invoice = cur
            cur += 1

    df["invoice_id"] = invoice_ids
    df["invoice_end_by_bv"] = df["has_bv"].astype(bool)

    # Pages après le dernier BV = facture incomplète (pas de BV de fin)
    df["is_orphan_after_last_bv"] = False
    if last_closed_invoice > 0:
        df.loc[df["invoice_id"] > last_closed_invoice, "is_orphan_after_last_bv"] = True
    else:
        # aucun BV du tout => tout est "incomplet"
        df["is_orphan_after_last_bv"] = True

    # Construction tableau factures
    invoices = []
    for inv_id, g in df.groupby("invoice_id", sort=True):
        pages = g["page"].tolist()
        has_end = bool(g["invoice_end_by_bv"].any())

        invoices.append(
            {
                "invoice_id": int(inv_id),
                "pages_count": int(len(pages)),
                "page_start": int(min(pages)),
                "page_end": int(max(pages)),
                "ended_by_bv": has_end,
                "bv_page": int(g.loc[g["invoice_end_by_bv"], "page"].max()) if has_end else None,
                "bv_reference": (
                    g.loc[g["invoice_end_by_bv"], "bv_reference"].dropna().iloc[-1]
                    if has_end and g.loc[g["invoice_end_by_bv"], "bv_reference"].dropna().shape[0] > 0
                    else None
                ),
                "bv_amount": (
                    float(g.loc[g["invoice_end_by_bv"], "bv_amount"].dropna().iloc[-1])
                    if has_end and g.loc[g["invoice_end_by_bv"], "bv_amount"].dropna().shape[0] > 0
                    else None
                ),
                "bv_currency": (
                    g.loc[g["invoice_end_by_bv"], "bv_currency"].dropna().iloc[-1]
                    if has_end and g.loc[g["invoice_end_by_bv"], "bv_currency"].dropna().shape[0] > 0
                    else None
                ),
                "status": "CLOSED_BY_BV" if has_end else "INCOMPLETE_NO_BV_END",
            }
        )

    df_invoices = pd.DataFrame(invoices)

    # Le "comptage factures" v22 = nb factures clôturées par BV
    invoices_closed = int((df_invoices["status"] == "CLOSED_BY_BV").sum())

    return df, df_invoices, invoices_closed


# ------------------------------------------------------------
# Config Streamlit
# ------------------------------------------------------------
st.set_page_config(
    page_title="BV v22 – Détection + Comptage factures",
    layout="wide",
)
st.title("🧾 BV v22 – Détection des BV + Comptage de factures (BV = fin de facture)")

status_parts = [
    f"OpenCV QR {'✅' if HAS_CV2 else '❌'}",
    f"Tesseract OCR {'✅' if TESSERACT_AVAILABLE else '❌'}",
]
st.caption(" | ".join(status_parts))

st.markdown(
    """
**Objectifs v22 :**

- Conserver l'analyse **page par page** (BV, Référence, Montant, etc.).
- Ajouter un **comptage de factures** avec une règle simple (v22) :

> **Si une page contient un BV, alors c'est la dernière page de la facture.**

Donc chaque BV détecté **clôture** une facture, et on affecte un `invoice_id` à toutes les pages.
"""
)

# ------------------------------------------------------------
# Paramètres modèle YOLO (optionnel, on garde)
# ------------------------------------------------------------
MODEL_DEFAULT = "runs_tampon/yolov8s_tampon/weights/best.pt"

model_path_text = st.text_input(
    "Chemin du modèle YOLO (.pt) – optionnel (tampons non utilisés pour le comptage v22)",
    value=str(Path(MODEL_DEFAULT).resolve()),
)

use_yolo = st.checkbox("Activer la détection de tampons YOLO (info dans le tableau)", value=False)

conf_thres = st.slider(
    "Seuil de confiance (conf tampons)",
    min_value=0.10,
    max_value=1.00,
    value=0.94,
    step=0.01,
)
iou_thres = st.slider("Seuil IoU (NMS)", 0.1, 0.9, 0.45, 0.05)
dpi = st.select_slider("DPI rendu PDF", options=[150, 200, 250, 300, 350, 400], value=300)
show_images = st.checkbox("Afficher les pages avec boîtes de détection (YOLO)", value=False)

uploaded_pdf = st.file_uploader("📄 Dépose un PDF multipages", type=["pdf"])


# ------------------------------------------------------------
# Pipeline principal v22
# ------------------------------------------------------------
if uploaded_pdf:
    tmp_dir = Path("data/tmp_bv_v22")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / uploaded_pdf.name
    pdf_path.write_bytes(uploaded_pdf.getbuffer())
    st.info(f"PDF enregistré : {pdf_path.resolve()}")

    # YOLO (optionnel)
    model = None
    if use_yolo:
        try:
            model = load_yolo_model(model_path_text)
            st.success("Modèle YOLO chargé ✅")
        except Exception as e:
            st.error(f"Impossible de charger le modèle YOLO (tampons désactivés) : {e}")
            model = None
            use_yolo = False

    # PDF
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

    page_qr_texts = []
    page_ref_ocr_texts = []

    # ----------------- boucle pages -----------------
    for i in range(n_pages):
        img_rgb = render_pdf_page_to_rgb(pdf_path, page_index=i, dpi=dpi)

        # 1) Tampon YOLO (facultatif, uniquement pour info)
        tampon_pred = None
        proba_tampon = None
        n_det = None
        max_conf = None

        if use_yolo and model is not None:
            results = model(img_rgb, conf=conf_thres, iou=iou_thres, verbose=False)
            r = results[0]
            boxes = r.boxes
            n_det = len(boxes)

            if n_det > 0:
                confs = boxes.conf.cpu().numpy()
                max_conf_val = float(confs.max())
                tampon_pred = 1
                proba_tampon = max_conf_val
                max_conf = max_conf_val
            else:
                tampon_pred = 0
                proba_tampon = 0.0
                max_conf = 0.0

            if show_images:
                im_plot = r.plot()
                images_to_show.append((i + 1, im_plot))

        # 2) QR-code BV
        qr_text, qr_pts = detect_bv_qr(img_rgb)
        has_bv_page = qr_pts is not None
        page_qr_texts.append(qr_text)

        # 3) OCR zone à droite du QR (Référence)
        ref_zone_text = ""
        bv_reference = None
        ref_source = None

        if qr_pts is not None:
            crop = crop_region_right_of_qr(img_rgb, qr_pts)
            if crop is not None:
                ref_zone_text = ocr_image_rgb(crop)

            # 3a) Référence via OCR
            if ref_zone_text.strip():
                bv_reference = extract_reference_from_text(ref_zone_text)
                if bv_reference:
                    ref_source = "OCR_DROITE_QR"

            # 3b) Fallback : via payload QR
            if not bv_reference and qr_text:
                bv_reference_payload = extract_reference_from_qr_payload(qr_text)
                if bv_reference_payload:
                    bv_reference = bv_reference_payload
                    ref_source = "QR_PAYLOAD"

        page_ref_ocr_texts.append(ref_zone_text)

        # 4) Monnaie + Montant
        bv_currency = None
        bv_amount = None
        if qr_text:
            bv_currency, bv_amount = extract_currency_amount_from_qr_text(qr_text)

        row = {
            "page": i + 1,
            "has_bv": bool(has_bv_page),
            "bv_reference": bv_reference,
            "ref_source": ref_source,
            "bv_currency": bv_currency,
            "bv_amount": bv_amount,
            "qr_text_preview": (qr_text[:80] + "…") if qr_text else None,
        }

        if use_yolo:
            row.update(
                {
                    "tampon_pred": tampon_pred,
                    "proba_tampon": round(proba_tampon, 4) if proba_tampon is not None else None,
                    "n_detections": int(n_det) if n_det is not None else None,
                    "max_conf": round(max_conf, 4) if max_conf is not None else None,
                }
            )

        rows.append(row)

        if (i + 1) % max(1, n_pages // 20) == 0 or i == n_pages - 1:
            progress.progress((i + 1) / n_pages, text=f"Page {i+1}/{n_pages}")

    df_pages = pd.DataFrame(rows)

    # --------------------------------------------------------
    # V22 : Comptage factures + affectation invoice_id
    # --------------------------------------------------------
    df_pages2, df_invoices, invoices_closed = assign_invoices_by_bv(df_pages)

    # --------------------------------------------------------
    # AFFICHAGE
    # --------------------------------------------------------
    st.subheader("🧮 Comptage de factures (règle v22 : BV = fin de facture)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Factures clôturées (BV détectés)", invoices_closed)
    col2.metric("BV détectés (pages)", int(df_pages2["has_bv"].sum()))
    col3.metric("Factures totales (incl. incomplètes)", int(df_invoices.shape[0]))

    if (df_invoices["status"] == "INCOMPLETE_NO_BV_END").any():
        st.warning(
            "Au moins une facture est **incomplète** (pages après le dernier BV, ou aucun BV détecté). "
            "Statut = `INCOMPLETE_NO_BV_END`."
        )

    st.subheader("📄 Résultats par page (BV + invoice_id)")
    st.dataframe(df_pages2, use_container_width=True)

    st.subheader("🧾 Résumé par facture (segmentation v22)")
    st.dataframe(df_invoices, use_container_width=True)

    # CSV exports
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📥 Télécharger résultats par page (CSV)",
            data=df_pages2.to_csv(index=False).encode("utf-8"),
            mime="text/csv",
            file_name=f"bv_v22_pages_{pdf_path.stem}.csv",
        )
    with c2:
        st.download_button(
            "📥 Télécharger résumé factures (CSV)",
            data=df_invoices.to_csv(index=False).encode("utf-8"),
            mime="text/csv",
            file_name=f"bv_v22_invoices_{pdf_path.stem}.csv",
        )

    # Debug : OCR zone Référence
    with st.expander("🔍 Debug Référence : texte OCR à droite du QR"):
        page_num_debug = st.number_input("Page (1 à n_pages)", min_value=1, max_value=n_pages, value=1)
        if st.button("Afficher le texte OCR de cette page (droite du QR)"):
            txt = page_ref_ocr_texts[page_num_debug - 1]
            st.text(txt if txt.strip() else "[aucun texte / pas de BV sur cette page]")

    # Debug : texte brut du QR
    with st.expander("🔍 Debug BV : texte brut du QR d'une page"):
        page_num_debug2 = st.number_input("Page pour le QR", min_value=1, max_value=n_pages, value=1, key="qr_debug_page")
        if st.button("Afficher le texte QR de cette page"):
            txt_qr = page_qr_texts[page_num_debug2 - 1]
            st.text(txt_qr if txt_qr else "[aucun QR BV décodé sur cette page]")

    # Aperçu visuel YOLO (si activé)
    if show_images and images_to_show:
        st.subheader("Aperçu des pages avec boîtes de détection (tampons YOLO)")
        per_row = 2
        k = 0
        while k < len(images_to_show):
            cols = st.columns(per_row)
            for col in cols:
                if k >= len(images_to_show):
                    break
                page_no, im_plot = images_to_show[k]
                col.image(im_plot, caption=f"Page {page_no}")
                k += 1

else:
    st.info("Dépose un PDF multipages pour analyser BV + compter les factures (v22).")
