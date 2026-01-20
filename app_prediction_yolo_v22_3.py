# app_prediction_yolo_v22_3.py
# v22.3 = BV validé (QR bas + Référence stricte) + segmentation factures
# - Si BV validé => FIN facture (page BV)
# - Sinon (factures sans BV) => on coupe quand on détecte un TAMPON (YOLO) sur une page,
#   car on suppose que le tampon est sur la 1ère page de la facture.
#
# Paramètres par défaut demandés :
# - Recherche QR candidate (ratio) = 0.45
# - OCR zones BV (ratio) = 0.45
# - DPI = 400

from pathlib import Path
import re
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from ultralytics import YOLO

# ------------------------------------------------------------
# OpenCV pour QR
# ------------------------------------------------------------
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# ------------------------------------------------------------
# Tesseract OCR
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
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)


# ------------------------------------------------------------
# YOLO – chargement
# ------------------------------------------------------------
@st.cache_resource
def load_yolo_model(model_path: str):
    return YOLO(model_path)


# ------------------------------------------------------------
# QR candidate dans le bas de page
# ------------------------------------------------------------
def detect_bottom_qr_candidates(img_rgb: np.ndarray, bottom_ratio: float = 0.45, max_keep: int = 6):
    """
    Détecte plusieurs QR dans le bas de page.
    Retourne une liste de tuples: (area, qr_text, qr_points_full)
    - On garde max_keep candidats triés par surface décroissante.
    """
    if not HAS_CV2:
        return []

    detector = cv2.QRCodeDetector()
    try:
        h, w, _ = img_rgb.shape
        y_start = int(h * bottom_ratio)
        roi = img_rgb[y_start:, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    except Exception:
        return []

    candidates = []

    # Multi
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
            candidates.append((area, data_i or None, pts_full))

    # Fallback single
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
        candidates.append((area, data_single or None, pts_full))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:max_keep]


# ------------------------------------------------------------
# OCR
# ------------------------------------------------------------
def ocr_image_rgb(img_rgb: np.ndarray) -> str:
    if not (TESSERACT_AVAILABLE and Image is not None):
        return ""
    try:
        pil_img = Image.fromarray(img_rgb)
        return pytesseract.image_to_string(pil_img, lang="fra+deu+ita+eng") or ""
    except Exception:
        return ""


# ------------------------------------------------------------
# Référence STRICTE (BV)
# ------------------------------------------------------------
REF_PATTERN_STRICT = r"\b\d{2}(?:\s+\d{5}){5}\b"

def normalize_spaces(s: str) -> str:
    return (
        (s or "")
        .replace("\xa0", " ")
        .replace("\u202f", " ")
        .replace("\u00a0", " ")
    )

def extract_reference_strict_from_text(text: str) -> Optional[str]:
    """
    IMPORTANT: uniquement pattern strict tel quel.
    (on NE reconstruit PAS à partir de longues suites de chiffres)
    """
    if not text or not text.strip():
        return None
    zone = re.sub(r"\s+", " ", normalize_spaces(text))
    m = re.search(REF_PATTERN_STRICT, zone)
    return m.group(0).strip() if m else None

def is_reference_strict(ref) -> bool:
    if not ref:
        return False
    ref_norm = re.sub(r"\s+", " ", str(ref)).strip()
    return re.fullmatch(REF_PATTERN_STRICT, ref_norm) is not None


# ------------------------------------------------------------
# Référence depuis payload QR (QRR uniquement)
# ------------------------------------------------------------
def extract_reference_from_qr_payload_qrr(txt: str) -> Optional[str]:
    """
    Valide seulement si 'QRR' + ligne suivante (27 digits) => format strict.
    """
    if not txt:
        return None
    norm = txt.replace("\r", "\n")
    lines = [l.strip() for l in norm.splitlines() if l.strip()]

    for i, line in enumerate(lines):
        if line == "QRR":
            if i + 1 >= len(lines):
                return None
            digits = re.sub(r"\D", "", lines[i + 1])
            if len(digits) < 27:
                return None
            digits = digits[:27]
            blocks = [digits[0:2], digits[2:7], digits[7:12], digits[12:17], digits[17:22], digits[22:27]]
            ref = " ".join(blocks)
            return ref if is_reference_strict(ref) else None

    return None


# ------------------------------------------------------------
# Monnaie + Montant depuis payload QR (debug)
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

def extract_currency_amount_from_qr_text(txt: str) -> Tuple[Optional[str], Optional[float]]:
    if not txt:
        return None, None
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
# OCR zones autour QR (plusieurs crops)
# ------------------------------------------------------------
def crop_reference_regions(img_rgb: np.ndarray, qr_points: np.ndarray, bottom_ratio: float = 0.45):
    """
    OCR sur plusieurs zones probables.
    """
    h, w, _ = img_rgb.shape
    xs = qr_points[:, 0]
    ys = qr_points[:, 1]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    margin_x = int(0.02 * w)
    margin_y = int(0.02 * h)

    crops = []

    # A) droite du QR (autour vertical)
    left = min(w - 1, x_max + margin_x)
    right = w
    top = max(0, y_min - margin_y)
    bottom = min(h, y_max + margin_y)
    if left < right and top < bottom:
        crops.append(("OCR_DROITE_QR", img_rgb[top:bottom, left:right]))

    # B) droite LARGE (du début section BV vers bas)
    y_start = int(h * bottom_ratio)
    top2 = max(0, y_start)
    bottom2 = h
    left2 = min(w - 1, x_max + margin_x)
    right2 = w
    if left2 < right2 and top2 < bottom2:
        crops.append(("OCR_DROITE_LARGE", img_rgb[top2:bottom2, left2:right2]))

    # C) gauche LARGE (récépissé)
    left3 = 0
    right3 = max(1, x_min - margin_x)
    top3 = max(0, y_start)
    bottom3 = h
    if left3 < right3 and top3 < bottom3:
        crops.append(("OCR_GAUCHE_LARGE", img_rgb[top3:bottom3, left3:right3]))

    return crops


# ------------------------------------------------------------
# Détection BV validé sur une page (multi-QR candidates)
# ------------------------------------------------------------
def detect_bv_valid_on_page(
    img_rgb: np.ndarray,
    qr_bottom_ratio: float,
    ocr_bottom_ratio: float,
    max_qr_keep: int = 6,
):
    """
    Retour:
      has_qr_candidate: bool
      has_bv_valid: bool
      bv_reference: str|None
      ref_source: str|None
      qr_text_best: str|None (pour debug)
      qr_text_preview_best: str|None (pour tableau)
      debug_ocr_concat: str (facultatif)
      debug_qr_texts: list[str] (facultatif)
    """
    qr_candidates = detect_bottom_qr_candidates(img_rgb, bottom_ratio=qr_bottom_ratio, max_keep=max_qr_keep)
    has_qr_candidate = len(qr_candidates) > 0

    # Debug: garder les textes QR (quand décodés)
    debug_qr_texts = []
    for _, t, _ in qr_candidates:
        if t:
            debug_qr_texts.append(t)

    # On va tester les QR candidates en ordre (plus gros d'abord),
    # et valider le premier qui donne une référence stricte (OCR ou QRR).
    bv_reference = None
    ref_source = None
    chosen_qr_text = None
    ocr_debug_concat = ""

    for area, qr_text, qr_pts in qr_candidates:
        # 1) OCR autour de CE QR
        crops = crop_reference_regions(img_rgb, qr_pts, bottom_ratio=ocr_bottom_ratio)
        for source_name, crop_img in crops:
            txt = ocr_image_rgb(crop_img)
            if txt and txt.strip():
                ocr_debug_concat += f"\n--- QR_area={area:.0f} {source_name} ---\n{txt}\n"
                ref = extract_reference_strict_from_text(txt)
                if ref and is_reference_strict(ref):
                    bv_reference = ref
                    ref_source = source_name
                    chosen_qr_text = qr_text
                    break
        if bv_reference:
            break

        # 2) Fallback payload QRR (si qr_text décodé)
        if qr_text:
            ref_payload = extract_reference_from_qr_payload_qrr(qr_text)
            if ref_payload and is_reference_strict(ref_payload):
                bv_reference = ref_payload
                ref_source = "QR_PAYLOAD_QRR"
                chosen_qr_text = qr_text
                break

    has_bv_valid = bool(has_qr_candidate and is_reference_strict(bv_reference))

    # Choisir un "best qr_text" pour preview (même si pas BV validé)
    best_text = None
    for _, t, _ in qr_candidates:
        if t:
            best_text = t
            break

    qr_text_for_preview = chosen_qr_text if chosen_qr_text else best_text
    qr_text_preview = (qr_text_for_preview[:80] + "…") if qr_text_for_preview else None

    return (
        has_qr_candidate,
        has_bv_valid,
        bv_reference,
        ref_source,
        qr_text_for_preview,
        qr_text_preview,
        ocr_debug_concat,
        debug_qr_texts,
    )


# ------------------------------------------------------------
# Segmentation factures robuste : BV prime, sinon tampon
# ------------------------------------------------------------
def assign_invoices_hybrid(df_pages: pd.DataFrame):
    """
    Règles v22.3:
      1) Si une page i contient un BV validé => page i = FIN facture (raison 'BV')
         => page i+1 démarre une nouvelle facture
      2) Si pas de BV pour finir, alors un TAMPON (sur page i) indique une NOUVELLE facture.
         => On coupe entre i-1 et i (raison 'STAMP_START')
      3) La dernière facture peut finir en 'EOF' si non clôturée par BV.
    """
    df = df_pages.copy()

    invoice_id = 1
    invoice_ids = []
    end_reason = [None] * len(df)  # reason attached to page boundary (end page)

    for idx in range(len(df)):
        if idx == 0:
            invoice_ids.append(invoice_id)
            continue

        prev_has_bv = bool(df.loc[idx - 1, "has_bv"])
        cur_has_stamp = bool(df.loc[idx, "has_tampon"])

        if prev_has_bv:
            # la facture se termine à la page précédente
            end_reason[idx - 1] = "BV"
            invoice_id += 1
        else:
            # pas de BV pour finir => si on voit un tampon sur la page courante,
            # on considère que c'est une nouvelle facture (facture précédente sans BV)
            if cur_has_stamp:
                end_reason[idx - 1] = "STAMP_START"
                invoice_id += 1

        invoice_ids.append(invoice_id)

    df["invoice_id"] = invoice_ids
    df["invoice_end_reason"] = end_reason
    df["invoice_end_by_bv"] = df["invoice_end_reason"].eq("BV")
    df["invoice_end_by_stamp"] = df["invoice_end_reason"].eq("STAMP_START")

    # Résumé factures
    invoices = []
    for inv_id, g in df.groupby("invoice_id", sort=True):
        pages = g["page"].tolist()
        page_start = int(min(pages))
        page_end = int(max(pages))

        # Une facture est "clôturée BV" si la dernière page de la facture a end_reason == BV
        # Une facture est "clôturée par tampon suivant" si sa dernière page a end_reason == STAMP_START
        g_sorted = g.sort_values("page")
        last_row = g_sorted.iloc[-1]
        last_page_idx = last_row.name  # index dans df

        reason = df.loc[last_page_idx, "invoice_end_reason"]
        ended_by_bv = reason == "BV"
        ended_by_stamp = reason == "STAMP_START"
        ended_by_eof = reason is None  # fin fichier

        status = "CLOSED_BY_BV" if ended_by_bv else ("CLOSED_BY_STAMP" if ended_by_stamp else "EOF_NO_BV")

        # infos BV si présentes dans la facture
        bv_rows = g_sorted[g_sorted["has_bv"] == True]
        bv_page = int(bv_rows["page"].max()) if len(bv_rows) > 0 else None

        bv_reference = None
        bv_amount = None
        bv_currency = None
        if bv_page is not None:
            last_bv = bv_rows.sort_values("page").iloc[-1]
            bv_reference = last_bv.get("bv_reference")
            bv_amount = last_bv.get("bv_amount")
            bv_currency = last_bv.get("bv_currency")

        invoices.append(
            {
                "invoice_id": int(inv_id),
                "pages_count": int(len(pages)),
                "page_start": page_start,
                "page_end": page_end,
                "end_reason": reason if reason else "EOF",
                "ended_by_bv": bool(ended_by_bv),
                "ended_by_stamp": bool(ended_by_stamp),
                "bv_page": bv_page,
                "bv_reference": bv_reference,
                "bv_currency": bv_currency,
                "bv_amount": bv_amount,
                "status": status,
            }
        )

    df_invoices = pd.DataFrame(invoices)

    # Métriques
    n_closed_bv = int((df_invoices["status"] == "CLOSED_BY_BV").sum())
    n_closed_stamp = int((df_invoices["status"] == "CLOSED_BY_STAMP").sum())
    n_eof = int((df_invoices["status"] == "EOF_NO_BV").sum())

    return df, df_invoices, n_closed_bv, n_closed_stamp, n_eof


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(
    page_title="BV v22.3 – BV + Tampon (factures sans BV) + Comptage",
    layout="wide",
)
st.title("🧾 BV v22.3 – Segmentation factures : BV (fin) + Tampon (début)")

status_parts = [
    f"OpenCV QR {'✅' if HAS_CV2 else '❌'}",
    f"Tesseract OCR {'✅' if TESSERACT_AVAILABLE else '❌'}",
]
st.caption(" | ".join(status_parts))

st.markdown(
    """
### Règles v22.3 (hybride)
1) **BV validé** (QR bas + Référence stricte) ⇒ **fin de facture** (page BV).  
2) Si une facture **n'a pas de BV**, on utilise le **tampon YOLO** :  
   - on suppose que **le tampon est sur la 1ère page** de la facture  
   - donc quand on détecte un **tampon sur une page**, on **démarre une nouvelle facture** à cette page  
3) Le fichier peut finir par une facture **EOF_NO_BV** si aucune clôture BV.

### Paramètres par défaut (fixés comme demandé)
- QR candidate ratio = **0.45**
- OCR zones BV ratio = **0.45**
- DPI = **400**
"""
)

# ------------------------------------------------------------
# Paramètres
# ------------------------------------------------------------
MODEL_DEFAULT = "runs_tampon/yolov8s_tampon/weights/best.pt"

model_path_text = st.text_input(
    "Chemin du modèle YOLO (.pt) – requis pour segmenter les factures sans BV",
    value=str(Path(MODEL_DEFAULT).resolve()),
)

use_yolo = st.checkbox(
    "Activer la détection de tampons YOLO (recommandé pour factures sans BV)",
    value=True,
)

tampon_conf_thres = st.slider(
    "Seuil de confiance tampon (YOLO)",
    min_value=0.10,
    max_value=1.00,
    value=0.94,
    step=0.01,
)

iou_thres = st.slider("Seuil IoU (NMS)", 0.1, 0.9, 0.45, 0.05)

dpi = st.select_slider("DPI rendu PDF", options=[150, 200, 250, 300, 350, 400], value=400)

qr_bottom_ratio = st.slider(
    "Recherche QR candidate : début du bas de page (ratio hauteur)",
    min_value=0.30,
    max_value=0.80,
    value=0.45,
    step=0.01,
)

ocr_bottom_ratio = st.slider(
    "OCR zones BV : début de zone (ratio hauteur)",
    min_value=0.30,
    max_value=0.80,
    value=0.45,
    step=0.01,
)

max_qr_keep = st.slider("Nombre max de QR candidates testés par page", 1, 10, 6, 1)

show_yolo_images = st.checkbox("Afficher les pages avec boîtes de détection (YOLO)", value=False)

uploaded_pdf = st.file_uploader("📄 Dépose un PDF multipages", type=["pdf"])


# ------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------
if uploaded_pdf:
    tmp_dir = Path("data/tmp_bv_v22_3")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / uploaded_pdf.name
    pdf_path.write_bytes(uploaded_pdf.getbuffer())
    st.info(f"PDF enregistré : {pdf_path.resolve()}")

    # Charger YOLO si demandé
    model = None
    if use_yolo:
        try:
            model = load_yolo_model(model_path_text)
            st.success("Modèle YOLO chargé ✅ (tampons actifs)")
        except Exception as e:
            st.error(f"Impossible de charger YOLO : {e}")
            st.warning("➡️ Les factures SANS BV ne seront pas segmentées correctement sans tampon.")
            model = None
            use_yolo = False

    # PDF open
    try:
        with fitz.open(str(pdf_path)) as doc:
            n_pages = len(doc)
    except Exception as e:
        st.error(f"Impossible d'ouvrir le PDF : {e}")
        st.stop()

    st.write(f"Pages détectées : **{n_pages}**")
    progress = st.progress(0, text="Analyse des pages…")

    rows = []
    yolo_images = []

    # debug
    debug_qr_texts_by_page: List[List[str]] = []
    debug_ocr_by_page: List[str] = []

    for i in range(n_pages):
        img_rgb = render_pdf_page_to_rgb(pdf_path, page_index=i, dpi=dpi)

        # --- YOLO tampon (si activé) ---
        has_tampon = False
        tampon_pred = None
        proba_tampon = None
        n_det = None
        max_conf = None

        if use_yolo and model is not None:
            results = model(img_rgb, conf=tampon_conf_thres, iou=iou_thres, verbose=False)
            r = results[0]
            boxes = r.boxes
            n_det = len(boxes)

            if n_det > 0:
                confs = boxes.conf.cpu().numpy()
                max_conf_val = float(confs.max())
                has_tampon = True
                tampon_pred = 1
                proba_tampon = max_conf_val
                max_conf = max_conf_val
            else:
                has_tampon = False
                tampon_pred = 0
                proba_tampon = 0.0
                max_conf = 0.0

            if show_yolo_images:
                yolo_images.append((i + 1, r.plot()))
        else:
            # YOLO non dispo -> on laisse False
            has_tampon = False

        # --- BV validé (multi-QR + Référence stricte) ---
        (
            has_qr_candidate,
            has_bv_valid,
            bv_reference,
            ref_source,
            qr_text_best,
            qr_text_preview,
            ocr_debug_concat,
            qr_texts_list,
        ) = detect_bv_valid_on_page(
            img_rgb,
            qr_bottom_ratio=qr_bottom_ratio,
            ocr_bottom_ratio=ocr_bottom_ratio,
            max_qr_keep=max_qr_keep,
        )

        debug_qr_texts_by_page.append(qr_texts_list)
        debug_ocr_by_page.append(ocr_debug_concat)

        # Debug monnaie/montant si qr_text_best existe
        bv_currency, bv_amount = (None, None)
        if qr_text_best:
            bv_currency, bv_amount = extract_currency_amount_from_qr_text(qr_text_best)

        row = {
            "page": i + 1,

            # BV
            "has_qr_candidate": bool(has_qr_candidate),
            "has_bv": bool(has_bv_valid),
            "bv_reference": bv_reference,
            "ref_source": ref_source,
            "bv_currency": bv_currency,
            "bv_amount": bv_amount,
            "qr_text_preview": qr_text_preview,

            # Tampon
            "has_tampon": bool(has_tampon),
            "tampon_pred": tampon_pred,
            "proba_tampon": round(proba_tampon, 4) if proba_tampon is not None else None,
            "n_detections": int(n_det) if n_det is not None else None,
            "max_conf": round(max_conf, 4) if max_conf is not None else None,
        }

        rows.append(row)

        if (i + 1) % max(1, n_pages // 20) == 0 or i == n_pages - 1:
            progress.progress((i + 1) / n_pages, text=f"Page {i+1}/{n_pages}")

    df_pages = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Segmentation HYBRIDE
    # --------------------------------------------------------
    df_pages2, df_invoices, n_closed_bv, n_closed_stamp, n_eof = assign_invoices_hybrid(df_pages)

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------
    st.subheader("🧮 Résumé segmentation (v22.3)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Factures clôturées par BV", n_closed_bv)
    c2.metric("Factures clôturées par tampon", n_closed_stamp)
    c3.metric("Factures fin fichier (EOF)", n_eof)
    c4.metric("BV validés (pages)", int(df_pages2["has_bv"].sum()))
    c5.metric("Tampons détectés (pages)", int(df_pages2["has_tampon"].sum()))

    if not use_yolo:
        st.warning(
            "YOLO est désactivé / indisponible. "
            "➡️ La segmentation des factures SANS BV ne peut pas être robuste sans tampons."
        )

    # Faux QR (QR détecté mais pas BV)
    false_qr = df_pages2[(df_pages2["has_qr_candidate"]) & (~df_pages2["has_bv"])]
    if len(false_qr) > 0:
        st.info(
            f"{len(false_qr)} page(s) ont un QR en bas de page mais pas de référence stricte "
            f"→ ignorées comme BV."
        )

    # --------------------------------------------------------
    # TABLES
    # --------------------------------------------------------
    st.subheader("📄 Résultats par page (BV + tampon + invoice_id)")
    st.dataframe(df_pages2, use_container_width=True)

    st.subheader("🧾 Résumé par facture")
    st.dataframe(df_invoices, use_container_width=True)

    # --------------------------------------------------------
    # EXPORTS
    # --------------------------------------------------------
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button(
            "📥 Télécharger résultats par page (CSV)",
            data=df_pages2.to_csv(index=False).encode("utf-8"),
            mime="text/csv",
            file_name=f"bv_v22_3_pages_{pdf_path.stem}.csv",
        )
    with e2:
        st.download_button(
            "📥 Télécharger résumé factures (CSV)",
            data=df_invoices.to_csv(index=False).encode("utf-8"),
            mime="text/csv",
            file_name=f"bv_v22_3_invoices_{pdf_path.stem}.csv",
        )
    with e3:
        st.download_button(
            "📥 Télécharger QR ignorés (CSV)",
            data=false_qr.to_csv(index=False).encode("utf-8") if len(false_qr) > 0 else "page\n".encode("utf-8"),
            mime="text/csv",
            file_name=f"bv_v22_3_false_qr_{pdf_path.stem}.csv",
        )

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------
    with st.expander("🔍 Debug : texte(s) QR détecté(s) sur une page"):
        page_num_debug = st.number_input("Page (1 à n_pages)", min_value=1, max_value=n_pages, value=1, key="dbg_qr")
        if st.button("Afficher les textes QR détectés (page)"):
            lst = debug_qr_texts_by_page[page_num_debug - 1]
            if not lst:
                st.text("[aucun QR décodé sur cette page]")
            else:
                st.text("\n\n--- QR ---\n\n".join(lst))

    with st.expander("🔍 Debug : OCR zones BV (par page)"):
        page_num_debug2 = st.number_input("Page (1 à n_pages)", min_value=1, max_value=n_pages, value=1, key="dbg_ocr")
        if st.button("Afficher OCR (zones) de cette page"):
            txt = debug_ocr_by_page[page_num_debug2 - 1]
            st.text(txt if txt.strip() else "[aucun texte OCR / pas de QR candidate ou OCR vide]")

    # Aperçu YOLO
    if show_yolo_images and yolo_images:
        st.subheader("Aperçu des pages avec boîtes YOLO (tampons)")
        per_row = 2
        k = 0
        while k < len(yolo_images):
            cols = st.columns(per_row)
            for col in cols:
                if k >= len(yolo_images):
                    break
                page_no, im_plot = yolo_images[k]
                col.image(im_plot, caption=f"Page {page_no}")
                k += 1

else:
    st.info("Dépose un PDF multipages pour analyser BV + segmenter les factures (v22.3).")
