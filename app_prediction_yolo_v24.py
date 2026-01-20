# app_prediction_yolo_v24.py
# v24 (rewrite complet) + colonne "début facture" éditable par l’utilisateur
#
# BV validé = (QR détecté en bas de page) AND (Référence STRICTE "XX XXXXX XXXXX XXXXX XXXXX XXXXX")
# IMPORTANT: on NE reconstruit PLUS une référence à partir de longues suites de chiffres (trop de faux positifs).
#
# v24:
# - Segmentation automatique:
#   - fin facture si BV validé (QR+référence stricte)
#   - optionnel: split sur "tampon" YOLO (supposé 1ère page)
# - NOUVEAU: une colonne "invoice_start" (case à cocher) pré-remplie automatiquement
#   -> l'utilisateur peut éditer ces coches
#   -> puis cliquer "Recalculer segmentation" pour valider la structure finale

from pathlib import Path
import re

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
# YOLO – chargement (facultatif)
# ------------------------------------------------------------
@st.cache_resource
def load_yolo_model(model_path: str):
    return YOLO(model_path)


# ------------------------------------------------------------
# Détection QR bas de page (candidate)
# ------------------------------------------------------------
def detect_bottom_qr_candidate(img_rgb: np.ndarray, bottom_ratio: float = 0.55):
    """
    Détecte un QR candidate dans le bas de page.
    - ROI = [bottom_ratio*h : h]
    - Choisit le QR de plus grande surface.
    Retour: (qr_text, qr_points_full, y_start)
    """
    if not HAS_CV2:
        return None, None, None

    detector = cv2.QRCodeDetector()

    try:
        h, w, _ = img_rgb.shape
        y_start = int(h * bottom_ratio)
        roi = img_rgb[y_start:, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    except Exception:
        return None, None, None

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
            candidates.append((area, data_i, pts_full))

    # Single fallback
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
        return None, None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, data_best, pts_best = candidates[0]
    return (data_best if data_best else None), pts_best, y_start


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

def extract_reference_strict_from_text(text: str):
    """
    IMPORTANT: uniquement pattern strict tel quel.
    On NE reconstruit PAS à partir de longs digits (ça crée des faux positifs).
    """
    if not text or not text.strip():
        return None
    zone = re.sub(r"\s+", " ", normalize_spaces(text))
    m = re.search(REF_PATTERN_STRICT, zone)
    return m.group(0).strip() if m else None

def is_reference_strict(ref):
    if not ref:
        return False
    ref_norm = re.sub(r"\s+", " ", str(ref)).strip()
    return re.fullmatch(REF_PATTERN_STRICT, ref_norm) is not None


# ------------------------------------------------------------
# Référence depuis payload QR (QRR uniquement)
# ------------------------------------------------------------
def extract_reference_from_qr_payload_qrr(txt: str):
    """
    BV Swiss QR: on valide seulement si on trouve 'QRR' et la ligne suivante
    (27 digits) -> formatage 'XX XXXXX XXXXX XXXXX XXXXX XXXXX'
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

def extract_currency_amount_from_qr_text(txt: str):
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
# OCR zones autour BV (sans label "Référence")
# ------------------------------------------------------------
def crop_reference_regions(img_rgb: np.ndarray, qr_points: np.ndarray, bottom_ratio: float = 0.55):
    """
    OCR sur plusieurs zones probables. Mais la validation BV exigera
    une référence STRICTE trouvée dans ces OCR (pas reconstruite).
    """
    h, w, _ = img_rgb.shape
    xs = qr_points[:, 0]
    ys = qr_points[:, 1]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    margin_x = int(0.02 * w)
    margin_y = int(0.02 * h)

    crops = []

    # A) droite du QR (autour vertical QR)
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
# Pré-remplissage "début facture" + segmentation depuis ces starts
# ------------------------------------------------------------
def compute_invoice_starts_default(df_pages: pd.DataFrame, use_tampon_split: bool, tampon_flag_col: str = "tampon_pred"):
    """
    Crée une colonne booléenne invoice_start (pré-remplie):
    - page 1 = True
    - page après un BV = True
    - optionnel: si tampon_split, page avec tampon = True (sauf page 1 déjà True)
    """
    df = df_pages.copy()
    n = df.shape[0]
    starts = [False] * n
    if n == 0:
        df["invoice_start"] = starts
        return df

    starts[0] = True

    # start après BV
    for i in range(1, n):
        if bool(df.loc[i - 1, "has_bv"]):
            starts[i] = True

    # start via tampon
    if use_tampon_split and (tampon_flag_col in df.columns):
        for i in range(1, n):
            if bool(df.loc[i, tampon_flag_col]):
                starts[i] = True

    df["invoice_start"] = starts
    return df


def assign_invoices_from_user_starts(df_pages: pd.DataFrame, start_col: str = "invoice_start"):
    """
    Segmentation "finale" : invoice_id basé uniquement sur les coches invoice_start.
    - Chaque True démarre une nouvelle facture.
    - On conserve les infos BV (si une page a BV, c’est une info de fin potentielle mais la structure est pilotée par l’utilisateur).
    """
    df = df_pages.copy()
    starts = df[start_col].fillna(False).astype(bool).tolist()

    # sécurité: la 1ère page doit être start
    if len(starts) > 0 and not starts[0]:
        starts[0] = True
        df.loc[df.index[0], start_col] = True

    invoice_ids = []
    cur = 0
    for i, s in enumerate(starts):
        if s:
            cur += 1
        invoice_ids.append(cur)

    df["invoice_id"] = invoice_ids

    invoices = []
    for inv_id, g in df.groupby("invoice_id", sort=True):
        pages = g["page"].tolist()
        has_bv = bool(g["has_bv"].any())
        bv_pages = g.loc[g["has_bv"], "page"].tolist()

        # Référence/montant/currency depuis la DERNIERE page BV de la facture (si existante)
        last_bv_page = None
        if len(bv_pages) > 0:
            last_bv_page = int(max(bv_pages))

        def _last_non_null(series):
            s2 = series.dropna()
            return s2.iloc[-1] if len(s2) > 0 else None

        invoices.append(
            {
                "invoice_id": int(inv_id),
                "pages_count": int(len(pages)),
                "page_start": int(min(pages)),
                "page_end": int(max(pages)),
                "has_bv_in_invoice": has_bv,
                "bv_pages": ",".join(map(str, bv_pages)) if bv_pages else None,
                "bv_last_page": last_bv_page,
                "bv_reference": _last_non_null(g.loc[g["has_bv"], "bv_reference"]) if has_bv else None,
                "bv_amount": float(_last_non_null(g.loc[g["has_bv"], "bv_amount"])) if has_bv and _last_non_null(g.loc[g["has_bv"], "bv_amount"]) is not None else None,
                "bv_currency": _last_non_null(g.loc[g["has_bv"], "bv_currency"]) if has_bv else None,
                "status": "HAS_BV" if has_bv else "NO_BV",
            }
        )

    df_invoices = pd.DataFrame(invoices)
    return df, df_invoices


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(
    page_title="BV v24 – QR + Référence strict + validation manuelle des débuts de facture",
    layout="wide",
)
st.title("🧾 BV v24 – Détection BV + cases à cocher 'Début facture' (validation manuelle)")

status_parts = [
    f"OpenCV QR {'✅' if HAS_CV2 else '❌'}",
    f"Tesseract OCR {'✅' if TESSERACT_AVAILABLE else '❌'}",
]
st.caption(" | ".join(status_parts))

st.markdown(
    """
**Règle BV (anti faux-positifs QR) :**
✅ **BV validé** si et seulement si : QR détecté en bas de page **ET** Référence strict `XX XXXXX XXXXX XXXXX XXXXX XXXXX`.

**Nouveau : validation manuelle de la structure**
- La colonne **`invoice_start`** (✅/⬜) est **pré-remplie** automatiquement
- Tu peux **modifier** les coches
- Puis cliquer **Recalculer segmentation** pour obtenir la structure finale (invoice_id, résumé par facture).
"""
)

# Paramètres YOLO (optionnel)
MODEL_DEFAULT = "runs_tampon/yolov8s_tampon/weights/best.pt"
model_path_text = st.text_input(
    "Chemin du modèle YOLO (.pt) – optionnel",
    value=str(Path(MODEL_DEFAULT).resolve()),
)
use_yolo = st.checkbox("Activer la détection de tampons YOLO", value=False)
use_tampon_split = st.checkbox(
    "Pré-cocher aussi 'Début facture' quand un tampon est détecté (hypothèse 1ère page)",
    value=False,
    help="Nécessite YOLO activé. Influence uniquement le pré-remplissage de invoice_start.",
)

conf_thres = st.slider("Seuil de confiance (conf tampons)", 0.10, 1.00, 0.94, 0.01)
iou_thres = st.slider("Seuil IoU (NMS)", 0.1, 0.9, 0.45, 0.05)

# DPI par défaut = 400
dpi = st.select_slider("DPI rendu PDF", options=[150, 200, 250, 300, 350, 400], value=400)

# ROI bas de page pour chercher le QR candidate
qr_bottom_ratio = st.slider(
    "Recherche QR candidate : début du bas de page (ratio hauteur)",
    min_value=0.35,
    max_value=0.80,
    value=0.45,
    step=0.01,
)

# OCR large zone start
ocr_bottom_ratio = st.slider(
    "OCR zones BV : début de zone (ratio hauteur)",
    min_value=0.35,
    max_value=0.80,
    value=0.45,
    step=0.01,
)

show_images = st.checkbox("Afficher les pages avec boîtes de détection (YOLO)", value=False)

uploaded_pdf = st.file_uploader("📄 Dépose un PDF multipages", type=["pdf"])


# ------------------------------------------------------------
# Session state helpers
# ------------------------------------------------------------
def ss_init(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


if uploaded_pdf:
    tmp_dir = Path("data/tmp_bv_v24")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / uploaded_pdf.name
    pdf_path.write_bytes(uploaded_pdf.getbuffer())
    st.info(f"PDF enregistré : {pdf_path.resolve()}")

    # cohérence options
    if use_tampon_split and not use_yolo:
        st.warning("Option 'pré-cocher via tampon' activée mais YOLO est désactivé → ignorée.")
        use_tampon_split = False

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
            use_tampon_split = False

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
    images_to_show = []
    page_qr_texts = []
    page_ocr_debug = []

    for i in range(n_pages):
        img_rgb = render_pdf_page_to_rgb(pdf_path, page_index=i, dpi=dpi)

        # YOLO (optionnel)
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
                images_to_show.append((i + 1, r.plot()))

        # QR candidate (bas de page)
        qr_text, qr_pts, _ = detect_bottom_qr_candidate(img_rgb, bottom_ratio=qr_bottom_ratio)
        has_qr_candidate = qr_pts is not None
        page_qr_texts.append(qr_text)

        # Référence STRICTE (OCR multi-zones)
        bv_reference = None
        ref_source = None
        ocr_debug = ""

        if has_qr_candidate:
            crops = crop_reference_regions(img_rgb, qr_pts, bottom_ratio=ocr_bottom_ratio)
            for source_name, crop_img in crops:
                txt = ocr_image_rgb(crop_img)
                if txt and txt.strip():
                    ocr_debug += f"\n--- {source_name} ---\n{txt}\n"
                    ref = extract_reference_strict_from_text(txt)
                    if ref and is_reference_strict(ref):
                        bv_reference = ref
                        ref_source = source_name
                        break

            # Fallback QR payload QRR
            if not bv_reference and qr_text:
                ref_payload = extract_reference_from_qr_payload_qrr(qr_text)
                if ref_payload and is_reference_strict(ref_payload):
                    bv_reference = ref_payload
                    ref_source = "QR_PAYLOAD_QRR"

        page_ocr_debug.append(ocr_debug)

        # Monnaie/Montant (debug)
        bv_currency, bv_amount = (None, None)
        if qr_text:
            bv_currency, bv_amount = extract_currency_amount_from_qr_text(qr_text)

        # Validation BV = QR + Référence STRICTE
        has_reference_strict = is_reference_strict(bv_reference)
        has_bv_valid = bool(has_qr_candidate and has_reference_strict)

        row = {
            "page": i + 1,
            "has_qr_candidate": bool(has_qr_candidate),
            "has_reference_strict": bool(has_reference_strict),
            "has_bv": bool(has_bv_valid),
            "bv_reference": bv_reference,
            "ref_source": ref_source,
            "bv_currency": bv_currency,
            "bv_amount": bv_amount,
            "qr_text_preview": (qr_text[:80] + "…") if qr_text else None,
        }

        if use_yolo:
            row.update(
                {
                    "tampon_pred": int(tampon_pred) if tampon_pred is not None else None,
                    "proba_tampon": round(proba_tampon, 4) if proba_tampon is not None else None,
                    "n_detections": int(n_det) if n_det is not None else None,
                    "max_conf": round(max_conf, 4) if max_conf is not None else None,
                }
            )

        rows.append(row)

        if (i + 1) % max(1, n_pages // 20) == 0 or i == n_pages - 1:
            progress.progress((i + 1) / n_pages, text=f"Page {i+1}/{n_pages}")

    df_pages = pd.DataFrame(rows)

    # Pré-remplissage invoice_start (début facture)
    df_pages_prefill = compute_invoice_starts_default(
        df_pages,
        use_tampon_split=bool(use_tampon_split),
        tampon_flag_col="tampon_pred",
    )

    # Persist dans session_state pour édition
    ss_init("df_pages_editable", df_pages_prefill)

    st.subheader("✅ Étape 1 — Valider / corriger les débuts de facture (cases à cocher)")
    st.caption("Astuce: coche uniquement les pages qui démarrent une nouvelle facture. La page 1 doit rester cochée.")

    # Data editor: colonne invoice_start éditable
    editable_cols = [
        "invoice_start",
        "page",
        "has_bv",
        "has_qr_candidate",
        "has_reference_strict",
        "bv_reference",
        "ref_source",
        "bv_currency",
        "bv_amount",
    ]
    if use_yolo:
        editable_cols += ["tampon_pred", "proba_tampon", "n_detections", "max_conf"]

    df_for_edit = st.session_state["df_pages_editable"].copy()

    # Force bool for editor
    if "invoice_start" in df_for_edit.columns:
        df_for_edit["invoice_start"] = df_for_edit["invoice_start"].fillna(False).astype(bool)

    edited_df = st.data_editor(
        df_for_edit[editable_cols],
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "invoice_start": st.column_config.CheckboxColumn(
                "Début facture",
                help="Coche = cette page démarre une nouvelle facture",
                default=False,
            )
        },
        disabled=[c for c in editable_cols if c != "invoice_start"],
        key="editor_pages_v24",
    )

    # Remettre la colonne éditée dans df complet
    df_updated = st.session_state["df_pages_editable"].copy()
    df_updated["invoice_start"] = edited_df["invoice_start"].fillna(False).astype(bool).values
    # sécurité page 1
    if df_updated.shape[0] > 0 and not bool(df_updated.loc[df_updated.index[0], "invoice_start"]):
        df_updated.loc[df_updated.index[0], "invoice_start"] = True
        st.warning("La page 1 doit être un début de facture → elle a été recochée automatiquement.")

    st.session_state["df_pages_editable"] = df_updated

    st.subheader("🧾 Étape 2 — Recalculer la segmentation finale (invoice_id)")
    colA, colB = st.columns([1, 2])

    with colA:
        recalc = st.button("🔁 Recalculer segmentation", type="primary")

    if recalc:
        df_final_pages, df_final_invoices = assign_invoices_from_user_starts(
            st.session_state["df_pages_editable"],
            start_col="invoice_start",
        )
        st.session_state["df_final_pages"] = df_final_pages
        st.session_state["df_final_invoices"] = df_final_invoices
        st.success("Segmentation recalculée ✅")

    # Affichage des résultats finaux si disponibles
    if "df_final_pages" in st.session_state and "df_final_invoices" in st.session_state:
        df_pages2 = st.session_state["df_final_pages"]
        df_invoices = st.session_state["df_final_invoices"]

        st.subheader("📄 Résultats finaux par page")
        st.dataframe(df_pages2, use_container_width=True)

        st.subheader("🧾 Résumé final par facture")
        st.dataframe(df_invoices, use_container_width=True)

        # KPIs simples
        st.subheader("📊 KPI")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Factures totales", int(df_invoices.shape[0]))
        k2.metric("Factures avec BV", int((df_invoices["status"] == "HAS_BV").sum()))
        k3.metric("Factures sans BV", int((df_invoices["status"] == "NO_BV").sum()))
        k4.metric("BV validés (pages)", int(df_pages2["has_bv"].sum()))

        false_qr = df_pages2[(df_pages2["has_qr_candidate"]) & (~df_pages2["has_bv"])]
        if len(false_qr) > 0:
            st.warning(
                f"{len(false_qr)} page(s) ont un QR en bas de page mais **PAS** de référence stricte "
                f"→ ignorées comme BV."
            )

        # Exports
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "📥 Télécharger résultats finaux par page (CSV)",
                data=df_pages2.to_csv(index=False).encode("utf-8"),
                mime="text/csv",
                file_name=f"bv_v24_final_pages_{pdf_path.stem}.csv",
            )
        with c2:
            st.download_button(
                "📥 Télécharger résumé final factures (CSV)",
                data=df_invoices.to_csv(index=False).encode("utf-8"),
                mime="text/csv",
                file_name=f"bv_v24_final_invoices_{pdf_path.stem}.csv",
            )
        with c3:
            st.download_button(
                "📥 Télécharger QR ignorés (CSV)",
                data=false_qr.to_csv(index=False).encode("utf-8") if len(false_qr) > 0 else "page\n".encode("utf-8"),
                mime="text/csv",
                file_name=f"bv_v24_false_qr_{pdf_path.stem}.csv",
            )

    # Debug OCR
    with st.expander("🔍 Debug : texte OCR (zones multiples)"):
        page_num_debug = st.number_input("Page (1 à n_pages)", min_value=1, max_value=n_pages, value=1)
        if st.button("Afficher OCR (zones) de cette page"):
            txt = page_ocr_debug[page_num_debug - 1]
            st.text(txt if txt.strip() else "[aucun texte OCR / pas de QR candidate sur cette page]")

    # Debug QR payload
    with st.expander("🔍 Debug : texte brut du QR d'une page"):
        page_num_debug2 = st.number_input("Page pour le QR", min_value=1, max_value=n_pages, value=1, key="qr_debug_page")
        if st.button("Afficher le texte QR de cette page"):
            txt_qr = page_qr_texts[page_num_debug2 - 1]
            st.text(txt_qr if txt_qr else "[aucun QR décodé sur cette page]")

    # Aperçu YOLO
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
    st.info("Dépose un PDF multipages pour analyser BV + valider les débuts de facture (v24).")

