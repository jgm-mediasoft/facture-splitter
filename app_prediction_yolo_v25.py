# app_prediction_yolo_v25_com.py

from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import fitz
from ultralytics import YOLO

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    Image = None
    TESSERACT_AVAILABLE = False


def render_pdf_page_to_rgb(pdf_path: Path, page_index: int = 0, dpi: int = 300):
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)


@st.cache_resource
def load_yolo_model(model_path: str):
    return YOLO(model_path)


def detect_bottom_qr_candidate(img_rgb: np.ndarray, bottom_ratio: float = 0.55):
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


def ocr_image_rgb(img_rgb: np.ndarray, mode: str = "text") -> str:
    if not (TESSERACT_AVAILABLE and Image is not None):
        return ""
    try:
        pil_img = Image.fromarray(img_rgb)
        if mode == "amount":
            cfg = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
            return pytesseract.image_to_string(pil_img, lang="fra+deu+ita+eng", config=cfg) or ""
        return pytesseract.image_to_string(pil_img, lang="fra+deu+ita+eng") or ""
    except Exception:
        return ""


REF_PATTERN_STRICT = r"\b\d{2}(?:\s+\d{5}){5}\b"


def normalize_spaces(s: str) -> str:
    return (
        (s or "")
        .replace("\xa0", " ")
        .replace("\u202f", " ")
        .replace("\u00a0", " ")
    )


def extract_reference_strict_from_text(text: str):
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


def extract_reference_from_qr_payload_qrr(txt: str):
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


def parse_amount_extended(raw: str):
    if raw is None:
        return None

    s = str(raw).strip()
    if not s:
        return None

    s = s.replace("’", "'").replace(" ", "")
    s = re.sub(r"[^0-9,.\-']", "", s)

    s = s.replace(".-", "")
    s = s.replace(",-", "")

    s = s.replace("'", "")

    if not s:
        return None

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        s = s.replace(",", ".")

    if s in ("-", ".", "-.", ".-"):
        return None

    try:
        v = float(s)
        if v < 0:
            return None
        return v
    except Exception:
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
            amount = parse_amount_extended(ln)
            if amount is not None:
                break
    return currency, amount


def crop_reference_regions(img_rgb: np.ndarray, qr_points: np.ndarray, bottom_ratio: float = 0.55):
    h, w, _ = img_rgb.shape
    xs = qr_points[:, 0]
    ys = qr_points[:, 1]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    margin_x = int(0.02 * w)
    margin_y = int(0.02 * h)

    crops = []

    left = min(w - 1, x_max + margin_x)
    right = w
    top = max(0, y_min - margin_y)
    bottom = min(h, y_max + margin_y)
    if left < right and top < bottom:
        crops.append(img_rgb[top:bottom, left:right])

    y_start = int(h * bottom_ratio)
    top2 = max(0, y_start)
    bottom2 = h
    left2 = min(w - 1, x_max + margin_x)
    right2 = w
    if left2 < right2 and top2 < bottom2:
        crops.append(img_rgb[top2:bottom2, left2:right2])

    left3 = 0
    right3 = max(1, x_min - margin_x)
    top3 = max(0, y_start)
    bottom3 = h
    if left3 < right3 and top3 < bottom3:
        crops.append(img_rgb[top3:bottom3, left3:right3])

    return crops


def crop_amount_regions_invoice(img_rgb: np.ndarray):
    h, w, _ = img_rgb.shape
    crops = []

    y0 = int(h * 0.65)
    crops.append(img_rgb[y0:h, 0:w])

    y1 = int(h * 0.00)
    y2 = int(h * 0.45)
    x1 = int(w * 0.50)
    crops.append(img_rgb[y1:y2, x1:w])

    y3 = int(h * 0.50)
    x2 = int(w * 0.50)
    crops.append(img_rgb[y3:h, x2:w])

    return [c for c in crops if c.size > 0]


_AMOUNT_RX = re.compile(r"(\d{1,3}(?:[ '\u2019]\d{3})*(?:[.,]\d{2})|\d{1,9}(?:[.,]\d{2})|\d{1,9}(?:\.\-|\,-)?)")
_CCY_RX = re.compile(r"\b(CHF|EUR)\b", flags=re.IGNORECASE)


def extract_total_amount_from_ocr_text(text: str):
    if not text:
        return None, None

    t = normalize_spaces(text)
    lines = [l.strip() for l in t.splitlines() if l.strip()]

    candidates = []

    def add_candidate(amount, currency, score):
        if amount is None:
            return
        candidates.append((score, amount, currency))

    for idx, line in enumerate(lines):
        low = line.lower()

        curr = None
        mccy = _CCY_RX.search(line)
        if mccy:
            curr = mccy.group(1).upper()

        amts = _AMOUNT_RX.findall(line)
        amt_line = None
        if amts:
            amt_line = parse_amount_extended(amts[-1])

        score = 0
        if "total" in low:
            score += 6
        if "ttc" in low:
            score += 3
        if curr is not None:
            score += 2
        if low.startswith("total"):
            score += 1

        if score > 0 and amt_line is not None:
            add_candidate(amt_line, curr, score)

        if score >= 6 and amt_line is None and idx + 1 < len(lines):
            nxt = lines[idx + 1]
            amts2 = _AMOUNT_RX.findall(nxt)
            if amts2:
                amt2 = parse_amount_extended(amts2[-1])
                curr2 = None
                mccy2 = _CCY_RX.search(nxt)
                if mccy2:
                    curr2 = mccy2.group(1).upper()
                add_candidate(amt2, curr2 or curr, score - 1)

        if ("chf" in low or "eur" in low) and amt_line is not None and score == 0:
            add_candidate(amt_line, curr, 2)

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = candidates[0]
        return best[1], best[2]

    amounts = []
    for line in lines:
        amts = _AMOUNT_RX.findall(line)
        for a in amts:
            v = parse_amount_extended(a)
            if v is not None:
                amounts.append(v)

    if not amounts:
        return None, None

    amounts = [v for v in amounts if v >= 1.0]
    if not amounts:
        return None, None

    return max(amounts), None


def extract_invoice_total_from_page(img_rgb: np.ndarray):
    if not (TESSERACT_AVAILABLE and Image is not None):
        return None, None

    best_amt = None
    best_ccy = None
    best_score = -1

    for crop in crop_amount_regions_invoice(img_rgb):
        txt = ocr_image_rgb(crop, mode="amount")
        amt, ccy = extract_total_amount_from_ocr_text(txt)
        if amt is None:
            continue

        score = 1
        if ccy in ("CHF", "EUR"):
            score += 1
        if amt >= 100:
            score += 1
        if amt >= 1000:
            score += 1

        if score > best_score:
            best_score = score
            best_amt = amt
            best_ccy = ccy

    return best_amt, best_ccy


def compute_invoice_starts_default(df_pages: pd.DataFrame, use_tampon_split: bool, tampon_flag_col: str = "tampon_pred"):
    df = df_pages.copy()
    n = df.shape[0]
    starts = [False] * n
    if n == 0:
        df["invoice_start"] = starts
        return df

    starts[0] = True

    for i in range(1, n):
        if bool(df.loc[i - 1, "has_bv"]):
            starts[i] = True

    if use_tampon_split and (tampon_flag_col in df.columns):
        for i in range(1, n):
            if bool(df.loc[i, tampon_flag_col]):
                starts[i] = True

    df["invoice_start"] = starts
    return df


def assign_invoices_from_user_starts(df_pages: pd.DataFrame, start_col: str = "invoice_start"):
    df = df_pages.copy()
    starts = df[start_col].fillna(False).astype(bool).tolist()

    if len(starts) > 0 and not starts[0]:
        starts[0] = True
        df.loc[df.index[0], start_col] = True

    invoice_ids = []
    cur = 0
    for s in starts:
        if s:
            cur += 1
        invoice_ids.append(cur)

    df["invoice_id"] = invoice_ids

    def last_non_null(series):
        s2 = series.dropna()
        return s2.iloc[-1] if len(s2) > 0 else None

    invoices = []
    for inv_id, g in df.groupby("invoice_id", sort=True):
        pages = g["page"].tolist()
        has_bv = bool(g["has_bv"].any())
        bv_pages = g.loc[g["has_bv"], "page"].tolist()
        last_bv_page = int(max(bv_pages)) if bv_pages else None

        amt_last = last_non_null(g.loc[g["has_bv"], "bv_amount"]) if has_bv else None
        amt_last = float(amt_last) if amt_last is not None else None

        invoices.append(
            {
                "invoice_id": int(inv_id),
                "pages_count": int(len(pages)),
                "page_start": int(min(pages)),
                "page_end": int(max(pages)),
                "has_bv": has_bv,
                "bv_last_page": last_bv_page,
                "bv_reference": last_non_null(g.loc[g["has_bv"], "bv_reference"]) if has_bv else None,
                "bv_amount": amt_last,
                "bv_currency": last_non_null(g.loc[g["has_bv"], "bv_currency"]) if has_bv else None,
                "status": "HAS_BV" if has_bv else "NO_BV",
                "invoice_amount": None,
                "invoice_currency": None,
                "amount_source": None,
            }
        )

    return df, pd.DataFrame(invoices)


def apply_theme(mode: str):
    if mode == "Sombre":
        st.markdown(
            """
            <style>
            .stApp { background: #0e1117; color: #e6e6e6; }
            [data-testid="stSidebar"] { background: #0b0f14; }
            .stDataFrame, .stTable { background: #0e1117; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp { background: #ffffff; color: #111111; }
            [data-testid="stSidebar"] { background: #f7f7f9; }
            </style>
            """,
            unsafe_allow_html=True,
        )


def reset_scan():
    for k in [
        "df_pages_editable",
        "df_final_pages",
        "df_final_invoices",
        "last_pdf_name",
    ]:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    st.rerun()


def compute_invoice_amounts(df_invoices: pd.DataFrame, df_pages2: pd.DataFrame, pdf_path: Path, dpi: int):
    inv = df_invoices.copy()

    amounts = []
    currencies = []
    sources = []

    for _, row in inv.iterrows():
        if bool(row.get("has_bv")) and row.get("bv_amount") is not None:
            amounts.append(float(row.get("bv_amount")))
            currencies.append(row.get("bv_currency") or None)
            sources.append("BV_QR")
            continue

        page_start = int(row["page_start"])
        page_end = int(row["page_end"])
        pages = list(range(page_start, page_end + 1))

        best_amt = None
        best_ccy = None

        for p in pages[::-1]:
            try:
                img_rgb = render_pdf_page_to_rgb(pdf_path, page_index=p - 1, dpi=dpi)
            except Exception:
                continue

            amt, ccy = extract_invoice_total_from_page(img_rgb)
            if amt is not None:
                best_amt, best_ccy = amt, ccy
                break

        amounts.append(float(best_amt) if best_amt is not None else None)
        currencies.append(best_ccy)
        sources.append("FACT_OCR" if best_amt is not None else None)

    inv["invoice_amount"] = amounts
    inv["invoice_currency"] = currencies
    inv["amount_source"] = sources
    return inv


st.set_page_config(page_title="Scan facture PDF", layout="wide")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

with st.sidebar:
    theme_mode = st.radio("Mode", ["Clair", "Sombre"], horizontal=True)
    apply_theme(theme_mode)

    st.markdown("### Paramètres")
    dpi = st.select_slider("DPI", options=[150, 200, 250, 300, 350, 400], value=400)

    qr_bottom_ratio = st.slider("QR bas (ratio)", 0.35, 0.80, 0.45, 0.01)
    ocr_bottom_ratio = st.slider("OCR BV (ratio)", 0.35, 0.80, 0.45, 0.01)

    st.divider()
    st.markdown("### Tampon (optionnel)")
    MODEL_DEFAULT = "runs_tampon/yolov8s_tampon/weights/best.pt"
    use_yolo = st.checkbox("Activer YOLO", value=False)
    model_path_text = st.text_input("Modèle .pt", value=str(Path(MODEL_DEFAULT).resolve()))
    use_tampon_prefill = st.checkbox("Pré-cocher début si tampon", value=False)
    conf_thres = st.slider("Seuil conf", 0.10, 1.00, 0.94, 0.01)
    iou_thres = st.slider("IoU", 0.10, 0.90, 0.45, 0.05)

    st.divider()
    st.markdown("### Montant (si pas de BV)")
    read_amount_no_bv = st.checkbox("Lire montant sur facture", value=True, disabled=not TESSERACT_AVAILABLE)

    st.divider()
    if st.button("🧹 Réinitialiser le scan", use_container_width=True):
        reset_scan()

st.title("Scan facture PDF")
st.caption("v25")

status_parts = [
    f"QR OpenCV: {'OK' if HAS_CV2 else 'OFF'}",
    f"OCR: {'OK' if TESSERACT_AVAILABLE else 'OFF'}",
]
st.caption(" • ".join(status_parts))

uploaded_pdf = st.file_uploader(
    "Dépose un PDF",
    type=["pdf"],
    key=f"uploader_{st.session_state['uploader_key']}",
)

if not uploaded_pdf:
    st.info("Importe un PDF pour lancer l’analyse.")
    st.stop()

tmp_dir = Path("data/tmp_bv_v25_com")
tmp_dir.mkdir(parents=True, exist_ok=True)
pdf_path = tmp_dir / uploaded_pdf.name
pdf_path.write_bytes(uploaded_pdf.getbuffer())
st.session_state["last_pdf_name"] = uploaded_pdf.name

if use_tampon_prefill and not use_yolo:
    use_tampon_prefill = False

model = None
if use_yolo:
    try:
        model = load_yolo_model(model_path_text)
    except Exception:
        model = None
        use_yolo = False
        use_tampon_prefill = False

try:
    with fitz.open(str(pdf_path)) as doc:
        n_pages = len(doc)
except Exception as e:
    st.error(f"PDF invalide: {e}")
    st.stop()

progress = st.progress(0, text="Analyse…")
rows = []

for i in range(n_pages):
    img_rgb = render_pdf_page_to_rgb(pdf_path, page_index=i, dpi=dpi)

    tampon_pred = None
    proba_tampon = None
    if use_yolo and model is not None:
        results = model(img_rgb, conf=conf_thres, iou=iou_thres, verbose=False)
        r = results[0]
        boxes = r.boxes
        if len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            max_conf = float(confs.max())
            tampon_pred = 1
            proba_tampon = max_conf
        else:
            tampon_pred = 0
            proba_tampon = 0.0

    qr_text, qr_pts, _ = detect_bottom_qr_candidate(img_rgb, bottom_ratio=qr_bottom_ratio)
    has_qr_candidate = qr_pts is not None

    bv_reference = None
    ref_source = None

    if has_qr_candidate:
        for crop_img in crop_reference_regions(img_rgb, qr_pts, bottom_ratio=ocr_bottom_ratio):
            txt = ocr_image_rgb(crop_img)
            ref = extract_reference_strict_from_text(txt)
            if ref and is_reference_strict(ref):
                bv_reference = ref
                ref_source = "OCR"
                break

        if not bv_reference and qr_text:
            ref_payload = extract_reference_from_qr_payload_qrr(qr_text)
            if ref_payload and is_reference_strict(ref_payload):
                bv_reference = ref_payload
                ref_source = "QR_QRR"

    bv_currency, bv_amount = (None, None)
    if qr_text:
        bv_currency, bv_amount = extract_currency_amount_from_qr_text(qr_text)

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
    }

    if use_yolo:
        row.update(
            {
                "tampon_pred": int(tampon_pred) if tampon_pred is not None else None,
                "proba_tampon": round(float(proba_tampon), 4) if proba_tampon is not None else None,
            }
        )

    rows.append(row)
    progress.progress((i + 1) / n_pages, text=f"Analyse… {i+1}/{n_pages}")

df_pages = pd.DataFrame(rows)
df_pages = compute_invoice_starts_default(df_pages, use_tampon_split=bool(use_tampon_prefill), tampon_flag_col="tampon_pred")

if "df_pages_editable" not in st.session_state or st.session_state.get("last_pdf_name") != uploaded_pdf.name:
    st.session_state["df_pages_editable"] = df_pages.copy()

st.subheader("Structure")
cols = ["invoice_start", "page", "has_bv", "has_qr_candidate", "has_reference_strict", "bv_reference", "ref_source", "bv_currency", "bv_amount"]
if use_yolo:
    cols += ["tampon_pred", "proba_tampon"]

df_edit = st.session_state["df_pages_editable"].copy()
df_edit["invoice_start"] = df_edit["invoice_start"].fillna(False).astype(bool)

edited_df = st.data_editor(
    df_edit[cols],
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    column_config={
        "invoice_start": st.column_config.CheckboxColumn("Début", default=False),
        "bv_amount": st.column_config.NumberColumn("Montant BV", format="%.2f"),
    },
    disabled=[c for c in cols if c != "invoice_start"],
    key="editor_pages_v25_com",
)

df_updated = st.session_state["df_pages_editable"].copy()
df_updated["invoice_start"] = edited_df["invoice_start"].fillna(False).astype(bool).values
if df_updated.shape[0] > 0 and not bool(df_updated.loc[df_updated.index[0], "invoice_start"]):
    df_updated.loc[df_updated.index[0], "invoice_start"] = True
st.session_state["df_pages_editable"] = df_updated

c1, c2 = st.columns([1, 2])
with c1:
    recalc = st.button("Valider la segmentation", type="primary", use_container_width=True)

if recalc:
    df_final_pages, df_final_invoices = assign_invoices_from_user_starts(st.session_state["df_pages_editable"], start_col="invoice_start")

    if read_amount_no_bv and TESSERACT_AVAILABLE and df_final_invoices.shape[0] > 0:
        df_final_invoices = compute_invoice_amounts(df_final_invoices, df_final_pages, pdf_path, dpi=dpi)

    st.session_state["df_final_pages"] = df_final_pages
    st.session_state["df_final_invoices"] = df_final_invoices

if "df_final_pages" in st.session_state and "df_final_invoices" in st.session_state:
    df_pages2 = st.session_state["df_final_pages"]
    df_invoices = st.session_state["df_final_invoices"]

    st.subheader("Indicateurs")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Factures", int(df_invoices.shape[0]))
    k2.metric("Factures avec BV", int((df_invoices["status"] == "HAS_BV").sum()))
    k3.metric("Factures sans BV", int((df_invoices["status"] == "NO_BV").sum()))
    k4.metric("BV (pages)", int(df_pages2["has_bv"].sum()))

    st.subheader("Résultats par facture")
    st.dataframe(df_invoices, use_container_width=True)

    st.subheader("Résultats par page")
    st.dataframe(df_pages2, use_container_width=True)

    false_qr = df_pages2[(df_pages2["has_qr_candidate"]) & (~df_pages2["has_bv"])]

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Télécharger pages (CSV)",
            data=df_pages2.to_csv(index=False).encode("utf-8"),
            mime="text/csv",
            file_name=f"scan_v25_pages_{pdf_path.stem}.csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Télécharger factures (CSV)",
            data=df_invoices.to_csv(index=False).encode("utf-8"),
            mime="text/csv",
            file_name=f"scan_v25_factures_{pdf_path.stem}.csv",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            "Télécharger QR ignorés (CSV)",
            data=false_qr.to_csv(index=False).encode("utf-8") if len(false_qr) > 0 else "page\n".encode("utf-8"),
            mime="text/csv",
            file_name=f"scan_v25_qr_ignores_{pdf_path.stem}.csv",
            use_container_width=True,
        )
