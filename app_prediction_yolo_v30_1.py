# app_prediction_yolo_v30_1_com.py
# v30_1 = v30 + accélération + qualité (moins d'erreurs) SANS changer tes paramètres UI
# Améliorations v30_1:
# 1) Rendu "clip" PyMuPDF pour OCR (BV/date/montant) => beaucoup plus rapide que page entière
# 2) "Text-first" PDF: tentative extraction depuis texte natif avant OCR (gratuit si PDF texte)
# 3) BV: payload QRR prioritaire, OCR fallback sur clip bas de page (pas page entière)
# 4) Montant: OCR d'abord sur zone bas-droite (clip) puis fallback bas complet + stratégie 2 pages
# 5) Confiance: image_to_data seulement si candidat montant détecté
# 6) Correction bug scroll/jump: Structure via st.form (appliquer/valider)

from pathlib import Path
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import fitz
from ultralytics import YOLO

try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    Image = None
    TESSERACT_AVAILABLE = False


# =========================
# Cache + Render (full & clip)
# =========================
class RenderCache:
    def __init__(self, doc: fitz.Document):
        self.doc = doc
        self._full: Dict[Tuple[int, int], np.ndarray] = {}
        self._clip: Dict[Tuple[int, int, int, int, int, int, int, int], np.ndarray] = {}

    def get_full_rgb(self, page_index: int, dpi: int) -> np.ndarray:
        key = (page_index, int(dpi))
        if key in self._full:
            return self._full[key]
        page = self.doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        self._full[key] = img
        return img

    def get_clip_rgb_ratio(self, page_index: int, dpi: int, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        """
        clip en coordonnées relatives (0..1) basé sur la page PDF
        """
        page = self.doc.load_page(page_index)
        rect = page.rect  # coordonnées PDF
        rx0 = rect.x0 + (rect.width * float(x0))
        ry0 = rect.y0 + (rect.height * float(y0))
        rx1 = rect.x0 + (rect.width * float(x1))
        ry1 = rect.y0 + (rect.height * float(y1))
        clip = fitz.Rect(rx0, ry0, rx1, ry1)

        # clé cache discrétisée
        k = (
            page_index, int(dpi),
            int(x0 * 10000), int(y0 * 10000), int(x1 * 10000), int(y1 * 10000),
            int(rect.width), int(rect.height),
        )
        if k in self._clip:
            return self._clip[k]

        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        self._clip[k] = img
        return img

    def get_pdf_text(self, page_index: int) -> str:
        try:
            page = self.doc.load_page(page_index)
            return page.get_text("text") or ""
        except Exception:
            return ""


# =========================
# YOLO
# =========================
@st.cache_resource
def load_yolo_model(model_path: str):
    return YOLO(model_path)


# =========================
# QR detect (ROI bottom) - needs full image array
# =========================
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


# =========================
# OCR helpers
# =========================
def ocr_image_rgb(img_rgb: np.ndarray, mode: str = "text") -> str:
    if not (TESSERACT_AVAILABLE and Image is not None):
        return ""
    try:
        pil_img = Image.fromarray(img_rgb)
        cfg = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
        if mode in ("amount", "date"):
            return pytesseract.image_to_string(pil_img, lang="fra+deu+ita+eng", config=cfg) or ""
        return pytesseract.image_to_string(pil_img, lang="fra+deu+ita+eng") or ""
    except Exception:
        return ""


def ocr_image_rgb_conf_if_useful(img_rgb: np.ndarray, mode: str = "text", require_amount_candidate: bool = False):
    if not (TESSERACT_AVAILABLE and Image is not None):
        return "", None
    try:
        pil_img = Image.fromarray(img_rgb)
        cfg = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
        lang = "fra+deu+ita+eng"

        if mode in ("amount", "date"):
            txt = pytesseract.image_to_string(pil_img, lang=lang, config=cfg) or ""
        else:
            txt = pytesseract.image_to_string(pil_img, lang=lang) or ""

        if require_amount_candidate and not _AMOUNT_RX.search(normalize_spaces(txt or "")):
            return txt, None

        if mode in ("amount", "date"):
            df = pytesseract.image_to_data(
                pil_img, lang=lang, config=cfg, output_type=pytesseract.Output.DATAFRAME
            )
        else:
            df = pytesseract.image_to_data(
                pil_img, lang=lang, output_type=pytesseract.Output.DATAFRAME
            )

        conf = None
        if df is not None and "conf" in df.columns:
            c = pd.to_numeric(df["conf"], errors="coerce")
            c = c[(c.notna()) & (c >= 0)]
            if len(c) > 0:
                conf = float(c.mean())

        return txt, conf
    except Exception:
        return "", None


# =========================
# Parsing BV / Amount / Date
# =========================
REF_PATTERN_STRICT = r"\b\d{2}(?:\s+\d{5}){5}\b"

_AMOUNT_RX = re.compile(
    r"(\d{1,3}(?:[ '\u2019]\d{3})*(?:[.,]\d{2})|\d{1,9}(?:[.,]\d{2})|\d{1,9}(?:\.\-|\,-|\-))"
)
_CCY_RX = re.compile(r"\b(CHF|EUR)\b", flags=re.IGNORECASE)
_CCY_AMOUNT_RX = re.compile(r"\b(CHF|EUR)\b[\s:]*([0-9][0-9 '\u2019.,\-]{0,20})", flags=re.IGNORECASE)

_DATE_NUM_RXES = [
    re.compile(r"\b(0?[1-9]|[12]\d|3[01])[./-](0?[1-9]|1[0-2])[./-]((?:19|20)\d{2})\b"),
    re.compile(r"\b((?:19|20)\d{2})[./-](0?[1-9]|1[0-2])[./-](0?[1-9]|[12]\d|3[01])\b"),
]

_MONTHS = {
    "jan": 1, "janv": 1, "janvier": 1,
    "fev": 2, "fevr": 2, "févr": 2, "fevrier": 2, "février": 2,
    "mar": 3, "mars": 3,
    "avr": 4, "avril": 4,
    "mai": 5,
    "jun": 6, "juin": 6,
    "jul": 7, "juil": 7, "juillet": 7,
    "aou": 8, "août": 8, "aout": 8,
    "sep": 9, "sept": 9, "septembre": 9,
    "oct": 10, "octobre": 10,
    "nov": 11, "novembre": 11,
    "dec": 12, "déc": 12, "decembre": 12, "décembre": 12,
}

_DATE_WORD_RX = re.compile(
    r"\b(0?[1-9]|[12]\d|3[01])\s+([A-Za-zéèêëàâäîïôöùûüç\.]{3,12})\s+((?:19|20)\d{2})\b",
    flags=re.IGNORECASE,
)


def normalize_spaces(s: str) -> str:
    return (s or "").replace("\xa0", " ").replace("\u202f", " ").replace("\u00a0", " ")


def is_reference_strict(ref):
    if not ref:
        return False
    ref_norm = re.sub(r"\s+", " ", str(ref)).strip()
    return re.fullmatch(REF_PATTERN_STRICT, ref_norm) is not None


def extract_reference_strict_from_text(text: str):
    if not text or not text.strip():
        return None
    zone = re.sub(r"\s+", " ", normalize_spaces(text))
    m = re.search(REF_PATTERN_STRICT, zone)
    return m.group(0).strip() if m else None


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

    s = s.replace("\u2014", "-").replace("\u2013", "-").replace("—", "-").replace("–", "-")
    s = s.replace("’", "'").replace(" ", "")
    s = re.sub(r"[^0-9,.\-']", "", s)

    s = s.replace(".-", "").replace(",-", "").rstrip("-")
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


# =========================
# Amount/date scoring
# =========================
def extract_total_amount_from_text(text: str):
    if not text:
        return None, None

    t = normalize_spaces(text)
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    candidates = []

    def add(amount, currency, score):
        if amount is None:
            return
        candidates.append((score, amount, currency))

    for line in lines:
        low = line.lower()
        base_penalty = -2 if ("tva" in low and "total" not in low) else 0
        base_penalty += -5 if ("acompte" in low or "avance" in low) else 0

        keywords_boost = 0
        if "total" in low:
            keywords_boost += 6
        if "ttc" in low:
            keywords_boost += 3
        if "net" in low and ("payer" in low or "à payer" in low or "a payer" in low):
            keywords_boost += 6
        if "à payer" in low or "a payer" in low or "montant dû" in low or "montant du" in low:
            keywords_boost += 6

        m_ca = _CCY_AMOUNT_RX.search(line)
        if m_ca:
            curr = m_ca.group(1).upper()
            amt = parse_amount_extended(m_ca.group(2))
            score = 3 + base_penalty + keywords_boost
            add(amt, curr, score)

        curr2 = None
        mccy = _CCY_RX.search(line)
        if mccy:
            curr2 = mccy.group(1).upper()

        amts = _AMOUNT_RX.findall(line)
        amt_line = parse_amount_extended(amts[-1]) if amts else None

        score2 = base_penalty + keywords_boost
        if curr2 is not None:
            score2 += 2
        if low.startswith("total"):
            score2 += 2

        if score2 > 0 and amt_line is not None:
            add(amt_line, curr2, score2)

        if curr2 is not None and amt_line is not None and score2 <= 0:
            add(amt_line, curr2, 2 + base_penalty)

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _, amt, ccy = candidates[0]
        return amt, ccy

    amounts = []
    for line in lines:
        for a in _AMOUNT_RX.findall(line):
            v = parse_amount_extended(a)
            if v is not None:
                amounts.append(v)
    amounts = [v for v in amounts if v >= 1.0]
    return (max(amounts), None) if amounts else (None, None)


def _fmt_date_iso(d: int, m: int, y: int):
    return f"{y:04d}-{m:02d}-{d:02d}"


def extract_invoice_date_from_text(text: str):
    if not text:
        return None

    t = normalize_spaces(text)
    lines = [l.strip() for l in t.splitlines() if l.strip()]

    best = None
    best_score = -10

    for line in lines:
        low = line.lower()

        penalty = -8 if ("échéance" in low or "echeance" in low or "due" in low) else 0
        penalty += -6 if ("livraison" in low or "delivery" in low) else 0

        boost = 0
        if "date facture" in low or ("date" in low and "facture" in low):
            boost += 10
        elif "facture" in low:
            boost += 4
        elif "date" in low:
            boost += 2
        if " le " in f" {low} ":
            boost += 1

        for rx in _DATE_NUM_RXES:
            m = rx.search(line)
            if not m:
                continue
            if rx is _DATE_NUM_RXES[0]:
                d = int(m.group(1))
                mm = int(m.group(2))
                y = int(m.group(3))
            else:
                y = int(m.group(1))
                mm = int(m.group(2))
                d = int(m.group(3))

            if not (1 <= mm <= 12 and 1 <= d <= 31 and 1900 <= y <= 2099):
                continue

            cand = _fmt_date_iso(d, mm, y)
            score = 1 + boost + penalty
            if score > best_score:
                best_score = score
                best = cand

        mw = _DATE_WORD_RX.search(line)
        if mw:
            d = int(mw.group(1))
            mon_raw = mw.group(2).lower().strip(".")
            y = int(mw.group(3))
            mon_key = re.sub(r"[^a-zàâäçéèêëîïôöùûü]", "", mon_raw)
            mm = _MONTHS.get(mon_key, _MONTHS.get(mon_key[:4], _MONTHS.get(mon_key[:3], None)))
            if mm and (1 <= d <= 31) and (1900 <= y <= 2099):
                cand = _fmt_date_iso(d, mm, y)
                score = 2 + boost + penalty + 2
                if score > best_score:
                    best_score = score
                    best = cand

    return best


# =========================
# Structure helpers
# =========================
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
                "amount_confidence": None,
                "invoice_date": None,
                "date_source": None,
            }
        )

    return df, pd.DataFrame(invoices)


# =========================
# Theme / reset
# =========================
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
        "df_final_invoices_editable",
        "df_transfer_erp",
        "last_pdf_signature",
    ]:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    st.rerun()


# =========================
# Formatting / ERP table
# =========================
def _fmt_amount_2dec(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        return f"{float(x):.2f}"
    except Exception:
        return ""


def _fmt_conf_4dec(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        return f"{float(x):.4f}"
    except Exception:
        return ""


def style_invoice_amount_by_conf(df: pd.DataFrame, threshold: float = 0.7):
    cols = list(df.columns)
    if "invoice_amount" not in cols or "amount_confidence" not in cols:
        return df.style

    def _row_styles(row: pd.Series):
        conf = row.get("amount_confidence", None)
        if conf is None or (isinstance(conf, float) and np.isnan(conf)) or conf == "":
            return pd.Series({c: "" for c in cols})

        try:
            conf_val = float(conf)
        except Exception:
            return pd.Series({c: "" for c in cols})

        color = "green" if conf_val >= float(threshold) else "red"
        return pd.Series({c: (f"color:{color}; font-weight:700;" if c == "invoice_amount" else "") for c in cols})

    return df.style.apply(_row_styles, axis=1)


def _parse_date_iso_or_none(s: str):
    if s is None:
        return None
    txt = str(s).strip()
    if not txt:
        return None
    try:
        dt = datetime.strptime(txt, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def build_transfer_erp_table(df_invoices_num: pd.DataFrame):
    tz = ZoneInfo("Europe/Zurich")
    now = datetime.now(tz)
    date_jour = now.strftime("%Y-%m-%d")
    heure = now.strftime("%H:%M:%S")

    cols = [
        "invoice_id",
        "pages_count",
        "has_bv",
        "bv_reference",
        "invoice_amount",
        "invoice_currency",
        "invoice_date",
    ]
    base = df_invoices_num.copy()
    for c in cols:
        if c not in base.columns:
            base[c] = None
    out = base[cols].copy()
    out.insert(0, "Heure", heure)
    out.insert(0, "Date du jour", date_jour)
    out["has_bv"] = out["has_bv"].fillna(False).astype(bool)
    out["invoice_id"] = pd.to_numeric(out["invoice_id"], errors="coerce").astype("Int64")
    out["pages_count"] = pd.to_numeric(out["pages_count"], errors="coerce").astype("Int64")
    out["invoice_amount"] = pd.to_numeric(out["invoice_amount"], errors="coerce")
    out["invoice_date"] = out["invoice_date"].apply(_parse_date_iso_or_none)
    return out


# =========================
# Invoice meta extraction (clip + text-first)
# =========================
def extract_bv_reference_strict(
    cache: RenderCache,
    page_index: int,
    dpi: int,
    qr_text: Optional[str],
    qr_pts_detect: Optional[np.ndarray],
    dpi_detect: int,
    ocr_bottom_ratio: float,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Retourne (bv_reference, ref_source)
    """
    # 1) payload QRR
    if qr_text:
        ref_payload = extract_reference_from_qr_payload_qrr(qr_text)
        if ref_payload and is_reference_strict(ref_payload):
            return ref_payload, "QR_QRR"

    # 2) OCR fallback: clip bas de page (55%..100%)
    # On ne recalcule pas QR en 400; on OCR une zone bas-droite/bas.
    # Qualité conservée car DPI 400 sur clip.
    img_clip = cache.get_clip_rgb_ratio(page_index, dpi, x0=0.0, y0=max(0.0, float(ocr_bottom_ratio)), x1=1.0, y1=1.0)
    txt = ocr_image_rgb(img_clip)
    ref = extract_reference_strict_from_text(txt)
    if ref and is_reference_strict(ref):
        return ref, "OCR"

    return None, None


def extract_invoice_date_clip_textfirst(cache: RenderCache, page_index: int, dpi: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Date: text-first sur page entière, puis OCR clip haut 1/3
    """
    # text-first (rapide si PDF texte)
    t = cache.get_pdf_text(page_index)
    d = extract_invoice_date_from_text(t)
    if d:
        return d, "PDF_TEXT"

    # OCR clip haut 1/3
    img_top = cache.get_clip_rgb_ratio(page_index, dpi, x0=0.0, y0=0.0, x1=1.0, y1=1.0/3.0)
    txt = ocr_image_rgb(img_top, mode="date")
    d2 = extract_invoice_date_from_text(txt)
    return (d2, "FACT_OCR_TOP_1_3") if d2 else (None, None)


def extract_invoice_amount_clip_textfirst(cache: RenderCache, page_index: int, dpi: int) -> Tuple[Optional[float], Optional[str], Optional[float], Optional[str]]:
    """
    Montant: text-first sur page entière, puis OCR clip bas-droite, puis bas complet
    Retourne: (amount, currency, conf(0..1), source)
    """
    # text-first
    t = cache.get_pdf_text(page_index)
    a, c = extract_total_amount_from_text(t)
    if a is not None:
        return float(a), c, 1.0, "PDF_TEXT"

    # OCR bas-droite (zone très rentable)
    img_br = cache.get_clip_rgb_ratio(page_index, dpi, x0=0.45, y0=0.55, x1=1.0, y1=1.0)
    txt, conf = ocr_image_rgb_conf_if_useful(img_br, mode="amount", require_amount_candidate=True)
    amt, ccy = extract_total_amount_from_text(txt)
    if amt is not None:
        conf01 = (float(conf) / 100.0) if conf is not None else None
        return float(amt), ccy, conf01, "FACT_OCR_BR"

    # OCR bas complet (fallback)
    img_bottom = cache.get_clip_rgb_ratio(page_index, dpi, x0=0.0, y0=0.55, x1=1.0, y1=1.0)
    txt2, conf2 = ocr_image_rgb_conf_if_useful(img_bottom, mode="amount", require_amount_candidate=True)
    amt2, ccy2 = extract_total_amount_from_text(txt2)
    if amt2 is not None:
        conf01 = (float(conf2) / 100.0) if conf2 is not None else None
        return float(amt2), ccy2, conf01, "FACT_OCR_BOTTOM"

    return None, None, None, None


def compute_invoice_meta_v30_1(
    df_invoices: pd.DataFrame,
    cache: RenderCache,
    dpi: int,
    read_amount: bool,
    read_date: bool,
) -> pd.DataFrame:
    inv = df_invoices.copy()

    amounts, currencies, amount_sources, amount_confs = [], [], [], []
    dates, date_sources = [], []

    for _, row in inv.iterrows():
        page_start = int(row["page_start"])
        page_end = int(row["page_end"])
        has_bv = bool(row.get("has_bv"))
        bv_amt = row.get("bv_amount", None)

        inv_amt = None
        inv_ccy = None
        inv_amt_src = None
        inv_amt_conf = None

        if has_bv and bv_amt is not None:
            inv_amt = float(bv_amt)
            inv_ccy = row.get("bv_currency") or None
            inv_amt_src = "BV_QR"
            inv_amt_conf = 1.0

        elif read_amount and TESSERACT_AVAILABLE:
            # stratégie 2 pages: last puis last-1, puis fallback complet
            candidates = [page_end]
            if page_end - 1 >= page_start:
                candidates.append(page_end - 1)

            found = False
            for p in candidates:
                amt, ccy, conf01, src = extract_invoice_amount_clip_textfirst(cache, page_index=p - 1, dpi=dpi)
                if amt is not None:
                    inv_amt = float(amt)
                    inv_ccy = ccy
                    inv_amt_conf = conf01
                    inv_amt_src = src if src else ("FACT_OCR_NO_BV" if not has_bv else "FACT_OCR_BV_NO_AMOUNT")
                    found = True
                    break

            if not found:
                for p in range(page_end, page_start - 1, -1):
                    if p in candidates:
                        continue
                    amt, ccy, conf01, src = extract_invoice_amount_clip_textfirst(cache, page_index=p - 1, dpi=dpi)
                    if amt is not None:
                        inv_amt = float(amt)
                        inv_ccy = ccy
                        inv_amt_conf = conf01
                        inv_amt_src = src if src else ("FACT_OCR_NO_BV" if not has_bv else "FACT_OCR_BV_NO_AMOUNT")
                        break

        inv_date = None
        inv_date_src = None
        if read_date and TESSERACT_AVAILABLE:
            d, src = extract_invoice_date_clip_textfirst(cache, page_index=page_start - 1, dpi=dpi)
            inv_date = d
            inv_date_src = src

        amounts.append(inv_amt)
        currencies.append(inv_ccy)
        amount_sources.append(inv_amt_src)
        amount_confs.append(inv_amt_conf)
        dates.append(inv_date)
        date_sources.append(inv_date_src)

    inv["invoice_amount"] = amounts
    inv["invoice_currency"] = currencies
    inv["amount_source"] = amount_sources
    inv["amount_confidence"] = amount_confs
    inv["invoice_date"] = dates
    inv["date_source"] = date_sources
    return inv


# ==========================================================
# UI
# ==========================================================
st.set_page_config(page_title="Scan facture PDF", layout="wide")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

with st.sidebar:
    theme_mode = st.radio("Mode", ["Clair", "Sombre"], horizontal=True)
    apply_theme(theme_mode)

    st.markdown("### Paramètres (scan rapide)")
    dpi_fast = st.select_slider("DPI rapide", options=[120, 150, 180, 200, 250], value=180)
    qr_bottom_ratio_fast = st.slider("QR bas (ratio) rapide", 0.35, 0.80, 0.45, 0.01)

    st.divider()
    st.markdown("### Paramètres (après validation)")
    dpi = st.select_slider("DPI", options=[150, 200, 250, 300, 350, 400], value=400)
    qr_bottom_ratio = st.slider("QR bas (ratio)", 0.35, 0.80, 0.45, 0.01)
    ocr_bottom_ratio = st.slider("OCR BV (ratio)", 0.35, 0.80, 0.45, 0.01)

    st.divider()
    st.markdown("### Accélération (phase 2)")
    dpi_detect = st.select_slider("DPI détection QR (phase 2)", options=[120, 150, 180, 200, 250, 300], value=200)

    st.divider()
    st.markdown("### Tampon (optionnel)")
    MODEL_DEFAULT = "runs_tampon/yolov8s_tampon/weights/best.pt"
    use_yolo = st.checkbox("Activer YOLO", value=False)
    model_path_text = st.text_input("Modèle .pt", value=str(Path(MODEL_DEFAULT).resolve()))
    use_tampon_prefill = st.checkbox("Pré-cocher début si tampon", value=False)
    conf_thres = st.slider("Seuil conf", 0.10, 1.00, 0.94, 0.01)
    iou_thres = st.slider("IoU", 0.10, 0.90, 0.45, 0.05)

    st.divider()
    st.markdown("### Lecture facture")
    read_amount_if_missing = st.checkbox(
        "Lire montant (si BV sans montant / pas de BV)",
        value=True,
        disabled=not TESSERACT_AVAILABLE,
    )
    read_date_top = st.checkbox(
        "Lire date (haut de page, 1/3)",
        value=True,
        disabled=not TESSERACT_AVAILABLE,
    )

    st.divider()
    if st.button("🧹 Réinitialiser le scan", use_container_width=True):
        reset_scan()

st.title("Scan facture PDF")
st.caption("v30_1")

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

tmp_dir = Path("data/tmp_bv_v30_1_com")
tmp_dir.mkdir(parents=True, exist_ok=True)
pdf_path = tmp_dir / uploaded_pdf.name
pdf_path.write_bytes(uploaded_pdf.getbuffer())

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
    doc = fitz.open(str(pdf_path))
    n_pages = len(doc)
except Exception as e:
    st.error(f"PDF invalide: {e}")
    st.stop()

cache = RenderCache(doc)

pdf_signature = f"{uploaded_pdf.name}::{len(uploaded_pdf.getbuffer())}"
if st.session_state.get("last_pdf_signature") != pdf_signature:
    for k in ["df_pages_editable", "df_final_pages", "df_final_invoices", "df_final_invoices_editable", "df_transfer_erp"]:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state["last_pdf_signature"] = pdf_signature

# ==========================================================
# PHASE 1 (rapide) - structure uniquement
# ==========================================================
if "df_pages_editable" not in st.session_state:
    progress_fast = st.progress(0, text="Scan rapide…")
    rows_fast = []

    for i in range(n_pages):
        img_rgb = cache.get_full_rgb(page_index=i, dpi=dpi_fast)

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

        _, qr_pts, _ = detect_bottom_qr_candidate(img_rgb, bottom_ratio=qr_bottom_ratio_fast)
        has_qr_candidate = qr_pts is not None

        row = {
            "page": i + 1,
            "has_qr_candidate": bool(has_qr_candidate),
            "has_reference_strict": False,
            "has_bv": bool(has_qr_candidate),
            "bv_reference": None,
            "ref_source": None,
            "bv_currency": None,
            "bv_amount": None,
        }
        if use_yolo:
            row.update(
                {
                    "tampon_pred": int(tampon_pred) if tampon_pred is not None else None,
                    "proba_tampon": round(float(proba_tampon), 4) if proba_tampon is not None else None,
                }
            )

        rows_fast.append(row)
        progress_fast.progress((i + 1) / n_pages, text=f"Scan rapide… {i+1}/{n_pages}")

    df_fast = pd.DataFrame(rows_fast)
    df_fast = compute_invoice_starts_default(df_fast, use_tampon_split=bool(use_tampon_prefill), tampon_flag_col="tampon_pred")
    st.session_state["df_pages_editable"] = df_fast.copy()

# ==========================================================
# STRUCTURE (form => pas de jump)
# ==========================================================
st.subheader("Structure")

cols_fast = ["invoice_start", "page", "has_qr_candidate"]
if use_yolo:
    cols_fast += ["tampon_pred", "proba_tampon"]

df_edit = st.session_state["df_pages_editable"].copy()
df_edit["invoice_start"] = df_edit["invoice_start"].fillna(False).astype(bool)

with st.form("structure_form_v30_1", clear_on_submit=False):
    st.caption("Coche/décoche plusieurs lignes, puis clique **Appliquer** (pas de retour en haut).")

    edited_df = st.data_editor(
        df_edit[cols_fast],
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={"invoice_start": st.column_config.CheckboxColumn("Début", default=False)},
        disabled=[c for c in cols_fast if c != "invoice_start"],
        key="editor_pages_v30_1",
    )

    cA, cB, cC = st.columns([1, 1, 3])
    with cA:
        apply_only = st.form_submit_button("Appliquer les coches", use_container_width=True)
    with cB:
        validate_seg = st.form_submit_button("Valider la segmentation", type="primary", use_container_width=True)
    with cC:
        st.write("")

if apply_only or validate_seg:
    df_updated = st.session_state["df_pages_editable"].copy()
    df_updated["invoice_start"] = edited_df["invoice_start"].fillna(False).astype(bool).values
    if df_updated.shape[0] > 0 and not bool(df_updated.loc[df_updated.index[0], "invoice_start"]):
        df_updated.loc[df_updated.index[0], "invoice_start"] = True
    st.session_state["df_pages_editable"] = df_updated

# ==========================================================
# PHASE 2 strict (accéléré + clip)
# ==========================================================
if validate_seg:
    progress = st.progress(0, text="Analyse…")
    rows = []

    user_starts = st.session_state["df_pages_editable"]["invoice_start"].fillna(False).astype(bool).values

    for i in range(n_pages):
        # DPI_detect full (QR + éventuellement YOLO)
        img_detect = cache.get_full_rgb(page_index=i, dpi=dpi_detect)
        qr_text, qr_pts_detect, _ = detect_bottom_qr_candidate(img_detect, bottom_ratio=qr_bottom_ratio)
        has_qr_candidate = qr_pts_detect is not None

        # YOLO (sur dpi_detect)
        tampon_pred = None
        proba_tampon = None
        if use_yolo and model is not None:
            results = model(img_detect, conf=conf_thres, iou=iou_thres, verbose=False)
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

        bv_currency, bv_amount = (None, None)
        if qr_text:
            bv_currency, bv_amount = extract_currency_amount_from_qr_text(qr_text)

        bv_reference = None
        ref_source = None
        has_reference_strict = False
        has_bv_valid = False

        if has_qr_candidate:
            bv_reference, ref_source = extract_bv_reference_strict(
                cache=cache,
                page_index=i,
                dpi=dpi,
                qr_text=qr_text,
                qr_pts_detect=qr_pts_detect,
                dpi_detect=dpi_detect,
                ocr_bottom_ratio=ocr_bottom_ratio,
            )
            has_reference_strict = is_reference_strict(bv_reference)
            has_bv_valid = bool(has_reference_strict)

        row = {
            "page": i + 1,
            "has_qr_candidate": bool(has_qr_candidate),
            "has_reference_strict": bool(has_reference_strict),
            "has_bv": bool(has_bv_valid),
            "bv_reference": bv_reference,
            "ref_source": ref_source,
            "bv_currency": bv_currency,
            "bv_amount": bv_amount,
            "invoice_start": bool(user_starts[i]) if i < len(user_starts) else (i == 0),
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
    if df_pages.shape[0] > 0 and not bool(df_pages.loc[df_pages.index[0], "invoice_start"]):
        df_pages.loc[df_pages.index[0], "invoice_start"] = True

    df_final_pages, df_final_invoices = assign_invoices_from_user_starts(df_pages, start_col="invoice_start")

    if df_final_invoices.shape[0] > 0:
        df_final_invoices = compute_invoice_meta_v30_1(
            df_final_invoices,
            cache=cache,
            dpi=dpi,
            read_amount=bool(read_amount_if_missing),
            read_date=bool(read_date_top),
        )

    if "bv_last_page" in df_final_invoices.columns:
        df_final_invoices["bv_last_page"] = pd.to_numeric(df_final_invoices["bv_last_page"], errors="coerce").astype("Int64")

    st.session_state["df_final_pages"] = df_final_pages
    st.session_state["df_final_invoices"] = df_final_invoices.copy()
    st.session_state["df_final_invoices_editable"] = df_final_invoices.copy()
    st.session_state["df_transfer_erp"] = build_transfer_erp_table(st.session_state["df_final_invoices"])

# ==========================================================
# Affichage v27
# ==========================================================
if "df_final_pages" in st.session_state and "df_final_invoices" in st.session_state:
    df_pages2 = st.session_state["df_final_pages"]
    df_invoices_num = st.session_state["df_final_invoices"].copy()

    st.subheader("Indicateurs")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Factures", int(df_invoices_num.shape[0]))
    k2.metric("Factures avec BV", int((df_invoices_num["status"] == "HAS_BV").sum()))
    k3.metric("Factures sans BV", int((df_invoices_num["status"] == "NO_BV").sum()))
    k4.metric("BV (pages)", int(df_pages2["has_bv"].sum()))

    st.subheader("Résultats par facture")

    if "df_final_invoices_editable" not in st.session_state:
        st.session_state["df_final_invoices_editable"] = df_invoices_num.copy()

    df_inv_edit = st.session_state["df_final_invoices_editable"].copy()

    inv_cols = [
        "invoice_id", "pages_count", "page_start", "page_end",
        "has_bv", "bv_last_page", "bv_reference", "bv_amount", "bv_currency",
        "invoice_amount", "invoice_currency", "amount_source", "amount_confidence",
        "invoice_date", "date_source", "status",
    ]
    inv_cols = [c for c in inv_cols if c in df_inv_edit.columns]

    edited_invoices = st.data_editor(
        df_inv_edit[inv_cols],
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "invoice_amount": st.column_config.NumberColumn("Montant facture", format="%.2f"),
            "invoice_date": st.column_config.TextColumn("Date facture (YYYY-MM-DD)"),
            "bv_last_page": st.column_config.NumberColumn("Dernière page BV", format="%d"),
            "bv_amount": st.column_config.NumberColumn("Montant BV", format="%.2f"),
            "amount_confidence": st.column_config.NumberColumn("Confiance", format="%.4f"),
        },
        disabled=[c for c in inv_cols if c not in ("invoice_amount", "invoice_date")],
        key="editor_invoices_v30_1",
    )

    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        apply_inv_edits = st.button("Valider corrections facture", use_container_width=True)
    with cB:
        rebuild_transfer = st.button("Recréer tableau ERP", use_container_width=True)
    with cC:
        st.caption("")

    if apply_inv_edits:
        df_apply = st.session_state["df_final_invoices"].copy()
        if "invoice_id" in df_apply.columns and "invoice_id" in edited_invoices.columns:
            df_apply = df_apply.set_index("invoice_id", drop=False)
            tmp = edited_invoices.copy().set_index("invoice_id", drop=False)

            if "invoice_amount" in tmp.columns:
                df_apply.loc[tmp.index, "invoice_amount"] = pd.to_numeric(tmp["invoice_amount"], errors="coerce")
            if "invoice_date" in tmp.columns:
                df_apply.loc[tmp.index, "invoice_date"] = tmp["invoice_date"].apply(_parse_date_iso_or_none)

            df_apply = df_apply.reset_index(drop=True)

        st.session_state["df_final_invoices"] = df_apply.copy()
        st.session_state["df_final_invoices_editable"] = df_apply.copy()
        st.session_state["df_transfer_erp"] = build_transfer_erp_table(df_apply)

    if rebuild_transfer:
        st.session_state["df_transfer_erp"] = build_transfer_erp_table(st.session_state["df_final_invoices"])

    df_invoices_num = st.session_state["df_final_invoices"].copy()

    df_show = df_invoices_num.copy()
    if "bv_last_page" in df_show.columns:
        df_show["bv_last_page"] = pd.to_numeric(df_show["bv_last_page"], errors="coerce").astype("Int64")
    if "bv_amount" in df_show.columns:
        df_show["bv_amount"] = df_show["bv_amount"].apply(_fmt_amount_2dec)
    if "invoice_amount" in df_show.columns:
        df_show["invoice_amount"] = df_show["invoice_amount"].apply(_fmt_amount_2dec)
    if "amount_confidence" in df_show.columns:
        df_show["amount_confidence"] = df_show["amount_confidence"].apply(_fmt_conf_4dec)

    st.dataframe(style_invoice_amount_by_conf(df_show, threshold=0.7), use_container_width=True)

    st.subheader("Transfert ERP")
    df_transfer = st.session_state.get("df_transfer_erp", build_transfer_erp_table(df_invoices_num))
    st.dataframe(df_transfer, use_container_width=True)

    false_qr = df_pages2[(df_pages2["has_qr_candidate"]) & (~df_pages2["has_bv"])]

    with st.expander("Résultats par page", expanded=False):
        st.dataframe(df_pages2, use_container_width=True)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            "Télécharger pages (CSV)",
            data=df_pages2.to_csv(index=False).encode("utf-8"),
            mime="text/csv",
            file_name=f"scan_v30_1_pages_{pdf_path.stem}.csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Télécharger factures (CSV)",
            data=df_invoices_num.to_csv(index=False).encode("utf-8"),
            mime="text/csv",
            file_name=f"scan_v30_1_factures_{pdf_path.stem}.csv",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            "Télécharger QR ignorés (CSV)",
            data=false_qr.to_csv(index=False).encode("utf-8") if len(false_qr) > 0 else "page\n".encode("utf-8"),
            mime="text/csv",
            file_name=f"scan_v30_1_qr_ignores_{pdf_path.stem}.csv",
            use_container_width=True,
        )
    with d4:
        st.download_button(
            "Télécharger transfert ERP (CSV)",
            data=df_transfer.to_csv(index=False).encode("utf-8"),
            mime="text/csv",
            file_name=f"scan_v30_1_transfert_erp_{pdf_path.stem}.csv",
            use_container_width=True,
        )

try:
    doc.close()
except Exception:
    pass
