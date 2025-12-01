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
# YOLO – chargement
# ------------------------------------------------------------
@st.cache_resource
def load_yolo_model(model_path: str):
    return YOLO(model_path)


# ------------------------------------------------------------
# Détection du QR du BV (grand QR en bas de page)
# ------------------------------------------------------------
def detect_bv_qr(img_rgb: np.ndarray):
    """
    Détecte le QR du Bulletin de versement (BV) et retourne (data, points).

    - data  : texte décodé du QR (peut être None si non décodable)
    - points: tableau (4,2) des coins du QR dans l'image (coordonnées d'origine)

    Stratégie :
      1. detectAndDecodeMulti sur image originale
      2. detectMulti sur image originale (points même si data vide)
      3. detectAndDecodeMulti sur image agrandie x1.7
      4. detectMulti sur image agrandie x1.7
      5. fallback detectAndDecode simple

    Filtrage BV :
      - centre vertical dans la moitié basse : cy_ratio >= 0.35
      - surface >= 1.5% de la surface page : area_ratio >= 0.015
      - on prend le QR éligible avec la plus grande surface
    """
    if not HAS_CV2:
        return None, None

    detector = cv2.QRCodeDetector()

    try:
        gray_base = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    except Exception:
        return None, None

    h, w = gray_base.shape
    page_area = float(h * w)

    def collect_candidates_from_gray(gray, scale_factor: float):
        """Retourne une liste de (area_ratio, data, pts_orig, cy_ratio)."""
        res = []

        # 1) detectAndDecodeMulti : points + data
        try:
            retval, decoded_infos, points, _ = detector.detectAndDecodeMulti(gray)
        except Exception:
            retval, decoded_infos, points = False, None, None

        if retval and points is not None and len(points) > 0:
            for i, pts in enumerate(points):
                data_i = decoded_infos[i] if decoded_infos is not None else None
                pts_orig = pts / scale_factor
                xs = pts_orig[:, 0]
                ys = pts_orig[:, 1]
                cx, cy = xs.mean(), ys.mean()
                w_qr = xs.max() - xs.min()
                h_qr = ys.max() - ys.min()
                area_qr = float(w_qr * h_qr)

                cy_ratio = cy / h
                area_ratio = area_qr / page_area
                res.append((area_ratio, data_i, pts_orig, cy_ratio))

        # 2) detectMulti : points sans data (data = None)
        try:
            ok, points_multi = detector.detectMulti(gray)
        except Exception:
            ok, points_multi = False, None

        if ok and points_multi is not None and len(points_multi) > 0:
            for pts in points_multi:
                pts_orig = pts / scale_factor
                xs = pts_orig[:, 0]
                ys = pts_orig[:, 1]
                cx, cy = xs.mean(), ys.mean()
                w_qr = xs.max() - xs.min()
                h_qr = ys.max() - ys.min()
                area_qr = float(w_qr * h_qr)

                cy_ratio = cy / h
                area_ratio = area_qr / page_area
                res.append((area_ratio, None, pts_orig, cy_ratio))

        return res

    candidates = []

    # 1) image originale
    candidates.extend(collect_candidates_from_gray(gray_base, scale_factor=1.0))

    # 2) image agrandie x1.7 pour les QR difficiles
    try:
        gray_big = cv2.resize(
            gray_base, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_LINEAR
        )
        candidates.extend(collect_candidates_from_gray(gray_big, scale_factor=1.7))
    except Exception:
        pass

    # Filtrage BV : bas de page + taille suffisante
    bv_candidates = [
        (area_ratio, data, pts_orig)
        for (area_ratio, data, pts_orig, cy_ratio) in candidates
        if cy_ratio >= 0.35 and area_ratio >= 0.015
    ]

    if bv_candidates:
        bv_candidates.sort(key=lambda x: x[0], reverse=True)
        _, data_best, pts_best = bv_candidates[0]
        return data_best, pts_best

    # 3) Fallback : detectAndDecode simple sur l'image de base
    try:
        data, pts, _ = detector.detectAndDecode(gray_base)
    except Exception:
        return None, None

    if pts is not None and (data or True):
        pts_arr = pts[0] if pts.ndim == 3 else pts
        xs = pts_arr[:, 0]
        ys = pts_arr[:, 1]
        cx, cy = xs.mean(), ys.mean()
        w_qr = xs.max() - xs.min()
        h_qr = ys.max() - ys.min()
        area_qr = float(w_qr * h_qr)

        cy_ratio = cy / h
        area_ratio = area_qr / page_area

        if cy_ratio >= 0.35 and area_ratio >= 0.015:
            # data peut être "", on le garde tel quel
            return data if data else None, pts_arr

    return None, None


# ------------------------------------------------------------
# Crop : zone à droite du QR (pour Référence)
# ------------------------------------------------------------
def crop_region_right_of_qr(img_rgb: np.ndarray, qr_points: np.ndarray):
    """
    Découpe un rectangle à droite du QR-code.
    Zone où se trouve la Référence imprimée.
    """
    h, w, _ = img_rgb.shape
    xs = qr_points[:, 0]
    ys = qr_points[:, 1]

    x_min = int(xs.min())
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

    crop = img_rgb[top:bottom, left:right]
    return crop


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
    On cherche d'abord autour des mots 'Référence/Referenz/...',
    puis un pattern XX XXXXX XXXXX XXXXX XXXXX XXXXX.
    """
    norm = (
        text.replace("\xa0", " ")
        .replace("\u202f", " ")
        .replace("\u00a0", " ")
    )
    norm_low = norm.lower()

    # Localiser un label 'Référence'
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

    # 1) pattern strict avec espaces
    zone_one_space = re.sub(r"\s+", " ", zone)
    m = re.search(REF_PATTERN_STRICT, zone_one_space)
    if m:
        return m.group(0).strip()

    # 2) fallback : longue séquence de chiffres près de la zone
    digit_sequences = []
    for m2 in re.finditer(r"[0-9\D]{20,}", zone):
        raw = m2.group(0)
        digits = re.sub(r"\D", "", raw)
        if len(digits) >= 27:
            digit_sequences.append(digits)

    if not digit_sequences:
        return None

    best = max(digit_sequences, key=len)
    best = best[:27]
    if len(best) < 27:
        return None

    blocks = [
        best[0:2],
        best[2:7],
        best[7:12],
        best[12:17],
        best[17:22],
        best[22:27],
    ]
    return " ".join(blocks)


# ------------------------------------------------------------
# Fallback : Référence dans le payload du QR (QRR / SCOR)
# ------------------------------------------------------------
def extract_reference_from_qr_payload(txt: str):
    """
    Lit la référence directement dans la structure Swiss QR :

      ... QRR\n210000000003139471430009017\n...

    ou SCOR + ligne suivante.
    Retourne une référence formatée comme imprimée sur le BV.
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
                    blocks = [
                        digits[0:2],
                        digits[2:7],
                        digits[7:12],
                        digits[12:17],
                        digits[17:22],
                        digits[22:27],
                    ]
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
    """
    À partir du texte complet du QR (payload Swiss QR-bill),
    on récupère :
      - currency (CHF/EUR)
      - amount (float ou None)
    """
    norm = txt.replace("\r", "\n")

    # Devise CHF / EUR
    m_curr = re.search(r"\b(CHF|EUR)\b", norm, flags=re.IGNORECASE)
    currency = m_curr.group(1).upper() if m_curr else None

    # Montant : première ligne qui ressemble à 123.45 ou 123,45
    amount = None
    for line in norm.splitlines():
        ln = line.strip().replace(" ", "")
        if re.fullmatch(r"\d{1,9}[.,]\d{2}", ln):
            amount = parse_amount(ln)
            if amount is not None:
                break

    return currency, amount


# ------------------------------------------------------------
# Config Streamlit
# ------------------------------------------------------------
st.set_page_config(
    page_title="BV only – Tampons, factures & BV",
    layout="wide",
)
st.title("🧾 BV only – Tampons, factures & Bulletins de versement")

status_parts = [
    f"OpenCV QR {'✅' if HAS_CV2 else '❌'}",
    f"Tesseract OCR {'✅' if TESSERACT_AVAILABLE else '❌'}",
]
st.caption(" | ".join(status_parts))

st.markdown(
    """
**Règles :**

1. **Tampon YOLO**  
   - Première page avec tampon → début facture 1.  
   - Pages suivantes sans tampon → même facture.  
   - Nouveau tampon → nouvelle facture.

2. **BV**  
   - Un BV est présent s'il y a un **grand QR en bas de page**.  
   - On indique si **chaque page** contient un BV (`has_bv_page`).  
   - Une facture a un BV (`has_bv_facture`) si sa **dernière page** contient un BV.

3. **Référence**  
   - Lue **d'abord** dans le texte situé **à droite du QR-code** (OCR Tesseract).  
   - Si l'OCR échoue → **fallback** dans le payload du QR (QRR/SCOR).

4. **Monnaie + Montant**  
   - Lus dans le **texte du QR** (CHF / EUR et montant).
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
dpi = st.select_slider("DPI rendu PDF", options=[150, 200, 250, 300, 350, 400], value=300)
show_images = st.checkbox("Afficher les pages avec boîtes de détection (YOLO)", value=False)

uploaded_pdf = st.file_uploader("📄 Dépose un PDF multipages", type=["pdf"])


# ------------------------------------------------------------
# Pipeline principal
# ------------------------------------------------------------
if uploaded_pdf:
    tmp_dir = Path("data/tmp_bv_only")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / uploaded_pdf.name
    pdf_path.write_bytes(uploaded_pdf.getbuffer())
    st.info(f"PDF enregistré : {pdf_path.resolve()}")

    # YOLO
    try:
        model = load_yolo_model(model_path_text)
        st.success("Modèle YOLO chargé ✅")
    except Exception as e:
        st.error(f"Impossible de charger le modèle : {e}")
        st.stop()

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

        # 1) Tampon YOLO
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

        # 2) QR-code BV
        qr_text, qr_pts = detect_bv_qr(img_rgb)
        has_bv_page = qr_pts is not None  # BV si on a AU MOINS les points du QR

        page_qr_texts.append(qr_text)

        # 3) OCR de la zone à droite du QR (pour Référence)
        ref_zone_text = ""
        if qr_pts is not None:
            crop = crop_region_right_of_qr(img_rgb, qr_pts)
            if crop is not None:
                ref_zone_text = ocr_image_rgb(crop)
        page_ref_ocr_texts.append(ref_zone_text)

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

        if show_images:
            im_plot = r.plot()
            images_to_show.append((i + 1, im_plot))

        if (i + 1) % max(1, n_pages // 20) == 0 or i == n_pages - 1:
            progress.progress((i + 1) / n_pages, text=f"Page {i+1}/{n_pages}")

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Découpage factures par tampons
    # --------------------------------------------------------
    facture_index = 0
    facture_index_per_page = []
    for _, row in df.iterrows():
        if row["tampon_pred"] == 1:
            facture_index += 1
            facture_index_per_page.append(facture_index)
        else:
            if facture_index > 0:
                facture_index_per_page.append(facture_index)
            else:
                facture_index_per_page.append(None)

    df["facture_index"] = facture_index_per_page
    nb_factures = facture_index

    # --------------------------------------------------------
    # Résumé par facture (BV sur dernière page + données BV)
    # --------------------------------------------------------
    data_invoices = []
    if nb_factures > 0:
        for idx in range(1, nb_factures + 1):
            sub = df[df["facture_index"] == idx]
            if sub.empty:
                continue

            start_page = int(sub["page"].min())
            end_page = int(sub["page"].max())
            nb_pages_fact = end_page - start_page + 1

            last_page = end_page
            last_row = df[df["page"] == last_page].iloc[0]
            has_bv_facture = bool(last_row["has_bv_page"])

            bv_reference = None
            bv_currency = None
            bv_amount = None

            if has_bv_facture:
                # Référence : OCR à droite du QR
                ref_text_zone = page_ref_ocr_texts[last_page - 1] or ""
                if ref_text_zone.strip():
                    bv_reference = extract_reference_from_text(ref_text_zone)

                # Monnaie + Montant + fallback Référence via payload QR
                qr_text_last = page_qr_texts[last_page - 1]
                if qr_text_last:
                    # Montant + devise
                    bv_currency, bv_amount = extract_currency_amount_from_qr_text(
                        qr_text_last
                    )
                    # Si l'OCR n'a pas trouvé la référence, on essaie le payload
                    if not bv_reference:
                        bv_reference = extract_reference_from_qr_payload(qr_text_last)

            data_invoices.append(
                {
                    "facture_index": idx,
                    "page_debut": start_page,
                    "page_fin": end_page,
                    "nb_pages": nb_pages_fact,
                    "has_bv_facture": has_bv_facture,
                    "bv_page": last_page if has_bv_facture else None,
                    "bv_reference": bv_reference,
                    "bv_currency": bv_currency,
                    "bv_amount": bv_amount,
                }
            )

    df_factures = (
        pd.DataFrame(data_invoices)
        if data_invoices
        else pd.DataFrame(
            columns=[
                "facture_index",
                "page_debut",
                "page_fin",
                "nb_pages",
                "has_bv_facture",
                "bv_page",
                "bv_reference",
                "bv_currency",
                "bv_amount",
            ]
        )
    )

    # --------------------------------------------------------
    # AFFICHAGE
    # --------------------------------------------------------
    st.subheader("📄 Résultats par page")
    st.dataframe(df, use_container_width=True)

    st.subheader("🧾 Résumé par facture")
    st.metric("Nombre de factures détectées", nb_factures)
    if not df_factures.empty:
        st.dataframe(df_factures, use_container_width=True)
    else:
        st.info("Aucune facture détectée (aucun tampon).")

    # CSV
    st.download_button(
        "📥 Télécharger les résultats par page (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        mime="text/csv",
        file_name=f"bv_only_pages_{pdf_path.stem}.csv",
    )

    st.download_button(
        "📥 Télécharger le résumé par facture (CSV)",
        data=df_factures.to_csv(index=False).encode("utf-8"),
        mime="text/csv",
        file_name=f"bv_only_factures_{pdf_path.stem}.csv",
    )

    # Debug : OCR de la zone Référence
    with st.expander("🔍 Debug Référence : texte OCR à droite du QR"):
        page_num_debug = st.number_input(
            "Page (1 à n_pages)", min_value=1, max_value=n_pages, value=1
        )
        if st.button("Afficher le texte OCR de cette page (droite du QR)"):
            txt = page_ref_ocr_texts[page_num_debug - 1]
            st.text(txt if txt.strip() else "[aucun texte / pas de BV sur cette page]")

    # Debug : texte brut du QR
    with st.expander("🔍 Debug BV : texte brut du QR d'une page"):
        page_num_debug2 = st.number_input(
            "Page pour le QR", min_value=1, max_value=n_pages, value=1, key="qr_debug_page"
        )
        if st.button("Afficher le texte QR de cette page"):
            txt_qr = page_qr_texts[page_num_debug2 - 1]
            st.text(txt_qr if txt_qr else "[aucun QR BV décodé sur cette page]")

    # Aperçu visuel YOLO
    if show_images and images_to_show:
        st.subheader("Aperçu des pages avec boîtes de détection (tampons YOLO)")
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


