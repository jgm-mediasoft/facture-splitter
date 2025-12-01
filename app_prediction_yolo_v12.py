# app_prediction_yolo_v12.py

from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF

from ultralytics import YOLO

# ------------------------------------------------------------
# OpenCV pour les QR-codes
# ------------------------------------------------------------
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# ------------------------------------------------------------
# OCR (pytesseract + Pillow) – optionnel
# ------------------------------------------------------------
try:
    import pytesseract
    from PIL import Image

    # 💡 Si besoin, décommente la ligne ci-dessous et adapte le chemin :
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False


def ocr_page_from_image(img_rgb):
    """
    Retourne le texte OCR d'une page à partir de l'image RGB.
    Si pytesseract/Tesseract n'est pas disponible, retourne "".
    """
    if not HAS_TESSERACT:
        return ""
    pil_img = Image.fromarray(img_rgb)
    try:
        # langues BV typiques : FR, DE, IT + EN
        text = pytesseract.image_to_string(pil_img, lang="fra+deu+ita+eng")
    except Exception:
        text = pytesseract.image_to_string(pil_img)
    return text


# ------------------------------------------------------------
# Config Streamlit
# ------------------------------------------------------------
st.set_page_config(page_title="YOLO – Tampons & factures (v12)", layout="wide")
st.title("🧭 Détection de tampons + découpe des factures + BV (v12)")

if not HAS_TESSERACT:
    st.warning(
        "⚠️ OCR désactivé : `pytesseract` ou Tesseract-OCR n'ont pas été trouvés.\n"
        "Le système utilisera uniquement le QR-code pour lire Montant / Devise / Référence des BV."
    )

st.markdown(
    """
Cette application :

1. Utilise un modèle **YOLOv8** pour détecter les **tampons** sur chaque page d’un PDF.  
2. Déduit les **factures** à partir des tampons :
   - une facture commence sur une page qui contient un tampon ;
   - les pages suivantes sans tampon appartiennent à la même facture
     jusqu’au prochain tampon.
3. Pour chaque facture :
   - cherche un **BV** parmi ses pages (QR-code + texte OCR éventuel) ;
   - si BV trouvé : lit **Référence**, **Monnaie/Devise** et **Montant** (d’abord dans le QR, puis dans le texte) ;
   - sinon (ou si pas de montant sur BV) : cherche un **montant total** dans le texte de la facture.
"""
)


# ------------------------------------------------------------
# Helpers génériques montants + monnaie (fallback global facture)
# ------------------------------------------------------------
def _parse_amount_currency_from_text(text: str):
    """
    Parse générique (fallback) sur tout le texte de la facture :
    essaie de trouver (montant, monnaie).
    """
    t = re.sub(r"[ \t]+", " ", text)

    candidates = []

    # monnaie explicite
    patterns_currency = [
        r"(CHF|EUR)[^0-9]{0,20}([0-9][0-9\s.,']*)",
        r"([0-9][0-9\s.,']*)[^0-9A-Z]{0,20}(CHF|EUR)",
    ]

    for pat in patterns_currency:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            if pat.startswith("(CHF|EUR)"):
                curr = m.group(1).upper()
                raw_val = m.group(2)
            else:
                raw_val = m.group(1)
                curr = m.group(2).upper()

            cleaned = re.sub(r"[^\d,\.']", "", raw_val)
            cleaned = cleaned.replace(" ", "").replace("'", "")
            cleaned = cleaned.replace(",", ".")
            try:
                val = float(cleaned)
                candidates.append((val, curr))
            except ValueError:
                continue

    if candidates:
        return max(candidates, key=lambda x: x[0])

    # Totaux génériques sans monnaie explicite
    patterns_total = [
        r"montant\s+total[^0-9]{0,40}([0-9][0-9\s.,']*)",
        r"total\s+chf[^0-9]{0,40}([0-9][0-9\s.,']*)",
        r"total\s+(?:avec\s+tva|tva\s+incluse)[^0-9]{0,40}([0-9][0-9\s.,']*)",
        r"total\s+chf\s+ttc[^0-9]{0,40}([0-9][0-9\s.,']*)",
        r"total\s*\(eur\)[^0-9]{0,40}([0-9][0-9\s.,']*)",
        r"total\s+ttc[^0-9]{0,40}([0-9][0-9\s.,']*)",
    ]

    amount_only = []

    for pat in patterns_total:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            raw_val = m.group(1)
            cleaned = re.sub(r"[^\d,\.']", "", raw_val)
            cleaned = cleaned.replace(" ", "").replace("'", "")
            cleaned = cleaned.replace(",", ".")
            try:
                val = float(cleaned)
                amount_only.append(val)
            except ValueError:
                continue

    if amount_only:
        curr = None
        if re.search(r"\bEUR\b", t, flags=re.IGNORECASE):
            curr = "EUR"
        elif re.search(r"\bCHF\b", t, flags=re.IGNORECASE):
            curr = "CHF"
        return max(amount_only), curr

    return None, None


# ------------------------------------------------------------
# QR-code – décodage
# ------------------------------------------------------------
def decode_qr_data(img_rgb):
    """
    Tente de décoder un QR code sur l'image.
    Retourne la chaine décodée (str) ou None si rien.
    """
    if not HAS_CV2:
        return None

    try:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(img_gray)
        if points is not None and len(points) > 0 and data:
            return data
    except Exception:
        return None

    return None


# ------------------------------------------------------------
# BV – détection de la page BV
# ------------------------------------------------------------
BV_KEYWORDS = [
    "récépissé", "recepisse",
    "section paiement", "partie paiement", "bulletin de versement",
    "empfangsschein", "zahlteil",
    "ricevuta", "sezione pagamento",
    "référence", "reference",
    "monnaie", "monnale", "devise", "währung", "waehrung", "currency", "valuta",
    "montant", "betrag", "importo", "amount",
]


def has_bv_keywords(text_low: str) -> bool:
    return any(kw in text_low for kw in BV_KEYWORDS)


def is_bv_page(text: str, img_rgb) -> bool:
    """
    Page BV si :
      - QR-code décodable, OU
      - mots-clés BV dans le texte OCR.
    """
    if decode_qr_data(img_rgb):
        return True
    return has_bv_keywords(text.lower())


# ------------------------------------------------------------
# BV – Référence / Monnaie / Montant (texte & QR)
# ------------------------------------------------------------
REF_PATTERN_STRICT = r"\b\d{2}\s\d{5}\s\d{5}\s\d{5}\s\d{5}\s\d{5}\b"


def extract_reference_from_text(text: str):
    """
    Extrait une Référence BV au format obligatoire :
        XX XXXXX XXXXX XXXXX XXXXX XXXXX   (27 chiffres)

    Logique :
    1) Cherche d'abord le format exact avec espaces (cas normal)
    2) Sinon cherche toutes les séquences de ≥ 20 caractères contenant des chiffres
    3) Sélectionne la séquence contenant le plus de chiffres
    4) Si elle contient au moins 27 chiffres → on prend les 27 premiers
    5) Reformattage strict en 2 + 5×5 chiffres
    """

    # 1) Format déjà parfait
    m = re.search(REF_PATTERN_STRICT, text)
    if m:
        return m.group(0).strip()

    # 2) Chercher des longues séquences contenant chiffres, espaces, apostrophes
    digit_sequences = []
    for m2 in re.finditer(r"[0-9\s'’]{20,}", text):
        raw = m2.group(0)
        digits = re.sub(r"\D", "", raw)
        if len(digits) >= 20:
            digit_sequences.append(digits)

    if not digit_sequences:
        return None

    # 3) On prend la séquence contenant le plus de chiffres
    best = max(digit_sequences, key=len)

    # 4) On doit obtenir exactement 27 chiffres → format BV suisse
    if len(best) < 27:
        return None  # trop court, pas une référence valide

    # On garde les 27 premiers chiffres
    best = best[:27]

    # 5) Reformattage : 2 chiffres + cinq blocs de 5 chiffres
    blocks = [
        best[0:2],
        best[2:7],
        best[7:12],
        best[12:17],
        best[17:22],
        best[22:27],
    ]

    reference_formatted = " ".join(blocks)
    return reference_formatted


CURRENCY_LABELS = ["monnaie", "monnale", "devise", "währung", "waehrung", "valuta", "currency"]


def extract_currency_from_lines(lines):
    """
    Cherche CHF / EUR autour des labels Monnaie/Devise/Währung...
    """
    for i, line in enumerate(lines):
        low = line.lower()
        if any(lbl in low for lbl in CURRENCY_LABELS):
            m = re.search(r"\b(CHF|EUR)\b", line, flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()
            for j in range(i + 1, min(i + 4, len(lines))):
                l2 = lines[j]
                m2 = re.search(r"\b(CHF|EUR)\b", l2, flags=re.IGNORECASE)
                if m2:
                    return m2.group(1).upper()

    # fallback : première occurrence CHF/EUR dans la page
    joined = "\n".join(lines)
    m = re.search(r"\b(CHF|EUR)\b", joined, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


AMOUNT_LABELS = ["montant", "betrag", "amount", "importo"]


def parse_amount(raw: str):
    cleaned = re.sub(r"[^\d,\.']", "", raw)
    cleaned = cleaned.replace(" ", "").replace("'", "")
    cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_amount_from_lines(lines):
    """
    Cherche le montant autour des labels Montant/Betrag/Importo/Amount,
    sinon tous les nombres avec 2 décimales dans la page.
    """
    candidates = []

    # 1) autour des labels
    for i, line in enumerate(lines):
        low = line.lower()
        if any(lbl in low for lbl in AMOUNT_LABELS):
            zone = [line] + lines[i + 1 : i + 4]
            for l in zone:
                for m in re.finditer(r"[0-9][0-9\s.,']*\d", l):
                    val = parse_amount(m.group(0))
                    if val is not None:
                        candidates.append(val)
            break

    # 2) si rien, nombres à 2 décimales (bas de page typiquement)
    if not candidates:
        joined = "\n".join(lines)
        for m in re.finditer(r"[0-9][0-9\s']*[.,]\d{2}", joined):
            val = parse_amount(m.group(0))
            if val is not None:
                candidates.append(val)

    return max(candidates) if candidates else None


def extract_bv_fields_from_qr(img_rgb):
    """
    Essaie d'extraire (reference, currency, amount) à partir du contenu du QR.
    Retourne (ref, curr, amount) ou (None, None, None).
    """
    data = decode_qr_data(img_rgb)
    if not data:
        return None, None, None

    # On travaille sur tout le texte du QR
    txt = data.replace("\r", "\n")

    # Référence : même logique que pour le texte BV
    ref = extract_reference_from_text(txt)

    # Monnaie : CHF / EUR dans le payload
    m_curr = re.search(r"\b(CHF|EUR)\b", txt, flags=re.IGNORECASE)
    curr = m_curr.group(1).upper() if m_curr else None

    # Montant : tous les nombres avec 2 décimales, on prend le plus grand
    candidates = []
    for m in re.finditer(r"[0-9][0-9\s']*[.,]\d{2}", txt):
        val = parse_amount(m.group(0))
        if val is not None:
            candidates.append(val)
    amount = max(candidates) if candidates else None

    return ref, curr, amount


def extract_bv_fields(page_text: str, img_rgb):
    """
    Extraction complète BV pour une page déjà identifiée comme BV :
      - on essaie d'abord via le QR (très robuste, même si le papier est abîmé)
      - ensuite fallback via le texte OCR (Monnaie/Montant/Référence imprimés).
    """
    # 1) Tentative via QR
    ref_qr, curr_qr, amt_qr = extract_bv_fields_from_qr(img_rgb)

    # Si on a déjà un montant via le QR, on considère que c'est suffisant
    if amt_qr is not None:
        return ref_qr, curr_qr, amt_qr

    # 2) Fallback via texte OCR de la page
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in page_text.splitlines()]

    reference = extract_reference_from_text(page_text)
    currency = extract_currency_from_lines(lines)
    amount = extract_amount_from_lines(lines)

    # Si le QR donnait au moins une info, on peut combiner
    if ref_qr and not reference:
        reference = ref_qr
    if curr_qr and not currency:
        currency = curr_qr
    if amt_qr and not amount:
        amount = amt_qr

    return reference, currency, amount


# ------------------------------------------------------------
# Extraction Montant + Monnaie pour UNE facture
# ------------------------------------------------------------
def extract_amount_currency_for_invoice(page_texts, page_images, start_idx: int, end_idx: int):
    """
    Cherche Montant + Monnaie pour UNE facture (pages [start_idx, end_idx]).
    1) Priorité aux pages BV (QR + texte OCR) → on lit les champs BV.
    2) Sinon, parseur générique sur tout le texte de la facture.
    """
    facture_texts = page_texts[start_idx : end_idx + 1]
    facture_imgs = page_images[start_idx : end_idx + 1]
    full_text = "\n".join(facture_texts)

    has_bv = False
    bv_reference = None
    bv_amount = None
    bv_currency = None

    for txt, img in zip(facture_texts, facture_imgs):
        if is_bv_page(txt, img):
            has_bv = True
            ref, curr, amt = extract_bv_fields(txt, img)
            bv_reference = ref
            bv_currency = curr
            bv_amount = amt
            break

    if has_bv and bv_amount is not None:
        return bv_amount, bv_currency, True, bv_reference

    # fallback texte global facture
    val, curr = _parse_amount_currency_from_text(full_text)
    return val, curr, has_bv, bv_reference


# ------------------------------------------------------------
# YOLO + PDF helpers
# ------------------------------------------------------------
@st.cache_resource
def load_yolo_model(model_path: str):
    return YOLO(model_path)


def render_pdf_page_to_rgb(pdf_path: Path, page_index: int = 0, dpi: int = 300):
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        return img


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
dpi = st.select_slider("DPI rendu PDF", options=[150, 200, 250, 300], value=300)
show_images = st.checkbox("Afficher les pages avec boîtes de détection", value=False)

uploaded_pdf = st.file_uploader("📄 Dépose un PDF multipages", type=["pdf"])


# ------------------------------------------------------------
# Pipeline principal
# ------------------------------------------------------------
if uploaded_pdf:
    tmp_dir = Path("data/tmp_pred_yolo")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / uploaded_pdf.name
    pdf_path.write_bytes(uploaded_pdf.getbuffer())
    st.info(f"PDF enregistré : {pdf_path.resolve()}")

    # Modèle YOLO
    try:
        model = load_yolo_model(model_path_text)
        st.success("Modèle YOLO chargé ✅")
    except Exception as e:
        st.error(f"Impossible de charger le modèle : {e}")
        st.stop()

    # Lecture PDF pour connaître le nombre de pages
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
    page_images = []
    page_texts = []  # texte OCR par page (ou vide si pas d'OCR)

    for i in range(n_pages):
        img_rgb = render_pdf_page_to_rgb(pdf_path, page_index=i, dpi=dpi)
        page_images.append(img_rgb)

        # OCR text (si dispo)
        text_ocr = ocr_page_from_image(img_rgb)
        page_texts.append(text_ocr)

        # YOLO tampon
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

        rows.append(
            {
                "page": i + 1,
                "tampon_pred": tampon_pred,
                "proba_tampon": round(proba_tampon, 4),
                "n_detections": int(n_det),
                "max_conf": round(max_conf, 4),
            }
        )

        if show_images:
            im_plot = r.plot()
            images_to_show.append((i + 1, im_plot))

        if (i + 1) % max(1, n_pages // 20) == 0 or i == n_pages - 1:
            progress.progress((i + 1) / n_pages, text=f"Page {i+1}/{n_pages}")

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Découpage des factures par tampons
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
    # Montant / Monnaie / BV par facture
    # --------------------------------------------------------
    facture_montant_map = {}
    facture_monnaie_map = {}
    facture_has_bv_map = {}
    facture_bv_ref_map = {}
    data_invoices = []

    if nb_factures > 0:
        for idx in range(1, nb_factures + 1):
            sub = df[df["facture_index"] == idx]
            if sub.empty:
                continue

            start_page = int(sub["page"].min())
            end_page = int(sub["page"].max())
            start_idx = start_page - 1
            end_idx = end_page - 1

            montant, monnaie, has_bv, bv_ref = extract_amount_currency_for_invoice(
                page_texts, page_images, start_idx, end_idx
            )

            facture_montant_map[idx] = montant
            facture_monnaie_map[idx] = monnaie
            facture_has_bv_map[idx] = has_bv
            facture_bv_ref_map[idx] = bv_ref

            data_invoices.append(
                {
                    "facture_index": idx,
                    "page_debut": start_page,
                    "page_fin": end_page,
                    "nb_pages": end_page - start_page + 1,
                    "montant": montant,
                    "monnaie": monnaie,
                    "has_bv": has_bv,
                    "bv_reference": bv_ref,
                }
            )

    df_factures = pd.DataFrame(data_invoices) if data_invoices else pd.DataFrame(
        columns=[
            "facture_index",
            "page_debut",
            "page_fin",
            "nb_pages",
            "montant",
            "monnaie",
            "has_bv",
            "bv_reference",
        ]
    )

    # Reporter montant/monnaie sur chaque page
    facture_montant_per_page = []
    facture_monnaie_per_page = []
    for _, row in df.iterrows():
        idx = row["facture_index"]
        if idx is None:
            facture_montant_per_page.append(None)
            facture_monnaie_per_page.append(None)
        else:
            facture_montant_per_page.append(facture_montant_map.get(idx))
            facture_monnaie_per_page.append(facture_monnaie_map.get(idx))

    df["facture_montant"] = facture_montant_per_page
    df["facture_monnaie"] = facture_monnaie_per_page

    # --------------------------------------------------------
    # Affichages
    # --------------------------------------------------------
    st.subheader("📊 Résultats par page")
    st.dataframe(df, use_container_width=True)

    st.subheader("🧾 Factures détectées (basées sur les tampons)")
    st.metric("Nombre de factures détectées", nb_factures)
    if not df_factures.empty:
        st.dataframe(df_factures, use_container_width=True)
    else:
        st.info("Aucune facture détectée (aucun tampon).")

    # Export CSV
    st.download_button(
        "📥 Télécharger les résultats par page (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        mime="text/csv",
        file_name=f"yolo_tampons_factures_v12_{pdf_path.stem}.csv",
    )

    if not df_factures.empty:
        st.download_button(
            "📥 Télécharger le résumé par facture (CSV)",
            data=df_factures.to_csv(index=False).encode("utf-8"),
            mime="text/csv",
            file_name=f"factures_resume_v12_{pdf_path.stem}.csv",
        )

    # Debug : afficher le texte OCR d'une page
    with st.expander("🔍 Debug : afficher le texte OCR d'une page"):
        page_num_debug = st.number_input(
            "Page à afficher (1 à n_pages)", min_value=1, max_value=n_pages, value=1
        )
        if st.button("Afficher le texte OCR de cette page"):
            text_dbg = page_texts[page_num_debug - 1]
            st.text(text_dbg if text_dbg.strip() else "[Texte OCR vide]")

    # Miniatures YOLO
    if show_images and images_to_show:
        st.subheader("Aperçu des pages avec boîtes de détection")
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



