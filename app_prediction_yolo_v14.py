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
    from PIL import Image  # utile quand même pour HuggingFace
    pytesseract = None
    HAS_TESSERACT = False

# ------------------------------------------------------------
# OCR HuggingFace (TrOCR) – optionnel
# ------------------------------------------------------------
try:
    from transformers import pipeline

    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

HF_OCR_MODEL_NAME = "microsoft/trocr-base-printed"


@st.cache_resource
def load_hf_ocr_pipeline(model_name: str = HF_OCR_MODEL_NAME):
    """
    Charge le pipeline HuggingFace pour OCR (image -> texte).
    Nécessite : transformers, torch, etc.
    """
    return pipeline("image-to-text", model=model_name)


def ocr_page_from_image(
    img_rgb: np.ndarray,
    use_hf: bool = False,
    hf_pipeline=None,
) -> str:
    """
    Retourne le texte OCR d'une page à partir de l'image RGB.

    Priorité :
    1) Si use_hf=True et pipeline HF disponible -> HuggingFace TrOCR.
    2) Sinon, si Tesseract dispo -> pytesseract.
    3) Sinon -> texte vide.
    """
    # 1) HuggingFace OCR
    if use_hf and hf_pipeline is not None:
        try:
            pil_img = Image.fromarray(img_rgb)
            out = hf_pipeline(pil_img, max_new_tokens=256)
            if isinstance(out, list) and out:
                txt = out[0].get("generated_text", "")
                return txt if txt is not None else ""
        except Exception:
            pass  # fallback sur Tesseract

    # 2) Tesseract
    if HAS_TESSERACT and pytesseract is not None:
        try:
            pil_img = Image.fromarray(img_rgb)
            text = pytesseract.image_to_string(pil_img, lang="fra+deu+ita+eng")
        except Exception:
            text = pytesseract.image_to_string(pil_img)
        return text

    # 3) Rien de dispo
    return ""


# ------------------------------------------------------------
# Config Streamlit
# ------------------------------------------------------------
st.set_page_config(page_title="YOLO – Tampons & factures (v14 HF)", layout="wide")
st.title("🧭 Détection de tampons + découpe des factures + BV (v14, HuggingFace OCR)")

if not HAS_TESSERACT and not HF_AVAILABLE:
    st.warning(
        "⚠️ Aucun moteur OCR disponible : ni Tesseract, ni HuggingFace.\n"
        "Installe soit Tesseract + pytesseract, soit transformers + torch pour activer l’OCR."
    )
elif not HAS_TESSERACT and HF_AVAILABLE:
    st.info(
        "ℹ️ Tesseract non disponible, mais HuggingFace OCR l’est.\n"
        "L’app utilisera uniquement le modèle TrOCR pour l’OCR."
    )
elif HAS_TESSERACT and not HF_AVAILABLE:
    st.info(
        "ℹ️ HuggingFace OCR non disponible, mais Tesseract l’est.\n"
        "L’app utilisera uniquement Tesseract pour l’OCR."
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
   - si BV trouvé : lit **Référence**, **Monnaie/Devise** et **Montant** (priorité au QR) ;
   - si **aucun BV** : lit **Montant + Monnaie** uniquement dans le texte de la facture (ex. `CHF 330.-`).

Nouveauté v14 : possibilité d’utiliser **HuggingFace TrOCR** comme moteur OCR (option dans l’interface).
"""
)

# ------------------------------------------------------------
# Helpers génériques montants + monnaie (fallback global facture)
# ------------------------------------------------------------

def _parse_amount_currency_from_text(text: str):
    """
    Parse générique (fallback) sur tout le texte de la facture.
    Objectif : trouver (montant, monnaie), en priorité le nombre
    immédiatement autour de CHF / EUR (ex : 'CHF 330.-').

    Stratégie :
      1) Normalise les espaces.
      2) Pour chaque 'CHF' / 'EUR', cherche d'abord un nombre APRÈS la monnaie,
         sinon un nombre AVANT la monnaie (le plus proche).
      3) Ignore les "années" typiques (1900–2100) sans décimales.
      4) Si plusieurs candidats, on prend le plus grand.
      5) Si rien trouvé, on retombe sur des patterns 'Total ...' classiques.
    """

    norm = (
        text.replace("\xa0", " ")
        .replace("\u202f", " ")
        .replace("\u00a0", " ")
    )

    def clean_to_float(raw: str):
        cleaned = re.sub(r"[^\d,\.']", "", raw)
        cleaned = cleaned.replace(" ", "").replace("'", "").replace("’", "")
        cleaned = cleaned.replace(",", ".")
        if not cleaned:
            return None
        try:
            val = float(cleaned)
        except ValueError:
            return None
        # Filtre les années typiques sans décimales (ex. 2024, 2025)
        if val.is_integer() and 1900 <= val <= 2100 and "." not in cleaned:
            return None
        return val

    candidates = []

    # 1) Parcourt toutes les occurrences de CHF / EUR
    for m in re.finditer(r"\b(CHF|EUR)\b", norm, flags=re.IGNORECASE):
        curr = m.group(1).upper()

        # 1a) Cherche d'abord un nombre APRÈS la monnaie
        after = norm[m.end(): m.end() + 80]
        m_after = re.search(r"[0-9][0-9\s'’]*(?:[.,]\d{1,2})?", after)
        val_after = None
        if m_after:
            val_after = clean_to_float(m_after.group(0))

        if val_after is not None:
            candidates.append((val_after, curr))
            continue  # pas besoin de chercher avant si on a déjà un nombre après

        # 1b) Sinon, cherche un nombre AVANT la monnaie (le plus proche)
        before = norm[max(0, m.start() - 80): m.start()]
        matches_before = list(re.finditer(r"[0-9][0-9\s'’]*(?:[.,]\d{1,2})?", before))
        if matches_before:
            last = matches_before[-1]
            val_before = clean_to_float(last.group(0))
            if val_before is not None:
                candidates.append((val_before, curr))

    if candidates:
        return max(candidates, key=lambda x: x[0])

    # 2) Fallback : patterns "Total ..."
    norm2 = re.sub(r"[ \t]+", " ", norm)

    patterns_total = [
        r"montant\s+total[^0-9]{0,40}([0-9][0-9\s.,'-]*)",
        r"total\s+chf[^0-9]{0,40}([0-9][0-9\s.,'-]*)",
        r"total\s+(?:avec\s+tva|tva\s+incluse)[^0-9]{0,40}([0-9][0-9\s.,'-]*)",
        r"total\s+chf\s+ttc[^0-9]{0,40}([0-9][0-9\s.,'-]*)",
        r"total\s*\(eur\)[^0-9]{0,40}([0-9][0-9\s.,'-]*)",
        r"total\s+ttc[^0-9]{0,40}([0-9][0-9\s.,'-]*)",
    ]

    amount_only = []
    for pat in patterns_total:
        for m in re.finditer(pat, norm2, flags=re.IGNORECASE):
            raw_val = m.group(1)
            val = clean_to_float(raw_val)
            if val is not None:
                amount_only.append(val)

    if amount_only:
        curr = None
        if re.search(r"\bEUR\b", norm2, flags=re.IGNORECASE):
            curr = "EUR"
        elif re.search(r"\bCHF\b", norm2, flags=re.IGNORECASE):
            curr = "CHF"
        return max(amount_only), curr

    return None, None


# ------------------------------------------------------------
# QR-code – décodage
# ------------------------------------------------------------

def decode_qr_data(img_rgb: np.ndarray):
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
    "référence", "reference", "referenz",
    "monnaie", "monnale", "devise", "währung", "waehrung", "currency", "valuta",
    "montant", "betrag", "importo", "amount",
]


def has_bv_keywords(text_low: str) -> bool:
    return any(kw in text_low for kw in BV_KEYWORDS)


def is_bv_page(text: str, img_rgb: np.ndarray) -> bool:
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

# 2 chiffres, puis 5 blocs de 5 chiffres, séparés par au moins un espace
REF_PATTERN_STRICT = r"\b\d{2}(?:\s+\d{5}){5}\b"


def extract_reference_from_text(text: str):
    """
    Extrait une Référence BV au format obligatoire :
        XX XXXXX XXXXX XXXXX XXXXX XXXXX   (27 chiffres)

    Hypothèses :
    - la référence se trouve sur le BV, à proximité d'un libellé de type
      'Référence', 'Reference', 'Referenz'.
    - la forme est toujours 2 chiffres + 5 blocs de 5 chiffres.
    """

    norm = (
        text.replace("\xa0", " ")
        .replace("\u202f", " ")
        .replace("\u00a0", " ")
    )

    norm_low = norm.lower()
    labels = ["référence", "reference", "referenz"]

    idx_label = -1
    for lab in labels:
        pos = norm_low.find(lab)
        if pos != -1 and (idx_label == -1 or pos < idx_label):
            idx_label = pos

    if idx_label != -1:
        start = max(0, idx_label - 50)
        end = min(len(norm), idx_label + 200)
        zone = norm[start:end]
    else:
        zone = norm

    # 1) Essai strict avec un seul espace
    zone_one_space = re.sub(r"\s+", " ", zone)
    m = re.search(REF_PATTERN_STRICT, zone_one_space)
    if m:
        return m.group(0).strip()

    # 2) Essai plus souple : on extrait toutes les séquences contenant
    # suffisamment de chiffres et on recompose 27 chiffres.
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

    joined = "\n".join(lines)
    m = re.search(r"\b(CHF|EUR)\b", joined, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


AMOUNT_LABELS = ["montant", "betrag", "amount", "importo"]


def parse_amount(raw: str):
    cleaned = re.sub(r"[^\d,\.']", "", raw)
    cleaned = cleaned.replace(" ", "").replace("'", "").replace("’", "")
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

    for i, line in enumerate(lines):
        low = line.lower()
        if any(lbl in low for lbl in AMOUNT_LABELS):
            zone = [line] + lines[i + 1 : i + 4]
            for l in zone:
                for m in re.finditer(r"[0-9][0-9\s.,'-]*\d", l):
                    val = parse_amount(m.group(0))
                    if val is not None:
                        candidates.append(val)
            break

    if not candidates:
        joined = "\n".join(lines)
        for m in re.finditer(r"[0-9][0-9\s']*[.,]\d{2}", joined):
            val = parse_amount(m.group(0))
            if val is not None:
                candidates.append(val)

    return max(candidates) if candidates else None


def extract_bv_fields_from_qr(img_rgb: np.ndarray):
    """
    Essaie d'extraire (reference, currency, amount) à partir du contenu du QR.
    Retourne (ref, curr, amount) ou (None, None, None).
    """
    data = decode_qr_data(img_rgb)
    if not data:
        return None, None, None

    txt = data.replace("\r", "\n")

    ref = extract_reference_from_text(txt)

    m_curr = re.search(r"\b(CHF|EUR)\b", txt, flags=re.IGNORECASE)
    curr = m_curr.group(1).upper() if m_curr else None

    candidates = []
    for m in re.finditer(r"[0-9][0-9\s']*[.,]\d{2}", txt):
        val = parse_amount(m.group(0))
        if val is not None:
            candidates.append(val)
    amount = max(candidates) if candidates else None

    return ref, curr, amount


def extract_bv_fields(page_text: str, img_rgb: np.ndarray):
    """
    Extraction complète BV pour une page déjà identifiée comme BV :
      - on essaie d'abord via le QR (très robuste)
      - ensuite fallback via le texte OCR (Monnaie/Montant/Référence imprimés).
    """
    ref_qr, curr_qr, amt_qr = extract_bv_fields_from_qr(img_rgb)

    if amt_qr is not None:
        return ref_qr, curr_qr, amt_qr

    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in page_text.splitlines()]

    reference = extract_reference_from_text(page_text)
    currency = extract_currency_from_lines(lines)
    amount = extract_amount_from_lines(lines)

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

    1) Si une page BV est trouvée : on lit les champs BV (QR + texte).
    2) Sinon : parseur générique sur tout le texte de la facture (ex: 'CHF 330.-').
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

    # Aucun BV : pas de référence, on parse dans le texte général
    val, curr = _parse_amount_currency_from_text(full_text)
    return val, curr, False, None


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

use_hf_ocr = st.checkbox(
    "Utiliser OCR HuggingFace (TrOCR) si disponible",
    value=HF_AVAILABLE,
)

uploaded_pdf = st.file_uploader("📄 Dépose un PDF multipages", type=["pdf"])

# Chargement éventuel du pipeline HF
hf_ocr_pipe = None
if use_hf_ocr:
    if HF_AVAILABLE:
        try:
            hf_ocr_pipe = load_hf_ocr_pipeline()
            st.success(f"OCR HuggingFace prêt (modèle : {HF_OCR_MODEL_NAME}) ✅")
        except Exception as e:
            st.error(f"Erreur chargement OCR HuggingFace : {e}")
            hf_ocr_pipe = None
    else:
        st.warning("HuggingFace OCR indisponible (install 'transformers', 'torch', etc.).")


# ------------------------------------------------------------
# Pipeline principal
# ------------------------------------------------------------

if uploaded_pdf:
    tmp_dir = Path("data/tmp_pred_yolo")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / uploaded_pdf.name
    pdf_path.write_bytes(uploaded_pdf.getbuffer())
    st.info(f"PDF enregistré : {pdf_path.resolve()}")

    try:
        model = load_yolo_model(model_path_text)
        st.success("Modèle YOLO chargé ✅")
    except Exception as e:
        st.error(f"Impossible de charger le modèle : {e}")
        st.stop()

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
    page_texts = []

    for i in range(n_pages):
        img_rgb = render_pdf_page_to_rgb(pdf_path, page_index=i, dpi=dpi)
        page_images.append(img_rgb)

        text_ocr = ocr_page_from_image(
            img_rgb,
            use_hf=use_hf_ocr,
            hf_pipeline=hf_ocr_pipe,
        )
        page_texts.append(text_ocr)

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

    # Découpage factures par tampons
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

    # Montant / Monnaie / BV par facture
    facture_montant_map = {}
    facture_monnaie_map = {}
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

    # Affichage résultats
    st.subheader("📊 Résultats par page")
    st.dataframe(df, use_container_width=True)

    st.subheader("🧾 Factures détectées (basées sur les tampons)")
    st.metric("Nombre de factures détectées", nb_factures)
    if not df_factures.empty:
        st.dataframe(df_factures, use_container_width=True)
    else:
        st.info("Aucune facture détectée (aucun tampon).")

    st.download_button(
        "📥 Télécharger les résultats par page (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        mime="text/csv",
        file_name=f"yolo_tampons_factures_v14_{pdf_path.stem}.csv",
    )

    if not df_factures.empty:
        st.download_button(
            "📥 Télécharger le résumé par facture (CSV)",
            data=df_factures.to_csv(index=False).encode("utf-8"),
            mime="text/csv",
            file_name=f"factures_resume_v14_{pdf_path.stem}.csv",
        )

    with st.expander("🔍 Debug : afficher le texte OCR d'une page"):
        page_num_debug = st.number_input(
            "Page à afficher (1 à n_pages)", min_value=1, max_value=n_pages, value=1
        )
        if st.button("Afficher le texte OCR de cette page"):
            text_dbg = page_texts[page_num_debug - 1]
            st.text(text_dbg if text_dbg.strip() else "[Texte OCR vide]")

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
