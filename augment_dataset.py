from __future__ import annotations
from pathlib import Path
import random

import numpy as np
import pandas as pd
import cv2
import fitz  # PyMuPDF
from PIL import Image


# ---------- CONFIG ----------
DATA_DIR = Path("data")
PAGES_DIR = DATA_DIR / "pages"
CSV_PATH = DATA_DIR / "tableau_page.csv"
CSV_OUT = DATA_DIR / "tableau_page_augmented.csv"

N_AUG_PER_PAGE = 10      # 👈 nombre de copies par page
DPI_RENDER = 200         # cohérent avec ton apprentissage


# ---------- RENDU / AUGMENTATION ----------
def render_pdf_to_bgr(pdf_path: Path, dpi: int = 200) -> np.ndarray:
    """Rend la première page du PDF en image BGR (OpenCV)."""
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(0)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)  # RGB
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def random_augmentation(img: np.ndarray) -> np.ndarray:
    """Applique une combinaison aléatoire de petites augmentations réalistes."""
    h, w = img.shape[:2]
    aug = img.copy()

    # 1) Rotation légère
    angle = random.uniform(-4, 4)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 2) Luminosité / contraste
    alpha = random.uniform(0.9, 1.1)  # contraste
    beta = random.uniform(-15, 15)    # luminosité
    aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    # 3) Légère blur ou sharpen
    if random.random() < 0.5:
        aug = cv2.GaussianBlur(aug, (3, 3), 0)
    else:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        aug = cv2.filter2D(aug, -1, kernel)

    # 4) Un peu de bruit gaussien
    if random.random() < 0.7:
        noise = np.random.normal(0, 5, aug.shape).astype(np.float32)
        aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return aug


def save_bgr_as_pdf(img_bgr: np.ndarray, out_path: Path) -> None:
    """Enregistre une image BGR en PDF 1 page."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img.save(out_path, "PDF")


# ---------- NUMÉROTATION ----------
def parse_page_num(nom: str) -> int:
    """Extrait le numéro à partir de 'page_XXX'."""
    try:
        return int(nom.split("_")[-1])
    except Exception:
        raise ValueError(f"Nom de page inattendu: {nom}")


def format_page_name(num: int) -> str:
    """Formate le nom de page suivant: page_001, ..., page_999, page_1000, etc."""
    if num < 1000:
        return f"page_{num:03d}"
    else:
        return f"page_{num:04d}"


# ---------- PROGRAMME PRINCIPAL ----------
def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV introuvable: {CSV_PATH}")
    if not PAGES_DIR.exists():
        raise FileNotFoundError(f"Dossier pages introuvable: {PAGES_DIR}")

    df = pd.read_csv(CSV_PATH)
    if "nom" not in df.columns:
        raise ValueError("Le CSV doit contenir la colonne 'nom'.")
    if "tampon" not in df.columns and "sans" not in df.columns:
        raise ValueError("Le CSV doit contenir 'tampon' ou 'sans'.")

    # Déterminer le max existant
    nums = df["nom"].astype(str).apply(parse_page_num)
    current_max = int(nums.max())
    next_num = current_max + 1

    print(f"Pages existantes jusqu'à: {current_max} → prochaine nouvelle page: {next_num}")

    new_rows = []

    for idx, row in df.iterrows():
        nom = str(row["nom"]).strip()
        if "tampon" in row:
            tampon = int(row["tampon"])
            sans = int(row["sans"]) if "sans" in row else int(1 - tampon)
        else:
            sans = int(row["sans"])
            tampon = int(1 - sans)

        pdf_path = PAGES_DIR / f"{nom}.pdf"
        if not pdf_path.exists():
            print(f"[WARN] PDF manquant pour {nom}, on saute.")
            continue

        try:
            img = render_pdf_to_bgr(pdf_path, dpi=DPI_RENDER)
        except Exception as e:
            print(f"[ERR] Impossible de rendre {pdf_path}: {e}")
            continue

        for k in range(N_AUG_PER_PAGE):
            aug_img = random_augmentation(img)
            new_name = format_page_name(next_num)
            out_pdf = PAGES_DIR / f"{new_name}.pdf"
            save_bgr_as_pdf(aug_img, out_pdf)

            new_rows.append({
                "nom": new_name,
                "tampon": tampon,
                "sans": sans,
            })

            print(f"Créé: {out_pdf} (tampon={tampon}, sans={sans})")
            next_num += 1

    # Construire le nouveau CSV
    df_new = pd.DataFrame(new_rows)
    df_all = pd.concat([df, df_new], ignore_index=True)

    df_all.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"\n✅ Terminé. Nouveau CSV écrit dans: {CSV_OUT}")
    print(f"   Lignes originales: {len(df)}")
    print(f"   Nouvelles lignes : {len(df_new)}")
    print(f"   Total            : {len(df_all)}")


if __name__ == "__main__":
    main()
