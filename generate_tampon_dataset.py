
from __future__ import annotations
from pathlib import Path
import random

import numpy as np
import cv2
import fitz  # PyMuPDF

# ---------------- CONFIG ----------------
DATA_DIR = Path("data")
PAGES_RAW_DIR = DATA_DIR / "pages_raw"
TAMPONS_DIR = DATA_DIR / "tampons_png"
YOLO_DIR = DATA_DIR / "yolo_tampons"

N_AUG_PER_PAGE = 10
PROBA_METTRE_TAMPON = 0.7
DPI_RENDER = 200             # 🔥 baisse DPI pour réduire mémoire
MAX_SIZE = 1600              # 🔥 taille max en pixels (largeur ou hauteur)

random.seed(42)


# ---------------- PDF → RGB + resize ----------------
def render_pdf_to_rgb(pdf_path: Path, dpi: int = 200, max_size: int = 1600):
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(0)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)  # RGB

    h, w = img.shape[:2]

    # 🔥 resize si image trop grande
    scale = min(max_size / h, max_size / w, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


# ---------------- Charge tampons ----------------
def load_tampons():
    tampons = []
    for p in TAMPONS_DIR.glob("*.png"):
        png = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if png is None:
            continue
        if png.shape[2] == 4:
            bgr = png[:, :, :3]
            alpha = png[:, :, 3] / 255.0
        else:
            bgr = png
            alpha = np.ones(bgr.shape[:2], float)
        tampons.append((bgr, alpha))
    if not tampons:
        raise RuntimeError(f"Aucun fichier .png dans {TAMPONS_DIR}")
    return tampons


# ---------------- Colle tampon + bbox ----------------
def paste_tampon(base, tampon_bgr, alpha, x, y, scale=1.0):
    h, w = base.shape[:2]
    th, tw = tampon_bgr.shape[:2]

    th_s = max(8, int(th * scale))
    tw_s = max(8, int(tw * scale))

    tampon_resized = cv2.resize(tampon_bgr, (tw_s, th_s))
    alpha_resized = cv2.resize(alpha, (tw_s, th_s))

    x = max(0, min(x, w - tw_s))
    y = max(0, min(y, h - th_s))

    roi = base[y:y + th_s, x:x + tw_s]
    if roi.shape[:2] != (th_s, tw_s):
        return base, None

    alpha_3 = np.dstack([alpha_resized] * 3)
    blended = (alpha_3 * tampon_resized + (1 - alpha_3) * roi).astype(np.uint8)
    base[y:y + th_s, x:x + tw_s] = blended

    return base, (x, y, x + tw_s, y + th_s)


# ---------------- Convertit vers format YOLO ----------------
def bbox_to_yolo(xmin, ymin, xmax, ymax, w, h):
    x_c = (xmin + xmax) / 2.0 / w
    y_c = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return (x_c, y_c, bw, bh)


# ---------------- MAIN ----------------
def main():
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (YOLO_DIR / sub).mkdir(parents=True, exist_ok=True)

    tampons = load_tampons()
    pdfs = sorted(PAGES_RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError("Aucun PDF dans data/pages_raw")

    dataset = []
    img_id = 0

    for pdf in pdfs:
        print(f"Page: {pdf.name}")

        try:
            base = render_pdf_to_rgb(pdf, dpi=DPI_RENDER, max_size=MAX_SIZE)
        except Exception as e:
            print(f"  [ERREUR RENDU] {e}")
            continue

        h, w = base.shape[:2]

        for _ in range(N_AUG_PER_PAGE):
            img = base.copy()
            labels = []

            if random.random() < PROBA_METTRE_TAMPON:
                for _ in range(random.randint(1, 2)):
                    bgr, alpha = random.choice(tampons)
                    scale = random.uniform(0.4, 1.0)
                    x = random.randint(0, w - 8)
                    y = random.randint(0, h - 8)
                    img, bbox = paste_tampon(img, bgr, alpha, x, y, scale)
                    if bbox:
                        xmin, ymin, xmax, ymax = bbox
                        labels.append(bbox_to_yolo(xmin, ymin, xmax, ymax, w, h))

            dataset.append((img, labels))
            img_id += 1

    # --- SPLIT TRAIN / VAL ---
    random.shuffle(dataset)
    n = len(dataset)
    n_val = max(1, int(0.2 * n))
    val_set = dataset[:n_val]
    train_set = dataset[n_val:]

    def save_set(items, split):
        img_dir = YOLO_DIR / f"images/{split}"
        lbl_dir = YOLO_DIR / f"labels/{split}"
        for i, (img, labels) in enumerate(items):
            img_name = f"{split}_{i:06d}.jpg"
            cv2.imwrite(str(img_dir / img_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            with open(lbl_dir / img_name.replace(".jpg", ".txt"), "w") as f:
                for (x_c, y_c, bw, bh) in labels:
                    f.write(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

    save_set(train_set, "train")
    save_set(val_set, "val")

    print("✅ Dataset YOLO généré avec succès.")
    print(f"Train: {len(train_set)} images  • Val: {len(val_set)} images")


if __name__ == "__main__":
    main()
