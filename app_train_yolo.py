from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

# Pour générer le dataset depuis l'interface
try:
    from generate_tampon_dataset import main as gen_dataset_main
except ImportError:
    gen_dataset_main = None


# ---------------- CONFIG DE BASE ----------------
BASE_DIR = Path(".").resolve()
DATA_DIR = BASE_DIR / "data"
YOLO_DATA_DIR = DATA_DIR / "yolo_tampons"
RUNS_DIR = BASE_DIR / "runs_tampon"
DEFAULT_YAML = BASE_DIR / "tampons.yaml"


# ----------------- PAGE CONFIG ------------------
st.set_page_config(
    page_title="Entraînement YOLO – Tampons",
    layout="wide",
)
st.title("🧠 Entraînement & Ré-entraînement YOLO – Détection de tampons")

st.markdown(
    """
Cette interface permet de :

1. **Générer / régénérer** le dataset YOLO à partir de `pages_raw` + `tampons_png`
2. **Lancer un entraînement YOLOv8n / YOLOv8s** (avec GPU si disponible)
3. **Suivre l'état d'un run** (courbes de loss, mAP, etc.)

Les chemins supposés :

- Dataset YOLO : `data/yolo_tampons`  
- Config : `tampons.yaml`  
- Runs : `runs_tampon/...`
"""
)

# Init session_state
for k in ["last_run_path", "last_results"]:
    if k not in st.session_state:
        st.session_state[k] = None


# ----------------- SIDEBAR PARAMÈTRES ------------------
st.sidebar.header("Paramètres d'entraînement")

# Modèle de base
variant = st.sidebar.selectbox(
    "Variant YOLO",
    options=["n (nano – très rapide)", "s (small – plus précis)"],
    index=1,
)
MODEL_VARIANT = "n" if variant.startswith("n") else "s"
base_model = f"yolov8{MODEL_VARIANT}.pt"

# Dataset
data_yaml = st.sidebar.text_input(
    "Chemin du fichier data YAML",
    value=str(DEFAULT_YAML),
)

# Hyperparams
epochs = st.sidebar.slider("Epochs", 10, 200, 50, 5)
imgsz = st.sidebar.select_slider("Taille images (imgsz)", options=[416, 512, 640, 800, 960], value=640)
batch = st.sidebar.slider("Batch size", 2, 32, 8, 2)
patience = st.sidebar.slider("Patience (early stop)", 3, 30, 10, 1)

# Device
if torch.cuda.is_available():
    dev_choice = st.sidebar.selectbox("Device", ["auto (GPU si dispo)", "cuda:0 (forcer GPU)", "cpu"], index=0)
else:
    dev_choice = st.sidebar.selectbox("Device", ["cpu"], index=0)

if dev_choice.startswith("cuda"):
    device = dev_choice
elif dev_choice.startswith("cpu"):
    device = "cpu"
else:
    device = 0 if torch.cuda.is_available() else "cpu"


st.sidebar.markdown("---")
st.sidebar.write("Dossiers utilisés :")
st.sidebar.code(
    f"DATA_DIR  = {DATA_DIR}\nYOLO_DIR  = {YOLO_DATA_DIR}\nRUNS_DIR  = {RUNS_DIR}",
    language="bash",
)


# ----------------- 1. GÉNÉRATION DATASET ------------------
st.header("1️⃣ Génération / régénération du dataset YOLO")

col_g1, col_g2 = st.columns([1, 2])

with col_g1:
    if gen_dataset_main is None:
        st.error("⚠️ `generate_tampon_dataset.py` introuvable ou sans `main()`. "
                 "Vérifie que le fichier est dans le même dossier et contient une fonction main().")
    else:
        gen_btn = st.button("📦 Générer / régénérer le dataset", type="primary")

        if gen_btn:
            with st.spinner("Génération du dataset en cours…"):
                try:
                    gen_dataset_main()
                    st.success(f"Dataset YOLO généré dans : {YOLO_DATA_DIR}")
                except Exception as e:
                    st.error(f"Erreur pendant la génération : {e}")

with col_g2:
    # Petit résumé du contenu actuel
    train_imgs = list((YOLO_DATA_DIR / "images" / "train").glob("*.jpg"))
    val_imgs = list((YOLO_DATA_DIR / "images" / "val").glob("*.jpg"))
    st.write("Contenu actuel du dataset :")
    st.write(f"- Train : **{len(train_imgs)}** images")
    st.write(f"- Val   : **{len(val_imgs)}** images")

    if train_imgs:
        st.caption(f"Exemple chemin train : {train_imgs[0]}")
    if val_imgs:
        st.caption(f"Exemple chemin val   : {val_imgs[0]}")


# ----------------- 2. LANCER L'ENTRAÎNEMENT ------------------
st.header("2️⃣ Lancement d'un nouvel entraînement YOLO")

train_col1, train_col2 = st.columns([1, 3])

with train_col1:
    run_prefix = st.text_input("Préfixe du run", value="ui")
    start_train = st.button("🚀 Lancer l'entraînement / ré-entraînement", type="primary")

with train_col2:
    st.write("Résumé de la config :")
    st.code(
        f"Modèle   : {base_model}\n"
        f"Data     : {data_yaml}\n"
        f"Epochs   : {epochs}\n"
        f"ImgSize  : {imgsz}\n"
        f"Batch    : {batch}\n"
        f"Patience : {patience}\n"
        f"Device   : {device}",
        language="bash",
    )

if start_train:
    data_path = Path(data_yaml)
    if not data_path.exists():
        st.error(f"Fichier data YAML introuvable : {data_path}")
    else:
        # Nom unique du run
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{run_prefix}_yolov8{MODEL_VARIANT}_{ts}"

        with st.spinner(f"Entraînement YOLOv8{MODEL_VARIANT} en cours… (run: {run_name})"):
            try:
                model = YOLO(base_model)

                results = model.train(
                    data=str(data_path),
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch,
                    patience=patience,
                    project=str(RUNS_DIR),
                    name=run_name,
                    device=device,
                    verbose=True,
                )

                run_path = RUNS_DIR / run_name
                st.session_state["last_run_path"] = str(run_path)

                st.success(f"✅ Entraînement terminé. Run : {run_path}")

            except Exception as e:
                st.error(f"Erreur pendant l'entraînement : {e}")


# ----------------- 3. SUIVI D'UN RUN EXISTANT ------------------
st.header("3️⃣ Suivi d'un run existant")

# liste des runs
RUNS_DIR.mkdir(exist_ok=True, parents=True)
runs = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()])

if not runs:
    st.info("Aucun run trouvé dans `runs_tampon`. Lance un premier entraînement.")
else:
    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        default_index = 0
        if st.session_state["last_run_path"]:
            for i, rp in enumerate(runs):
                if str(rp) == st.session_state["last_run_path"]:
                    default_index = i
                    break

        selected_run = st.selectbox(
            "Choisir un run",
            options=runs,
            index=default_index,
            format_func=lambda p: p.name,
        )

    with col_r2:
        refresh_metrics = st.button("🔄 Rafraîchir les métriques")

    if refresh_metrics:
        results_csv = selected_run / "results.csv"
        if not results_csv.exists():
            st.error(f"`results.csv` introuvable pour ce run : {results_csv}")
        else:
            try:
                df = pd.read_csv(results_csv)
                st.session_state["last_results"] = df
            except Exception as e:
                st.error(f"Erreur lors de la lecture de results.csv : {e}")

    df_res = st.session_state.get("last_results")

    if df_res is not None:
        st.subheader(f"Métriques – {selected_run.name}")
        st.dataframe(df_res.tail(), use_container_width=True)

        # Courbes de loss et mAP
        metrics_to_plot = ["train/box_loss", "train/cls_loss", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
        metrics_existing = [m for m in metrics_to_plot if m in df_res.columns]

        if metrics_existing:
            fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
            for m in metrics_existing:
                ax.plot(df_res.index, df_res[m], label=m)
            ax.set_xlabel("epoch")
            ax.set_title("Courbes d'entraînement")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("Aucune colonne standard (loss/mAP) trouvée dans results.csv.")

        # Chemin du meilleur modèle
        best_pt = selected_run / "weights" / "best.pt"
        if best_pt.exists():
            st.success(f"Modèle best.pt : {best_pt.resolve()}")
        else:
            st.warning("best.pt introuvable dans ce run.")
    else:
        st.info("Clique sur 🔄 *Rafraîchir les métriques* pour voir les résultats d'un run.")
