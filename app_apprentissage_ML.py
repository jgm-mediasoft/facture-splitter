
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from invoice_splitter.learner import (
    train_and_evaluate,
    save_model,
    load_model,
    load_labeled_dataset,
)

st.set_page_config(page_title="Apprentissage ML – Classification Tampon", layout="wide")
st.title("📚 Apprentissage ML – Classification des pages (tampon vs sans)")
st.markdown(
    """
- **CSV** : colonnes `nom`, `tampon` (1/0) et/ou `sans` (0/1)  
- **Pages** : fichiers `nom.pdf` (ex: `page_001.pdf`)  

Split **stratifié**, rééquilibrage via `class_weight="balanced"`.  
DPI par défaut : **300** (plus de détails pour détecter les tampons).
"""
)

DATA_DIR = Path("data")
PAGES_DIR = DATA_DIR / "pages"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PAGES_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1. Upload CSV + PDF
# ------------------------------------------------------------------
with st.expander("📤 Importer les données (CSV + PDF)", expanded=False):
    tab_csv, tab_pdf = st.tabs(["CSV", "Pages PDF"])

    # ---- CSV ----
    with tab_csv:
        up_csv = st.file_uploader(
            "CSV d’étiquettes (tableau_page.csv)",
            type=["csv"],
            key="upload_csv_file",
        )
        col_c1, col_c2 = st.columns(2)
        if col_c1.button("💾 Déposer le CSV dans ./data/"):
            if up_csv:
                p = DATA_DIR / "tableau_page.csv"
                p.write_bytes(up_csv.getbuffer())
                st.success(f"CSV enregistré ➜ {p.resolve()}")
                st.session_state["uploaded_csv_path"] = str(p.resolve())
            else:
                st.warning("Sélectionne un fichier CSV.")
        if col_c2.button("🗑️ Supprimer le CSV existant"):
            p = DATA_DIR / "tableau_page.csv"
            if p.exists():
                p.unlink()
                st.success("CSV supprimé.")
            else:
                st.info("Aucun CSV à supprimer.")

    # ---- PDF ----
    with tab_pdf:
        sub_tab_multi, sub_tab_zip = st.tabs(["Multi-PDF", "ZIP"])

        # Multi PDF
        with sub_tab_multi:
            uploaded_pdfs = st.file_uploader(
                "Dépose toutes tes pages PDF",
                type=["pdf"],
                accept_multiple_files=True,
            )
            col_p1, col_p2 = st.columns(2)
            clear_before = col_p1.checkbox(
                "Vider './data/pages' avant de copier", value=False
            )
            if col_p2.button("📥 Copier les PDF dans ./data/pages"):
                if uploaded_pdfs:
                    if clear_before:
                        for old in PAGES_DIR.glob("*.pdf"):
                            try:
                                old.unlink()
                            except Exception:
                                pass
                    count = 0
                    for f in uploaded_pdfs:
                        (PAGES_DIR / f.name).write_bytes(f.getbuffer())
                        count += 1
                    st.success(f"{count} PDF sauvegardés ➜ {PAGES_DIR.resolve()}")
                    st.session_state["uploaded_pages_dir"] = str(PAGES_DIR.resolve())
                else:
                    st.warning("Sélectionne des PDF.")

        # ZIP
        with sub_tab_zip:
            up_zip = st.file_uploader(
                "ZIP des pages (page_xxx.pdf)",
                type=["zip"],
            )
            col_z1, col_z2 = st.columns(2)
            clear_before_zip = col_z1.checkbox(
                "Vider './data/pages' avant d'extraire (ZIP)", value=False
            )
            if col_z2.button("🧩 Extraire le ZIP dans ./data/pages"):
                if up_zip:
                    if clear_before_zip:
                        for old in PAGES_DIR.glob("*.pdf"):
                            try:
                                old.unlink()
                            except Exception:
                                pass
                    with zipfile.ZipFile(io.BytesIO(up_zip.getbuffer())) as zf:
                        zf.extractall(PAGES_DIR)
                    count = len(list(PAGES_DIR.glob("*.pdf")))
                    st.success(
                        f"ZIP extrait ➜ {PAGES_DIR.resolve()} ({count} PDF détectés)"
                    )
                    st.session_state["uploaded_pages_dir"] = str(PAGES_DIR.resolve())
                else:
                    st.warning("Sélectionne un fichier ZIP.")

    if st.button("➡️ Utiliser ces chemins plus bas"):
        st.session_state["pages_dir_str_prefill"] = st.session_state.get(
            "uploaded_pages_dir", str(PAGES_DIR.resolve())
        )
        st.session_state["csv_path_str_prefill"] = st.session_state.get(
            "uploaded_csv_path", str((DATA_DIR / "tableau_page.csv").resolve())
        )
        st.success("Champs paramétrés — descends aux **Paramètres**.")

# ------------------------------------------------------------------
# 2. Paramètres
# ------------------------------------------------------------------
st.header("1) Paramètres")

pages_dir_str = st.text_input(
    "Dossier des pages PDF",
    value=st.session_state.get("pages_dir_str_prefill", str(PAGES_DIR.resolve())),
)
csv_path_str = st.text_input(
    "Chemin du CSV",
    value=st.session_state.get(
        "csv_path_str_prefill", str((DATA_DIR / "tableau_page.csv").resolve())
    ),
)

pages_dir = Path(pages_dir_str.replace("\\", "/"))
csv_path = Path(csv_path_str.replace("\\", "/"))

colA, colB, colC = st.columns(3)
with colA:
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
with colB:
    random_state = st.number_input("Random state", 0, 42, step=1)
with colC:
    dpi = st.select_slider(
        "DPI rendu PDF (features)",
        options=[100, 150, 200, 250, 300],
        value=300,  # quick fix : plus de détails → meilleure détection
    )

st.caption(
    f"CSV: {csv_path.resolve()} — existe: {'✅' if csv_path.exists() else '❌'}"
)
st.caption(
    f"Pages: {pages_dir.resolve()} — existe: {'✅' if pages_dir.exists() else '❌'}"
)

# ------------------------------------------------------------------
# 3. Aperçu CSV
# ------------------------------------------------------------------
st.header("2) Aperçu du CSV")
df_preview, missing = None, []
try:
    df_preview = load_labeled_dataset(pages_dir, csv_path)
    st.dataframe(df_preview, width="stretch")
    st.success(f"{len(df_preview)} lignes chargées.")
    for _, r in df_preview.iterrows():
        if not Path(r["chemin_pdf"]).exists():
            missing.append(Path(r["chemin_pdf"]).name)
    if missing:
        st.warning(f"⚠️ {len(missing)} PDF manquant(s). Exemples: {missing[:10]}")
    else:
        st.info("✅ Tous les PDF référencés existent.")
except Exception as e:
    st.error(f"Chargement impossible: {e}")

# ------------------------------------------------------------------
# 4. Entraînement / Évaluation
# ------------------------------------------------------------------
st.header("3) Entraînement + Évaluation")
col_train, col_save, col_reload, col_export = st.columns(4)
with col_train:
    do_train = st.button("🚀 Lancer l'entraînement")
with col_save:
    do_save = st.button("💾 Sauvegarder le modèle")
with col_reload:
    do_reload = st.button("♻️ Recharger un modèle")
with col_export:
    do_export = st.button("📥 Exporter prédictions test (CSV)")

for key in ["trained_model", "df_pred", "metrics", "feature_info"]:
    st.session_state.setdefault(key, None)

if do_train:
    try:
        metrics, df_pred, model, feature_info, _ = train_and_evaluate(
            pages_dir=pages_dir,
            csv_path=csv_path,
            test_size=float(test_size),
            random_state=int(random_state),
            dpi=int(dpi),
        )
        st.session_state.update(
            {
                "trained_model": model,
                "df_pred": df_pred,
                "metrics": metrics,
                "feature_info": feature_info,
            }
        )

        st.success(
            f"✅ Accuracy: {metrics['accuracy']:.3f} • Train: {metrics['n_train']} • Test: {metrics['n_test']}"
        )

        # Matrice de confusion compacte
        st.subheader("Matrice de confusion")
        cm = metrics["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        im = ax.imshow(cm)
        ax.set_title("Confusion matrix", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["sans_tampon", "tampon"], fontsize=9)
        ax.set_yticklabels(["sans_tampon", "tampon"], fontsize=9)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center", fontsize=9)
        ax.set_xlabel("Prédit", fontsize=9)
        ax.set_ylabel("Réel", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Rapport de classification")
        st.code(metrics["report"])

        st.subheader("4) Prédictions sur le test set")
        st.dataframe(df_pred, width="stretch")

    except Exception as e:
        st.error(f"Erreur entraînement: {e}")

if do_save:
    if st.session_state["trained_model"] is None:
        st.warning("Entraîne d'abord un modèle.")
    else:
        save_model(
            st.session_state["trained_model"],
            st.session_state.get("feature_info", {}),
        )
        st.success("Modèle sauvegardé dans ./models (tampon_model.pkl + meta).")

if do_reload:
    try:
        model, meta = load_model()
        st.session_state["trained_model"] = model
        st.session_state["feature_info"] = meta
        st.success(f"Modèle rechargé. Meta: {meta}")
    except Exception as e:
        st.error(f"Reload impossible: {e}")

if do_export:
    dfp = st.session_state.get("df_pred")
    if dfp is None:
        st.warning("Aucune prédiction à exporter.")
    else:
        st.download_button(
            "Télécharger le CSV des prédictions test",
            data=dfp.to_csv(index=False).encode("utf-8"),
            mime="text/csv",
            file_name="predictions_test.csv",
        )
