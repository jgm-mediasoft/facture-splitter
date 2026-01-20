# app_train_tampons_v3.py
from __future__ import annotations

import os
import re
import sys
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import streamlit as st


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def safe_name(s: str, maxlen: int = 80) -> str:
    s = (s or "").strip()
    if not s:
        s = "run"
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)
    return s[:maxlen]


def open_in_file_explorer(path: Path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass


def parse_yaml_basic(yaml_path: Path) -> dict:
    """
    Parser simple pour YAML YOLO (sans dépendance).
    Sert à afficher quelques champs.
    """
    info: dict = {}
    try:
        text = yaml_path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if ":" in s:
                k, v = s.split(":", 1)
                info[k.strip()] = v.strip().strip("'\"")
    except Exception:
        return {}
    return info


def detect_gpu_lines() -> list[str]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            return []
        return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def run_paths(project: str, name: str) -> tuple[Path, Path, Path]:
    run_dir = Path(project).resolve() / name
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    return run_dir, weights_dir, best_pt


def tail_text(s: str, limit: int = 18000) -> str:
    if len(s) <= limit:
        return s
    return s[-limit:]


def build_train_cmd(data_yaml: Path, args: dict) -> tuple[list[str], str]:
    """
    Mode 1 (priorité): `yolo detect train ...`
    Mode 2 (fallback): `python -c "from ultralytics import YOLO; ..."`
    """
    yolo_exe = shutil.which("yolo")

    common = [
        f"model={args['model']}",
        f"data={str(data_yaml)}",
        f"epochs={args['epochs']}",
        f"imgsz={args['imgsz']}",
        f"batch={args['batch']}",
        f"patience={args['patience']}",
        f"device={args['device']}",
        f"workers={args['workers']}",
        f"project={args['project']}",
        f"name={args['name']}",
    ]

    if args.get("cache"):
        common.append("cache=True")
    if args.get("amp"):
        common.append("amp=True")
    if args.get("resume"):
        common.append("resume=True")
    if args.get("plots"):
        common.append("plots=True")
    if args.get("verbose"):
        common.append("verbose=True")
    if args.get("cos_lr"):
        common.append("cos_lr=True")
    if args.get("close_mosaic") is not None:
        common.append(f"close_mosaic={args['close_mosaic']}")
    if args.get("seed") is not None:
        common.append(f"seed={args['seed']}")
    if args.get("val") is not None:
        common.append(f"val={str(args['val'])}")

    if yolo_exe:
        return [yolo_exe, "detect", "train", *common], "yolo"

    # Fallback python -c (évite python -m ultralytics)
    code = f"""
from ultralytics import YOLO
model = YOLO(r\"\"\"{args['model']}\"\"\")
model.train(
    data=r\"\"\"{str(data_yaml)}\"\"\",
    epochs={int(args['epochs'])},
    imgsz={int(args['imgsz'])},
    batch={int(args['batch'])},
    patience={int(args['patience'])},
    device=r\"\"\"{args['device']}\"\"\",
    workers={int(args['workers'])},
    project=r\"\"\"{args['project']}\"\"\",
    name=r\"\"\"{args['name']}\"\"\",
    cache={bool(args.get('cache'))},
    amp={bool(args.get('amp'))},
    resume={bool(args.get('resume'))},
    plots={bool(args.get('plots'))},
    verbose={bool(args.get('verbose'))},
    cos_lr={bool(args.get('cos_lr'))},
    close_mosaic={int(args.get('close_mosaic') or 0)},
    seed={int(args.get('seed') or 0)},
    val={bool(args.get('val'))},
)
"""
    return [sys.executable, "-c", code], "python-fallback"


def popen_with_safe_text(cmd: list[str]) -> subprocess.Popen:
    """
    Important Windows: éviter UnicodeDecodeError.
    On force un encodage et on met errors=replace.
    """
    # utf-8 marche souvent, errors=replace évite crash même si mix encodings
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        universal_newlines=True,
    )


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="YOLOv8 - Entraînement Tampons (v3)", layout="wide")
st.title("🧪 YOLOv8 — Entraînement Tampons (v3)")
st.caption("Robuste Windows (pas d'UnicodeDecodeError), logs temps réel, best.pt du run courant.")

BASE_DIR = Path(".").resolve()
DEFAULT_YAML = BASE_DIR / "tampons.yaml"

# Session state
if "log" not in st.session_state:
    st.session_state.log = ""
if "last_cmd" not in st.session_state:
    st.session_state.last_cmd = None
if "last_mode" not in st.session_state:
    st.session_state.last_mode = None
if "last_run_dir" not in st.session_state:
    st.session_state.last_run_dir = None


def add_log(s: str):
    st.session_state.log += s
    if len(st.session_state.log) > 400_000:
        st.session_state.log = st.session_state.log[-400_000:]


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Dataset")
    yaml_path_str = st.text_input("tampons.yaml", value=str(DEFAULT_YAML))
    data_yaml = Path(yaml_path_str).expanduser().resolve()

    st.subheader("Modèle")
    model_sel = st.selectbox(
        "Modèle de départ",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt", "Custom (.pt)"],
        index=1,
    )
    if model_sel == "Custom (.pt)":
        model_path = st.text_input("Chemin modèle .pt", value="yolov8n.pt").strip()
    else:
        model_path = model_sel

    st.subheader("Paramètres")
    epochs = st.slider("epochs", 1, 400, 50)
    imgsz = st.selectbox("imgsz", [320, 416, 512, 640, 768, 960, 1280], index=3)
    batch = st.selectbox("batch", [1, 2, 4, 8, 16, 32, 64], index=3)
    patience = st.slider("patience", 1, 100, 10)

    st.subheader("Device")
    gpus = detect_gpu_lines()
    if gpus:
        st.success("GPU détecté:\n- " + "\n- ".join(gpus))
        device = st.text_input("device (ex: 0 ou 0,1 ou cpu)", value="0")
    else:
        st.warning("GPU non détecté via nvidia-smi → CPU conseillé.")
        device = st.text_input("device", value="cpu")

    with st.expander("🔧 Avancé", expanded=False):
        workers = st.number_input("workers", min_value=0, max_value=32, value=4, step=1)
        cache = st.checkbox("cache", value=False)
        amp = st.checkbox("amp (mixed precision)", value=True)
        resume = st.checkbox("resume", value=False)

        cos_lr = st.checkbox("cos_lr", value=False)
        close_mosaic = st.number_input("close_mosaic", min_value=0, max_value=50, value=10, step=1)
        seed = st.number_input("seed", min_value=0, max_value=999999, value=0, step=1)

        val = st.checkbox("val", value=True)
        plots = st.checkbox("plots", value=True)
        verbose = st.checkbox("verbose", value=True)

        project = st.text_input("project", value="runs_tampon").strip() or "runs_tampon"
        default_name = f"ui_{Path(model_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        name = safe_name(st.text_input("name", value=default_name))

    st.divider()
    start_btn = st.button("🚀 Lancer l'entraînement", type="primary", use_container_width=True)
    st.caption("Stop : fermer Streamlit ou CTRL+C dans le terminal.")


# Main panels
left, right = st.columns([1, 1])

with left:
    st.subheader("📄 Aperçu YAML")
    if data_yaml.exists():
        st.code(data_yaml.read_text(encoding="utf-8", errors="ignore")[:4000], language="yaml")
        info = parse_yaml_basic(data_yaml)
        if info:
            st.json(info)
    else:
        st.error(f"Fichier introuvable : {data_yaml}")

with right:
    st.subheader("📦 Run courant")
    run_dir, weights_dir, best_pt = run_paths(project, name)

    st.write("**project :**", project)
    st.write("**name :**", name)
    st.write("**run_dir :**", str(run_dir))

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📂 Ouvrir run_dir"):
            open_in_file_explorer(run_dir)
    with c2:
        if st.button("📂 Ouvrir weights_dir"):
            open_in_file_explorer(weights_dir)

    if best_pt.exists():
        st.success(f"best.pt (run courant) : {best_pt}")
    else:
        st.info("best.pt pas encore créé (normal avant la fin).")

st.divider()
st.subheader("🖥️ Logs entraînement (temps réel)")
log_box = st.empty()
status_box = st.empty()

# Training
if start_btn:
    if not data_yaml.exists():
        st.error("❌ tampons.yaml introuvable. Corrige le chemin.")
        st.stop()

    args = dict(
        model=model_path,
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        patience=int(patience),
        device=str(device).strip(),
        workers=int(workers),
        cache=bool(cache),
        amp=bool(amp),
        resume=bool(resume),
        cos_lr=bool(cos_lr),
        close_mosaic=int(close_mosaic),
        seed=int(seed),
        val=bool(val),
        plots=bool(plots),
        verbose=bool(verbose),
        project=str(project),
        name=str(name),
    )

    cmd, mode = build_train_cmd(data_yaml=data_yaml, args=args)

    st.session_state.last_cmd = cmd
    st.session_state.last_mode = mode
    st.session_state.last_run_dir = str(run_dir)

    st.session_state.log = ""
    add_log("=== Mode ===\n" + mode + "\n\n")
    add_log("=== Commande ===\n" + " ".join(cmd) + "\n\n")

    status_box.info("Démarrage entraînement…")
    log_box.code(tail_text(st.session_state.log), language="text")

    try:
        proc = popen_with_safe_text(cmd)
    except Exception as e:
        status_box.error(f"Impossible de lancer l'entraînement : {e}")
        st.stop()

    assert proc.stdout is not None

    while True:
        line = proc.stdout.readline()
        if line:
            add_log(line)
            log_box.code(tail_text(st.session_state.log), language="text")

        if proc.poll() is not None:
            break

        time.sleep(0.02)

    rc = proc.returncode

    if rc == 0:
        status_box.success("✅ Entraînement terminé avec succès.")
    else:
        status_box.error(f"❌ Entraînement terminé avec erreur (code={rc}).")

    # Vérifier best.pt DU RUN COURANT
    _, _, best_pt2 = run_paths(project, name)
    if best_pt2.exists():
        st.success(f"✅ Modèle (run courant) : {best_pt2}")
    else:
        st.warning("Je ne trouve pas `best.pt` pour le run courant. Regarde les logs.")

# Logs persistants
if st.session_state.log and not start_btn:
    log_box.code(tail_text(st.session_state.log), language="text")

# Dernière commande
if st.session_state.last_cmd:
    with st.expander("🔎 Dernière commande lancée"):
        st.write("**Mode :**", st.session_state.last_mode)
        st.code(" ".join(st.session_state.last_cmd), language="bash")
        st.write("**Dernier run_dir :**", st.session_state.last_run_dir)

