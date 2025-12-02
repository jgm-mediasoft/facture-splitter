# Architecture du projet `facture_splitter`

## 1. Vue d’ensemble

Le projet **facture_splitter** permet de traiter des PDF de factures afin de :

1. Détecter les **tampons** (cachets) sur chaque page avec YOLO.
2. Déduire les **factures** (groupes de pages) à partir des tampons.
3. Détecter la présence d’un **Bulletin de Versement (BV)**.
4. Lire le **QR-code** du BV et extraire :
   - la **référence**,
   - le **montant**,
   - la **devise**.
5. Exposer ces résultats via des **applications Streamlit**.

L’architecture est organisée autour de :
- scripts **d’inférence** (Streamlit),
- scripts **d’apprentissage** YOLO,
- utilitaires de génération de datasets,
- et d’un socle commun de dépendances (requirements + modèles).

---

## 2. Composants principaux

### 2.1. Applications Streamlit

- **`app_prediction_yolo_v20.py`**  
  Application principale de **détection de tampons** et de découpe des factures :
  - charge un modèle YOLO,
  - rend les pages PDF en images,
  - détecte les tampons,
  - regroupe les pages en factures,
  - affiche un tableau (pages, factures).

- **`app_bv_only.py`**  
  Application focalisée sur les **BV** :
  - détection du QR-code,
  - OCR / parsing pour extraire :
    - Référence,
    - Monnaie,
    - Montant,
  - indique quelles pages et quelles factures contiennent un BV.

- **`app_apprentissage_ML.py`**  
  Application pour l’**apprentissage / expérimentation** (à adapter selon les besoins) :
  - tests de modèles,
  - visualisation de datasets,
  - préparation de futurs modèles ML.

---

### 2.2. Scripts d’apprentissage YOLO

- **`train_yolo_tampon.py`**  
  Script principal d’**entraînement YOLO** pour la détection de tampons :
  - lit la configuration `tampons.yaml`,
  - lance l’entraînement (Ultralytics),
  - enregistre les modèles dans `runs/` ou `runs_tampon/`.

- **`app_train_yolo.py`**  
  Interface / script d’**entraînement interactif** :
  - peut être utilisé pour lancer des entraînements avec des paramètres sélectionnés (chemin modèle, epochs, batch size, etc.).

- **`print_model_architecture.py`**  
  Utilitaire pour afficher l’architecture du modèle YOLO (couches, nombres de paramètres, etc.).

---

### 2.3. Génération et augmentation de données

- **`generate_tampon_dataset.py`**  
  Génération ou préparation de datasets spécifiques pour la détection de **tampons**.

- **`augment_dataset.py`**  
  Scripts d’**augmentation** (rotation, bruit, etc.) pour améliorer la robustesse du modèle.

---

## 3. Flux de traitement d’un PDF

### 3.1. Détection tampons + factures (`app_prediction_yolo_v20.py`)

1. L’utilisateur dépose un **PDF multipages**.
2. Chaque page est :
   - rendue en image RGB (`PyMuPDF` + `numpy`),
   - passée dans le modèle **YOLO**.
3. Pour chaque page :
   - si un tampon est détecté → début d’une nouvelle facture,
   - les pages suivantes sans tampon appartiennent à la même facture,
   - jusqu’au prochain tampon.
4. Un **DataFrame** par page est construit (page, tampon, proba, facture_index).
5. Un **résumé par facture** est construit (page de début, page de fin, nb de pages, etc.).
6. Les résultats sont affichés et exportables (CSV).

### 3.2. Détection des BV (`app_bv_only.py`)

1. L’utilisateur dépose un PDF.
2. Pour chaque page :
   - rendu en image,
   - recherche d’un **QR-code** (OpenCV),
   - si présent :
     - décodage du contenu,
     - extraction de référence / montant / devise,
     - marquage de la page comme **BV**.
3. Par facture (groupement de pages basé sur les tampons) :
   - la dernière page contenant un BV est considérée comme **page BV** de la facture.
4. Résultats :
   - tableau par page (BV ou non),
   - tableau par facture (contient un BV ?, référence BV, montant, devise).

---

## 4. Structure des dossiers (vue simplifiée)

```text
facture_splitter/
│
├── app_apprentissage_ML.py
├── app_bv_only.py
├── app_prediction_yolo_v20.py
├── app_train_yolo.py
├── augment_dataset.py
├── generate_tampon_dataset.py
├── train_yolo_tampon.py
├── print_model_architecture.py
│
├── models/                # Scripts ou configs liés aux modèles
├── invoice_splitter/      # (optionnel) logique de découpe avancée
│
├── runs/                  # sorties d’entraînement YOLO (ignoré par git)
├── runs_tampon/           # modèles tampon (ignoré par git)
├── data/                  # datasets / PDF (ignoré par git)
│
├── requirements.txt
├── .gitignore
├── README.md
├── CONTRIBUTING.md
└── docs/
    ├── architecture.md
    ├── guide_azeem.md
    └── git-workflow.png
