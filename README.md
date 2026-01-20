# Facture Splitter – Détection de tampons, factures et BV

Ce projet permet d’analyser automatiquement des factures PDF afin de :
- détecter les **tampons** via YOLO,
- découper un PDF multipages en **factures distinctes**,
- détecter la présence d’un **Bulletin de Versement (BV)**,
- lire le **QR code** du BV,
- extraire les champs importants : **référence**, **montant**, **devise**.

## 🚀 Principales applications

- `app_prediction_yolo_v30_1.py`  
  Détection des tampons + découpage des factures.

- `app_bv_only.py`  
  Détection des Bulletins de Versement (BV) + extraction QR / Référence / Montant / Devise.

- `app_apprentissage_ML.py`  
  Interface d’apprentissage / expérimentation ML.

## 🏗 Installation

```bash
git clone https://github.com/jgm-mediasoft/facture-splitter.git
cd facture-splitter

python -m venv venv
# Windows
venv\Scripts\activate

pip install -r requirements.txt


streamlit run app_bv_only.py


streamlit run app_prediction_yolo_v20.py

