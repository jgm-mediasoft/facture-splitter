# Guide rapide – Azeem

## 1. Cloner le projet

git clone https://github.com/jgm-mediasoft/facture-splitter.git
cd facture-splitter


## 2. Créer l’environnement Python

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


## 3. Travailler dans ta branche

git checkout test_aa


## 4. Sauvegarder ton travail

git add .
git commit -m "Ton message de commit"
git push


## 5. Intégrer dans main

Quand une fonctionnalité est prête :

Aller sur GitHub

Créer un Pull Request :

base : main

compare : test_aa

Jean-Gabriel valide / merge dans main.