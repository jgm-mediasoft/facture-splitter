
---

## 3️⃣ Contenu de `CONTRIBUTING.md`

Dans `C:\ml\facture_splitter\CONTRIBUTING.md` :

```markdown
# Guide de contribution – Facture Splitter

## 🌿 Branches

- `prod`  
  Branche **stable**, utilisée comme référence de production.  
  ✅ Mise à jour uniquement via Pull Request depuis `main`.

- `main`  
  Branche d’intégration.  
  ✅ Mise à jour via Pull Request depuis les branches de test.

- `test_jgm`  
  Branche personnelle de **Jean-Gabriel**.

- `test_aa`  
  Branche personnelle d’**Azeem**.

## 🔁 Workflow classique

1. Chaque développeur travaille sur **sa branche perso** :
   - Jean-Gabriel : `test_jgm`
   - Azeem : `test_aa`

2. Une fois une fonctionnalité prête :
   - ouvrir un **Pull Request** vers `main`
   - faire les tests / revue
   - merger dans `main` quand c’est validé

3. Quand `main` est stable :
   - ouvrir un Pull Request `main → prod`
   - tagger une nouvelle version (`v1.0.0`, `v1.1.0`, ...)

## 🧪 Tests avant Pull Request

Avant de créer un PR vers `main` :

- Tester l’application sur plusieurs PDF
- Vérifier :
  - découpage des factures correct
  - détection des BV
  - extraction correcte de la Référence / Montant / Devise

## 📝 Style des commits

Format recommandé :

```text
[BV] amélioration détection pages 12 et 13
[YOLO] ajustement seuil de confiance
[OCR] nettoyage du texte BV
