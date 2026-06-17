# Mourabaha Portfolio Optimization

## 📝 Description
Ce projet constitue un cadre méthodologique pour l'optimisation de portefeuille d'actifs **Mourabaha** (finance participative) au Maroc sur la période 2019-2025. Il applique la théorie moderne du portefeuille (Markowitz) en intégrant les contraintes spécifiques de conformité à la Charia.

## 👥 Équipe de recherche
Projet réalisé par les étudiants de la 1ère année Master **Ingénierie de la Finance Participative et l'IA** :
- **Soukaina Hlal**
- **Asma Daaou**
- **Abdelouadoud Elkhalfi**
- **Mohammed Elgharb**

**Encadré par :** [Dr.Asmae Faris]

## 🏗 Méthodologie
1. **Prétraitement :** Nettoyage et Winsorization des données pour atténuer les valeurs aberrantes.
2. **Modélisation :** Utilisation de l'algorithme d'optimisation sous contraintes **SLSQP** (Sequential Least Squares Programming).
3. **Analyse :** Évaluation des poids optimaux permettant de maximiser le rendement sous contrainte de non-négativité ($\sum w_i = 1$).

## 🛠 Prérequis
Ce projet nécessite les bibliothèques suivantes :
- `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`

---
*Ce dépôt est une documentation technique accompagnant les travaux de recherche du Master Ingénierie de la Finance Participative et l'IA.*
