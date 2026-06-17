import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.optimize import minimize

# --- Configuration de la page ---
st.set_page_config(page_title="Optimisation de Portefeuille Mourabaha", layout="wide")
st.title("📈 Tableau de Bord Mourabaha")

# --- Chargement des données ---
@st.cache_data
def load_data():
    # on_bad_lines='skip' ignore les lignes mal formatées
    # encoding='latin-1' gère les caractères spéciaux
    mu = pd.read_csv('mu.csv', index_col=0, encoding='latin-1', on_bad_lines='skip', engine='python')
    sigma = pd.read_csv('sigma.csv', index_col=0, encoding='latin-1', on_bad_lines='skip', engine='python')
    return mu, sigma

# --- Logique de l'application ---
try:
    mu, sigma = load_data()
    
    if st.button("Lancer l'optimisation"):
        num_assets = len(mu)
        
        # Contraintes : Somme des poids égale à 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # Bornes : Entre 0 et 1 (Pas de vente à découvert)
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = [1./num_assets] * num_assets
        
        # Objectif : Minimiser le risque (volatilité)
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(sigma.values, weights)))
        
        result = minimize(objective, initial_guess, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        
        # Résultats
        st.subheader("Allocation optimale du portefeuille")
        results = pd.DataFrame({'Actif': mu.index, 'Poids': result.x})
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Poids', y='Actif', data=results, palette="viridis", ax=ax)
        st.pyplot(fig)
        
        results['Poids (%)'] = (results['Poids'] * 100).round(2)
        st.write(results[['Actif', 'Poids (%)']])

except Exception as e:
    st.error("Erreur lors du chargement des données.")
    st.write("Vérifiez que 'mu.csv' et 'sigma.csv' sont dans le dossier racine et bien formés.")
    st.write("Détails techniques :", e)
