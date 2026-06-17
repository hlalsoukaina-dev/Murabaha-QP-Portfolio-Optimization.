import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# --- Page Configuration ---
st.set_page_config(page_title="Mourabaha Portfolio Optimizer", layout="wide")

st.title("📈 Mourabaha Portfolio Optimization Dashboard")

# --- Automated Data Loading ---
@st.cache_data
def load_data():
    # هذا الكود كايقلب فـ "كاع" الملفات اللي كاينين فـ الـ Repository
    # باش يلقى الملفات اللي كيبداو بكلمة Expected و Covariance
    files = os.listdir('.')
    mu_file = [f for f in files if 'Expected_Returns_mu' in f][0]
    sigma_file = [f for f in files if 'Covariance_Matrix_Sigma' in f][0]
    
    mu = pd.read_csv(mu_file, index_col=0)
    sigma = pd.read_csv(sigma_file, index_col=0)
    return mu, sigma

try:
    mu, sigma = load_data()
    
    # --- Optimization Logic ---
    if st.button("Run Portfolio Optimization"):
        num_assets = len(mu)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = [1./num_assets] * num_assets
        
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(sigma.values, weights)))
        
        result = minimize(objective, initial_guess, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        
        # --- Results ---
        st.subheader("Optimal Portfolio Allocation")
        results_df = pd.DataFrame({'Asset': mu.index, 'Weight': result.x})
        
        fig, ax = plt.subplots()
        sns.barplot(x='Weight', y='Asset', data=results_df, palette="viridis", ax=ax)
        st.pyplot(fig)
        
        st.write(results_df)

except Exception as e:
    st.error("Error: Could not load data automatically. Please ensure files are in the main directory.")
    st.write("Technical details:", e)
