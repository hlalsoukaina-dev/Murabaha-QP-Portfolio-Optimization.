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
st.markdown("""
This application utilizes **Quadratic Programming (SLSQP)** to determine the optimal asset allocation 
for a Sharia-compliant Mourabaha portfolio, based on historical financial data (2019-2025).
""")

# --- Sidebar ---
st.sidebar.header("Project Details")
st.sidebar.write("**Master Program:** Engineering in Participatory Finance and AI")
st.sidebar.write("**Team Members:** Soukaina, Asma, Abdelouadoud, Mohammed")
st.sidebar.write("**Supervisor:** Dr. Asmae Faris")

# --- Data Loading ---
@st.cache_data
def load_data():
    # Filenames as they appear in the repository
    mu_file = 'Murabaha_Quadratic_Ready_Data_2019_2025 (1).xlsx - Expected_Returns_mu.csv'
    sigma_file = 'Murabaha_Quadratic_Ready_Data_2019_2025 (1).xlsx - Covariance_Matrix_Sigma.csv'
    
    try:
        mu = pd.read_csv(mu_file, index_col=0)
        sigma = pd.read_csv(sigma_file, index_col=0)
        return mu, sigma
    except FileNotFoundError:
        return None, None

# --- Main Logic ---
mu, sigma = load_data()

if mu is not None and sigma is not None:
    if st.button("Run Portfolio Optimization"):
        num_assets = len(mu)
        
        # Constraints: Weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # Bounds: 0 to 1
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = [1./num_assets] * num_assets
        
        # Objective: Minimize Risk
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(sigma.values, weights)))
        
        result = minimize(objective, initial_guess, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        
        # Display Results
        st.subheader("Optimal Portfolio Allocation")
        results = pd.DataFrame({'Asset': mu.index, 'Weight': result.x})
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Weight', y='Asset', data=results, palette="viridis", ax=ax)
        st.pyplot(fig)
        
        st.write(results)
else:
    st.error("Error: Could not load the CSV files. Please ensure the files are in the root directory and names match exactly.")
