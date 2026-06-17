import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Page Configuration ---
st.set_page_config(page_title="Mourabaha Portfolio Optimizer", layout="wide")

# --- App Header ---
st.title("📈 Mourabaha Portfolio Optimization Dashboard")
st.markdown("""
This application utilizes **Quadratic Programming (SLSQP)** to determine the optimal asset allocation 
for a Sharia-compliant Mourabaha portfolio, based on historical financial data (2019-2025).
""")

# --- Sidebar Information ---
st.sidebar.header("Project Details")
st.sidebar.write("**Master Program:** Engineering in Participatory Finance and AI")
st.sidebar.write("**Team Members:** Soukaina, Asma, Abdelouadoud, Mohammed")
st.sidebar.write("**Supervisor:** Dr. Asmae Faris")

# --- Data Loading ---
@st.cache_data
def load_financial_data():
    # Using the exact filenames present in your GitHub repository
    file_mu = 'Murabaha_Quadratic_Ready_Data_2019_2025 (1).xlsx - Expected_Returns_mu.csv'
    file_sigma = 'Murabaha_Quadratic_Ready_Data_2019_2025 (1).xlsx - Covariance_Matrix_Sigma.csv'
    
    try:
        # Loading CSV files
        mu = pd.read_csv(file_mu, index_col=0)
        sigma = pd.read_csv(file_sigma, index_col=0)
        return mu, sigma
    except FileNotFoundError:
        st.error(f"Error: Could not find '{file_mu}' or '{file_sigma}'. "
                 "Please ensure these files are in the same directory as app.py in your GitHub repository.")
        return None, None

mu, sigma = load_financial_data()

# --- Logic ---
if mu is not None and sigma is not None:
    st.subheader("Asset Expected Returns")
    st.write(mu)

    if st.button("Run Portfolio Optimization"):
        # We assume the index of 'mu' contains the asset names
        num_assets = len(mu)
        
        # Optimization Constraints: Weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Optimization Bounds: No short-selling (0 to 1)
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = [1./num_assets] * num_assets
        
        # Objective Function: Minimize Portfolio Risk (Volatility)
        # Using the covariance matrix 'sigma'
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(sigma.values, weights)))
        
        # Solving
        result = minimize(objective, initial_guess, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        
        # Display Results
        st.subheader("Optimal Portfolio Allocation")
        results_df = pd.DataFrame({'Asset': mu.index, 'Weight': optimal_weights})
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Weight', y='Asset', data=results_df, palette="viridis", ax=ax)
        st.pyplot(fig)
        
        # Table of results
        results_df['Weight (%)'] = (results_df['Weight'] * 100).round(2)
        st.write(results_df[['Asset', 'Weight (%)']])

# --- Footer ---
st.markdown("---")
st.caption("Research conducted under the framework of Participatory Finance and AI Engineering.")
